param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
# Exercise the actual embedded inspector's admission and identity contracts.
# OS termination/HTTP are replaced with independent fixtures; no model is run.
$root = Split-Path -Parent $PSScriptRoot
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'install.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Installer parse failure.' }
$embedded = @($ast.FindAll({ param($node)
    $node -is [Management.Automation.Language.StringConstantExpressionAst] -and
    $node.Value.StartsWith('"""Embedded by the installer;')
}, $true))
if ($embedded.Count -ne 1) { throw 'Expected one shipping service inspector.' }
$directory = Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-update-contract-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($directory) | Out-Null
$modulePath = Join-Path $directory 'inspector.py'
$testPath = Join-Path $directory 'contracts.py'
$tests = @'
import pathlib,sys,unittest,tempfile,os,shutil
from unittest import mock
n={'__name__':'installer_contract_fixture'}
exec(compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),'shipping-inspector','exec'),n)
Path=pathlib.Path
class Contracts(unittest.TestCase):
 def test_ui_revision_uses_shipping_launcher_list_including_calculator_and_web_page(self):
  import importlib.util
  shipping=Path(sys.argv[2]);spec=importlib.util.spec_from_file_location('shipping_launcher_fixture',shipping/'ui/launch.py')
  launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
  with tempfile.TemporaryDirectory() as folder:
   source=Path(folder);ui=source/'ui';ui.mkdir()
   for name in launcher.UI_BACKEND_FILES:(ui/name).write_text('private revision fixture '+name,encoding='utf-8')
   shutil.copyfile(shipping/'ui/launch.py',ui/'launch.py')
   expected=launcher.backend_revision(source)
   self.assertEqual(n['revision'](source),expected)
   for name in ('calculator.py','web_page.py'):
    path=ui/name;original=path.read_bytes();path.write_bytes(original+b' changed')
    self.assertNotEqual(n['revision'](source),expected)
    self.assertEqual(n['revision'](source),launcher.backend_revision(source))
    path.write_bytes(original)
 def test_current_os_image_handles_store_alias_without_accepting_other_python_images(self):
  image=n['psutil'].Process(os.getpid()).exe();source=Path.cwd()
  expected=[sys.executable,'-B','server.py','--port','19191']
  actual=[image,*expected[1:]]
  metadata={'command':actual,'cwd':str(source),'executable':image}
  with mock.patch.object(sys,'_base_executable',str(source/'app-execution-alias/python.exe')):
   n['verify_process'](metadata,expected,source,Path(sys.executable))
   with tempfile.TemporaryDirectory() as folder:
    foreign=Path(folder)/'python.exe';foreign.write_bytes(b'not this interpreter image')
    with self.assertRaisesRegex(RuntimeError,'interpreter image'):
     n['verify_process']({**metadata,'executable':str(foreign)},expected,source,Path(sys.executable))
    self.assertFalse(n['same_invocation']([str(foreign),*actual[1:]],expected,Path(sys.executable)))
 def test_redirector_changes_only_the_verified_interpreter(self):
  expected=[sys.executable,'-B','server.py','--port','19191']
  actual=[sys._base_executable,*expected[1:]]
  self.assertTrue(n['same_invocation'](actual,expected,Path(sys.executable)))
  self.assertFalse(n['same_invocation']([*actual[:-1],'19192'],expected,Path(sys.executable)))
  self.assertFalse(n['same_invocation'](['unknown.exe',*actual[1:]],expected,Path(sys.executable)))
 def test_unrecognized_scripts_options_and_extra_arguments_are_rejected(self):
  source=Path.cwd()
  for command,kind in (([sys.executable,'-B','-m','another.server'],'api'),
                       ([sys.executable,'-B',str(source/'other.py')],'ui'),
                       ([sys.executable,'-B','-m','serve.server','--unknown','x'],'api'),
                       ([sys.executable,'-B','-m','serve.server','--port'],'api')):
   with self.subTest(command=command),self.assertRaises(RuntimeError):n['options'](command,kind,source)
 def responses(self,*,status=None,control=None,library=None,job=None):
  seen=[]
  def request(port,path,*args,**kwargs):
   seen.append(path)
   if path=='/status':return status or {'busy':False,'queued':0}
   if path=='/v1/model/switch':return control or {'switching':False,'in_flight':0,'downloading':False}
   if path=='/v1/models':return library or {'data':[]}
   if path.startswith('/v1/model/download?model='):return job or {'state':'ready'}
   raise AssertionError('No mutation or unrelated endpoint is allowed: '+path)
  return request,seen
 def api(self):return {'Kind':'api','Port':19191,'OriginalConfig':'fixture.json'}
 def test_busy_queued_leases_and_downloads_refuse_without_mutations(self):
  for status,control in (({'busy':True},{'downloading':False}),({'queued':2},{'downloading':False}),
                         ({},{'in_flight':1}),({},{'switching':True}),({},{'downloading':True})):
   request,seen=self.responses(status=status,control=control)
   with self.subTest(status=status,control=control),mock.patch.dict(n,request=request),self.assertRaises(RuntimeError):n['idle'](self.api())
   self.assertTrue(all(not p.startswith('/unload') for p in seen))
 def test_legacy_active_download_is_not_hidden_by_active_delete_lock(self):
  request,seen=self.responses(control={'in_flight':0,'switching':False},
      library={'data':[{'id':'gemma-4-26b-a4b-it','management':{'delete_reason':'active'}}]},job={'state':'verifying'})
  with mock.patch.dict(n,request=request),self.assertRaisesRegex(RuntimeError,'download'):n['idle'](self.api())
  self.assertIn('/v1/model/download?model=gemma-4-26b-a4b-it',seen)
 def test_empty_previous_snapshot_does_not_start_any_service(self):
  source=Path.cwd();work=source.parent/'work-fixture'
  with mock.patch.dict(n,listeners=mock.Mock(side_effect=AssertionError('No port probe needed'))),mock.patch.object(n['subprocess'],'Popen') as spawn:
   result=n['resume'](source,work,{'SourceRoot':str(source),'WorkRoot':str(work),'Apis':[],'Uis':[]})
  self.assertEqual(result,[]);spawn.assert_not_called()
 def test_foreign_replacement_listener_is_preserved(self):
  source=Path.cwd();work=source.parent/'work-fixture'
  snapshot={'SourceRoot':str(source),'WorkRoot':str(work),'Apis':[{'Kind':'api','Port':19191}],'Uis':[]}
  with mock.patch.dict(n,listeners=lambda port:{99},collect=lambda source,work:[]),mock.patch.object(n['subprocess'],'Popen') as spawn,self.assertRaisesRegex(RuntimeError,'another service'):
   n['resume'](source,work,snapshot)
  spawn.assert_not_called()
 def test_all_admission_checks_precede_unload_or_termination(self):
  items=[{'Kind':'api','Port':19191,'Processes':[]},{'Kind':'api','Port':19192,'Processes':[]}]
  def idle(item):
   if item['Port']==19192:raise RuntimeError('second service busy')
  request=mock.Mock(side_effect=AssertionError('No unload before all idle checks'))
  with mock.patch.dict(n,collect=lambda s,w:items,idle=idle,request=request),self.assertRaisesRegex(RuntimeError,'second service busy') as raised:n['stop'](Path.cwd(),Path.cwd().parent/'work-fixture')
  request.assert_not_called();self.assertEqual(len(raised.exception.snapshot['Apis']),2)
 def test_partial_stop_failure_retains_snapshot_and_releases_exact_handles(self):
  class Kernel:
   def __init__(self):self.closed=[]
   def WaitForSingleObject(self,*args):return 258
   def TerminateProcess(self,*args):return 0
   def CloseHandle(self,handle):self.closed.append(handle)
  k=Kernel();m={'pid':42};item={'Kind':'ui','Port':19191,'Processes':[m]}
  with mock.patch.dict(n,collect=lambda s,w:[item],idle=lambda i:None,hold=lambda m:(k,123),process=lambda pid:object(),meta=lambda p:m,active_children=lambda p,known:[]),self.assertRaisesRegex(RuntimeError,'did not stop') as raised:n['stop'](Path.cwd(),Path.cwd().parent/'work-fixture')
  self.assertEqual(raised.exception.snapshot['Uis'],[item]);self.assertEqual(k.closed,[123])
 def stop_api_fixture(self,*,loaded=False,resident=False,unload_status='unloaded',remove_children=True,late_child=False,busy=False):
  events=[];state={'stopped':False,'resident':resident,'reads':0}
  child=mock.Mock(pid=99);child.name.return_value='tinystrata-vision.exe'
  m={'pid':42};item={'Kind':'api','Port':19191,'OriginalConfig':'fixture.json','Loaded':loaded,'Processes':[m]}
  class Kernel:
   def WaitForSingleObject(self,*args):return 0 if state['stopped'] else 258
   def TerminateProcess(self,*args):events.append('terminate');state['stopped']=True;return 1
   def CloseHandle(self,handle):events.append('close')
  def request(port,path,*args,**kwargs):
   events.append(path)
   if path=='/status':return {'busy':busy,'queued':0}
   if path=='/v1/model/switch':return {'in_flight':0,'switching':False,'downloading':False}
   if path=='/v1/models':return {'data':[]}
   if path=='/unload':
    if remove_children:state['resident']=False
    return {'status':unload_status}
   raise AssertionError('Unexpected update request: '+path)
  def children(p,known):
   self.assertEqual(known,{42});state['reads']+=1
   return [child] if state['resident'] or (late_child and state['reads']>1) else []
  with mock.patch.dict(n,collect=lambda s,w:[item],request=request,hold=lambda m:(Kernel(),123),
       process=lambda pid:object(),meta=lambda p:m,active_children=children,
       drain=lambda items:[n['idle'](x) for x in items]):
   try:result=n['stop'](Path.cwd(),Path.cwd().parent/'work-fixture');error=None
   except RuntimeError as caught:result=None;error=caught
  return result,error,events
 def test_unloaded_lazy_or_bootstrap_api_needs_no_unload_endpoint(self):
  result,error,events=self.stop_api_fixture(unload_status='unsupported')
  self.assertIsNone(error);self.assertTrue(result['Stopped']);self.assertFalse(result['Api']['Loaded'])
  self.assertNotIn('/unload',events);self.assertEqual(events.count('terminate'),1);self.assertEqual(events.count('close'),1)
  self.assertEqual(events.count('/status'),4)
 def test_loaded_api_still_unloads_even_if_no_child_is_visible(self):
  result,error,events=self.stop_api_fixture(loaded=True)
  self.assertIsNone(error);self.assertTrue(result['Stopped']);self.assertIn('/unload',events)
  self.assertLess(events.index('/unload'),events.index('terminate'))
 def test_unloaded_api_with_resident_vision_or_native_child_still_unloads(self):
  result,error,events=self.stop_api_fixture(resident=True)
  self.assertIsNone(error);self.assertTrue(result['Stopped']);self.assertIn('/unload',events)
  self.assertLess(events.index('/unload'),events.index('terminate'))
 def test_resident_child_after_not_loaded_response_blocks_termination(self):
  result,error,events=self.stop_api_fixture(resident=True,unload_status='not loaded',remove_children=False)
  self.assertIsNone(result);self.assertIn('active child processes',str(error));self.assertIn('/unload',events)
  self.assertNotIn('terminate',events);self.assertEqual(events.count('close'),1)
 def test_unloaded_api_with_unsupported_unload_and_live_child_is_not_killed(self):
  result,error,events=self.stop_api_fixture(resident=True,unload_status='unsupported')
  self.assertIsNone(result);self.assertIn('did not unload safely',str(error));self.assertNotIn('terminate',events)
  self.assertEqual(events.count('close'),1)
 def test_child_appearing_after_unloaded_check_still_blocks_termination(self):
  result,error,events=self.stop_api_fixture(late_child=True)
  self.assertIsNone(result);self.assertIn('active child processes',str(error))
  self.assertNotIn('/unload',events);self.assertNotIn('terminate',events);self.assertEqual(events.count('close'),1)
 def test_unloaded_api_still_refuses_busy_work_before_any_mutation(self):
  result,error,events=self.stop_api_fixture(busy=True)
  self.assertIsNone(result);self.assertIn('busy',str(error));self.assertNotIn('/unload',events);self.assertNotIn('terminate',events)
 def test_active_jobs_drain_under_guard_without_cancelling_or_starting_work(self):
  events=[];item=self.api();clock=iter([0.,.1,.2])
  def idle(item):
   events.append('read-only-idle')
   if len(events)<3:raise n['BusyServer']('existing generation/download')
  with mock.patch.dict(n,idle=idle),mock.patch.object(n['time'],'monotonic',side_effect=lambda:next(clock)),mock.patch.object(n['time'],'sleep') as sleep:
   n['drain']([item],timeout=1)
  self.assertEqual(events,['read-only-idle']*3);self.assertEqual(sleep.call_count,2)
 def test_job_timeout_retains_every_service_without_unload_or_termination(self):
  with mock.patch.dict(n,idle=mock.Mock(side_effect=n['BusyServer']('still generating'))),mock.patch.object(n['time'],'monotonic',side_effect=[0.,2.]),mock.patch.object(n['time'],'sleep') as sleep,self.assertRaises(n['BusyServer']):
   n['drain']([self.api()],timeout=1)
  sleep.assert_not_called()
 def test_lazy_flag_is_preserved_without_accepting_unknown_switches(self):
  command=[sys.executable,'-B','-m','serve.server','--engine','tinystrata','--config','x','--lazy','--port','19191']
  self.assertIs(n['options'](command,'api',Path.cwd())['--lazy'],True)
  with self.assertRaises(RuntimeError):n['options'](command+['--lazy'],'api',Path.cwd())
 def test_collect_shared_root_keeps_intel_active_profile_unloaded_and_both_ui_directories(self):
  import json,datetime
  with tempfile.TemporaryDirectory() as folder:
   source=Path(folder)/'source';work=Path(folder)/'work';runtime=work/'runtime';configs=runtime/'.tinystrata';configs.mkdir(parents=True);source.mkdir()
   python=runtime/'.venv/Scripts/python.exe';created=1000.;stamp=datetime.datetime.fromtimestamp(created,datetime.timezone.utc).isoformat()
   python.parent.mkdir(parents=True);python.write_bytes(b'inert fixture identity; never executed')
   records={};health={};states=[]
   for pid,backend,config_name,model,port in ((10,'sycl','config-sycl.json','qwen3.6-35b-a3b',18085),(20,'cuda','config.json','qwen3.6-35b-a3b',18080)):
    config=configs/config_name;config.write_text(json.dumps({'backend':backend,'model_name':model,'intel_profile':'b390-qwen36'}))
    state=runtime/('server-state-sycl.json' if backend=='sycl' else 'server-state.json')
    state.write_text(json.dumps({'pid':pid,'started':stamp,'source':str(source),'config':str(config),'port':port}))
    command=[str(python),'-B','-m','serve.server','--engine','tinystrata','--config',str(config),'--port',str(port)]
    records[pid]={'pid':pid,'parent':1,'started':created,'command':command,'cwd':str(source),'executable':str(python)}
    health[port]={'status':'ok','service':'strata','model':model,'loaded':True}
    if backend=='sycl':health[port].update(backend='sycl',model='tiel-coder-35b-a3b',intel_profile='b390-tiel-coder',loaded=False)
   (configs/'config-tiel-coder-sycl.json').write_text(json.dumps({'backend':'sycl','model_name':'tiel-coder-35b-a3b','intel_profile':'b390-tiel-coder'}))
   for pid,port,api,data in ((30,3005,18085,'ui-sycl'),(40,3000,18080,'ui')):
    directory=runtime/data;directory.mkdir()
    command=[str(python),'-B',str(source/'ui/server.py'),'--port',str(port),'--api-port',str(api),'--data-dir',str(directory)]
    (directory/'ui-process.json').write_text(json.dumps({'pid':pid,'started':stamp,'command':command}))
    records[pid]={'pid':pid,'parent':1,'started':created,'command':command,'cwd':str(source),'executable':str(python)}
    health[port]={'app':'tinystrata-ui','pid':pid,'api_port':api,'source_dir':str(source),'data_dir':str(directory)}
   with mock.patch.dict(n,process=lambda pid:pid if pid in records else None,meta=lambda p:records[p],listeners=lambda port:{next(pid for pid,r in records.items() if r['command'][r['command'].index('--port')+1]==str(port))},
       request=lambda port,*a:health[port],verify_process=mock.Mock(),verify_connection_settings=mock.Mock()):
    result=n['collect'](source,work)
   intel=next(x for x in result if x.get('Backend')=='sycl' and x['Kind']=='api')
   self.assertEqual(intel['Model'],'tiel-coder-35b-a3b');self.assertEqual(intel['Family'],'tiel');self.assertFalse(intel['Loaded'])
   self.assertEqual(Path(intel['Config']).name,'config-tiel-coder-sycl.json');self.assertEqual(Path(intel['OriginalConfig']).name,'config-sycl.json')
   self.assertEqual(Path(intel['StatePath']).name,'server-state-tiel-coder-sycl.json')
   self.assertEqual({Path(x['DataDir']).name for x in result if x['Kind']=='ui'},{'ui','ui-sycl'})
   self.assertEqual({x['Port'] for x in result},{18080,18085,3000,3005})
 def test_resume_unloaded_intel_uses_saved_active_profile_and_lazy_without_loading(self):
  with tempfile.TemporaryDirectory() as folder:
   source=Path(folder)/'source';work=Path(folder)/'work';source.mkdir();config=work/'runtime/.tinystrata/config-tiel-coder-sycl.json';config.parent.mkdir(parents=True);config.write_text('{}')
   item={'Kind':'api','Port':19191,'Config':str(config),'StatePath':str(work/'runtime/server-state-tiel-coder-sycl.json'),'Model':'tiel-coder-35b-a3b','Loaded':False,'Backend':'sycl','IntelProfile':'b390-tiel-coder'}
   snapshot={'SourceRoot':str(source),'WorkRoot':str(work),'Apis':[item],'Uis':[]}
   child=mock.Mock(pid=88);child.poll.return_value=None
   with mock.patch.dict(n,listeners=lambda port:set(),collect=lambda s,w:[item],request=mock.Mock(side_effect=AssertionError('An unloaded snapshot must not POST /load'))),mock.patch.object(n['subprocess'],'Popen',return_value=child) as spawn,mock.patch.object(n['psutil'],'Process') as process:
    process.return_value.create_time.return_value=1000.
    result=n['resume'](source,work,snapshot,'receipt')
   command=spawn.call_args.args[0];self.assertIn('--lazy',command);self.assertEqual(command[command.index('--config')+1],str(config));self.assertFalse(result[0]['reused'])
   self.assertEqual(n['read'](item['StatePath'])['api_source_fingerprint'],'receipt')
 def test_verified_redirector_and_console_do_not_hide_a_resident_vision_child(self):
  known=mock.Mock(pid=43)
  console=mock.Mock(pid=44);console.ppid.return_value=42
  console.exe.return_value=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/conhost.exe')
  vision=mock.Mock(pid=45);vision.ppid.return_value=42;vision.exe.return_value='C:/TinyStrata/tinystrata-vision.exe'
  parent=mock.Mock();parent.children.return_value=[known,console,vision]
  self.assertEqual(n['active_children'](parent,{42,43}),[vision])
 def test_different_profile_connection_fields_fail_closed_without_logging_values(self):
  for field,before,after in (('api_key','original-secret','different-secret'),
      ('allowed_hosts',['original.local'],['different.local']),('cors_origins',['https://original.local'],[]),
      ('trusted_origins',[],['https://different.local'])):
   owner=mock.Mock()
   with self.subTest(field=field),mock.patch.dict(n,read=lambda p:{field:before if p=='original' else after}),self.assertRaisesRegex(RuntimeError,'different per-profile connection settings') as raised:
    n['verify_connection_settings']('original','active',owner)
   self.assertNotIn(str(before),str(raised.exception));self.assertNotIn(str(after),str(raised.exception))
   owner.environ.assert_not_called()
 def test_empty_profile_auth_and_equivalent_config_backed_environment_are_preserved(self):
  for before,after in (({}, {'api_key':''}),({'api_key':None},{'api_key':[]})):
   with self.subTest(before=before,after=after),mock.patch.dict(n,read=lambda p:before if p=='original' else after),mock.patch.dict(n['os'].environ,{'STRATA_API_KEY':'','STRATA_ALLOWED_HOSTS':''}):
    n['verify_connection_settings']('original','active',mock.Mock(environ=lambda:{}))
  cfg={'api_key':'same-secret','allowed_hosts':['safe.local']}
  with mock.patch.dict(n,read=lambda p:cfg),mock.patch.dict(n['os'].environ,{'STRATA_API_KEY':'','STRATA_ALLOWED_HOSTS':''}):
   n['verify_connection_settings']('original','active',mock.Mock(environ=lambda:{'STRATA_API_KEY':'same-secret','STRATA_ALLOWED_HOSTS':'safe.local'}))
 def test_different_inherited_connection_settings_are_refused_without_secret_values(self):
  cfg={'api_key':'profile-secret','allowed_hosts':['safe.local']}
  for environment in ({'STRATA_API_KEY':'inherited-secret'}, {'STRATA_ALLOWED_HOSTS':'extra.local'}):
   with self.subTest(environment=environment),mock.patch.dict(n,read=lambda p:cfg),mock.patch.dict(n['os'].environ,{'STRATA_API_KEY':'','STRATA_ALLOWED_HOSTS':''}),self.assertRaisesRegex(RuntimeError,'different inherited connection settings') as raised:
    n['verify_connection_settings']('original','active',mock.Mock(environ=lambda:environment))
   for value in environment.values():self.assertNotIn(value,str(raised.exception))
unittest.main(argv=[sys.argv[0]])
'@
try {
    [IO.File]::WriteAllText($modulePath, $embedded[0].Value, (New-Object Text.UTF8Encoding($false)))
    [IO.File]::WriteAllText($testPath, $tests, (New-Object Text.UTF8Encoding($false)))
    & $Python -I -X utf8 -B $testPath $modulePath $root
    if ($LASTEXITCODE -ne 0) { throw 'Service update contract regression failed.' }
    # The Desktop and installer write/read the same receipt. Compare the two
    # actual implementations over a Unicode fixture, including a template and
    # upper-case Python extension, without starting any application.
    $definition = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-TinyStrataApiSourceFingerprint'}, $true)
    Invoke-Expression $definition.Extent.Text
    $receiptRoot = Join-Path $directory ('receipt-' + [char]0x65E5)
    [IO.Directory]::CreateDirectory((Join-Path $receiptRoot 'serve')) | Out-Null
    foreach ($name in @('requirements.txt','run.py','serve/z.py','serve/A.PY','serve/chat.jinja')) {
        [IO.File]::WriteAllText((Join-Path $receiptRoot $name), ('receipt fixture ' + $name), (New-Object Text.UTF8Encoding($false)))
    }
    $pythonReceipt = @(& $Python -I -B -c 'import sys; sys.path.insert(0,sys.argv[1]); from ui.launch import api_source_fingerprint; print(api_source_fingerprint(sys.argv[2]))' $root $receiptRoot)
    if ($LASTEXITCODE -ne 0 -or ($pythonReceipt -join '').Trim() -cne (Get-TinyStrataApiSourceFingerprint $receiptRoot)) { throw 'Desktop and installer API source receipts differ.' }
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $receiptRoot 'serve') -File) { [IO.File]::Delete($file.FullName) }
    [IO.Directory]::Delete((Join-Path $receiptRoot 'serve'))
    foreach ($name in @('requirements.txt','run.py')) { [IO.File]::Delete((Join-Path $receiptRoot $name)) }
    [IO.Directory]::Delete($receiptRoot)
} finally {
    foreach ($path in @($modulePath, $testPath)) { if (Test-Path -LiteralPath $path -PathType Leaf) { [IO.File]::Delete($path) } }
    if (Test-Path -LiteralPath $directory -PathType Container) { [IO.Directory]::Delete($directory) }
}
