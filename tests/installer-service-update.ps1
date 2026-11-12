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
import pathlib,sys,unittest
from unittest import mock
n={'__name__':'installer_contract_fixture'}
exec(compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),'shipping-inspector','exec'),n)
Path=pathlib.Path
class Contracts(unittest.TestCase):
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
    & $Python -I -X utf8 -B $testPath $modulePath
    if ($LASTEXITCODE -ne 0) { throw 'Service update contract regression failed.' }
} finally {
    foreach ($path in @($modulePath, $testPath)) { if (Test-Path -LiteralPath $path -PathType Leaf) { [IO.File]::Delete($path) } }
    if (Test-Path -LiteralPath $directory -PathType Container) { [IO.Directory]::Delete($directory) }
}
