"""CPU-only managed Clover decode policy regression; no downloads or native processes."""
from __future__ import annotations
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
import run as launcher
from serve import backend, config, model_setup, tinyconfig
from serve.model_profiles import ModelProfiles


def stock(**changes):
    return {"arch":"clover_moe","model_source":"managed","vision_mode":"no",
            "args":["--model","fixture.gguf","--kv-type","auto","--clover-mode","causal","--clover-draft","0"],
            "sampling":dict(tinyconfig.CLOVER_SAMPLING),**changes}


class DecodeCompatibility(unittest.TestCase):
    def test_stock_saved_profiles_keep_original_defaults(self):
        for origin in ('managed','local',None):
            original=stock(model_source=origin,env={'USER_FIXTURE':'keep'})
            before=copy.deepcopy(original)
            new,_=tinyconfig.migrate_settings(original)
            self.assertEqual(original,before)
            self.assertEqual(config.value_of(new,'clover_mode'),'causal')
            self.assertEqual(config.value_of(new,'clover_draft'),0)
            self.assertEqual(config.value_of(new,'kv_type'),'auto')
            self.assertEqual(new['env'],original['env'])
            self.assertNotIn('--clover-accept',new['args'])
            self.assertEqual(new['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
            again,changed=tinyconfig.migrate_settings(new)
            self.assertFalse(changed);self.assertEqual(again,new)

    def test_custom_settings_and_environment_are_retained(self):
        for choices in (dict(clover_mode='self-spec',clover_draft=3,clover_accept='exact'),
                        dict(clover_mode='self-spec',clover_draft=5,clover_accept='threshold',clover_accept_delta=.03),
                        dict(clover_mode='diffusion',clover_draft=0,clover_block_size=4,clover_threshold=.95)):
            original,_=config.apply(stock(env={'STRATA_IQ_MT_MIN':'1','USER_FIXTURE':'keep'}),choices)
            new,_=tinyconfig.migrate_settings(original)
            for key,value in choices.items():self.assertEqual(config.value_of(new,key),value)
            self.assertEqual(config.value_of(new,'kv_type'),'auto')
            self.assertEqual(new['env'],original['env'])

    def test_legacy_managed_stock_signature_upgrades_once(self):
        # Historical same-value manual edits had no provenance and cannot be
        # distinguished from the stock tuple. Only this narrow tuple upgrades.
        original=stock()
        original['args']+=['--max-context','auto','--chunk','auto']
        before=copy.deepcopy(original)
        with patch.dict(os.environ,{},clear=True):new,changed=tinyconfig.migrate_settings(original)
        self.assertTrue(changed);self.assertEqual(original,before)
        self.assertEqual(config.value_of(new,'clover_mode'),'self-spec')
        self.assertEqual(config.value_of(new,'clover_draft'),3)
        self.assertEqual(config.value_of(new,'clover_accept'),'exact')
        self.assertEqual(config.value_of(new,'kv_type'),'int8')
        self.assertEqual(new['env'],{'STRATA_IQ_MT_MIN':'1'})
        again,changed=tinyconfig.migrate_settings(new)
        self.assertFalse(changed);self.assertEqual(again,new)

    def test_custom_legacy_choices_are_preserved(self):
        for key,value in (('kv_type','int8'),('kv_type','fp8-mma'),('kv_type','fp8'),
                          ('clover_block_size',4),('clover_accept','exact'),('clover_threshold',.95)):
            original,_=config.apply(stock(),{key:value})
            new,_=tinyconfig.migrate_settings(original)
            self.assertEqual(config.value_of(new,'clover_mode'),'causal')
            self.assertEqual(config.value_of(new,key),value)
            self.assertNotIn('env',new)
        for origin in ('local',None):
            new,_=tinyconfig.migrate_settings(stock(model_source=origin))
            self.assertEqual(config.value_of(new,'clover_mode'),'causal')
        for env in ({'STRATA_IQ_MT_MIN':'1'},{'STRATA_IQ_MT_MIN':'2'},{'STRATA_CPU_IQ_ALL_EXACT1':'1'}):
            new,_=tinyconfig.migrate_settings(stock(env=env))
            self.assertEqual(config.value_of(new,'clover_mode'),'causal');self.assertEqual(new['env'],env)
        for flag,value in (('--cpu-threads','9'),('--chunk','1024'),('--max-context','65536')):
            original=stock();original['args'] += [flag,value]
            new,_=tinyconfig.migrate_settings(original)
            self.assertEqual(config.value_of(new,'clover_mode'),'causal')
        original=stock(sampling={**tinyconfig.CLOVER_SAMPLING,'temperature':.7})
        new,_=tinyconfig.migrate_settings(original)
        self.assertEqual(config.value_of(new,'clover_mode'),'causal')

    def test_saved_policy_marker_preserves_same_value_manual_causal(self):
        original=stock(clover_decode_policy_version=tinyconfig.CLOVER_DECODE_POLICY_VERSION)
        new,_=tinyconfig.migrate_settings(original)
        self.assertEqual(config.value_of(new,'clover_mode'),'causal')
        self.assertEqual(config.value_of(new,'kv_type'),'auto')
        self.assertNotIn('env',new)

    def test_process_cpu_override_is_not_replaced(self):
        with patch.dict(os.environ,{'STRATA_IQ_MT_MIN':'2'}):
            new,_=tinyconfig.migrate_settings(stock())
            self.assertEqual(config.value_of(new,'clover_mode'),'causal')
            fresh=tinyconfig.clover_decode_defaults({'arch':'clover_moe','args':[]})
            self.assertEqual(fresh['env']['STRATA_IQ_MT_MIN'],'2')

    def test_vision_experiment_keeps_causal(self):
        original=stock(vision_mode='gpu')
        with patch.dict(os.environ,{'TINYSTRATA_CLOVER_VISION':'1'}):
            new,_=tinyconfig.migrate_settings(original)
        self.assertEqual(config.value_of(new,'clover_mode'),'causal')
        self.assertEqual(config.value_of(new,'kv_type'),'auto')

    def test_backend_preserves_new_clover_kv_for_both_families(self):
        for saved in (None,'cuda','hip'):
            for requested in ('cuda','hip'):
                original=tinyconfig.clover_decode_defaults({'arch':'clover_moe','args':[],**({'backend':saved} if saved else {})})
                changed=backend.configure_backend(original,requested)
                self.assertEqual(config.value_of(changed,'kv_type'),'int8')
                for selected in ('auto','fp8-mma'):
                    manual,_=config.apply(original,{'kv_type':selected})
                    expected='auto' if requested=='hip' and saved!='hip' else selected
                    self.assertEqual(config.value_of(backend.configure_backend(manual,requested),'kv_type'),expected)
        qwen={'arch':'qwen35moe','backend':'cuda','args':['--kv-type','int8']}
        self.assertEqual(config.value_of(backend.configure_backend(qwen,'hip'),'kv_type'),'auto')

    def test_cross_backend_custom_fp8_keeps_existing_auto_migration(self):
        original=tinyconfig.clover_decode_defaults({'arch':'clover_moe','backend':'cuda','args':[]})
        original,_=config.apply(original,{'kv_type':'fp8-mma'})
        self.assertEqual(config.value_of(backend.configure_backend(original,'hip'),'kv_type'),'auto')
        self.assertEqual(config.value_of(backend.configure_backend(original,'hip',explicit_kv='fp8-mma'),'kv_type'),'fp8-mma')
        original['backend']='hip'
        self.assertEqual(config.value_of(backend.configure_backend(original,'hip'),'kv_type'),'fp8-mma')

    def test_other_model_migrations_remain_unmarked(self):
        for arch in ('qwen35moe','gemma4','lfm2moe'):
            original={'arch':arch,'args':['--model','fixture.gguf','--kv-type','auto'],'vision_mode':'no'}
            new,_=tinyconfig.migrate_settings(original)
            self.assertNotIn('clover_decode_policy_version',new);self.assertNotIn('env',new)
            self.assertEqual(config.value_of(new,'kv_type'),'auto')

    def test_factories_use_clover_policy_only(self):
        profiles=object.__new__(ModelProfiles);profiles._hardware={'backend':'hip'};profiles._engine=None
        for name,arch in (('clover-35b-a3b-preview','clover_moe'),('qwen3.6-35b-a3b','qwen35moe'),
                          ('gemma-4-26b-a4b-it','gemma4'),('lfm2.5-24b-a2b','lfm2moe')):
            new=profiles._defaults(name,arch)
            if arch=='clover_moe':
                self.assertEqual(config.value_of(new,'clover_mode'),'self-spec')
                self.assertEqual(config.value_of(new,'clover_draft'),3)
                self.assertEqual(config.value_of(new,'clover_accept'),'exact')
                self.assertEqual(config.value_of(new,'kv_type'),'int8')
                self.assertEqual(new['env'],{'STRATA_IQ_MT_MIN':'1'})
                self.assertEqual(new['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
            else:
                self.assertEqual(config.value_of(new,'kv_type'),'auto')
                self.assertNotIn('env',new)
                self.assertNotIn('clover_decode_policy_version',new)
                self.assertFalse(any(arg.startswith('--clover-') for arg in new['args']))


class LauncherPolicy(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='tinystrata-decode-test-')
        self.root=Path(self.temp.name).resolve()
        # Verify the exact owned target before TemporaryDirectory cleans it up.
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.root.name.startswith('tinystrata-decode-test-'))
        self.addCleanup(self.temp.cleanup)
        self.path=self.root/'config-clover.json';self.model=self.root/'fixture.gguf';self.model.write_bytes(b'fixture')
        self.exe=self.root/'tinystrata.exe';self.exe.write_bytes(b'fixture')
        self.base=stock(exe=str(self.exe),backend='hip',hip_ordinal=0,cwd=str(self.root))
        self.base['args'][self.base['args'].index('--model')+1]=str(self.model)

    def prepare(self,cfg=None,**overrides):
        if cfg is not None:self.path.write_text(json.dumps(cfg),encoding='utf-8')
        a=SimpleNamespace(model_family='clover',config=self.path,model=None,exe=None,backend='hip',kv_type=None,
            lib_dir=None,port=None,host=None,api_key=None,gpu=None,vision=None,vision_exe=None,vision_tokens=None,
            draft_model=None,max_context=None,mtp=None,expert_vram=None,vram_reserve_mib=None,chunk=None,kv_grow=None,
            **{key:None for key in config.CLOVER_REQUEST_KEYS})
        for key,value in overrides.items():setattr(a,key,value)
        def download(cfg,path):
            new={**cfg,'args':list(cfg['args'])}
            if '--model' not in new['args']:
                new['args']+=['--model',str(self.model)];new['model_source']='managed'
            return new
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(backend,'configure_backend',side_effect=lambda cfg,*args,**kw:{**cfg,'backend':'hip'}))
            stack.enter_context(patch.object(backend,'executable_backend',return_value='hip'))
            stack.enter_context(patch.object(backend,'hip_ordinal',return_value=0))
            stack.enter_context(patch.object(model_setup,'model_architecture',return_value='clover_moe'))
            stack.enter_context(patch.object(model_setup,'configure_download',side_effect=download))
            stack.enter_context(patch.object(model_setup,'configure_vision',side_effect=lambda cfg,*args,**kw:cfg))
            stack.enter_context(patch.object(model_setup,'shared_library_dirs',side_effect=lambda cfg:cfg))
            stack.enter_context(patch.object(model_setup,'resolve_executable',return_value=self.exe))
            stack.enter_context(patch.object(model_setup,'prepare_installed_model',side_effect=lambda cfg,*args,**kw:cfg))
            stack.enter_context(patch.object(model_setup,'save_settings'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return launcher.prepare(a)[0]

    def test_fresh_managed_factory(self):
        cfg=self.prepare()
        self.assertEqual(config.value_of(cfg,'clover_mode'),'self-spec')
        self.assertEqual(config.value_of(cfg,'clover_draft'),3)
        self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
        self.assertEqual(config.value_of(cfg,'kv_type'),'int8')
        self.assertEqual(cfg['env'],{'STRATA_IQ_MT_MIN':'1'})
        self.assertEqual(cfg['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)

    def test_saved_stock_and_explicit_same_value_cli(self):
        migrated=self.prepare(self.base)
        self.assertEqual(config.value_of(migrated,'clover_mode'),'self-spec')
        self.assertEqual(config.value_of(migrated,'kv_type'),'int8')
        for explicit in ({'clover_mode':'causal'},{'clover_draft':0},{'kv_type':'auto'}):
            cfg=self.prepare(self.base,**explicit)
            self.assertEqual(config.value_of(cfg,'clover_mode'),'causal')
            self.assertEqual(config.value_of(cfg,'clover_draft'),0)
            self.assertEqual(config.value_of(cfg,'kv_type'),'auto')
            again,_=tinyconfig.migrate_settings(cfg)
            self.assertEqual(again,cfg)

    def test_mode_only_self_spec_uses_three_drafts(self):
        cfg=self.prepare(self.base,clover_mode='self-spec')
        self.assertEqual(config.value_of(cfg,'clover_draft'),3)

    def test_fresh_explicit_causal_and_diffusion_modes(self):
        for mode in ('causal','diffusion'):
            self.path.unlink(missing_ok=True)
            cfg=self.prepare(clover_mode=mode)
            self.assertEqual(config.value_of(cfg,'clover_mode'),mode)
            self.assertEqual(config.value_of(cfg,'clover_draft'),0)
            self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
            self.assertEqual(cfg['env'],{'STRATA_IQ_MT_MIN':'1'})
        self.path.unlink(missing_ok=True)
        cfg=self.prepare(clover_draft=7,kv_type='fp8-mma')
        self.assertEqual(config.value_of(cfg,'clover_mode'),'self-spec')
        self.assertEqual(config.value_of(cfg,'clover_draft'),7)
        self.assertEqual(config.value_of(cfg,'kv_type'),'fp8-mma')

    def test_explicit_vision_off_and_cpu_survive_stock_migration(self):
        for mode in ('no','cpu'):
            cfg=self.prepare(self.base,vision=mode)
            self.assertEqual(cfg['vision_mode'],mode)
            self.assertEqual(cfg['clover_vision_policy_version'],tinyconfig.CLOVER_VISION_POLICY_VERSION)
            self.path.unlink(missing_ok=True)
            cfg=self.prepare(vision=mode)
            self.assertEqual(cfg['vision_mode'],mode)
            self.assertEqual(config.value_of(cfg,'clover_draft'),3)

    def inherited_threshold(self):
        cfg,_=config.apply(self.base,dict(clover_mode='self-spec',clover_draft=3,clover_accept='threshold',
                                        clover_accept_ratio=.9,clover_accept_delta=.6))
        return cfg

    def test_explicit_local_guard_precedes_migration(self):
        selected=self.root/'custom.gguf';selected.write_bytes(b'custom')
        cfg=self.prepare(self.base,model=selected)
        self.assertEqual(cfg['model_source'],'local')
        self.assertEqual(config.value_of(cfg,'clover_mode'),'causal')
        self.assertEqual(config.value_of(cfg,'clover_draft'),0)
        self.assertNotIn('--clover-accept',cfg['args'])

    def test_explicit_cli_does_not_trigger_default_upgrade(self):
        cfg=self.prepare(self.base,clover_mode='self-spec',clover_draft=7,clover_accept='exact')
        self.assertEqual(config.value_of(cfg,'clover_draft'),7)
        self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
        self.assertEqual(cfg['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
        again,changed=tinyconfig.migrate_settings(cfg)
        self.assertFalse(changed);self.assertEqual(again,cfg)
        # A threshold-only request must not silently acquire self-spec mode.
        with self.assertRaises(ValueError):self.prepare(self.base,clover_accept='threshold')

    def test_inherited_threshold_falls_back_for_cli_mode(self):
        inherited=self.inherited_threshold()
        for choice in ({'clover_mode':'causal'},{'clover_draft':0},{'clover_mode':'diffusion'}):
            cfg=self.prepare(inherited,**choice)
            self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
            self.assertEqual(config.value_of(cfg,'clover_draft'),0)
        with self.assertRaises(ValueError):self.prepare(inherited,clover_mode='causal',clover_accept='threshold')

    def test_inherited_threshold_falls_back_for_saved_positive_temperature(self):
        inherited=self.inherited_threshold()
        inherited['sampling']={**tinyconfig.CLOVER_SAMPLING,'temperature':.7}
        cfg=self.prepare(inherited)
        self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
        self.assertEqual(cfg['sampling']['temperature'],.7)
        with self.assertRaises(ValueError):self.prepare(inherited,clover_accept='threshold')

    def test_saved_null_temperature_uses_model_default(self):
        inherited=self.inherited_threshold()
        inherited['sampling']={**tinyconfig.CLOVER_SAMPLING,'temperature':None}
        tinyconfig.validate(inherited,allow_uninstalled=True)
        for choices in ({},{'clover_accept':'threshold'}):
            cfg=self.prepare(inherited,**choices)
            self.assertEqual(config.value_of(cfg,'clover_accept'),'threshold')
            self.assertIsNone(cfg['sampling']['temperature'])


class HotSwitchPolicy(unittest.TestCase):
    def test_fresh_and_saved_profiles_prepare_and_persist_same_defaults(self):
        with tempfile.TemporaryDirectory(prefix='tinystrata-profile-policy-') as temp:
            folder=Path(temp);model=folder/'fixture.gguf';model.write_bytes(b'fixture')
            qwen={'arch':'qwen35moe','backend':'hip','vision_mode':'no','args':['--model',str(folder/'qwen.gguf'),'--kv-type','auto']}
            with patch.object(model_setup,'resolve_executable',return_value=None), \
                 patch.object(model_setup,'shared_library_dirs',side_effect=lambda cfg:cfg), \
                 patch.object(tinyconfig,'check_files'):
                for saved in (None,stock()):
                    path=folder/'config-clover.json'
                    path.unlink(missing_ok=True)
                    if saved is not None:
                        saved['args'][saved['args'].index('--model')+1]=str(model)
                        path.write_text(json.dumps(saved),encoding='utf-8')
                    profiles=ModelProfiles(folder/'config.json',qwen)
                    cfg=profiles.get('clover').config
                    self.assertEqual(config.value_of(cfg,'clover_mode'),'self-spec')
                    self.assertEqual(config.value_of(cfg,'clover_draft'),3)
                    self.assertEqual(config.value_of(cfg,'kv_type'),'int8')
                    self.assertEqual(cfg['env'],{'STRATA_IQ_MT_MIN':'1'})
                    def prepared(cfg,path,model,**kwargs):
                        model_setup.save_settings(path,cfg)
                        return cfg
                    with patch.object(model_setup,'prepare_installed_model',side_effect=prepared):
                        ready=profiles.prepare('clover',model_path=model)
                    self.assertEqual(config.load(path),ready)
                    reopened=ModelProfiles(folder/'config.json',qwen).get('clover').config
                    self.assertEqual(config.value_of(reopened,'clover_mode'),'self-spec')
                    self.assertEqual(config.value_of(reopened,'clover_draft'),3)
                    self.assertEqual(reopened['env'],{'STRATA_IQ_MT_MIN':'1'})

if __name__=='__main__':unittest.main(verbosity=2)
