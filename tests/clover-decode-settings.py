"""CPU-only managed Clover decode policy regression; no downloads or native processes."""
from __future__ import annotations
import contextlib
import copy
import io
import json
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
            self.assertNotIn('clover_decode_policy_version',new)
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

    def test_factories_keep_causal_auto_kv_and_no_cpu_override(self):
        profiles=object.__new__(ModelProfiles);profiles._hardware={'backend':'hip'};profiles._engine=None
        for name,arch in (('clover-35b-a3b-preview','clover_moe'),('qwen3.6-35b-a3b','qwen35moe'),
                          ('gemma-4-26b-a4b-it','gemma4'),('lfm2.5-24b-a2b','lfm2moe')):
            new=profiles._defaults(name,arch)
            self.assertEqual(config.value_of(new,'kv_type'),'auto')
            self.assertNotIn('env',new)
            self.assertNotIn('clover_decode_policy_version',new)
            if arch=='clover_moe':
                self.assertEqual(config.value_of(new,'clover_mode'),'causal')
                self.assertEqual(config.value_of(new,'clover_draft'),0)
            else:self.assertFalse(any(arg.startswith('--clover-') for arg in new['args']))


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
        self.assertEqual(config.value_of(cfg,'clover_mode'),'causal')
        self.assertEqual(config.value_of(cfg,'clover_draft'),0)
        self.assertEqual(config.value_of(cfg,'kv_type'),'auto')
        self.assertNotIn('env',cfg)
        self.assertNotIn('clover_decode_policy_version',cfg)

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
        self.assertNotIn('clover_decode_policy_version',cfg)
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


if __name__=='__main__':unittest.main(verbosity=2)
