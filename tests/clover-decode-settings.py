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


class DecodePolicy(unittest.TestCase):
    def test_stock_managed_migrates_once(self):
        self.assertEqual(tinyconfig.CLOVER_DECODE_DEFAULTS['clover_accept'],'exact')
        original=stock();before=copy.deepcopy(original)
        new,changed=tinyconfig.migrate_clover_decode_settings(original)
        self.assertTrue(changed);self.assertEqual(original,before)
        self.assertEqual(new['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
        for key,value in tinyconfig.CLOVER_DECODE_DEFAULTS.items():self.assertEqual(config.value_of(new,key),value)
        self.assertEqual(config.value_of(new,'kv_type'),'int8')
        again,changed=tinyconfig.migrate_clover_decode_settings(new)
        self.assertFalse(changed);self.assertEqual(again,new)

    def test_later_explicit_causal_stays_causal(self):
        cfg,_=tinyconfig.migrate_clover_decode_settings(stock())
        cfg,_=config.apply(cfg,dict(clover_mode='causal',clover_draft=0,clover_accept='exact'))
        again,changed=tinyconfig.migrate_clover_decode_settings(cfg)
        self.assertFalse(changed);self.assertEqual(again,cfg)

    def test_custom_generation_is_retained(self):
        variants=(dict(clover_mode='self-spec',clover_draft=7),dict(clover_mode='diffusion',clover_draft=0),
                  dict(clover_accept='exact'),dict(clover_accept_ratio=.9),dict(clover_accept_delta=.6),
                  dict(clover_block_size=4),dict(clover_threshold=.95))
        for choices in variants:
            cfg,_=config.apply(stock(),choices)
            new,changed=tinyconfig.migrate_clover_decode_settings(cfg)
            with self.subTest(choices=choices):
                self.assertTrue(changed);self.assertEqual(new['args'],cfg['args'])
                self.assertEqual(new['sampling'],cfg['sampling'])
                self.assertEqual(new['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
        for field,value in (('temperature',.7),('presence_penalty',0),('repetition_penalty',1),('seed',7)):
            cfg=stock(sampling={**tinyconfig.CLOVER_SAMPLING,field:value})
            new,_=tinyconfig.migrate_clover_decode_settings(cfg)
            self.assertEqual(new['args'],cfg['args']);self.assertEqual(new['sampling'],cfg['sampling'])

    def test_local_unknown_and_vision_profiles_are_retained(self):
        for extra in ({'model_source':'local'},{'model_source':None},{'vision_mode':'gpu'},
                      {'vision':{'exe':'fixture'}},{'args':stock()['args']+['--vision']}):
            cfg=stock(**extra);new,_=tinyconfig.migrate_clover_decode_settings(cfg)
            self.assertEqual(new['args'],cfg['args'])
            self.assertEqual(new['model_source'],cfg['model_source'])

    def test_custom_kv_is_retained(self):
        for kv in ('int8','fp8','fp8-mma'):
            cfg,_=config.apply(stock(),{'kv_type':kv})
            new,_=tinyconfig.migrate_clover_decode_settings(cfg)
            self.assertEqual(config.value_of(new,'kv_type'),kv)
            self.assertEqual(config.value_of(new,'clover_mode'),'self-spec')
        cfg,_=config.apply(stock(),{'clover_accept':'exact'})
        new,_=tinyconfig.migrate_clover_decode_settings(cfg)
        self.assertEqual(config.value_of(new,'kv_type'),'auto')

    def test_other_families_are_untouched(self):
        for arch in ('qwen35moe','gemma4','lfm2moe'):
            cfg={'arch':arch,'args':[]}
            self.assertEqual(tinyconfig.migrate_clover_decode_settings(cfg),(cfg,False))
            self.assertIs(tinyconfig.clover_decode_defaults(cfg),cfg)

    def test_model_profiles_factory_uses_shared_defaults(self):
        profiles=object.__new__(ModelProfiles);profiles._hardware={'backend':'hip'};profiles._engine=None
        new=profiles._defaults('clover-35b-a3b-preview','clover_moe')
        for key,value in tinyconfig.CLOVER_DECODE_DEFAULTS.items():self.assertEqual(config.value_of(new,key),value)
        self.assertEqual(config.value_of(new,'kv_type'),'int8')
        self.assertEqual(new['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)


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
        for key,value in tinyconfig.CLOVER_DECODE_DEFAULTS.items():self.assertEqual(config.value_of(cfg,key),value)
        self.assertEqual(config.value_of(cfg,'kv_type'),'int8')

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
        again,changed=tinyconfig.migrate_clover_decode_settings(cfg)
        self.assertFalse(changed);self.assertEqual(again,cfg)
        # A threshold-only request must not silently acquire self-spec mode.
        with self.assertRaises(ValueError):self.prepare(self.base,clover_accept='threshold')

    def test_inherited_threshold_falls_back_for_cli_mode(self):
        inherited=tinyconfig.clover_decode_defaults(self.base)
        inherited,_=config.apply(inherited,{'clover_accept':'threshold'})
        for choice in ({'clover_mode':'causal'},{'clover_draft':0},{'clover_mode':'diffusion'}):
            cfg=self.prepare(inherited,**choice)
            self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
            self.assertEqual(config.value_of(cfg,'clover_draft'),0)
        with self.assertRaises(ValueError):self.prepare(inherited,clover_mode='causal',clover_accept='threshold')

    def test_inherited_threshold_falls_back_for_saved_positive_temperature(self):
        inherited=tinyconfig.clover_decode_defaults(self.base)
        inherited,_=config.apply(inherited,{'clover_accept':'threshold'})
        inherited['sampling']={**tinyconfig.CLOVER_SAMPLING,'temperature':.7}
        cfg=self.prepare(inherited)
        self.assertEqual(config.value_of(cfg,'clover_accept'),'exact')
        self.assertEqual(cfg['sampling']['temperature'],.7)
        with self.assertRaises(ValueError):self.prepare(inherited,clover_accept='threshold')

    def test_saved_null_temperature_uses_model_default(self):
        inherited=tinyconfig.clover_decode_defaults(self.base)
        inherited,_=config.apply(inherited,{'clover_accept':'threshold'})
        inherited['sampling']={**tinyconfig.CLOVER_SAMPLING,'temperature':None}
        tinyconfig.validate(inherited,allow_uninstalled=True)
        for choices in ({},{'clover_accept':'threshold'}):
            cfg=self.prepare(inherited,**choices)
            self.assertEqual(config.value_of(cfg,'clover_accept'),'threshold')
            self.assertIsNone(cfg['sampling']['temperature'])


if __name__=='__main__':unittest.main(verbosity=2)
