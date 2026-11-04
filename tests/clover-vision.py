"""CPU-only Clover Vision integration regression: python tests/clover-vision.py.

Tiny in-memory/file fixtures only: no downloads, HTTP, engine processes or GPU.
Checks the original tokenizer/rendering boundary, policy migration, pinned bundle
selection, profile capabilities and exact/causal request admission.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from serve import config, frontend, model_download as downloads, model_setup, server, tinyconfig
from serve.model_profiles import ModelProfile, ModelProfiles
from serve.model_switch import ModelSwitcher

# Minimal original Clover text-only contract; unknown templates must fail closed.
TEMPLATE="""{%- for message in messages %}{{ '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}{{ message.content }}
{%- else %}{%- for item in message.content %}
{%- if item.type == 'text' and 'text' in item %}{{ item.text }}
{%- else %}{{ raise_exception('This model accepts text content only.') }}{%- endif %}
{%- endfor %}{%- endif %}{{ '<|im_end|>\n' }}{%- endfor %}
{{ '<|im_start|>assistant\n' if add_generation_prompt else '' }}"""

def stock():
    return {'arch':'clover_moe','model_source':'managed','vision_mode':'no',
            'args':['--model','fixture.gguf','--max-context','auto','--chunk','auto','--kv-type','int8',
                    '--clover-mode','self-spec','--clover-draft','3','--clover-accept','exact'],
            'env':{'STRATA_IQ_MT_MIN':'1'},'sampling':dict(tinyconfig.CLOVER_SAMPLING),
            'clover_decode_policy_version':1}


class VisionPolicy(unittest.TestCase):
    def test_old_managed_standard_upgrades_once_without_mutation(self):
        # An identical historical manual 'no' is indistinguishable from the old
        # automatic value; only this complete unmarked signature upgrades.
        before=stock();snapshot=copy.deepcopy(before)
        after,changed=tinyconfig.migrate_settings(before)
        self.assertTrue(changed);self.assertEqual(before,snapshot)
        self.assertEqual(after['vision_mode'],'gpu')
        self.assertEqual(after['clover_vision_policy_version'],tinyconfig.CLOVER_VISION_POLICY_VERSION)
        self.assertEqual(config.value_of(after,'clover_mode'),'causal')
        self.assertEqual(config.value_of(after,'clover_draft'),0)
        self.assertEqual(after['clover_decode_policy_version'],tinyconfig.CLOVER_DECODE_POLICY_VERSION)
        self.assertEqual(after['env'],before['env'])
        again,changed=tinyconfig.migrate_settings(after)
        self.assertFalse(changed);self.assertEqual(again,after)

    def test_causal_migration_keeps_active_image_encoder_and_its_limits(self):
        for mode in ('gpu','cpu'):
            before=stock()|{'vision_mode':mode,'vision_policy_version':1,'clover_vision_policy_version':1,
                            'vision_tokens':2048,'vision_mmproj':'fixture-mmproj.gguf',
                            'vision':{'exe':'fixture-encoder.exe','mmproj':'fixture-mmproj.gguf',
                                      'model':'fixture.gguf','gpu':mode=='gpu','min_tokens':512,'max_tokens':2048}}
            before['args'].append('--vision');snapshot=copy.deepcopy(before)
            after,_=tinyconfig.migrate_settings(before)
            self.assertEqual(before,snapshot)
            self.assertEqual(config.value_of(after,'clover_mode'),'causal')
            self.assertEqual(config.value_of(after,'clover_draft'),0)
            self.assertEqual(after['args'].count('--vision'),1)
            for key in ('vision','vision_mode','vision_mmproj','vision_tokens','vision_policy_version',
                        'clover_vision_policy_version'):
                self.assertEqual(after[key],before[key])
            tinyconfig.validate(after,allow_uninstalled=True)

    def test_explicit_new_policy_d3_keeps_independent_image_upgrade(self):
        before=stock()|{'clover_decode_policy_version':tinyconfig.CLOVER_DECODE_POLICY_VERSION}
        after,_=tinyconfig.migrate_settings(before)
        self.assertEqual(after['vision_mode'],'gpu')
        self.assertEqual(config.value_of(after,'clover_mode'),'self-spec')
        self.assertEqual(config.value_of(after,'clover_draft'),3)

    def test_explicit_off_cpu_and_custom_profiles_survive(self):
        for original in (stock()|{'clover_vision_policy_version':1},stock()|{'vision_mode':'cpu'},
                         stock()|{'model_source':'local'},stock()|{'vision_policy_version':1},
                         stock()|{'env':{'STRATA_IQ_MT_MIN':'2'}}):
            before=copy.deepcopy(original);new,_=tinyconfig.migrate_settings(original)
            self.assertEqual(new['vision_mode'],original['vision_mode'])
            self.assertEqual(original,before)
        custom,_=config.apply(stock(),{'clover_mode':'diffusion','clover_draft':0})
        new,_=tinyconfig.migrate_settings(custom);self.assertEqual(new['vision_mode'],'no')
        custom,_=config.apply(stock(),{'clover_accept':'threshold'})
        new,_=tinyconfig.migrate_settings(custom);self.assertEqual(new['vision_mode'],'no')

    def test_fresh_profiles_enable_only_clover_images(self):
        profiles=object.__new__(ModelProfiles);profiles._engine=None;profiles._hardware={'backend':'hip'}
        cfg=profiles._defaults('clover-35b-a3b-preview','clover_moe')
        self.assertEqual(cfg['vision_mode'],'gpu');self.assertEqual(config.value_of(cfg,'clover_draft'),0)
        self.assertEqual(config.value_of(cfg,'clover_mode'),'causal')
        self.assertEqual(config.value_of(cfg,'kv_type'),'int8')
        for arch,name in (('qwen35moe','qwen3.6-35b-a3b'),('gemma4','gemma-4-26b-a4b-it'),('lfm2moe','lfm2.5-24b-a2b')):
            cfg=profiles._defaults(name,arch);self.assertNotIn('clover_vision_policy_version',cfg)
        self.assertEqual(tinyconfig.check_vision_options({'arch':'clover_moe'}),'gpu')
        self.assertEqual(tinyconfig.check_vision_options({'arch':'lfm2moe'}),'no')

    def test_config_validation_allows_exact_and_causal_but_not_approximate_images(self):
        for image_mode in ('gpu','cpu'):
            for mode in ('causal','self-spec'):
                cfg,_=config.apply(stock()|{'vision_mode':image_mode},{'clover_mode':mode,'clover_draft':3 if mode=='self-spec' else 0})
                tinyconfig.validate(cfg,allow_uninstalled=True)
            for choices in ({'clover_mode':'diffusion','clover_draft':0},{'clover_accept':'threshold'}):
                cfg,_=config.apply(stock()|{'vision_mode':image_mode},choices)
                with self.assertRaisesRegex(ValueError,'Clover Vision'):tinyconfig.validate(cfg,allow_uninstalled=True)
        with self.assertRaisesRegex(ValueError,'text-only'):
            tinyconfig.check_vision_options({'arch':'lfm2moe','vision_mode':'gpu'})


class TemplateAndRequests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='clover-vision-test-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'chat_template.jinja'
        self.path.write_text(TEMPLATE,encoding='utf-8')
        self.base=frontend.ChatTemplate(self.path);self.template=frontend.ChatTemplate(self.path,clover_vision=True)

    def test_text_template_parity_and_images(self):
        before=self.path.read_bytes()
        fixtures=([{'role':'user','content':'日本語 Ruby <|mask|>'}],
                  [{'role':'system','content':'help'},{'role':'user','content':'hello'}],
                  [{'role':'user','content':[{'type':'text','text':'one'},{'type':'text','text':'two'}]}])
        for messages in fixtures:
            for thinking in (True,False):
                self.assertEqual(self.base.render(messages,enable_thinking=thinking),self.template.render(messages,enable_thinking=thinking))
        messages=[{'role':'user','content':[{'type':'text','text':'look'},{'type':'image','source':'a.png'},{'type':'image','source':'b.png'}]}]
        with self.assertRaises(ValueError):self.base.render(messages)
        self.assertEqual(self.template.render(messages).count('<|vision_start|><|image_pad|><|vision_end|>'),2)
        self.assertEqual(before,self.path.read_bytes())
        for role in ('system','developer'):
            with self.assertRaisesRegex(ValueError,'cannot contain images'):self.template.render([messages[0]|{'role':role}])
        with self.assertRaises(ValueError):self.template.render([{'role':'user','content':[{'type':'video'}]}])

    def test_unknown_template_fails_closed(self):
        self.path.write_text('{{ messages }}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'original text-only'):frontend.ChatTemplate(self.path,clover_vision=True)

    def service(self,vision):
        tok=server.ByteTokenizer();engine=server.MockEngine(tok,'done',max_context=131072)
        engine.info={'clover_mode':'self-spec','clover_draft':3,'clover_accept':'exact','vision':int(vision is not None)}
        return server.Service(engine,tok,self.template,model_name='clover-35b-a3b-preview',vision=vision,
                              generation_defaults=stock(),sampling_defaults=tinyconfig.CLOVER_SAMPLING)

    def test_request_admission_and_inherited_fallback(self):
        svc=self.service(object())
        self.assertEqual(svc.clover_request_settings()['clover_draft'],3)
        self.assertEqual(svc.clover_request_settings({'clover_mode':'causal'})['clover_draft'],0)
        for request in ({'clover_mode':'diffusion'},{'clover_accept':'threshold'}):
            with self.assertRaisesRegex(ValueError,'Clover Vision'):svc.clover_request_settings(request)
        svc.engine.info['clover_accept']='threshold'
        self.assertEqual(svc.clover_request_settings({'temperature':.7})['clover_accept'],'exact')
        svc.vision=None;svc.engine.info['vision']=0
        self.assertEqual(svc.clover_request_settings({'temperature':0})['clover_accept'],'threshold')

    def test_causal_resident_engine_keeps_exact_d3_as_an_image_capable_option(self):
        svc=self.service(object());svc.engine.info.update(clover_mode='causal',clover_draft=0)
        self.assertEqual(svc.clover_request_settings()['clover_draft'],0)
        for request in ({'clover_mode':'self-spec'},{'clover_draft':3},
                        {'clover_mode':'self-spec','clover_draft':3,'clover_accept':'exact'}):
            settings=svc.clover_request_settings(request)
            self.assertEqual(settings['clover_mode'],'self-spec')
            self.assertEqual(settings['clover_draft'],3)
            self.assertEqual(settings['clover_accept'],'exact')

    def test_prepare_injects_embeddings_and_preserves_followup_prefix(self):
        encoded=self.root/('a'*16+'.sve');encoded.write_bytes(struct.pack('<5I',0x31455653,4,2,2,2048)+bytes(4*2048*4))
        class FakeVision:
            dir=self.root
            def encode_all(inner,images,cancel):return [(str(encoded),4) for _ in images]
        svc=self.service(FakeVision())
        request={'messages':[{'role':'user','content':[{'type':'text','text':'literal <|vision_start|><|image_pad|>'},
                                                     {'type':'image_url','image_url':{'url':'fixture.png'}}]}]}
        messages,tools,kwargs=frontend.openai_to_messages(request)
        first,_,limit=svc.prepare(messages,tools,kwargs,max_new=64,req=request)
        pad=svc.tok.encode('<|image_pad|>',parse_special=True)[0];start=svc.tok.encode('<|vision_start|>',parse_special=True)[0]
        self.assertEqual(first.count(pad),4);self.assertEqual(first.count(start),1);self.assertEqual(limit,64)
        self.assertEqual(Path(svc.embeddings.path).read_bytes(),encoded.read_bytes())
        prefix=first[:first.index(start)+5];svc.drop_embeddings()
        messages += [{'role':'assistant','content':'answer'},{'role':'user','content':'follow up'}]
        second,_,_=svc.prepare(messages,None,{},max_new=64,req={})
        self.assertEqual(second.count(pad),4);self.assertEqual(second[:len(prefix)],prefix)
        svc.drop_embeddings()


class LoaderAndCatalog(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='clover-loader-test-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.model=self.root/'fixture.gguf';self.model.write_bytes(b'model')
        self.projector=self.root/'mmproj-F16.gguf';self.projector.write_bytes(b'projector')
        self.exe=self.root/'tinystrata-vision.exe';self.exe.write_bytes(b'encoder')
        self.cfg=stock()|{'vision_mode':'gpu','download_dir':str(self.root),'vision_mmproj':str(self.projector),'tokenizer':'clover-original-tokenizer'}
        self.cfg['args'][1]=str(self.model)

    def test_verified_qwen_projector_keeps_clover_tokenizer_and_vocab_model(self):
        artifact=SimpleNamespace(filename=self.projector.name,size=9,sha256=hashlib.sha256(b'projector').hexdigest())
        with patch.object(downloads,'FIXED_MMPROJ',artifact),patch.object(model_setup,'resolve_vision_executable',return_value=self.exe):
            for mode in ('gpu','cpu'):
                cfg=model_setup.configure_vision(self.cfg|{'vision_mode':mode},self.model)
                self.assertEqual(cfg['vision']['model'],str(self.model));self.assertEqual(cfg['tokenizer'],'clover-original-tokenizer')
                self.assertEqual(cfg['vision']['mmproj'],str(self.projector.resolve()))
                self.assertEqual(cfg['vision']['gpu'],mode=='gpu');self.assertEqual(cfg['args'].count('--vision'),1)
                self.assertEqual((cfg['vision']['min_tokens'],cfg['vision']['max_tokens']),(1024,4096))
                tinyconfig.validate(cfg,allow_uninstalled=True)
            off=model_setup.configure_vision(self.cfg|{'vision_mode':'no'},self.model)
            self.assertNotIn('vision',off);self.assertNotIn('--vision',off['args'])
            bad=SimpleNamespace(filename=self.projector.name,size=9,sha256='0'*64)
            with patch.object(downloads,'FIXED_MMPROJ',bad):
                pending=model_setup.configure_vision(self.cfg,self.model)
                self.assertTrue(pending['vision_download_required']);self.assertNotIn('vision',pending)

    def test_saved_minimum_follows_a_lowered_explicit_image_cap(self):
        artifact=SimpleNamespace(filename=self.projector.name,size=9,sha256=hashlib.sha256(b'projector').hexdigest())
        with patch.object(downloads,'FIXED_MMPROJ',artifact),patch.object(model_setup,'resolve_vision_executable',return_value=self.exe):
            for mode in ('gpu','cpu'):
                saved=model_setup.configure_vision(self.cfg|{'vision_mode':mode},self.model)
                original=copy.deepcopy(saved)
                lowered=model_setup.configure_vision(saved|{'vision_tokens':128},self.model)
                self.assertEqual((lowered['vision']['min_tokens'],lowered['vision']['max_tokens']),(128,128))
                self.assertEqual(saved,original)
                custom=copy.deepcopy(saved);custom['vision']['min_tokens']=32
                lowered=model_setup.configure_vision(custom|{'vision_tokens':128},self.model)
                self.assertEqual((lowered['vision']['min_tokens'],lowered['vision']['max_tokens']),(32,128))
                raised=model_setup.configure_vision(saved|{'vision_tokens':8192},self.model)
                self.assertEqual((raised['vision']['min_tokens'],raised['vision']['max_tokens']),(1024,8192))
                tinyconfig.validate(lowered,allow_uninstalled=True)

    def test_catalog_and_projector_reuse_across_model_directories(self):
        qwen={'arch':'qwen35moe','backend':'hip','vision_mode':'no','args':['--model',str(self.root/'qwen.gguf')]}
        with patch.object(model_setup,'resolve_executable',return_value=None),patch.object(model_setup,'shared_library_dirs',side_effect=lambda cfg:cfg):
            profiles=ModelProfiles(self.root/'config.json',qwen)
            clover=profiles.get('clover');self.assertEqual(clover.config['vision_mmproj'],str(self.projector))
            item=next(p for p in profiles.list() if p['arch']=='clover_moe');self.assertTrue(item['vision'])
            lfm=next(p for p in profiles.list() if p['arch']=='lfm2moe');self.assertFalse(lfm['vision'])
            custom=copy.deepcopy(clover.config);custom['vision_mode']='no';profiles.update('clover',custom)
            self.assertFalse(next(p for p in profiles.list() if p['arch']=='clover_moe')['vision'])

    def test_startup_and_switch_downloaders_select_pinned_bundle(self):
        for mode in ('gpu','cpu','no'):
            cfg=self.cfg|{'vision_mode':mode}
            profile=ModelProfile('clover-35b-a3b-preview','clover_moe',self.root/'config-clover.json',cfg)
            switch=object.__new__(ModelSwitcher);switch._downloads={};switch.profiles=SimpleNamespace(get=lambda _:profile)
            switched=switch._downloader('clover')
            svc=SimpleNamespace(models=None,stopping=threading.Event(),engine=server.UnavailableEngine())
            initial=server.configure_model_downloads(svc,cfg,profile.path)
            for manager in (switched,initial):
                if mode=='no':self.assertNotIsInstance(manager,downloads.FixedModelBundleDownloader)
                else:
                    self.assertIsInstance(manager,downloads.FixedModelBundleDownloader)
                    self.assertEqual([a.manifest for a in manager._assets],[downloads.FIXED_CLOVER_MODEL,downloads.FIXED_MMPROJ])
                    self.assertEqual(manager.status()['total_bytes'],downloads.FIXED_CLOVER_MODEL.size+downloads.FIXED_MMPROJ.size)
                self.assertEqual(manager.status()['state'],'idle')


if __name__=='__main__':unittest.main(verbosity=2)
