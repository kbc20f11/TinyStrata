'use strict';
// Run: node tests/ui-model-download.cjs
// DOM, timers, storage and model requests are simulated. No API/GPU is started.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert'),path=require('node:path');
const webDirectory=path.join(__dirname,'..','ui','web');
const localeDirectory=path.join(webDirectory,'locales');
const source=fs.readFileSync(path.join(webDirectory,'app.js'),'utf8');
const sectionStart=source.indexOf('let modelCatalog ='),sectionEnd=source.indexOf('$("composer-effort").onchange');
assert(sectionStart>=0&&sectionEnd>sectionStart,'Model controls section not found in app.js');
const snippet=source.slice(sectionStart,sectionEnd);
const IDS=['qwen3.6-35b-a3b','gemma-4-26b-a4b-it','lfm2.5-24b-a2b','clover-35b-a3b-preview'];
let checks=0;
function ok(value,label){assert(value,label);checks++;}
function fixture(saved=[]){
 const elements=new Map(),calls=[],toasts=[],timers=new Map(),storage={'model-downloads':saved};let tid=0;
 function element(id){
  if(!elements.has(id))elements.set(id,{id,hidden:false,disabled:false,textContent:'',value:'',open:false,attributes:{},
    options:[],children:[],setAttribute(k,v){this.attributes[k]=v;},removeAttribute(k){delete this.attributes[k];if(k==='value')this.value=undefined;},
    querySelector(){return {disabled:false,title:''};},showModal(){this.open=true;},close(){this.open=false;}});
  return elements.get(id);
 }
 element('composer-model').options=IDS.map(value=>({value}));
 element('composer-model').value=IDS[0];
 element('s-thinking').children=[{dataset:{v:'none'}},{dataset:{v:'high'}}];
 const jobs=Object.fromEntries(IDS.map(model_id=>[model_id,{state:'idle',bytes_downloaded:0,total_bytes:100,progress:0,error:null,can_cancel:false,model_id}]));
 const backend={model:IDS[0],jobs,nextError:null,rejectSwitch:false,rollback:false};
 const ctx={console,Map,Promise,Number,JSON,Array,String,Error,encodeURIComponent,AbortSignal,
  $:element,health:{model:IDS[0],images:true},settings:{thinking:'high'},messages:[],busy:null,pending:null,
  sharedSaving:false,modelOp:null,panel:null,lastMetrics:{live:{state:'idle'}},inflight:false,
  tr:(key,en,values={})=>en.replace(/\{(\w+)\}/g,(_,k)=>values[k]),fmt:(v,d=0)=>Number(v).toFixed(d),
  store:{get:(key,d)=>storage[key]??d,set:(key,value)=>{storage[key]=JSON.parse(JSON.stringify(value));return true;}},
  turnBlocksUnload:()=>ctx.inflight||ctx.busy!==null,pageBusy:()=>ctx.busy!==null,
  toast:(...args)=>toasts.push(args),paintLive(){},setPill(){},
  setTimeout:(fn,delay)=>{timers.set(++tid,{fn,delay});return tid;},clearTimeout:id=>timers.delete(id),
  headers:()=>({}),fetch:()=>{throw Error('Real fetch prohibited');},
  loadHealth:async()=>{ctx.health.model=backend.model;return true;},
  setBusy:()=>{},backend,calls,storage};
 vm.createContext(ctx);vm.runInContext(snippet,ctx);
 vm.runInContext(`modelCatalog=${JSON.stringify(IDS.map((id,i)=>({id,installed:i===0,active:i===0})))};
 modelSwitchAvailable=true;
 refreshModelCatalog=async()=>syncComposerControls();
 modelControl=async(path,options={})=>{
   const method=options.method||'GET',body=options.body?JSON.parse(options.body):null;
   calls.push({path,method,body});
   if(backend.nextError){const e=backend.nextError;backend.nextError=null;throw e;}
   if(path==='v1/model/switch'){
     if(method==='POST'){
       if(backend.rejectSwitch){const e=new Error('busy');e.status=409;throw e;}
       backend.target=body.model;
       if(!backend.rollback)backend.model=body.model;
       return {state:'ready',switching:false,model:backend.model,target:body.model};
     }
     return {state:backend.rollback?'error':'ready',switching:false,model:backend.model,target:backend.target,error:backend.rollback?(backend.switchError||'activation failed'):null,rolled_back:backend.rollback};
   }
   const target=body?body.model:decodeURIComponent(path.split('?model=')[1]);
   const job=backend.jobs[target];
   if(!job)throw new Error('Unexpected target');
   if(method==='POST'&&path==='v1/model/download')Object.assign(job,{state:'downloading',can_cancel:true,error:null});
   return JSON.parse(JSON.stringify(job));
 };
 globalThis.test={flows:modelDownloads,open:openModelDownload,start:startModelDownload,cancel:cancelModelDownload,
  poll:pollModelDownload,status:acceptDownloadStatus,render:renderModelDownload,recover:recoverModelDownloads,
  controls:syncComposerControls,dismiss:dismissModelDownload,switch:requestModelSwitch,pollSwitch:pollModelSwitch,
  state:()=>({switching:modelSwitching,dialogTarget:modelDownloadDialogTarget,target:modelSwitchTarget}),
  install:(id)=>{modelCatalog.find(row=>row.id===id).installed=true;}};`,ctx);
 return {ctx,element,backend,calls,toasts,timers,storage,t:ctx.test};
}
const posts=(f,path)=>f.calls.filter(c=>c.method==='POST'&&c.path===path);
const drain=()=>new Promise(resolve=>setImmediate(resolve));
function healthFixture({controlled=false,registration=null}={}){
 const start=source.indexOf('let health ='),end=source.indexOf('// ------------------------------------------------------------------ Monitor',start);
 assert(start>=0&&end>start,'Health controls section not found in app.js');
 const paths=[],elements=new Map();let updates=0;
 const context={URL,Date:{now:()=>12345},location:{origin:'http://127.0.0.1:3000',hash:''},
  navigator:{},sharedOn:true,loadShared:async()=>{},syncComposerControls(){},showTab(){},setTimeout(){},
  $:(id)=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);},
  tr:(_key,en,values={})=>en.replace(/\{(\w+)\}/g,(_,key)=>values[key]),
  fetch:async(path,options)=>{
   paths.push({path,options});
   // Reproduce the old cache-first worker: bare health is stale even though
   // the native switch succeeded. A new query URL reaches authoritative data.
   return {ok:true,json:async()=>({status:'ok',model:path==='health'?IDS[0]:IDS[1],images:true})};
  }};
 if(controlled)context.navigator.serviceWorker={controller:{},getRegistration:async()=>registration};
 if(registration)registration.update=async()=>{updates++;};
 vm.createContext(context);vm.runInContext(source.slice(start,end),context);
 context.currentHealth=vm.runInContext('()=>health',context);
 return {context,paths,updates:()=>updates};
}
async function workerRegressions(){
 let f=healthFixture();
 ok(f.context.healthRequestPath()==='health','uncontrolled health keeps ordinary same-origin path');
 const origin='http://127.0.0.1:3000';
 const registration={scope:origin+'/',active:{scriptURL:origin+'/sw.js'}};
 f=healthFixture({controlled:true,registration});await drain();
 const first=f.context.healthRequestPath(),second=f.context.healthRequestPath();
 ok(first.startsWith('health?_fresh=')&&first!==second,'old service-worker control gets a unique health URL even at the same timestamp');
 await f.context.loadHealth();
 ok(f.context.currentHealth().model===IDS[1],'stale cached Qwen health cannot report a failed successful Gemma switch');
 ok(f.paths[0].options.cache==='no-store'&&f.paths[0].path.startsWith('health?_fresh='),'loadHealth actually uses the nonce path and no-store');
 ok(f.updates()===1,'same-origin root sw.js registration is updated for retirement');
 f.context.navigator.serviceWorker.controller=null;
 ok(f.context.healthRequestPath()==='health','health nonce is unnecessary once worker control ends');
 for(const other of [
  {scope:origin+'/another-app/',active:{scriptURL:origin+'/sw.js'}},
  {scope:origin+'/',active:{scriptURL:origin+'/other-worker.js'}},
  {scope:origin+'/',active:{scriptURL:'https://example.invalid/sw.js'}},
  {scope:origin+'/',active:null}
 ]){
  const untouched=healthFixture({controlled:true,registration:other});await drain();
  ok(untouched.updates()===0,'unrelated worker registration is untouched');
 }
 const events=new Map(),sequence=[];
 const worker={self:{addEventListener:(name,fn)=>events.set(name,fn),
  skipWaiting:async()=>{sequence.push('skipWaiting');},
  clients:{claim:async()=>{sequence.push('claim');}},
  registration:{unregister:async()=>{sequence.push('unregister');}}}};
 vm.createContext(worker);vm.runInContext(fs.readFileSync(path.join(webDirectory,'sw.js'),'utf8'),worker);
 await events.get('install')();let activation;
 events.get('activate')({waitUntil:task=>{activation=task;}});await activation;
 ok(sequence.join(',')==='skipWaiting,claim,unregister','retirement worker activates, claims old pages, then unregisters itself');
 ok(!events.has('fetch'),'retirement worker cannot continue returning cached health responses');
}
(async()=>{
 let f=fixture();f.t.controls();
 ok(f.element('composer-model').options.every(o=>!o.disabled),'uninstalled catalog options are selectable');
 await f.t.open(IDS[1]);
 ok(f.element('model-download-dialog').open&&f.element('model-download-description').textContent.includes('0.00 GB'),'confirmation includes API size');
 ok(posts(f,'v1/model/download').length===0,'opening confirmation never starts a download');
 f.t.dismiss();ok(posts(f,'v1/model/download').length===0&&!f.element('model-download-dialog').open,'confirmation cancel is mutation-free');
 await f.t.open(IDS[1]);let flow=f.t.flows.get(IDS[1]);await f.t.start(flow);
 ok(posts(f,'v1/model/download').length===1&&flow.stage==='downloading','confirmation starts one fixed target download');
 f.t.dismiss();ok(f.ctx.health.model===IDS[0]&&!f.t.state().switching,'closing progress preserves active model and chat switch guard');
 f.t.install(IDS[1]);f.element('composer-model').value=IDS[1];await f.element('composer-model').onchange();
 ok(f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===1&&posts(f,'v1/model/switch').length===0,'partial installed bundle reopens existing job without duplicate POST/switch');
 Object.assign(f.backend.jobs[IDS[1]],{bytes_downloaded:50,progress:.5});await f.t.poll(flow);
 ok(f.element('model-download-progress').value===50&&f.element('model-download-percent').textContent==='50.0%','byte progress renders real percentage');
 await f.t.cancel(flow);ok(posts(f,'v1/model/download/cancel').length===1&&flow.stage==='cancelling'&&!flow.autoSwitch,'explicit cancellation is separate and asynchronous');
 Object.assign(f.backend.jobs[IDS[1]],{state:'cancelled',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='cancelled'&&posts(f,'v1/model/switch').length===0,'cancelled transfer never switches');
 await f.t.start(flow);ok(posts(f,'v1/model/download').length===2,'retry resumes same target through existing API');
 f.ctx.inflight=true;Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false,progress:1,bytes_downloaded:100});await f.t.poll(flow);
 ok(flow.stage==='waiting'&&posts(f,'v1/model/switch').length===0&&f.ctx.health.model===IDS[0],'ready download waits for active conversation');
 f.ctx.inflight=false;await f.t.poll(flow);await drain();
 ok(posts(f,'v1/model/switch').length===1&&f.ctx.health.model===IDS[1]&&flow.stage==='done','ready idle target switches and confirms actual health');
 ok(f.storage['model-downloads'].length===0,'completed intent is cleared');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);
 let seen=false;const base=f.ctx.modelControl;
 f.ctx.modelControl=async(path,options={})=>{
   if(path==='v1/model/download'&&options.method==='POST'&&!seen){seen=true;f.calls.push({path,method:'POST'});Object.assign(f.backend.jobs[IDS[1]],{state:'downloading',can_cancel:true});throw Error('lost POST response');}
   return base(path,options);
 };
 await f.t.start(flow);ok(flow.stage==='recovering','ambiguous POST failure enters read-only recovery');
 await f.t.poll(flow);ok(flow.stage==='downloading'&&posts(f,'v1/model/download').length===1,'GET recovery never duplicates accepted POST');
 Object.assign(f.backend.jobs[IDS[1]],{state:'error',error:'disk full',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='error'&&flow.error==='disk full'&&posts(f,'v1/model/switch').length===0,'HTTP200 job error is a failure, not ready');

 f=fixture([{model:IDS[1],switchWhenReady:true}]);Object.assign(f.backend.jobs[IDS[1]],{state:'downloading',can_cancel:true});
 await f.t.recover();flow=f.t.flows.get(IDS[1]);
 ok(flow&&flow.confirmed&&flow.autoSwitch&&f.element('model-download-dialog').open,'reload restores confirmed target and progress');
 ok(posts(f,'v1/model/download').length===0,'reload recovery is read-only');
 f.ctx.inflight=true;Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='waiting'&&posts(f,'v1/model/switch').length===0,'restored intent also respects busy guard');

 f=fixture();Object.assign(f.backend.jobs[IDS[2]],{state:'downloading',can_cancel:true});await f.t.recover();
 flow=f.t.flows.get(IDS[2]);ok(flow&&!flow.autoSwitch&&!flow.confirmed,'externally started jobs do not acquire switch intent');
 Object.assign(f.backend.jobs[IDS[2]],{state:'ready',can_cancel:false});await f.t.poll(flow);
 ok(posts(f,'v1/model/switch').length===0,'external ready job does not switch spontaneously');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);f.backend.nextError=Object.assign(Error('need key'),{status:401});
 await f.t.start(flow);ok(flow.stage==='error'&&posts(f,'v1/model/download').length===0,'authentication failure cannot start download');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);await f.t.start(flow);f.backend.rollback=true;
 Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);await drain();
 ok(f.ctx.health.model===IDS[0]&&flow.stage==='ready'&&flow.error==='activation failed','failed activation follows rollback health and offers retry');
 ok(!f.toasts.some(t=>t[0]==='success'),'rollback never announces switch success');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);await f.t.start(flow);f.backend.rejectSwitch=true;
 Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);await drain();
 ok(flow.stage==='waiting'&&f.ctx.health.model===IDS[0],'server-side busy rejection waits without changing current model');
 ok(!f.toasts.some(t=>t[0]==='error'),'busy retry does not spam switch errors');

 f=fixture();f.t.install(IDS[1]);Object.assign(f.backend.jobs[IDS[1]],{state:'cancelled'});
 f.element('composer-model').value=IDS[1];await f.element('composer-model').onchange();
 ok(f.element('model-download-dialog').open&&posts(f,'v1/model/switch').length===0,'cancelled partial bundle does not use installed flag as full readiness');
 f=fixture();let resolveConfirmation;
 const original=f.ctx.modelControl;
 f.ctx.modelControl=(path,options)=>path.includes('download?')?new Promise(resolve=>{resolveConfirmation=resolve;}):original(path,options);
 const opening=f.t.open(IDS[1]);
 f.t.dismiss();resolveConfirmation({...f.backend.jobs[IDS[1]]});await opening;
 ok(!f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===0&&f.ctx.health.model===IDS[0],'late confirmation GET after cancel does not reopen or mutate model');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);await f.t.start(flow);
 f.ctx.busy={};f.ctx.turnBlocksUnload=()=>false;f.ctx.pageBusy=()=>false;
 Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);f.t.controls();
 ok(flow.stage==='waiting'&&posts(f,'v1/model/switch').length===0&&f.element('composer-model').disabled,'local busy before metrics directly blocks ready switch and selector');
 f.ctx.busy=null;f.t.install(IDS[2]);f.element('composer-model').value=IDS[2];await f.element('composer-model').onchange();await drain();
 await f.t.poll(flow);await drain();
 ok(f.ctx.health.model===IDS[2]&&!flow.autoSwitch&&posts(f,'v1/model/switch').length===1,'explicit installed-model choice revokes older automatic switch intent');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);await f.t.start(flow);
 Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.cancel(flow);
 ok(flow.cancelRequested&&flow.stage==='ready','cancel racing transfer completion leaves ready files without auto-switch');
 await f.element('model-download-action').onclick();await drain();
 ok(!flow.cancelRequested&&flow.stage==='done'&&f.ctx.health.model===IDS[1],'explicit ready Switch overrides previous cancellation intent');

 f=fixture();f.t.install(IDS[1]);f.backend.rollback=true;
 f.backend.switchError='the matching Gemma QAT assistant is missing or invalid; download the model bundle first';
 f.element('composer-model').value=IDS[1];await f.element('composer-model').onchange();await drain();
 flow=f.t.flows.get(IDS[1]);
 ok(flow&&flow.stage==='confirm'&&f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===0,'installed idle partial bundle after server restart offers download confirmation');
 ok(f.ctx.health.model===IDS[0],'missing bundle switch failure retains original active model');

 f=fixture([{model:IDS[1],switchWhenReady:true}]);
 Object.assign(f.backend.jobs[IDS[1]],{state:'downloading',can_cancel:true});
 Object.assign(f.backend.jobs[IDS[2]],{state:'downloading',can_cancel:true});await f.t.recover();
 flow=f.t.flows.get(IDS[1]);Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='waiting'&&posts(f,'v1/model/switch').length===0,'automatic switch waits for other model download jobs');
 Object.assign(f.backend.jobs[IDS[2]],{state:'ready',can_cancel:false});await f.t.poll(f.t.flows.get(IDS[2]));await f.t.poll(flow);await drain();
 ok(flow.stage==='done'&&f.ctx.health.model===IDS[1],'latest intended model switches after all tracked transfers finish');
 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);
 const lostBeforeAcceptance=f.ctx.modelControl;
 f.ctx.modelControl=async(path,options={})=>{
   if(path==='v1/model/download'&&options.method==='POST'){f.calls.push({path,method:'POST'});throw Error('POST was not accepted');}
   return lostBeforeAcceptance(path,options);
 };
 await f.t.start(flow);await f.t.poll(flow);
 ok(flow.stage==='confirm'&&flow.confirmed,'unknown start with idle job returns to confirmation');
 f.t.dismiss();
 ok(!f.t.flows.has(IDS[1])&&f.storage['model-downloads'].length===0&&posts(f,'v1/model/download').length===1,'cancelling renewed confirmation clears persisted intent without another POST');
 await workerRegressions();
 const required=[...snippet.matchAll(/tr\("(model\.download[^\"]+)"/g)].map(m=>m[1]);
 for(const name of fs.readdirSync(localeDirectory)){
   if(!name.endsWith('.json'))continue;
   const locale=JSON.parse(fs.readFileSync(path.join(localeDirectory,name),'utf8'));
   ok(required.every(key=>typeof locale[key]==='string'),'all dynamic labels localized '+name);
 }
 console.log(JSON.stringify({checks,passed:true,api_started:false,download_started:false,gpu_started:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
