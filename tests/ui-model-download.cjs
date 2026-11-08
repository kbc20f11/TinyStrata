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
const IDS=['qwen3.6-35b-a3b','gemma-4-26b-a4b-it','lfm2.5-24b-a2b','clover-35b-a3b-preview','tiel-coder-35b-a3b'];
let checks=0;
function ok(value,label){assert(value,label);checks++;}
function fixture(saved=[]){
 const elements=new Map(),calls=[],toasts=[],timers=new Map(),storage={'model-downloads':saved};let tid=0;
 class Element {
  constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.parentNode=null;this.dataset={};this.style={};this.attributes={};this.hidden=false;this.disabled=false;this.value='';this._text='';this.open=false;this.writes=0;}
  get options(){return this.children.filter(n=>n.tagName==='OPTION');}
  set options(rows){this.children=rows;}
  get textContent(){return this._text+this.children.map(n=>n.textContent||'').join('');}
  set textContent(value){this._text=String(value);this.children=[];}
  set innerHTML(value){
   this.children=[];const stack=[this];
   for(const token of value.matchAll(/<([^>]+)>|([^<]+)/g)){
    if(token[2]){stack.at(-1)._text+=token[2];continue;}
    if(token[1][0]==='/'){stack.pop();continue;}
    const tag=/^([\w-]+)/.exec(token[1]);if(!tag)continue;
    const child=new Element(tag[1]);
    for(const attr of token[1].matchAll(/([\w-]+)(?:="([^"]*)")?/g))if(attr.index>0)child.setAttribute(attr[1],attr[2]??'');
    stack.at(-1).appendChild(child);if(!token[1].endsWith('/')&&!['INPUT','BR','IMG'].includes(child.tagName))stack.push(child);
   }
  }
  appendChild(node){if(node.parentNode)node.remove();this.children.push(node);node.parentNode=this;return node;}
  insertBefore(node,next){if(node.parentNode)node.remove();const at=next?this.children.indexOf(next):-1;this.children.splice(at<0?this.children.length:at,0,node);node.parentNode=this;return node;}
  remove(){if(this.parentNode){this.parentNode.children.splice(this.parentNode.children.indexOf(this),1);this.parentNode=null;}}
  setAttribute(k,v){this.attributes[k]=String(v);this.writes++;if(k==='class')this.className=String(v);if(k==='hidden')this.hidden=true;}
  getAttribute(k){return Object.hasOwn(this.attributes,k)?this.attributes[k]:null;}
  removeAttribute(k){delete this.attributes[k];if(k==='value')this.value=undefined;}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  querySelectorAll(selector){
   const parts=selector.split(' '),leaf=parts.at(-1),matches=n=>leaf.startsWith('.')?(n.className||'').split(' ').includes(leaf.slice(1)):leaf.startsWith('[value=')?n.value===leaf.slice(8,-2):n.tagName===leaf.toUpperCase();
   const all=[];const walk=n=>{for(const c of n.children){if(matches(c))all.push(c);walk(c);}};walk(this);
   if(parts.length===1)return all;
   return all.filter(n=>{let p=n.parentNode;while(p&&p!==this){if((p.className||'').split(' ').includes(parts[0].slice(1)))return true;p=p.parentNode;}return false;});
  }
  showModal(){this.open=true;}close(){this.open=false;}focus(){ctx.document.activeElement=this;}
 }
 function element(id){if(!elements.has(id)){const el=new Element(id==='composer-model'?'select':'div');el.id=id;elements.set(id,el);}return elements.get(id);}
 const noThinking=new Element('option');noThinking.value='none';element('composer-effort').appendChild(noThinking);
 element('composer-model').value=IDS[0];
 const jobs=Object.fromEntries(IDS.map(model_id=>[model_id,{state:'idle',bytes_downloaded:0,total_bytes:100,progress:0,error:null,can_cancel:false,model_id}]));
 const backend={model:IDS[0],jobs,nextError:null,rejectSwitch:false,rollback:false};
 const ctx={console,Map,Promise,Number,JSON,Array,String,Error,encodeURIComponent,AbortSignal,
  document:{createElement:tag=>new Element(tag)},StrataI18n:{lang:'en'},SPRITE:'web/sprite.svg',
  $:element,health:{model:IDS[0],images:true},settings:{thinking:'high'},messages:[],busy:null,pending:null,
  sharedSaving:false,toolSettingsSaving:false,modelOp:null,resourceOperation:null,panel:null,lastMetrics:{live:{state:'idle'}},inflight:false,
  tr:(key,en,values={})=>en.replace(/\{(\w+)\}/g,(_,k)=>values[k]),fmt:(v,d=0)=>Number(v).toFixed(d),
  store:{get:(key,d)=>storage[key]??d,set:(key,value)=>{storage[key]=JSON.parse(JSON.stringify(value));return true;}},
  turnBlocksUnload:()=>ctx.inflight||ctx.busy!==null,pageBusy:()=>ctx.busy!==null||ctx.resourceOperation!==null,
  toast:(...args)=>toasts.push(args),paintLive(){},setPill(){},
  setTimeout:(fn,delay)=>{timers.set(++tid,{fn,delay});return tid;},clearTimeout:id=>timers.delete(id),
  headers:()=>({}),fetch:()=>{throw Error('Real fetch prohibited');},
  loadHealth:async()=>{ctx.health.model=backend.model;return true;},
  setBusy:()=>{},backend,calls,storage};
 vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(webDirectory,'model-library.js'),'utf8'),ctx);
 vm.runInContext(fs.readFileSync(path.join(webDirectory,'thinking-controls.js'),'utf8'),ctx);vm.runInContext(snippet,ctx);
 vm.runInContext(`modelCatalog=${JSON.stringify(IDS.map((id,i)=>({id,installed:i===0,active:i===0,management:{source:'managed',can_delete:i!==0,can_download:i!==0,complete:i===0,disk_bytes:i===0?1000000000:0,reclaimable_bytes:1000000000,download_bytes:2000000000}})))};
 modelSwitchAvailable=true;
 const productionRefresh=refreshModelCatalog;
 refreshModelCatalog=async()=>{syncComposerControls();renderModelLibrary();};
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
   if(path==='v1/model/delete'){const row=modelCatalog.find(row=>row.id===body.model);row.installed=false;row.management.complete=false;row.management.disk_bytes=0;row.management.can_delete=false;row.management.can_download=true;return {model_id:body.model,deleted_bytes:1000000000,retained_shared_bytes:0,catalog:JSON.parse(JSON.stringify(modelCatalog))};}
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
  library:renderModelLibrary,openDelete:openModelDelete,delete:deleteModel,dismissDelete:dismissModelDelete,catalog:()=>modelCatalog,refresh:productionRefresh,useProductionRefresh:()=>{refreshModelCatalog=productionRefresh;},
  ready:activeModelNeedsDownload,setCatalog:(rows,available=true)=>{modelCatalog=rows;modelSwitchAvailable=available;},
  install:(id,complete=true)=>{const row=modelCatalog.find(row=>row.id===id);row.installed=true;row.management.complete=complete;row.management.can_download=!complete;row.management.disk_bytes=1000000000;}};`,ctx);
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
 const html=fs.readFileSync(path.join(webDirectory,'index.html'),'utf8');
 ok(html.includes('id="tab-btn-models"')&&html.includes('id="view-models"'),'Models has an explicit tab and panel');
 ok(!html.includes('<option value="tiel-coder-35b-a3b">'),'composer no longer embeds uninstalled model options');
 let f=fixture();f.t.controls();f.t.library();
 ok(f.element('composer-model').options.length===1&&f.element('composer-model').options[0].value===IDS[0],'picker contains only downloaded complete models');
 f.t.install(IDS[4],false);f.t.controls();
 ok(!f.element('composer-model').options.some(o=>o.value===IDS[4]),'main GGUF with missing required assets is excluded');
 f.t.install(IDS[1]);f.t.controls();
 ok(f.element('composer-model').options.some(o=>o.value===IDS[1]),'complete downloaded model joins picker');
 const pickerOption=f.element('composer-model').options[0];f.t.controls();
 ok(f.element('composer-model').options[0]===pickerOption,'unchanged picker options retain identity');
 const list=f.element('model-library-list'),row=list._modelRows.get(IDS[1]),image=row.image,button=row.download;
 button.focus();const sourceWrites=image.writes;
 for(let i=0;i<20;i++)f.t.library();
 ok(list.children.length===5&&list._modelRows.get(IDS[1])===row,'all five model rows retain keyed DOM');
 ok(row.image===image&&image.writes===sourceWrites&&row.download===button&&f.ctx.document.activeElement===button,'catalog polls preserve image src, button and focus');
 ok(image.tagName==='IMG'&&image.getAttribute('alt')===''&&image.getAttribute('aria-hidden')==='true','model images are decorative beside their text names');
 for(const id of IDS){const modelImage=list._modelRows.get(id).image;ok(modelImage.getAttribute('src')===f.ctx.StrataModelLibrary.get(id).iconSrc&&!modelImage.getAttribute('src').includes('#'),'each model renders its local standalone official or generic asset');}
 ok(list._modelRows.get(IDS[0]).remove.disabled&&list._modelRows.get(IDS[0]).status.textContent==='In use','active model clearly disables delete');
 f.element('composer-model').value=IDS[2];await f.element('composer-model').onchange();
 ok(f.calls.length===0&&!f.element('model-download-dialog').open,'invalid programmatic selection cannot fetch or download');
 f.element('composer-model').value=IDS[4];await f.element('composer-model').onchange();
 ok(f.calls.length===0,'incomplete programmatic selection cannot start loading or transfer');
 const noModels=fixture(),missingRows=JSON.parse(JSON.stringify(noModels.t.catalog()));
 for(const row of missingRows){row.installed=false;row.management.complete=false;row.management.can_download=true;row.management.can_delete=false;row.management.disk_bytes=0;}
 noModels.ctx.health.loaded=false;noModels.t.setCatalog(missingRows);noModels.t.controls();noModels.t.library();
 ok(noModels.element('composer-model').options.length===0&&noModels.element('composer-model').disabled,'zero installed models leaves an empty disabled picker');
 ok(noModels.t.ready()&&noModels.element('input').disabled&&noModels.element('send-btn').disabled&&noModels.element('input').placeholder==='Download a model to chat','known missing active model disables composer with a short placeholder');
 ok(noModels.element('model-library-list')._modelRows.get(IDS[0]).status.textContent==='Not downloaded'&&!noModels.element('model-library-list')._modelRows.get(IDS[0]).download.disabled,'missing active profile is downloadable in Models and is not labelled in use');
 const sendStart=source.indexOf('async function send(payload)'),replyStart=source.indexOf('async function generateReply()'),replyEnd=source.indexOf('\n$("composer").onsubmit',replyStart);
 vm.runInContext(source.slice(sendStart,replyStart)+'\n'+source.slice(replyStart,replyEnd),noModels.ctx);
 await noModels.ctx.send({text:'must not request',images:[],files:[]});await noModels.ctx.generateReply();
 ok(noModels.calls.length===0&&noModels.ctx.messages.length===0,'send and regenerate guards perform no fetch, download or chat mutation when the model is missing');
 noModels.t.setCatalog(missingRows,false);noModels.t.controls();
 ok(!noModels.t.ready()&&!noModels.element('input').disabled,'temporary catalog failure retains legacy/unknown chat behaviour');
 const legacyRows=JSON.parse(JSON.stringify(missingRows));delete legacyRows[0].installed;noModels.t.setCatalog(legacyRows);noModels.t.controls();
 ok(!noModels.t.ready()&&!noModels.element('input').disabled,'legacy unknown installation state does not block chat');
 const localRows=JSON.parse(JSON.stringify(missingRows));localRows[0].installed=true;localRows[0].management.complete=null;localRows[0].management.source='local';noModels.t.setCatalog(localRows);noModels.t.controls();
 ok(!noModels.t.ready()&&!noModels.element('input').disabled&&noModels.element('composer-model').options.length===1,'valid installed custom model with complete null stays usable');
 localRows[0].management.complete=false;noModels.t.setCatalog(localRows);noModels.t.controls();noModels.t.library();
 ok(noModels.t.ready()&&noModels.element('input').disabled&&noModels.element('model-library-list')._modelRows.get(IDS[0]).status.textContent==='Incomplete','known missing required assets blocks chat and labels the active profile incomplete');
 f.element('composer-model').value=IDS[1];await f.element('composer-model').onchange();await drain();
 ok(posts(f,'v1/model/switch').length===1&&posts(f,'v1/model/download').length===0&&f.ctx.health.model===IDS[1],'installed picker selection only switches the model');

 f=fixture();await f.t.open(IDS[1]);
 ok(f.element('model-download-dialog').open&&f.element('model-download-description').textContent.includes('0.00 GB'),'management confirmation includes reported size');
 ok(posts(f,'v1/model/download').length===0,'opening confirmation starts no transfer');
 f.t.dismiss();ok(!f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===0,'cancelling confirmation performs no mutation');
 await f.t.open(IDS[1]);let flow=f.t.flows.get(IDS[1]);await f.t.start(flow);
 ok(posts(f,'v1/model/download').length===1&&flow.stage==='downloading'&&!flow.autoSwitch,'confirm starts one download without switch intent');
 const downloadRow=f.element('model-library-list')._modelRows.get(IDS[1]),originalIcon=downloadRow.image,originalAction=downloadRow.download;
 f.t.dismiss();ok(f.ctx.health.model===IDS[0]&&!f.element('model-download-dialog').open,'background transfer leaves active model and modal alone');
 Object.assign(f.backend.jobs[IDS[1]],{bytes_downloaded:50,progress:.5});await f.t.poll(flow);
 ok(downloadRow.status.textContent.includes('50.0%')&&downloadRow.progress.value===50,'management row shows authoritative byte progress with modal closed');
 ok(downloadRow.image===originalIcon&&downloadRow.download===originalAction,'download progress preserves row images and controls');
 await f.t.open(IDS[1]);ok(f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===1,'management Progress reopens existing transfer without duplicate POST');
 await f.t.cancel(flow);ok(posts(f,'v1/model/download/cancel').length===1&&flow.stage==='cancelling','explicit cancel uses the separate endpoint');
 Object.assign(f.backend.jobs[IDS[1]],{state:'cancelled',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='cancelled'&&posts(f,'v1/model/switch').length===0,'cancelled transfer cannot switch');
 await f.t.start(flow);ok(posts(f,'v1/model/download').length===2,'resume restarts the same managed job');
 f.t.install(IDS[1]);Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false,progress:1,bytes_downloaded:100});await f.t.poll(flow);await drain();
 ok(flow.stage==='ready'&&posts(f,'v1/model/switch').length===0&&f.ctx.health.model===IDS[0],'completion never switches even while idle');
 ok(f.storage['model-downloads'].length===0,'completed transfer is removed from recovery state');
 ok(f.element('model-download-action').hidden,'completed progress offers no switch action');
 await f.element('model-download-action').onclick();
 ok(f.ctx.health.model===IDS[0]&&posts(f,'v1/model/switch').length===0,'even a programmatic completed action cannot switch');

 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);let seen=false;const base=f.ctx.modelControl;
 f.ctx.modelControl=async(path,options={})=>{if(path==='v1/model/download'&&options.method==='POST'&&!seen){seen=true;f.calls.push({path,method:'POST'});Object.assign(f.backend.jobs[IDS[1]],{state:'downloading',can_cancel:true});throw Error('lost POST response');}return base(path,options);};
 await f.t.start(flow);ok(flow.stage==='recovering','ambiguous POST failure enters recovery');
 await f.t.poll(flow);ok(flow.stage==='downloading'&&posts(f,'v1/model/download').length===1,'read-only recovery never duplicates POST');
 Object.assign(f.backend.jobs[IDS[1]],{state:'error',error:'disk full',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='error'&&flow.error==='disk full'&&posts(f,'v1/model/switch').length===0,'job errors remain errors without loading a model');

 f=fixture([{model:IDS[1],switchWhenReady:true}]);Object.assign(f.backend.jobs[IDS[1]],{state:'downloading',can_cancel:true});await f.t.recover();flow=f.t.flows.get(IDS[1]);
 ok(flow&&flow.confirmed&&!flow.autoSwitch&&!f.element('model-download-dialog').open,'reload restores progress while retiring old automatic switch intent');
 ok(posts(f,'v1/model/download').length===0,'reload recovery stays read-only');
 Object.assign(f.backend.jobs[IDS[1]],{state:'ready',can_cancel:false});await f.t.poll(flow);
 ok(flow.stage==='ready'&&posts(f,'v1/model/switch').length===0&&f.ctx.health.model===IDS[0],'legacy recovered completion never switches');
 f=fixture();Object.assign(f.backend.jobs[IDS[2]],{state:'downloading',can_cancel:true});await f.t.recover();flow=f.t.flows.get(IDS[2]);
 ok(flow&&!flow.autoSwitch&&!flow.confirmed,'externally started transfers appear without switch intent');

 f=fixture();f.t.install(IDS[1]);f.t.library();f.t.openDelete(IDS[0]);
 ok(!f.element('model-delete-dialog').open&&posts(f,'v1/model/delete').length===0,'active model cannot enter delete confirmation');
 f.t.openDelete(IDS[1]);ok(f.element('model-delete-dialog').open&&f.element('model-delete-title').textContent==='Delete Gemma 4?'&&f.element('model-delete-size').textContent==='The model files for Gemma 4 (1.00 GB) will be deleted.','delete confirmation explains the model files and reclaimable size');
 f.t.dismissDelete();ok(posts(f,'v1/model/delete').length===0,'delete cancellation is mutation-free');
 f.t.openDelete(IDS[1]);f.backend.nextError=Object.assign(Error('busy'),{status:409});await f.t.delete();
 ok(f.element('model-delete-dialog').open&&f.element('model-delete-error').textContent==='busy'&&f.t.catalog().find(r=>r.id===IDS[1]).installed,'rejected delete keeps files/catalog and exposes the error');
 await f.t.delete();
 ok(!f.element('model-delete-dialog').open&&!f.t.catalog().find(r=>r.id===IDS[1]).installed,'confirmed delete refreshes from the returned catalog');
 ok(!f.element('composer-model').options.some(o=>o.value===IDS[1])&&f.ctx.health.model===IDS[0],'deleted model leaves picker without switching current model');
 ok(posts(f,'v1/model/delete').at(-1).body.model===IDS[1]&&f.storage.chat===undefined,'delete targets only the model and never touches chat storage');
 f=fixture();f.t.install(IDS[1]);f.t.controls();
 let resolveCatalog;const beforeDelete=JSON.parse(JSON.stringify(f.t.catalog())),deleteBase=f.ctx.modelControl;
 f.ctx.modelControl=(path,options)=>path==='v1/models'?new Promise(resolve=>{resolveCatalog=resolve;}):deleteBase(path,options);
 const oldRead=f.t.refresh();f.t.openDelete(IDS[1]);await f.t.delete();resolveCatalog({data:beforeDelete});await oldRead;
 ok(!f.t.catalog().find(r=>r.id===IDS[1]).installed&&!f.element('composer-model').options.some(o=>o.value===IDS[1]),'catalog GET started before deletion cannot resurrect the model');
 f=fixture();f.t.useProductionRefresh();const beforeReady=JSON.parse(JSON.stringify(f.t.catalog())),readyBase=f.ctx.modelControl;let catalogReads=0,resolveOldCatalog;
 f.ctx.modelControl=(path,options)=>path==='v1/models'?(++catalogReads===1?new Promise(resolve=>{resolveOldCatalog=resolve;}):Promise.resolve({data:JSON.parse(JSON.stringify(f.t.catalog()))})):readyBase(path,options);
 const oldReadyRead=f.t.refresh();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);f.t.install(IDS[1]);
 await f.t.status(flow,{...f.backend.jobs[IDS[1]],state:'ready'});resolveOldCatalog({data:beforeReady});await oldReadyRead;await drain();
 ok(catalogReads===2&&f.element('composer-model').options.some(o=>o.value===IDS[1]),'ready download invalidates older catalog and queues a fresh read');
 f=fixture();f.t.install(IDS[1]);f.t.useProductionRefresh();
 const partialCatalog=JSON.parse(JSON.stringify(f.t.catalog())),partialRow=partialCatalog.find(r=>r.id===IDS[1]),partialBase=f.ctx.modelControl;
 partialRow.installed=false;partialRow.management.complete=false;partialRow.management.can_delete=false;
 let failedDeletes=0;
 f.ctx.modelControl=async(path,options)=>{
   if(path==='v1/model/delete'){failedDeletes++;throw Object.assign(Error('partially deleted'),{status:500});}
   if(path==='v1/models')return {data:partialCatalog};
   return partialBase(path,options);
 };
 f.t.openDelete(IDS[1]);await f.t.delete();
 ok(failedDeletes===1&&!f.element('composer-model').options.some(o=>o.value===IDS[1]),'partial delete failure refreshes file availability without retrying POST');
 ok(f.element('model-delete-dialog').open&&f.element('model-delete-error').textContent==='partially deleted'&&f.element('model-delete-confirm').disabled,'partial failure remains visible and cannot confirm a now-ineligible delete');
 f=fixture();f.t.install(IDS[1]);delete f.t.catalog().find(r=>r.id===IDS[1]).management;f.t.controls();f.t.library();f.t.openDelete(IDS[1]);
 ok(f.element('composer-model').options.some(o=>o.value===IDS[1])&&!f.element('model-delete-dialog').open,'older backend retains downloaded picker support but cannot delete');
 f=fixture();await f.t.open(IDS[1]);flow=f.t.flows.get(IDS[1]);f.backend.nextError=Object.assign(Error('need key'),{status:401});await f.t.start(flow);
 ok(flow.stage==='error'&&posts(f,'v1/model/download').length===0,'authentication failure cannot start a download');
 f=fixture();let resolveConfirmation;const original=f.ctx.modelControl;
 f.ctx.modelControl=(path,options)=>path.includes('download?')?new Promise(resolve=>{resolveConfirmation=resolve;}):original(path,options);
 const opening=f.t.open(IDS[1]);f.t.dismiss();resolveConfirmation({...f.backend.jobs[IDS[1]]});await opening;
 ok(!f.element('model-download-dialog').open&&posts(f,'v1/model/download').length===0,'late confirmation response cannot reopen or start transfer');
 await workerRegressions();
 f=fixture();f.ctx.resourceOperation='placement';f.t.controls();
 ok(f.element('composer-model').disabled&&f.element('composer-effort').disabled,'placement reload locks model and effort controls');
 ok(f.element('s-web-search').disabled&&f.element('s-apply').disabled,'resource operation keeps request settings stable');
 await f.t.open(IDS[1]);
 ok(posts(f,'v1/model/switch').length===0,'a resource operation cannot race a model switch');
 const required=[...snippet.matchAll(/tr\("(model\.download[^\"]+|models\.deleteDescription(?:Size)?)"/g)].map(m=>m[1]);
 for(const name of fs.readdirSync(localeDirectory)){
   if(!name.endsWith('.json'))continue;
   const locale=JSON.parse(fs.readFileSync(path.join(localeDirectory,name),'utf8'));
   ok(required.every(key=>typeof locale[key]==='string'),'all dynamic labels localized '+name);
 }
 console.log(JSON.stringify({checks,passed:true,api_started:false,download_started:false,gpu_started:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
