'use strict';
// Resource operations are injected. No network, model process, storage or GPU.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const R=require('../ui/web/resource-dialogs.js');
let checks=0;
const ok=(value,label)=>{assert.ok(value,label);checks++;};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Element {
 constructor(doc){this.doc=doc;this.nodes=new Map();this.listeners=new Map();this.attributes={};this.dataset={};this.style={};this.open=false;this.hidden=false;this.disabled=false;this.isConnected=true;this.textContent='';this.value='50';}
 setAttribute(k,v){this.attributes[k]=String(v);}
 removeAttribute(k){delete this.attributes[k];}
 querySelector(key){if(!this.nodes.has(key))this.nodes.set(key,new Element(this.doc));return this.nodes.get(key);}
 querySelectorAll(){if(!this.radios){this.radios=[new Element(this.doc),new Element(this.doc)];this.radios[0].value='auto';this.radios[1].value='manual';}return this.radios;}
 addEventListener(name,fn){if(!this.listeners.has(name))this.listeners.set(name,[]);this.listeners.get(name).push(fn);}
 emit(name){const event={defaultPrevented:false,preventDefault(){this.defaultPrevented=true;}};for(const fn of this.listeners.get(name)||[])fn(event);return event;}
 click(){if(!this.disabled)this.emit('click');}
 focus(){this.doc.activeElement=this;}
 showModal(){this.open=true;}
 close(){this.open=false;this.emit('close');}
 appendChild(node){node.isConnected=true;return node;}
 remove(){this.isConnected=false;}
}
function document(){const d={createElement(){return new Element(d);}};d.body=new Element(d);d.activeElement=new Element(d);return d;}
function fixture(extra={}){
 const doc=document(),state={model:'model-a',busy:false,kv:{used:750,total:1000,canCompact:true},placement:{gpuPct:65,cpuPct:35,requestedGpuPercent:null,adjustable:true}};
 const calls=[];
 const api=R.create({document:doc,getSnapshot:()=>state,onCompact:async(args)=>{calls.push(args);state.kv.used=250;return {beforeTokens:750,afterTokens:250,summary:'Remember the decision.'};},onPlacement:async(args)=>{calls.push(args);state.placement.requestedGpuPercent=args.gpuPercent;},...extra});
 return {doc,state,api,calls,find:s=>api.element.querySelector(s)};
}
async function resourcePolicyChecks(){
 const app=fs.readFileSync(path.join(__dirname,'../ui/web/app.js'),'utf8');
 const actions=fs.readFileSync(path.join(__dirname,'../ui/web/resource-actions.js'),'utf8');
 const block=(start,end)=>{const at=app.indexOf(start),stop=app.indexOf(end,at);assert(at>=0&&stop>at);return app.slice(at,stop);};
 const doc=document(),nodes=new Map(),requests=[];
 const ctx={console,Map,Number,Math,JSON,Error,Promise,AbortController,
  $:id=>{if(!nodes.has(id))nodes.set(id,new Element(doc));return nodes.get(id);},
  tr:(key,fallback,values={})=>fallback.replace(/\{(\w+)\}/g,(_,key)=>values[key]),fmt:String,kfmt:String,ctxfmt:String,
  lastMetrics:null,placementSampleValid:true,modelSwitching:false,modelOp:null,health:{model:'qwen3.6-35b-a3b'},messages:[],chat:{},
  modelOperationBusy:()=>false,StrataContextCompaction:{canCompact:()=>true},
  modelControl:async(url,options)=>{requests.push({url,options});return {placement:{model:ctx.health.model,supported:true,gpu_percent:null,expert_total:10240,count_step:40}};}};
 vm.createContext(ctx);vm.runInContext(block('function umaMemory(','function gpuMemoryReading(')+block('function expertPlacement(','function render(m)')+actions,ctx);
 const intel={engine:{backend:'sycl',memory_model:'uma',expert_slots:0,kv_used_tokens:7112,max_context:262144},hardware_static:{gpu_memory_model:'uma'},live:{state:'idle'}};
 const paint=metrics=>{ctx.lastMetrics=metrics;ctx.renderPlacement(metrics);ctx.renderContext(metrics);};
 paint(intel);
 ok(ctx.$('chat-placement').disabled&&ctx.$('chat-placement').attributes['aria-disabled']==='true','UMA footer is a disabled button, not a clickable placement control');
 ok(!Object.hasOwn(ctx.$('chat-placement').attributes,'aria-haspopup'),'fixed UMA policy does not announce a placement popup');
 ok(ctx.$('placement-gpu').textContent==='100%'&&ctx.$('placement-cpu').textContent==='0%'&&!ctx.$('placement-track').hidden,'Intel UMA shows the fixed GPU execution policy without expert residency counters');
 ok(ctx.$('placement-gpu-bar').style.width==='100%'&&ctx.$('placement-cpu-bar').style.width==='0%','fixed UMA execution draws a complete GPU share');
 ok(ctx.$('chat-placement').title.includes('policy')&&!ctx.$('chat-placement').title.includes('undefined'),'tooltip identifies policy rather than inventing counts');
 ok(ctx.$('context-value').textContent==='7112 / 262144'&&ctx.$('context-pct').textContent.startsWith('2.7'),'actual Intel committed KV usage and logical maximum remain visible');
 vm.runInContext('resourcePlacementSettings.set(health.model,{supported:true,gpu_percent:20,gpu_experts:2048,expert_total:10240,count_step:40})',ctx);
 let snapshot=ctx.resourceSnapshot();
 ok(snapshot.placement.fixed&&snapshot.placement.adjustable===false&&snapshot.placement.gpuPct===100&&snapshot.placement.cpuPct===0,'UMA policy overrides stale adjustable managed-endpoint settings');
 ok(snapshot.placement.total===undefined&&snapshot.placement.gpu===undefined&&snapshot.placement.requestedGpuExperts===undefined,'UMA policy never fabricates expert counts or retains a stale target');
 const request={model:ctx.health.model,signal:new AbortController().signal};
 ok((await ctx.readResourcePlacement(request)).adjustable===false&&requests.length===0,'read-only UMA settings do not query a discrete placement endpoint');
 for(const target of [{gpuPercent:0},{gpuExperts:0}]) await assert.rejects(ctx.applyResourcePlacement({...request,...target}),/policy/);
 ok(requests.length===0&&ctx.placementSampleValid,'direct UMA percentage and expert-count mutations fail before network or state invalidation');
 let opens=0;
 ctx.StrataResourceDialogs={create:()=>({openKV:()=>{opens++;},openPlacement:()=>{opens++;},refresh(){}})};
 ctx.initResourceDialogs();ctx.$('chat-placement').onclick();
 ok(opens===0,'even a direct invocation of the UMA click handler cannot open placement controls');
 ctx.$('chat-context').onclick();ok(opens===1,'KV usage and compaction stay independently accessible');
 for(const transition of [{state:'unloaded'}, {state:'idle',switching:true}, {state:'idle',operation:'load'}]){
  ctx.placementSampleValid=false;ctx.modelSwitching=!!transition.switching;ctx.modelOp=transition.operation||null;
  paint({...intel,engine:{memory_model:'uma',kv_used_tokens:999,max_context:262144},live:{state:transition.state}});
  ok(ctx.$('placement-gpu').textContent==='100%'&&ctx.$('chat-placement').disabled,'fixed UMA policy persists through unload, load and switch without claiming an execution sample');
  ok(ctx.$('context-value').textContent==='–'&&ctx.resourceSnapshot().kv.used===undefined,'stale KV usage is cleared during unload, load and switch');
 }
 ctx.placementSampleValid=true;ctx.modelSwitching=false;ctx.modelOp=null;
 for(const backend of ['cuda','hip','sycl']){
  paint({engine:{backend,memory_model:'discrete',expert_total:10000,expert_slots:3200,kv_used_tokens:0,max_context:32768},hardware_static:{gpu_memory_model:'discrete'},live:{state:'idle'}});
  ok(!ctx.$('chat-placement').disabled&&ctx.$('chat-placement').attributes['aria-haspopup']==='dialog'&&ctx.$('placement-gpu').textContent==='32%','discrete '+backend+' keeps measured placement and an operable dialog');
  ok(ctx.resourceSnapshot().placement.adjustable===true&&ctx.resourceSnapshot().placement.gpu===3200,'discrete '+backend+' retains managed settings and measured resident counts');
  ok(ctx.$('context-value').textContent==='0 / 32768','empty discrete KV remains a valid zero');
 }
 paint({engine:{memory_model:'discrete'},hardware_static:{},live:{state:'idle'}});
 ok(ctx.$('context-value').textContent==='–'&&ctx.$('placement-gpu').textContent==='–','missing discrete metrics stay unknown instead of inheriting UMA policy or KV usage');
 let finishRead;ctx.modelControl=(url,options)=>{requests.push({url,options});return new Promise(resolve=>finishRead=resolve);};
 const pending=ctx.readResourcePlacement(request);paint(intel);
 finishRead({placement:{model:ctx.health.model,supported:true,gpu_percent:20,expert_total:10240,count_step:40}});
 const late=await pending;
 ok(late.fixed&&late.adjustable===false&&late.total===undefined,'a managed endpoint read that completes after UMA detection cannot re-enable controls');
}
async function main(){
 ok(R.usage({used:0,total:1000}).available===1000,'an empty context has all capacity available');
 ok(R.usage({used:1200,total:1000}).pct===100&&R.usage({used:1200,total:1000}).available===0,'over-limit contexts do not draw an invalid pie or negative free space');
 ok(R.usage({used:10,total:0})===null&&R.usage({used:NaN,total:100})===null,'unknown limits and invalid counts stay unknown');
 ok(R.targetPercent(null)===null&&R.targetPercent(0)===0&&R.targetPercent(100)===100,'automatic, CPU-only and all-GPU requests stay distinct');
 const geometry=R.countGeometry({total:10240,countStep:40});
 ok(geometry.total===10240&&geometry.step===40&&R.countGeometry({total:100,countStep:3})===null,'count geometry exposes only legal whole-layer steps');
 ok(R.targetCount(5000,geometry)===5000&&R.targetCount(5015,geometry)===5000&&R.targetCount(5025,geometry)===5040,'expert counts preserve exact values and round to legal native slots');
 ok(R.targetCount(-4,geometry)===0&&R.targetCount(999999,geometry)===10240&&R.targetCount(null,geometry)===null,'expert counts clamp without conflating zero and automatic');
 assert.throws(()=>R.create({}),TypeError);checks++;
 let f=fixture(),opener=f.doc.activeElement;
 f.api.openKV(opener);ok(f.api.element.open&&f.find('[data-percent]').textContent==='75%','KV dialog displays current-chat usage');
 ok(f.find('[data-used]').textContent.includes('750')&&f.find('[data-remaining]').textContent.includes('250'),'used and available token counts are explicit');
 f.find('[data-action]').click();await tick();
 ok(f.calls.length===1&&f.calls[0].model==='model-a'&&f.calls[0].signal instanceof AbortSignal,'manual compaction delegates the owned model and cancellation signal');
 ok(f.find('[data-percent]').textContent==='25%'&&f.find('.resource-dialog__status').textContent.includes('750'),'successful compaction updates the pie and result');
 ok(f.find('.resource-dialog__summary').hidden===false&&f.find('.resource-dialog__summary').querySelector('div').textContent==='Remember the decision.','summary is available as plain text');
 f.api.close();ok(f.doc.activeElement===opener,'closing restores the original trigger focus');
 f.state.busy=true;f.api.openKV(opener);ok(f.find('[data-action]').disabled,'busy inference cannot start another compaction');f.api.close();
 f.state.busy=false;f.api.openPlacement(opener);const radios=f.api.element.radios;
 ok(radios[0].checked&&f.find('input[type="range"]').disabled&&f.find('[data-action]').disabled,'automatic mode initially retains its current setting');
 radios[0].checked=false;radios[1].checked=true;radios[1].emit('change');const range=f.find('input[type="range"]');range.value='0';range.emit('input');
 ok(!range.disabled&&f.find('.resource-dialog__range output').textContent==='0%','CPU-only zero is a selectable custom target');
 f.api.refresh();ok(range.value==='0','live metrics do not overwrite an edited target');
 f.find('[data-action]').click();await tick();ok(f.calls.at(-1).gpuPercent===0,'zero is sent as zero, not automatic');
 ok(f.find('[data-gpu-actual]').textContent==='65%'&&f.find('.resource-dialog__requested').textContent.includes('0%'),'current and requested placement are shown separately');
 radios[0].checked=true;radios[1].checked=false;radios[0].emit('change');f.find('[data-action]').click();await tick();ok(f.calls.at(-1).gpuPercent===null,'automatic mode explicitly clears the custom target');
 f.api.close();
 f.state.placement.legacyBudgetGiB=3.5;f.api.openPlacement();ok(!f.api.element.radios[0].checked&&!f.api.element.radios[1].checked&&f.find('[data-action]').disabled,'a legacy GiB budget is not misrepresented as automatic');
 ok(f.find('.resource-dialog__requested').textContent.includes('3.5 GiB'),'the existing explicit memory limit is visible');
 f.api.element.radios[1].checked=true;f.api.element.radios[1].emit('change');ok(f.find('input[type="range"]').value==='70','custom mode starts near the current actual GPU share');f.api.close();delete f.state.placement.legacyBudgetGiB;
 f.state.placement.legacyBudget=true;f.state.placement.legacyExpertSlots=0;f.api.openPlacement();ok(!f.api.element.radios[0].checked&&!f.api.element.radios[1].checked&&!f.find('.resource-dialog__requested').textContent.includes('0 GiB'),'legacy expert slots stay manual without inventing a memory size');f.api.close();delete f.state.placement.legacyBudget;delete f.state.placement.legacyExpertSlots;
 let resolveReload,args;
 f=fixture({onPlacement:input=>{args=input;return new Promise(resolve=>{resolveReload=resolve;});}});f.api.openPlacement();const modes=f.api.element.radios;modes[0].checked=false;modes[1].checked=true;modes[1].emit('change');f.find('input[type="range"]').value='100';f.find('input[type="range"]').emit('input');f.find('[data-action]').click();
 ok(args.gpuPercent===100&&f.find('[data-close]').disabled&&f.find('[data-dismiss]').disabled,'model reload holds the dialog and preserves a 100 percent request');
 ok(f.api.close()===false&&f.api.element.emit('cancel').defaultPrevented&&f.api.element.open,'Escape and programmatic close cannot orphan an in-flight reload');
 args.onProgress('Reading model weights');f.api.refresh();ok(f.find('.resource-dialog__status').textContent==='Reading model weights','polling preserves detailed reload progress');
 resolveReload();await tick();ok(!f.find('[data-close]').disabled,'reload completion releases the dialog');f.api.destroy();
 let compactArgs;
 f=fixture({onCompact:input=>{compactArgs=input;return new Promise((resolve,reject)=>input.signal.addEventListener('abort',()=>reject(new DOMException('cancelled','AbortError'))));}});f.api.openKV();f.find('[data-action]').click();f.find('[data-close]').click();await tick();
 ok(compactArgs.signal.aborted&&f.api.element.open&&f.find('.resource-dialog__status').textContent.includes('stopped'),'Stop cancels compaction while keeping its result dialog visible');
 ok(f.state.kv.used===750,'cancellation never edits resource state');f.api.destroy();
 f=fixture({onCompact:async()=>{throw new Error('Summary failed');}});f.api.openKV();f.find('[data-action]').click();await tick();ok(f.find('.resource-dialog__status').textContent==='Summary failed'&&!f.find('[data-action]').disabled,'failed compaction shows a recoverable error');f.api.destroy();
 let readArgs,resolveRead;
 f=fixture({readKV:input=>{readArgs=input;return new Promise(resolve=>resolveRead=resolve);}});f.api.openKV();ok(f.find('[data-action]').disabled,'current-chat counting finishes before the action is enabled');f.api.close();f.api.openPlacement();resolveRead({used:1,total:10,canCompact:true});await tick();ok(readArgs.signal.aborted&&f.api.element.querySelector('[data-kv]').hidden,'a late count cannot overwrite another dialog');
 f.state.model='model-b';f.api.refresh();ok(f.find('[data-action]').disabled&&f.find('.resource-dialog__reason').textContent.includes('model changed'),'model switches invalidate old resource actions');f.api.destroy();
 ok(f.api.element.isConnected===false,'destroy removes the modal');
 const countCalls=[];
 f=fixture({onPlacement:async args=>{countCalls.push(args);f.state.placement.requestedGpuExperts=args.gpuExperts;}});
 Object.assign(f.state.placement,{total:10240,countStep:40,gpu:5000,cpu:5240,gpuPct:48.8,cpuPct:51.2,requestedGpuExperts:null});
 f.api.openPlacement();let countRange=f.find('input[type="range"]'),numberInput=f.find('input[type="number"]');
 ok(countRange.max==='10240'&&countRange.step==='40'&&countRange.dataset.unit==='experts','count mode uses the model capacity and native granularity');
 ok(f.find('[data-gpu-actual]').textContent==='5,000'&&f.find('[data-gpu-share]').textContent==='48.8%'&&!f.find('[data-gpu-share]').hidden,'resident counts are primary and percentages remain visible');
 ok(!numberInput.hidden&&numberInput.disabled,'automatic count mode has a visible disabled numeric target');
 f.api.element.radios[0].checked=false;f.api.element.radios[1].checked=true;f.api.element.radios[1].emit('change');
 ok(numberInput.value==='5000'&&!numberInput.disabled,'custom target starts from the actual resident count');
 numberInput.value='5015';numberInput.emit('input');f.api.refresh();ok(numberInput.value==='5015','metrics do not rewrite a partially typed count');
 numberInput.emit('change');ok(numberInput.value==='5000'&&f.find('.resource-dialog__range output').textContent.includes('5,000 / 10,240'),'committed numeric targets show the legal rounded count');
 f.find('[data-action]').click();await tick();ok(countCalls.at(-1).gpuExperts===5000&&!Object.hasOwn(countCalls.at(-1),'gpuPercent'),'count mode submits an exact expert count without a percentage payload');
 numberInput.value='0';numberInput.emit('input');f.find('[data-action]').click();await tick();ok(countCalls.at(-1).gpuExperts===0,'Apply commits typed zero as CPU-only');
 const numberCalls=countCalls.length;numberInput.value='';numberInput.emit('input');f.find('[data-action]').click();await tick();ok(countCalls.length===numberCalls,'empty numeric input never submits a mistaken zero or automatic target');
 f.api.element.radios[0].checked=true;f.api.element.radios[1].checked=false;f.api.element.radios[0].emit('change');f.find('[data-action]').click();await tick();ok(countCalls.at(-1).gpuExperts===null,'automatic count mode explicitly clears the expert target');f.api.destroy();
 // A dialog opened before topology is known must obey a newly fixed policy,
 // even if an earlier managed endpoint read returns editable discrete data.
 let completePlacementRead;
 f=fixture({readPlacement:()=>new Promise(resolve=>completePlacementRead=resolve)});f.api.openPlacement();
 f.state.placement={fixed:true,gpuPct:100,cpuPct:0,adjustable:false,reason:'Fixed GPU execution uses shared system RAM.'};f.api.refresh();
 completePlacementRead({gpuPct:65,cpuPct:35,requestedGpuPercent:20,adjustable:true,total:10240,countStep:40});await tick();
 ok(f.find('[data-gpu-actual]').textContent==='100%'&&f.find('[data-cpu-actual]').textContent==='0%','live UMA policy overrides cached managed endpoint percentages and counts');
 ok(f.find('[data-actual-label]').textContent==='Fixed execution policy'&&f.find('#'+f.api.element.attributes['aria-describedby']).textContent===f.state.placement.reason,'read-only direct dialog explains the fixed policy');
 ok(f.find('.resource-dialog__mode').hidden&&f.find('.resource-dialog__range').hidden&&f.find('[data-action]').hidden&&f.find('[data-action]').disabled,'direct UMA dialog has no reload or placement controls');
 f.api.element.radios[1].checked=true;f.api.element.radios[1].emit('change');f.find('[data-action]').click();await tick();
 ok(f.calls.length===0&&f.find('input[type="range"]').disabled,'synthetic changes in a read-only dialog cannot call the placement mutation');
 f.api.close();f.api.openKV();
 ok(!f.find('[data-action]').hidden&&!f.find('[data-action]').disabled,'returning to KV restores its independent supported action');f.api.destroy();
 await resourcePolicyChecks();
 console.log(`PASS ${checks} resource dialog checks`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
