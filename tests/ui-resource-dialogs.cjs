'use strict';
// Resource operations are injected. No network, model process, storage or GPU.
const assert=require('node:assert/strict');
const R=require('../ui/web/resource-dialogs.js');
let checks=0;
const ok=(value,label)=>{assert.ok(value,label);checks++;};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Element {
 constructor(doc){this.doc=doc;this.nodes=new Map();this.listeners=new Map();this.attributes={};this.dataset={};this.style={};this.open=false;this.hidden=false;this.disabled=false;this.isConnected=true;this.textContent='';this.value='50';}
 setAttribute(k,v){this.attributes[k]=String(v);}
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
 console.log(`PASS ${checks} resource dialog checks`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
