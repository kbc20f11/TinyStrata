'use strict';
// Production preference/payload/DOM helpers only. No browser, API or model runs.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const web=path.join(__dirname,'../ui/web'),app=fs.readFileSync(path.join(web,'app.js'),'utf8');
const library=fs.readFileSync(path.join(web,'thinking-controls.js'),'utf8');
const helpers=app.slice(app.indexOf('function effortForModel('),app.indexOf('function syncComposerControls('));
const drawer=app.slice(app.indexOf('const checked = (id)'),app.indexOf('function renderLocalToolStatus('));
const composer=app.slice(app.indexOf('$("composer-effort").onchange'),app.indexOf('function timeStr('));
const shared=app.slice(app.indexOf('function sharedDefaults('),app.indexOf('async function saveShared('));
const reset=app.slice(app.indexOf('$("s-reset").onclick'),app.indexOf('$("s-apply").onclick'));
const apply=app.slice(app.indexOf('$("s-apply").onclick'),app.indexOf('$("sampling-btn").onclick'));
const ids=['qwen3.6-35b-a3b','clover-35b-a3b-preview','gemma-4-26b-a4b-it','lfm2.5-24b-a2b','tiel-coder-35b-a3b'];
const plain=value=>JSON.parse(JSON.stringify(value));let checks=0;
function equal(value,expected,label){assert.deepStrictEqual(plain(value),expected,label);checks++;}
function ok(value,label){assert.ok(value,label);checks++;}
function fixture(saved={thinking:'high'}){
 const elements=new Map(),storage={};let mutations=0;
 class Element{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.parentNode=null;this.dataset={};this.attributes={};this.style={};this.value='';this.disabled=false;this._text='';}
  get options(){return this.children;}
  get textContent(){return this._text;}
  set textContent(value){this._text=String(value);mutations++;}
  appendChild(node){return this.insertBefore(node,null);}
  insertBefore(node,before){if(node.parentNode)node.remove();const index=before?this.children.indexOf(before):this.children.length;this.children.splice(index,0,node);node.parentNode=this;mutations++;return node;}
  remove(){this.parentNode.children.splice(this.parentNode.children.indexOf(this),1);this.parentNode=null;mutations++;}
  setAttribute(name,value){this.attributes[name]=String(value);mutations++;}
  getAttribute(name){return this.attributes[name]??null;}
  focus(){ctx.document.activeElement=this;}
  querySelectorAll(){return [];}
 }
 const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 const ctx={console,document:{createElement:tag=>new Element(tag)},$:element,
  health:{model:ids[0]},modelCatalog:[],modelSwitching:false,sharedSaving:false,pageBusy:()=>false,
  tr:(_key,fallback)=>fallback,store:{set:(key,value)=>{storage[key]=plain(value);}},
  outputs(){},projectionLoaded:()=>false,samplingUntouched:()=>true,sharedOn:false,
  resourceOperation:null,busy:null,panel:null,pending:null,messages:[],toolSettingsSaving:false,
  renderLocalToolStatus(){},renderEmptyChat(){},StrataMath:{reset(){}},setBusy(){},updateAssistant(){},openDrawer(){},toast(){},
  saveGlobalTools:async()=>{},saveShared:async()=>{},setChatList:async()=>{},releasePending(){},
  DEFAULTS:{thinkingByModel:{},temperature:1,top_p:.95,top_k:20,max:'',seed:'',math:true,show:true,esp:true,mcp:true,autoCompact:true,chatList:true,whileBusy:'wait',webSearch:true}};
 vm.createContext(ctx);vm.runInContext(library,ctx);
 ctx.settings=ctx.StrataThinkingControls.migrate({...ctx.DEFAULTS,...saved});
 vm.runInContext(helpers+'\n'+drawer+'\n'+composer+'\n'+shared+'\n'+reset+'\n'+apply,ctx);
 ctx.syncComposerControls=()=>ctx.syncThinkingControls();
 ctx.setComposerTools=changes=>{ctx.settings={...ctx.settings,...changes};};
 const busy=element('s-busy');for(const value of ['wait','queue','interject']){const b=new Element('button');b.dataset.v=value;b.setAttribute('aria-checked',String(value==='wait'));busy.appendChild(b);}
 return {ctx,element,storage,mutations:()=>mutations,model(id,metadata){ctx.health.model=id;ctx.modelCatalog=metadata?[{id,thinking:metadata}]:[];ctx.syncThinkingControls();},
  values:()=>element('composer-effort').options.map(node=>node.value),radios:()=>element('s-thinking').children.map(node=>node.dataset.v)};
}
async function main(){
 const f=fixture(),T=f.ctx.StrataThinkingControls;
 for(const id of ids){
  f.model(id);const effort=id===ids[4],always=id===ids[3];
  equal(f.values(),effort?['none','low','medium','xhigh']:always?['on']:['none','on'],'model-native options');
  equal(f.radios(),f.values(),'drawer and composer use the same choices and order');
  equal(f.element('composer-effort').value,effort?'medium':'on','legacy factory high migrates to native defaults');
  equal(f.ctx.thinkingPayload(),effort?{reasoning_effort:'medium'}:{chat_template_kwargs:{enable_thinking:true}},'request uses canonical native controls');
  equal(f.ctx.sharedDefaults(f.ctx.settings),effort?{reasoning_effort:'medium'}:{enable_thinking:true},'shared defaults use boolean or effort, never on');
  equal(f.element('composer-effort').disabled,always,'always-on control is fixed');
 }
 ok(!Object.hasOwn(f.ctx.settings,'thinking'),'retired global preference removed');
 for(const value of [null,'high','max',3,{},false])equal(T.choice(ids[4],{thinkingByModel:{[ids[4]]:value}}),'medium','invalid saved Tiel level falls back safely');
 equal(T.migrate({thinking:'none',thinkingByModel:{[ids[0]]:'none',[ids[4]]:'xhigh',unknown:'on'},seed:'123'}),
  {thinkingByModel:{[ids[0]]:'none',[ids[1]]:'none',[ids[2]]:'none',[ids[4]]:'xhigh'},seed:'123'},'explicit legacy off is preserved; valid per-model choice wins and unrelated settings retained');
 for(const legacy of ['none','low','medium','high']){
  const migrated=T.migrate({thinking:legacy,seed:'keep',futureOption:{enabled:true}});
  for(const id of ids){
   const expected=id===ids[3]?'on':legacy==='none'?'none':id===ids[4]&&legacy!=='high'?legacy:id===ids[4]?'medium':'on';
   equal(T.choice(id,migrated),expected,'legacy '+legacy+' migrates meaningfully for '+id);
  }
  equal(migrated.futureOption,{enabled:true},'migration retains unrelated future settings');
 }
 equal(T.choice(ids[0],T.migrate({thinking:'none',thinkingByModel:{[ids[0]]:'on'}})),'on','explicit model ON takes precedence over legacy OFF');
 equal(T.choice(ids[4],T.migrate({thinking:'low',thinkingByModel:{[ids[4]]:'medium'}})),'medium','explicit model effort takes precedence over legacy effort');
 for(const metadata of [null,{}, {mode:'effort',default:'max',options:['none','max']},{mode:'toggle',default:'on',options:['none','on','max']}])
  equal(T.contract(ids[4],metadata).options,['none','low','medium','xhigh'],'invalid/older metadata has known-model fallback');
 f.model(ids[0]);f.element('composer-effort').value='none';await f.element('composer-effort').onchange();
 f.model(ids[4]);f.element('composer-effort').value='xhigh';await f.element('composer-effort').onchange();
 equal(f.ctx.thinkingPayload(),{reasoning_effort:'xhigh'},'High is xhigh on the actual request');
 f.model(ids[0]);equal(f.element('composer-effort').value,'none','switching restores Qwen preference');
 equal(f.ctx.thinkingPayload(),{chat_template_kwargs:{enable_thinking:false}},'toggle off is explicit false');
 f.model(ids[2]);equal(f.element('composer-effort').value,'on','Gemma does not inherit Qwen off');
 const reload=fixture(f.storage.sampling);reload.model(ids[4]);equal(reload.element('composer-effort').value,'xhigh','reload restores model-specific explicit choice');
 f.model(ids[3]);f.element('composer-effort').value='none';await f.element('composer-effort').onchange();
 equal(f.ctx.thinkingPayload(),{chat_template_kwargs:{enable_thinking:true}},'invalid programmatic LFM off cannot escape always-on');
 f.model(ids[4]);const selectNodes=[...f.element('composer-effort').options],radioNodes=[...f.element('s-thinking').children],before=f.mutations();
 for(let i=0;i<50;i++)f.ctx.syncThinkingControls();
 ok(f.element('composer-effort').options.every((node,i)=>node===selectNodes[i]),'polls retain select option identity');
 ok(f.element('s-thinking').children.every((node,i)=>node===radioNodes[i]),'polls retain radio node identity');
 equal(f.mutations(),before,'unchanged polls do not write text or attributes');
 radioNodes[1].onclick();f.ctx.syncThinkingControls();equal(f.ctx.segValue?f.ctx.segValue('s-thinking'):vm.runInContext('segValue("s-thinking")',f.ctx),'low','polls preserve an unapplied drawer choice');
 f.ctx.tr=(_key,text)=>'JP '+text;f.ctx.syncThinkingControls();
 ok(f.element('s-thinking').children[1]===radioNodes[1]&&radioNodes[1].textContent==='JP Low','language changes patch labels without replacing controls');
 radioNodes[1].focus();let prevented=false;radioNodes[1].onkeydown({key:'End',preventDefault(){prevented=true;}});
 ok(prevented&&f.ctx.document.activeElement===radioNodes[3],'radio keyboard selection moves focus to the selected endpoint');
 equal(vm.runInContext('segValue("s-thinking")',f.ctx),'xhigh','keyboard uses canonical xhigh');
 f.ctx.loadDrawer();f.element('s-reset').onclick();
 equal(vm.runInContext('segValue("s-thinking")',f.ctx),'medium','reset stages this model default');
 equal(f.ctx.settings.thinkingByModel[ids[4]],'xhigh','reset does not commit before Apply');
 await f.element('s-apply').onclick();
 equal(f.ctx.settings.thinkingByModel,{[ids[0]]:'none',[ids[4]]:'medium'},'drawer Apply/reset changes only current model preference');
 f.ctx.modelSwitching=true;f.ctx.syncThinkingControls();ok(f.element('composer-effort').disabled&&f.element('s-thinking').children.every(node=>node.disabled),'model switching locks thinking controls');
 const html=fs.readFileSync(path.join(web,'index.html'),'utf8');
 ok(html.indexOf('web/thinking-controls.js')<html.indexOf('web/app.js'),'native control module loads before app');
 for(const lang of ['en','ja','de','es','fr','pt-BR','zh-CN']){
  const locale=JSON.parse(fs.readFileSync(path.join(web,'locales',lang+'.json'),'utf8'));
  for(const key of ['thinking.on','thinking.alwaysOn','thinking.high'])ok(typeof locale[key]==='string'&&locale[key].length>0,'localized concise thinking labels');
 }
 const effort=f.ctx.StrataThinkingControls;
 equal(effort.contract('gpt-oss-20b'),{mode:'effort_always',default:'medium',options:['low','medium','xhigh']},'gpt-oss reasons at low, medium or high and cannot turn it off');
 equal(effort.payload('gpt-oss-20b',{thinkingByModel:{}}),{reasoning_effort:'medium'},'gpt-oss sends its effort, never a thinking switch');
 ok(effort.select({thinkingByModel:{}},'gpt-oss-20b','none').thinkingByModel['gpt-oss-20b']===undefined,'gpt-oss has no off choice');
 console.log(`ui thinking controls: ${checks} checks PASS`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
