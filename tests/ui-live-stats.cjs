'use strict';
// node tests/ui-live-stats.cjs — production functions with a simulated DOM,
// clock and streamed API. No API, model, GPU, browser or real timer is started.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert');
const source=fs.readFileSync(path.join(__dirname,'../ui/web/app.js'),'utf8');
const statsSource=source.slice(source.indexOf('const terminalWords ='),source.indexOf('function msgEl('));
const formatSource=source.slice(source.indexOf('const numberFormats ='),source.indexOf('const kfmt ='));
const replyStart=source.indexOf('async function generateReply()');
const replyEnd=source.indexOf('\n$("composer").onsubmit',replyStart);
const replySource=source.slice(replyStart,replyEnd);
const readinessSource=source.slice(source.indexOf('function modelIsSelectable('),source.indexOf('function syncComposerInput('));
assert(statsSource.includes('terminalStreamStats')&&replySource.includes('stopStatsClock()'));
let checks=0;
const ok=(value,label)=>{assert(value,label);checks++;};
const equal=(actual,expected,label)=>{assert.deepStrictEqual(actual,expected,label);checks++;};

function fixture(){
 let now=0,serial=0,writes=0;
 const frames=new Map();
 class Text{
  constructor(data){this.nodeType=3;this.parentNode=null;this._data=String(data);}
  get data(){return this._data;}
  set data(value){this._data=String(value);writes++;}
  get textContent(){return this.data;}
 }
 class Element{
  constructor(tag='div'){this.nodeType=1;this.tagName=tag;this.childNodes=[];this.parentNode=null;this.dataset={};this.hidden=false;this.title='';this.className='';this.isConnected=true;}
  get children(){return this.childNodes.filter(n=>n.nodeType===1);}
  append(...nodes){nodes.forEach(n=>this.appendChild(n));}
  appendChild(node){this.insertBefore(node,null);return node;}
  insertBefore(node,before){
   if(node.parentNode)node.remove?node.remove():node.parentNode.childNodes.splice(node.parentNode.childNodes.indexOf(node),1);
   const i=before===null?this.childNodes.length:this.childNodes.indexOf(before);
   assert(i>=0);this.childNodes.splice(i,0,node);node.parentNode=this;return node;
  }
  remove(){if(this.parentNode){this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this),1);this.parentNode=null;}}
  get textContent(){return this.childNodes.map(n=>n.textContent).join('');}
  set textContent(value){this.childNodes=[new Text(value)];this.childNodes[0].parentNode=this;writes++;}
  querySelector(selector){return this.children.find(n=>selector==='.'+n.className)||null;}
  replaceChildren(){throw Error('Telemetry must not replace all badge nodes');}
 }
 const el=new Element(),badges=new Element(),phase=new Element();
 badges.className='terminal-stats';phase.className='terminal-phase';el.dataset.i='0';el.append(badges,phase);
 const chat={lastElementChild:el,querySelector:()=>el},scroll={scrollTop:123,scrollHeight:999};
 const ctx={console,Map,Set,Intl,JSON,Number,String,Math,Date,Promise,Error,TextDecoder,AbortController,
  StrataI18n:{lang:'en'},busy:null,messages:[],performance:{now:()=>now},
  document:{createElement:tag=>new Element(tag),createTextNode:text=>new Text(text)},
  $:id=>id==='chat'?chat:scroll,
  tr:(key,fallback,values={})=>String(fallback).replace(/\{(\w+)\}/g,(_,k)=>values[k]),
  requestAnimationFrame:callback=>{frames.set(++serial,callback);return serial;},cancelAnimationFrame:id=>frames.delete(id),
  setInterval(){throw Error('No fixed-interval stats loop allowed');},clearInterval(){throw Error('No fixed-interval stats loop allowed');}};
 vm.createContext(ctx);vm.runInContext(formatSource+'\n'+statsSource,ctx);
 function message(stats={}){
  const m={text:'',reasoning:'',stats:{startedAt:1000,elapsedMs:0,firstTokenMs:null,...stats}};
  ctx.messages=[m];ctx.busy={msg:m,element:el,startedMono:0};return m;
 }
 function frame(ms){now=ms;const pending=[...frames.values()];frames.clear();pending.forEach(cb=>cb(ms));}
 return {ctx,el,badges,phase,scroll,frames,message,frame,setNow:ms=>now=ms,writes:()=>writes};
}
const sample=(generated,more={})=>({generated,tok_s:21,elapsed_s:.5,first_token_s:.1,done:false,...more});

// Request ownership, exact counts, invalid/stale updates and tool round offsets.
{
 const f=fixture(),s=f.message().stats;
 f.ctx.terminalBeginRound(s,0,0);
 ok(f.ctx.terminalStreamStats(s,sample(0,{tok_s:null,first_token_s:null}),0),'initial own-stream sample');
 equal(s.generated,0,'no character/event token estimate');
 equal(s.firstTokenMs,null,'native progress does not invent browser-visible first-token time');
 ok(f.ctx.terminalStreamStats(s,sample(7),0),'reasoning/held tool tokens are counted');
 equal(s.generated,7,'exact native count');
 for(const invalid of [sample(6),sample(7.5),sample(-1),sample(8,{elapsed_s:NaN}),sample(8,{tok_s:-1}),sample(8,{first_token_s:-1}),sample(8,{done:'false'})]){
  ok(!f.ctx.terminalStreamStats(s,invalid,0),'reject invalid or regressing count');equal(s.generated,7,'invalid sample leaves count intact');
 }
 ok(!f.ctx.terminalStreamStats(s,sample(100),1),'wrong stream round ignored');
 ok(f.ctx.terminalStreamStats(s,sample(8,{done:true}),0),'stream round completes');
 ok(!f.ctx.terminalStreamStats(s,sample(9),0),'late sample after done ignored');
 f.ctx.terminalBeginRound(s,1,8);
 equal(s.generated,8,'completed local-tool round offset retained');
 ok(!f.ctx.terminalStreamStats(s,sample(999),0),'previous local-tool HTTP stream ignored');
 ok(f.ctx.terminalStreamStats(s,sample(4),1),'new HTTP round accepted');
 equal(s.generated,12,'local-tool offset added once; server MCP already included');
 const done=f.ctx.terminalFinalStats(s,{completion_tokens:13,prompt_tokens:20,prompt_tokens_details:{cached_tokens:5}},
                                  {predicted_per_second:33,prompt_n:15,prompt_per_second:40});
 equal(done.generated,13,'final usage authoritative');equal(done.decodeRate,33,'final native timing authoritative');
 equal(done.finalCount,true,'final count marked complete');equal(done.rateSource,'native','final rate is not live');
 ok(!f.ctx.terminalStreamStats(done,sample(99),1),'finished message cannot be overwritten');
 const cancelled=f.ctx.terminalFinalStats({...s,finalCount:false},null,null);
 ok(!cancelled.finalCount,'cancelled stream not labelled final');equal(cancelled.generated,12,'interrupted exact count retained');
}
// Old backends still use guarded one-second /metrics; own stream wins thereafter.
{
 const f=fixture(),s=f.message({firstTokenMs:100}).stats;
 f.ctx.terminalBeginRound(s,0,0);s.roundHasToken=true;
 const metrics={time:2,live:{state:'generating',elapsed_s:1,generated:5,tok_s:10}};
 ok(f.ctx.terminalMetrics(s,metrics),'legacy metrics fallback');equal(s.generated,5,'legacy native count');
 ok(f.ctx.terminalStreamStats(s,sample(7),0),'own stream supersedes global sample');
 ok(!f.ctx.terminalMetrics(s,{...metrics,live:{...metrics.live,generated:999}}),'global sample cannot overwrite own stream');
 equal(s.generated,7,'concurrent global count ignored');
 f.ctx.terminalBeginRound(s,1,7);
 ok(!f.ctx.terminalMetrics(s,metrics),'new round cannot inherit another request before token');
 s.roundHasToken=true;
 ok(!f.ctx.terminalMetrics(s,{...metrics,live:{...metrics.live,parallel:2}}),'parallel global sample rejected');
}
// Existing badge elements AND text nodes survive live/final/localized updates.
{
 const f=fixture(),m=f.message({firstTokenMs:100,elapsedMs:500,generated:7,decodeRate:20,rateSource:'live'});
 f.ctx.renderTerminalStats(f.el,m,true);
 const nodes=new Map([...f.badges._terminalNodes].map(([key,node])=>[key,[node,node._name,node._value]]));
 m.stats.generated=8;m.stats.elapsedMs=600;m.stats.decodeRate=22;
 f.ctx.renderTerminalStats(f.el,m,true);
 for(const [key,old]of nodes){const n=f.badges._terminalNodes.get(key);equal([n,n._name,n._value],old,key+' node identity');}
 equal(f.badges._terminalNodes.get('tokens')._value.data,'8','only token text updates');
 f.ctx.tr=(key,en)=>'Translated '+en;
 f.ctx.renderTerminalStats(f.el,m,true);
 equal(f.badges._terminalNodes.get('output'),nodes.get('output')[0],'localization keeps same badge key');
 m.stats=f.ctx.terminalFinalStats(m.stats,{completion_tokens:8,prompt_tokens:10,prompt_tokens_details:{cached_tokens:2}},{predicted_per_second:23,prompt_n:8});
 f.ctx.busy=null;f.ctx.renderTerminalStats(f.el,m,false);
 equal(f.badges._terminalNodes.get('elapsed'),nodes.get('elapsed')[0],'elapsed-to-total keeps same node');
 equal(f.badges._terminalNodes.get('tokens')._value.data,'8','final count no incomplete marker');
 ok(f.badges._terminalNodes.has('input')&&f.badges._terminalNodes.has('cache'),'final fields added without rebuilding existing nodes');
 equal(f.scroll.scrollTop,123,'stats do not move scroll');
}
// Video-style cadence: rAF at60Hz, two decimals below10s, then ten updates/s.
{
 const f=fixture(),m=f.message({promptProgress:{processed:100,total:200}});
 f.setNow(7000);const stop=f.ctx.startTerminalClock(f.ctx.busy);
 const badge=f.badges._terminalNodes.get('elapsed'),value=badge._value,phaseText=f.phase._terminalText;
 let changes=0,last=value.data;
 for(let i=1;i<=60;i++){f.frame(7000+i*1000/60);if(value.data!==last){changes++;last=value.data;}}
 ok(changes>=45&&changes<=60,'7–8s numeral changes on nearly every video frame');
 equal(value.data,'8.00 s','under10s uses two decimals');
 ok(f.phase.textContent.endsWith('8.00 s'),'prefill phase clock shares fine cadence');
 equal(f.phase._terminalText,phaseText,'prefill phase text node remains stable');
 equal(f.phase.childNodes.length,1,'clock does not accumulate phase text nodes');
 equal(f.badges._terminalNodes.get('elapsed')._value,value,'rAF preserves elapsed text node');
 f.frame(16000);changes=0;last=value.data;
 for(let i=1;i<=60;i++){f.frame(16000+i*1000/60);if(value.data!==last){changes++;last=value.data;}}
 ok(changes>=9&&changes<=11,'16–17s stays readable at tenths');equal(value.data,'17.0 s','over10s uses one decimal');
 equal(f.frames.size,1,'one clock frame outstanding');equal(f.scroll.scrollTop,123,'clock does not move reader scroll');
 stop();equal(f.frames.size,0,'done/abort cancels clock');const old=f.writes();f.frame(18000);equal(f.writes(),old,'no post-stop writes');
 const stop2=f.ctx.startTerminalClock(f.ctx.busy);f.ctx.busy=null;f.frame(19000);equal(f.frames.size,0,'expired request does not reschedule');stop2();
}

// Exercise the actual generateReply SSE reader and local-tool loop. Only
// ordinary deltas schedule prose paints; stats-only packets never scroll.
async function streamingFixture({tool=false,cancel=false,legacy=false}={}){
 const f=fixture(),c=f.ctx;let fetches=0,paintCalls=0,scrollCalls=0,telemetryChecks=0;
 c.messages=[];c.busy=null;c.modelSwitching=false;c.modelSwitchAvailable=false;c.modelCatalog=[];c.resourceOperation=null;c.health={model:'clover-35b-a3b-preview'};
 c.settings={temperature:0,mcp:false};c.mcpInfo={tools:0};c.globalTools={max_concurrency:1};c.headers=()=>({});
 c.saveChat=()=>{};c.settlePending=()=>{};c.setBusy=()=>{};c.paintLive=()=>{};c.apiMessages=()=>[];c.apiMessageSerializer=()=>c.apiMessages;c.effortForModel=()=> 'none';c.projectionLoaded=()=>false;
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/web/thinking-controls.js'),'utf8')+'\n'+
   source.slice(source.indexOf('function effortForModel('),source.indexOf('function syncComposerControls(')),c);
 c.renderChat=()=>{};c.samplingUntouched=()=>true;c.toast=()=>{};
 c.prepareContext=async body=>({messages:body.messages});
 c.updateAssistant=(el,m,streaming)=>{paintCalls++;c.renderTerminalStats(el,m,streaming);};
 c.scrollDown=()=>{scrollCalls++;};c.onTool=(m,x)=>{m.tools=m.tools||[];let t=m.tools.find(t=>t.id===x.id);if(!t){t={id:x.id,at:0,rat:0};m.tools.push(t);}Object.assign(t,x,{state:x.event==='result'?'done':'running'});};
 c.StrataLocalTools={MAX_ROUNDS:2,session:()=>({run:async()=>({text:'ok',ok:true,images:[]})}),definitions:()=>tool?[{}]:[],
  combined:rounds=>({usage:rounds.length?{completion_tokens:rounds.reduce((n,r)=>n+(r.usage?.completion_tokens||0),0)}:null,timings:null}),
  mapLimit:async(items,limit,run)=>{for(const item of items)await run(item);},
  mergeCalls:(map,calls)=>calls.map(call=>{const full={index:0,id:'tool-1',name:'run_python',arguments:'{}'};map.set(0,full);return full;}),parseCall:()=>({}),safeURL:()=>null};
 c.fetch=async(url,options)=>{
  fetches++;const body=JSON.parse(options.body);ok(body.return_live_stats===true,'UI opts into request-bound live stats');
  ok(body.chat_template_kwargs?.enable_thinking===true&&!Object.hasOwn(body,'reasoning_effort'),'actual toggle-model chat request sends native boolean, never on as effort');
  const round=fetches-1,isTool=tool&&round===0;
  const chunk=delta=>({choices:[{index:0,delta,finish_reason:null}]});
  const packets=legacy?[]:[{...chunk({}),strata_stats:sample(0,{tok_s:null,first_token_s:null})},
                         {...chunk({}),strata_stats:sample(isTool?7:4)}];
  packets.push(chunk(isTool?{tool_calls:[{index:0,id:'tool-1',function:{name:'run_python',arguments:'{}'}}]}:{content:'391'}));
  if(!legacy)packets.push({...chunk({}),strata_stats:sample(isTool?7:4,{done:true})});
  packets.push({choices:[{index:0,delta:{},finish_reason:isTool?'tool_calls':'stop'}],usage:{completion_tokens:isTool?7:4,prompt_tokens:10},timings:{predicted_per_second:25,predicted_n:isTool?7:4}});
  let index=0,previousWasTelemetry=false,lastPaint=0,lastScroll=0;
  return {ok:true,body:{getReader:()=>({read:async()=>{
   f.frame(1000+index*100);
   if(previousWasTelemetry){equal(paintCalls,lastPaint,'stats-only packet does not repaint prose');equal(scrollCalls,lastScroll,'stats-only packet does not scroll');telemetryChecks++;}
   if(cancel&&index===2){const e=Error('cancelled');e.name='AbortError';throw e;}
   if(index>packets.length)return {done:true};
   const packet=index<packets.length?packets[index]:null;index++;
   previousWasTelemetry=!!packet?.strata_stats;lastPaint=paintCalls;lastScroll=scrollCalls;
   const data=packet?'data: '+JSON.stringify(packet)+'\n\n':'data: [DONE]\n\n';
   return {done:false,value:new TextEncoder().encode(data)};
  }})}};
 };
 vm.runInContext(readinessSource+'\n'+replySource,c);await c.generateReply();
 const m=c.messages[0];equal(c.busy,null,'request releases busy state');equal(f.frames.size,0,'all clock/prose frames cancelled on exit');
 if(cancel){ok(m.stopped,'cancelled SSE retained');ok(!m.stats.finalCount,'cancel cannot invent final usage');equal(m.stats.generated,4,'cancel retains last actual native count');}
 else{ok(!m.error,'no stream error: '+m.error);equal(m.stats.generated,tool?11:4,'HTTP local-tool offsets and final aggregate usage');ok(m.stats.finalCount,'final authoritative count saved');equal(m.text,'391','content unchanged by stats');}
 if(!legacy)ok(telemetryChecks>0,'actual SSE parser consumed stats-only packets');
 equal(fetches,tool?2:1,'no per-delta metrics requests');
}
(async()=>{
 await streamingFixture();await streamingFixture({tool:true});await streamingFixture({cancel:true});await streamingFixture({legacy:true});
 console.log('PASS '+checks+' live stats checks; no API/GPU/browser execution');
})().catch(error=>{console.error(error);process.exitCode=1;});
