'use strict';
// Production rendering helpers in a small DOM. No browser, network, timer,
// native process, model, GPU, storage or copied implementation is used.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../ui/web/app.js'),'utf8');
let checks=0;
const ok=(value,label)=>{assert.ok(value,label);checks++;};
const equal=(value,expected,label)=>{assert.deepStrictEqual(value,expected,label);checks++;};
const plain=value=>JSON.parse(JSON.stringify(value));

function functionSource(name){
 const start=source.indexOf('function '+name+'(');
 assert(start>=0,'Production function missing: '+name);
 const next=source.indexOf('\nfunction ',start+1);
 assert(next>start,'Missing next function after '+name);
 return source.slice(start,next);
}

function fixture(){
 let writes=0,appends=0;
 class Text{
  constructor(value){this.nodeType=3;this.parentNode=null;this._data=String(value);}
  get data(){return this._data;}
  set data(value){this._data=String(value);writes++;}
  get length(){return this._data.length;}
  get textContent(){return this.data;}
  appendData(value){this._data+=String(value);appends++;}
 }
 const matches=(el,selector)=>{
  const attr=/\[([\w-]+)(?:="([^"]*)")?\]/.exec(selector);
  if(attr){const key=attr[1].replace(/^data-/,'').replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
   if(!(key in el.dataset)||attr[2]!=null&&el.dataset[key]!==attr[2])return false;
   selector=selector.replace(attr[0],'');}
  const match=/^(\w+)?(?:\.([\w-]+))?$/.exec(selector);
  return !!match&&(!match[1]||el.tagName===match[1].toUpperCase())&&(!match[2]||el.className.split(/\s+/).includes(match[2]));
 };
 class Element{
  constructor(tag='div'){
   this.nodeType=1;this.tagName=tag.toUpperCase();this.childNodes=[];this.parentNode=null;this.dataset={};
   this.className='';this.hidden=false;this.open=false;this.scrollTop=0;this.scrollLeft=0;this.isConnected=true;
   this.classList={contains:name=>this.className.split(/\s+/).includes(name),remove:name=>{this.className=this.className.split(/\s+/).filter(x=>x!==name).join(' ');},
    add:name=>{if(!this.classList.contains(name))this.className+=' '+name;}};
  }
  get children(){return this.childNodes.filter(node=>node.nodeType===1);}
  get firstChild(){return this.childNodes[0]||null;}
  get nextSibling(){if(!this.parentNode)return null;return this.parentNode.childNodes[this.parentNode.childNodes.indexOf(this)+1]||null;}
  get textContent(){return this.childNodes.map(node=>node.textContent).join('');}
  set textContent(value){this.childNodes=[new Text(value)];this.childNodes[0].parentNode=this;writes++;}
  append(...nodes){nodes.forEach(node=>this.appendChild(node));}
  appendChild(node){return this.insertBefore(node,null);}
  insertBefore(node,before){
   if(node.parentNode)node.parentNode.childNodes.splice(node.parentNode.childNodes.indexOf(node),1);
   const index=before==null?this.childNodes.length:this.childNodes.indexOf(before);assert(index>=0);
   this.childNodes.splice(index,0,node);node.parentNode=this;return node;
  }
  remove(){if(this.parentNode){this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this),1);this.parentNode=null;}}
  setAttribute(name,value){
   if(name.startsWith('data-'))this.dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(value);
   else this[name]=String(value);
  }
  removeAttribute(name){if(name.startsWith('data-'))delete this.dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())];else delete this[name];}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  querySelectorAll(selector){
   const found=[];for(const child of this.children){if(matches(child,selector))found.push(child);found.push(...child.querySelectorAll(selector));}return found;
  }
  closest(selector){let node=this;while(node){if(matches(node,selector))return node;node=node.parentNode;}return null;}
 }
 const selection={anchorNode:null,focusNode:null,anchorOffset:0,focusOffset:0,
  setBaseAndExtent(anchor,ai,focus,fi){Object.assign(this,{anchorNode:anchor,focusNode:focus,anchorOffset:ai,focusOffset:fi});}};
 const handlers=new Map(),chat=new Element(),scroll={scrollTop:123};
 chat.addEventListener=(type,listener)=>handlers.set(type,listener);
 const ctx={console,Map,Set,JSON,Number,String,Object,Array,Math,Date,Node:{TEXT_NODE:3},busy:null,messages:[],
  settings:{show:true,math:false},StrataI18n:{lang:'en'},readingToolOutput:false,
  document:{createElement:tag=>new Element(tag),createTextNode:value=>new Text(value)},window:{getSelection:()=>selection},
  $:id=>id==='chat'?chat:scroll,
  tr:(key,fallback,values={})=>String(fallback).replace(/\{(\w+)\}/g,(_,key)=>values[key]),fmt:value=>String(value),
  clearTerminalCursor(){},setTerminalCursor(){},renderTerminalStats(){},saveChat(){},
  answerMarkdown:value=>value,patchToolPart(){},toolHtml:()=>'',toolImages:()=>'',icon:()=>'',esc:value=>String(value)};
 vm.createContext(ctx);
 for(const name of ['reasoningTimeline','reasoningActive','reasoningText','writeReasoningText','patchReasoningPart'])vm.runInContext(functionSource(name),ctx);
 function details(){
  const node=new Element('details');node.className='st-collapse think';
  const summary=new Element('summary'),title=new Element('span'),body=new Element();
  title.className='think-title';body.className='st-collapse__body thinking';summary.append(title);node.append(summary,body);
  return {node,title,body};
 }
 return {ctx,Element,Text,selection,chat,scroll,handlers,details,writes:()=>writes,appends:()=>appends};
}

function projection(m,timeline){
 return plain(timeline.map(part=>part.kind==='tool'?['tool',part.tool.id]:[part.kind,m[part.kind==='reasoning'?'reasoning':'text'].slice(part.start,part.end)]));
}

// A cumulative reasoning string is split at the moment each actual tool began.
{
 const f=fixture(),m={reasoning:'oneTWOthree',text:'answer',tools:[{id:'search',at:0,rat:3},{id:'python',at:0,rat:6}]};
 const before=JSON.stringify(m),timeline=f.ctx.reasoningTimeline(m);
 equal(projection(m,timeline),[['reasoning','one'],['tool','search'],['reasoning','TWO'],['tool','python'],['reasoning','three'],['text','answer']],
       'reasoning/tool/reasoning chronology follows tool offsets');
 equal(JSON.stringify(m),before,'display split does not alter saved cumulative API text');
 equal(new Set(timeline.map(part=>part.key)).size,timeline.length,'stable keys are unique');
 const firstKey=timeline[0].key,lastReasoning=timeline.findLast(part=>part.kind==='reasoning').key;
 m.reasoning+=' grows';
 const next=f.ctx.reasoningTimeline(m);
 equal(next[0].key,firstKey,'a frozen preceding thought keeps its key');
 equal(next.findLast(part=>part.kind==='reasoning').key,lastReasoning,'streamed tail keeps its key while it grows');
}
{
 const f=fixture(),m={reasoning:'one  two',text:'prePOST',tools:[{id:'a',at:3,rat:3},{id:'b',at:3,rat:3},{id:'c',at:3,rat:5}]};
 const timeline=f.ctx.reasoningTimeline(m);
 equal(projection(m,timeline),[['reasoning','one'],['text','pre'],['tool','a'],['tool','b'],['tool','c'],['reasoning','two'],['text','POST']],
       'same-offset parallel tools stay ordered and whitespace does not produce empty thought panels');
 equal(timeline.filter(part=>part.kind==='reasoning').length,2,'no duplicate thought for a parallel tool batch');
}

// Old saved chats and malformed offsets must fall back without guessing time order.
{
 const f=fixture(),base={reasoning:'abcdef',text:'answer',tools:[{id:'a',at:0,rat:3}]};
 for(const key of ['rat','at']){
  const invalid=[undefined,-1,.5,'3',NaN,Infinity,key==='rat'?7:7];
  for(const value of invalid){const m={...base,tools:[{...base.tools[0],[key]:value}]};equal(f.ctx.reasoningTimeline(m),null,'invalid '+key+' refuses inferred segmentation');}
 }
 for(const tools of [[{id:'a',at:3,rat:4},{id:'b',at:2,rat:5}],[{id:'a',at:0,rat:4},{id:'b',at:0,rat:3}]])
  equal(f.ctx.reasoningTimeline({...base,tools}),null,'decreasing offsets do not reorder saved text');
 const legacy=f.details().node;
 equal(f.ctx.reasoningText(base,legacy),'abcdef','unsegmented legacy panel still shows all reasoning');
}

// A UI-inserted boundary separator is display-only, not a whitespace trim.
{
 const f=fixture(),{node}=f.details();
 node.dataset.reasoningSegment='reasoning:after:a:0';
 const display=(reasoning,start)=>{
  node.dataset.reasoningStart=String(start);node.dataset.reasoningEnd=String(reasoning.length);
  return f.ctx.reasoningText({reasoning},node);
 };
 const raw='pre\n\n    indented code\n\ninternal paragraph\n';
 equal(display(raw,3),'    indented code\n\ninternal paragraph\n','only the UI boundary separator is removed from a later thought');
 equal(raw,'pre\n\n    indented code\n\ninternal paragraph\n','display cleanup leaves canonical reasoning unchanged');
 equal(display('pre\n\n\n\n\tcode\n',3),'\n\n\tcode\n','exactly one newline pair is removed, preserving additional leading whitespace');
 equal(display('pre\n    code\n',3),'\n    code\n','a single leading newline is preserved');
 equal(display('\n\n    first thought\n',0),'\n\n    first thought\n','the first thought preserves its original leading whitespace');
 delete node.dataset.reasoningSegment;
 equal(f.ctx.reasoningText({reasoning:raw},node),raw,'legacy single-panel text is never stripped');
}

// Incremental streamed thoughts retain text nodes, selection and reading position.
{
 const f=fixture(),{body}=f.details();body.scrollTop=77;body.scrollLeft=11;
 f.ctx.writeReasoningText(body,'abcdef');const text=body.firstChild;
 f.selection.setBaseAndExtent(text,1,text,4);const writes=f.writes(),appends=f.appends();
 f.ctx.writeReasoningText(body,'abcdefgh');
 equal(body.firstChild,text,'streaming appends to the same Text node');
 equal(f.writes(),writes,'appending does not replace a selected Text node');
 equal(f.appends(),appends+1,'only the new suffix is appended');
 equal([f.selection.anchorNode,f.selection.anchorOffset,f.selection.focusNode,f.selection.focusOffset],[text,1,text,4],'reader selection remains on the same span');
 equal([body.scrollTop,body.scrollLeft],[77,11],'thought scroll position remains fixed');
 f.ctx.writeReasoningText(body,'ab');
 equal(body.firstChild,text,'a corrected suffix also retains its text node');
 equal([f.selection.anchorOffset,f.selection.focusOffset],[1,2],'selection clamps safely when the text shrinks');
}

// Earlier thoughts freeze before the tool; only the actual current segment is active.
{
 const f=fixture(),m={reasoning:'oneTWOthree',text:'',thinkSecs:99};
 const a=f.details(),b=f.details();
 a.node.dataset.reasoningSegment='reasoning:start';b.node.dataset.reasoningSegment='reasoning:after:a:0';
 f.ctx.patchReasoningPart(a.node,m,0,3,true,true);
 equal(a.body.textContent,'one','the first active panel shows its own slice');
 ok(a.node.open,'active thought follows the show-thinking preference');
 f.ctx.patchReasoningPart(a.node,m,0,3,false,true);
 ok(!a.node.open,'untouched thought collapses after its tool begins');
 ok(!a.title.textContent.includes('99'),'segmented headers do not mislabel every panel with whole-message time');
 f.ctx.patchReasoningPart(b.node,m,3,6,true,true);
 equal(b.body.textContent,'TWO','new active panel receives only post-tool reasoning');
 equal(a.body.textContent,'one','earlier thought remains frozen when the next thought starts');
 a.node.dataset.touched='1';a.node.open=true;
 f.ctx.patchReasoningPart(a.node,m,0,3,false,true);
 ok(a.node.open,'completion/cancel does not close a panel the reader explicitly opened');
 f.ctx.patchReasoningPart(b.node,{...m,stopped:true},3,6,false,true);
 ok(!b.node.open,'cancel/final transition ends automatic active state');
 equal(f.ctx.reasoningText(m,b.node),'TWO','lazy segment lookup uses its own saved boundaries');
}

// Thinking resumes only from an actual reasoning delta, not because there is
// no final answer yet while tools or prompt preparation are still running.
{
 const f=fixture(),m={reasoning:'one',text:''};f.ctx.busy={msg:m,outputPhase:'preparing'};
 for(const phase of ['preparing','tool','content']){
  f.ctx.busy.outputPhase=phase;
  ok(!f.ctx.reasoningActive(m,true),phase+' is not an active thought');
 }
 f.ctx.busy.outputPhase='reasoning';
 ok(f.ctx.reasoningActive(m,true),'an actual post-tool reasoning delta resumes thinking');
 ok(!f.ctx.reasoningActive(m,false),'a completed stream freezes thought state');
 m.stopped=true;ok(!f.ctx.reasoningActive(m,true),'cancelled message is not thinking');delete m.stopped;
 m.error='failed';ok(!f.ctx.reasoningActive(m,true),'errored message is not thinking');delete m.error;
 ok(!f.ctx.reasoningActive({...m},true),'another message cannot inherit active thinking state');
 f.ctx.busy=null;ok(!f.ctx.reasoningActive(m,true),'idle state never appears to think');
}

// Run the actual delegated toggle handler: reopening a later lazy panel must
// display its own range, rather than the message's cumulative reasoning.
{
 const f=fixture(),m={reasoning:'oneTWOthree',text:'answer'},message=new f.Element();
 message.className='st-msg';message.dataset.i='0';f.ctx.messages=[m];
 const {node,body}=f.details();message.append(node);node.dataset.reasoningSegment='reasoning:after:a:0';
 f.ctx.patchReasoningPart(node,m,3,6,false,true);
 equal(body.dataset.pending,'1','closed thought defers its body');
 const start=source.indexOf('$("chat").addEventListener("toggle",');
 const end=source.indexOf('// UI-only guidance.',start);
 assert(start>=0&&end>start,'production lazy-toggle handler anchors');
 vm.runInContext(source.slice(start,end),f.ctx);
 node.open=true;f.handlers.get('toggle')({target:node});
 equal(body.textContent,'TWO','lazy open renders only the selected post-tool thought');
 ok(!body.dataset.pending,'lazy thought is marked rendered');
 equal(node.dataset.touched,'1','explicit toggle keeps reader control over collapse');
}

console.log(`PASS ${checks} reasoning chronology checks`);
