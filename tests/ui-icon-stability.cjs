'use strict';
// Production UI functions in a deterministic DOM. No browser, API, model,
// network, GPU, real timer or clipboard is accessed.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../ui/web/app.js'),'utf8');
let checks=0;
const equal=(value,expected,label)=>{assert.deepStrictEqual(value,expected,label);checks++;};
const ok=(value,label)=>{assert.ok(value,label);checks++;};
function functionSource(name){
 const found=new RegExp('(?:async )?function '+name+'\\(').exec(source);
 assert(found,'Missing production function '+name);
 const end=source.indexOf('\n}',found.index);
 assert(end>found.index,'Missing function end '+name);
 return source.slice(found.index,end+2);
}

function fixture(){
 let serial=0,now=0,innerWrites=0,fitCalls=0;
 const timers=new Map(),copied=[],notices=[],nodes=new Map();
 class Text{
  constructor(value=''){this.nodeType=3;this.parentNode=null;this.data=String(value);}
  get textContent(){return this.data;}
  set textContent(value){this.data=String(value);}
  get nextSibling(){if(!this.parentNode)return null;return this.parentNode.childNodes[this.parentNode.childNodes.indexOf(this)+1]||null;}
  remove(){if(this.parentNode){this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this),1);this.parentNode=null;}}
 }
 class Element{
  constructor(tag='div'){
   this.nodeType=1;this.tagName=tag.toUpperCase();this.parentNode=null;this.childNodes=[];this.dataset={};this.attributes={};
   this.className='';this.title='';this.disabled=false;this.hidden=false;this.writes=new Map();
  }
  get children(){return this.childNodes.filter(n=>n.nodeType===1);}
  get firstChild(){return this.childNodes[0]||null;}
  get lastChild(){return this.childNodes.at(-1)||null;}
  get nextSibling(){if(!this.parentNode)return null;return this.parentNode.childNodes[this.parentNode.childNodes.indexOf(this)+1]||null;}
  get isConnected(){return !!this.parentNode||this===document.body;}
  get textContent(){return this.childNodes.map(n=>n.textContent).join('');}
  set textContent(value){this.replaceChildren(new Text(value));}
  set innerHTML(value){
   innerWrites++;this.replaceChildren();
   const use=/<use\s+[^>]*href="([^"]+)"/.exec(value);
   if(use){const svg=new Element('svg'),symbol=new Element('use');symbol.setAttribute('href',use[1]);svg.append(symbol);this.append(svg);}
  }
  append(...children){children.forEach(n=>this.appendChild(typeof n==='string'?new Text(n):n));}
  appendChild(node){return this.insertBefore(node,null);}
  insertBefore(node,before){
   if(node.parentNode)node.remove();const index=before==null?this.childNodes.length:this.childNodes.indexOf(before);assert(index>=0);
   this.childNodes.splice(index,0,node);node.parentNode=this;return node;
  }
  replaceChildren(...children){for(const child of this.childNodes)child.parentNode=null;this.childNodes=[];this.append(...children);}
  remove(){if(this.parentNode){this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this),1);this.parentNode=null;}}
  setAttribute(key,value){this.attributes[key]=String(value);this.writes.set(key,(this.writes.get(key)||0)+1);}
  getAttribute(key){return Object.hasOwn(this.attributes,key)?this.attributes[key]:null;}
  hasAttribute(key){return Object.hasOwn(this.attributes,key);}
  removeAttribute(key){delete this.attributes[key];}
  toggleAttribute(key,force){const on=force??!Object.hasOwn(this.attributes,key);if(on)this.setAttribute(key,'');else this.removeAttribute(key);return on;}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  querySelectorAll(selector){
   const found=[];for(const child of this.children){
    const match=selector==='use'?child.tagName==='USE':selector==='button'?child.tagName==='BUTTON':selector==='code'?child.tagName==='CODE':
     selector==='svg'?child.tagName==='SVG':selector==='[data-copy]'?'copy'in child.dataset:false;
    if(match)found.push(child);found.push(...child.querySelectorAll(selector));
   }return found;
  }
  focus(){document.activeElement=this;}
 }
 const document={body:null,activeElement:null,createElement:tag=>new Element(tag),createTextNode:value=>new Text(value)};
 document.body=new Element('body');
 for(const id of ['pill','pill-symbol','pill-text','unload-btn','model-button-text','model-button-icon','model-button-symbol']){
  const node=new Element(id.includes('symbol')?'use':'div');nodes.set(id,node);document.body.append(node);
 }
 const ctx={console,JSON,Math,Number,String,Array,Object,Map,Set,document,
  navigator:{clipboard:{writeText:async text=>{copied.push(text);}}},
  $:id=>nodes.get(id),tr:(key,en)=>en,fitHeader(){fitCalls++;},toast:(...args)=>notices.push(args),
  setTimeout(callback,delay){const id=++serial;timers.set(id,{at:now+delay,callback});return id;},
  clearTimeout(id){timers.delete(id);}};
 const helpers=source.slice(source.indexOf('const SPRITE ='),source.indexOf('const numberFormats ='));
 vm.createContext(ctx);vm.runInContext(helpers+'\nlet lastPillText = null;\n'+
  ['setIconSymbol','setStableHtml','copyText','setPill','setModelButton','facts'].map(functionSource).join('\n'),ctx);
 const tick=ms=>{now+=ms;for(const[id,timer]of[...timers])if(timer.at<=now){timers.delete(id);timer.callback();}};
 return {ctx,Element,Text,document,nodes,copied,notices,timers,tick,innerWrites:()=>innerWrites,fitCalls:()=>fitCalls};
}

async function main(){
 // A state change can change a symbol; another poll of that state must not.
 {
  const f=fixture(),use=f.nodes.get('pill-symbol');
  f.ctx.setPill('idle','Ready');const writes=use.writes.get('href'),text=f.nodes.get('pill-text').firstChild;
  for(let i=0;i<60;i++)f.ctx.setPill('idle','Ready');
  equal(use.writes.get('href'),writes,'unchanged status never reloads the external SVG symbol');
  equal(f.nodes.get('pill-symbol'),use,'status symbol element survives repeated polls');
  f.ctx.setPill('generating','Generating · 50 tok/s');
  equal(use.getAttribute('href'),'web/sprite.svg#i-activity','real state changes update the symbol');
  const changed=use.writes.get('href');
  for(let i=0;i<20;i++)f.ctx.setPill('generating',`Generating · ${50+i} tok/s`);
  equal(use.writes.get('href'),changed,'changing speed text does not reload the same activity icon');
  equal(f.nodes.get('pill-symbol'),use,'rate updates retain symbol identity');
  f.ctx.setPill('error','Failed');equal(use.getAttribute('href'),'web/sprite.svg#i-warning','error has the distinct warning symbol');
  ok(text!==null,'status has a rendered label');
 }
 {
  const f=fixture(),use=f.nodes.get('model-button-symbol'),button=f.nodes.get('unload-btn');button.focus();
  f.ctx.setModelButton('Unload','Unload the model',false,'eject');const writes=use.writes.get('href');
  for(let i=0;i<30;i++)f.ctx.setModelButton('Unload','Unload the model',false,'eject');
  equal(use.writes.get('href'),writes,'unchanged model control does not reload eject SVG');
  equal(f.document.activeElement,button,'model control keeps keyboard focus');
  f.ctx.setModelButton('Load','Load the model',false,'download');
  equal(use.getAttribute('href'),'web/sprite.svg#i-download','load state uses download symbol');
  f.ctx.setModelButton('Switching','',true,null);
  ok(f.nodes.get('model-button-icon').getAttribute('hidden')!==null,'a state without an icon hides its existing slot');
  f.ctx.setModelButton('Unload','Unload the model',false,'eject');
  equal(f.nodes.get('model-button-symbol'),use,'showing the slot again reuses its symbol');
 }
 // About's repeated fact updates keep the copy button, SVG/use, focus and
 // temporary feedback even as labels/values are translated or changed.
 {
  const f=fixture(),facts=new f.Element('dl');f.document.body.append(facts);
  const rows=[['OpenAI URL','http://localhost/v1',true],['Model name','model-a',true],['Workers',16,false]];
  f.ctx.facts(facts,rows);
  equal(facts._factRows.length,3,'facts have persistent row records');
  const row=facts._factRows[0],button=row.button,use=button.querySelector('use'),svg=button.querySelector('svg'),code=row.code;
  const originalChildren=[...facts.childNodes],writes=use.writes.get('href'),htmlWrites=f.innerWrites();button.focus();
  await f.ctx.copyText(rows[0][1],button);
  equal(f.copied,['http://localhost/v1'],'copy writes the selected actual value');
  equal(use.getAttribute('href'),'web/sprite.svg#i-check','copy feedback uses the existing symbol');
  const feedbackWrites=use.writes.get('href');
  for(let i=0;i<20;i++)f.ctx.facts(facts,rows);
  equal(facts.childNodes,originalChildren,'unchanged polls retain every fact row node');
  equal(facts._factRows[0].button,button,'unchanged polls retain copy button');
  equal(button.querySelector('use'),use,'copy symbol is not remounted');
  equal(button.querySelector('svg'),svg,'copy SVG is not remounted');
  equal(use.writes.get('href'),feedbackWrites,'polling does not erase or restart check feedback');
  equal(f.innerWrites(),htmlWrites,'polling does not reconstruct copy-button markup');
  equal(f.document.activeElement,button,'copy focus survives repeated facts refresh');
  f.ctx.tr=(key,en)=>'Translated '+en;
  f.ctx.facts(facts,[['URL traduite','http://localhost/v1',true],['Modèle','model-b',true],['Workers',32,false]]);
  equal(facts._factRows[0].button,button,'language changes preserve copy control identity');
  equal(button.querySelector('use'),use,'language changes preserve icon identity');
  equal(use.getAttribute('href'),'web/sprite.svg#i-check','language changes preserve live copy feedback');
  ok(button.getAttribute('aria-label').startsWith('Translated '),'copy accessible name is translated');
  equal(facts._factRows[0].code,code,'value formatting keeps its code element');
  equal(facts._factRows[1].button.dataset.copy,'model-b','model changes update actual copied value');
  equal(facts._factRows[2].value.textContent,'32','ordinary numeric facts update in place');
  equal(facts._factRows[0].label.textContent,'URL traduite','translated row label updates');
  equal(f.document.activeElement,button,'relocalization does not steal focus');
  f.tick(1499);equal(use.getAttribute('href'),'web/sprite.svg#i-check','feedback remains for its stated interval');
  f.tick(1);equal(use.getAttribute('href'),'web/sprite.svg#i-copy','feedback expiry restores the same copy symbol');
  equal(button.querySelector('use'),use,'feedback expiry never replaces its use node');
  ok((use.writes.get('href')||0)>writes,'only meaningful copy feedback changes href');
  f.ctx.facts(facts,[['Remaining','value',false]]);
  equal(facts._factRows.length,1,'removed facts do not leave ghost rows');
  equal(facts.children.length,2,'remaining fact keeps only its term and value');
  ok(!facts.querySelector('button'),'copy controls disappear when the actual row is not copyable');
 }
 {
  const f=fixture(),facts=new f.Element('dl');f.ctx.facts(facts,[['URL','local',true]]);
  const button=facts._factRows[0].button,use=button.querySelector('use');
  await f.ctx.copyText('first',button);f.tick(1000);await f.ctx.copyText('second',button);
  equal(f.timers.size,1,'repeated copies have only one feedback timer');
  f.tick(900);equal(use.getAttribute('href'),'web/sprite.svg#i-check','an old timer cannot clear a newer copy confirmation');
  f.tick(900);equal(use.getAttribute('href'),'web/sprite.svg#i-copy','latest confirmation expires from its own copy time');
  equal(f.copied,['first','second'],'repeated copies preserve the requested values');
 }
 {
  const f=fixture(),list=new f.Element();
  const ready='<span><svg><use href="web/sprite.svg#i-check"/></svg>Ready</span>';
  f.ctx.setStableHtml(list,ready);const icon=list.querySelector('use'),writes=f.innerWrites();
  for(let i=0;i<30;i++)f.ctx.setStableHtml(list,ready);
  equal(f.innerWrites(),writes,'unchanged monitor/MCP markup is not remounted');
  equal(list.querySelector('use'),icon,'unchanged list symbols retain their actual nodes');
  f.ctx.setStableHtml(list,'<span><svg><use href="web/sprite.svg#i-warning"/></svg>Failed</span>');
  equal(list.querySelector('use').getAttribute('href'),'web/sprite.svg#i-warning','real list state changes still render');
  equal(f.innerWrites(),writes+1,'a changed list is rendered exactly once');
 }
 console.log(`PASS ${checks} icon stability checks; no browser/API/GPU execution`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
