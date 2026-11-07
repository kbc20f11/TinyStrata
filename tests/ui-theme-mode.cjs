'use strict';
// Production first-paint and theme controls with mocked storage/OS/DOM only.
// No browser, network, timers, model, process, API or GPU is started.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.join(__dirname,'..');
const app=fs.readFileSync(path.join(root,'ui/web/app.js'),'utf8');
const html=fs.readFileSync(path.join(root,'ui/web/index.html'),'utf8');
const firstPaint=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const theme=app.slice(app.indexOf('// ------------------------------------------------------------------ theme and tabs'),app.indexOf('\nlet tab ='));
const store=app.slice(app.indexOf('const store = {'),app.indexOf('\n// ------------------------------------------------------------------ toasts'));
let checks=0;
const equal=(value,expected,label)=>{assert.deepStrictEqual(value,expected,label);checks++;};
const ok=(value,label)=>{assert.ok(value,label);checks++;};

function fixture({values={},dark=false,readFails=false,writeFails=false,mediaFails=false,legacyMedia=false}={}){
 const data=new Map(Object.entries(values)),listeners=new Map(),elements=new Map();
 let previews=0,diagrams=0;
 const meta={content:''},doc={documentElement:{dataset:{}},activeElement:null,
  querySelector:selector=>selector==='meta[name="theme-color"]'?meta:null};
 const element=(id,value)=>{
  const node={id,dataset:value?{v:value}:{},children:[],attributes:{},title:'',tabIndex:-1,
   setAttribute(name,text){this.attributes[name]=String(text);},
   getAttribute(name){return this.attributes[name];},
   focus(){doc.activeElement=this;}};
  elements.set(id,node);return node;
 };
 for(const id of ['theme-icon','dark-toggle','theme-btn'])element(id);
 const group=element('theme-options');
 group.children=['system','light','dark'].map(mode=>element('theme-'+mode,mode));
 const media={matches:dark};
 if(legacyMedia)media.addListener=listener=>listeners.set('change',listener);
 else media.addEventListener=(name,listener)=>listeners.set(name,listener);
 const ctx={console,JSON,Number,String,Array,Map,Set,document:doc,
  localStorage:{getItem(key){if(readFails)throw Error('storage denied');return data.get(key)??null;},
   setItem(key,value){if(writeFails)throw Error('storage full');data.set(key,value);}},
  matchMedia(){if(mediaFails)throw Error('media unavailable');return media;},
  $:id=>elements.get(id),SPRITE:'web/sprite.svg',tr:(key,fallback)=>fallback,
  refreshPreviews(){previews++;},StrataDiagrams:{refresh(){diagrams++;}}};
 vm.createContext(ctx);vm.runInContext(firstPaint,ctx);
 const painted={...doc.documentElement.dataset};
 vm.runInContext(store+'\n'+theme,ctx);
 return {ctx,data,doc,meta,painted,buttons:group.children,previews:()=>previews,diagrams:()=>diagrams,
  mode:()=>vm.runInContext('themeMode',ctx),apply:()=>vm.runInContext('applyTheme()',ctx),
  os(value){media.matches=value;listeners.get('change')?.({matches:value});},
  key(index,key){let prevented=false;group.children[index].onkeydown({key,preventDefault(){prevented=true;}});return prevented;}};
}

// The inline pre-paint resolver and the full UI must make exactly the same choice.
for(const dark of [false,true]){
 for(const value of [undefined,'system','light','dark','invalid',null,{},true]){
  const values=value===undefined?{}:{'strata.themeMode':JSON.stringify(value)};
  const f=fixture({values,dark}),mode=['system','light','dark'].includes(value)?value:'system';
  const expected=mode==='system'?(dark?'dark':'light'):mode;
  equal(f.painted.theme,expected,'first paint honours saved mode or OS fallback');
  equal(f.doc.documentElement.dataset.theme,f.painted.theme,'main script does not flash a different colour');
  equal(f.mode(),mode,'stored mode stays distinct from its resolved colour');
 }
}
{
 const f=fixture({values:{'strata.themeMode':'broken json','strata.theme':'"light"'},dark:true});
 equal(f.mode(),'system','malformed new key falls back to system');
 equal(f.painted.theme,'dark','retired old key does not override OS');
 const denied=fixture({readFails:true,dark:true});equal(denied.painted.theme,'dark','unreadable storage still follows OS before paint');
 equal(denied.mode(),'system','unreadable storage is safe in the main UI too');
 const unsupported=fixture({mediaFails:true});equal(unsupported.painted.theme,'light','missing colour-scheme query has a deterministic fallback');
 equal(unsupported.doc.documentElement.dataset.theme,'light','missing matchMedia does not prevent startup');
}

// Manual choices are immediate, persistent and stable across OS and language changes.
{
 const f=fixture({dark:true,values:{'strata.apikey':'"untouched"'}});
 f.buttons[1].onclick();
 equal(f.mode(),'light','manual light is selected');
 equal(f.doc.documentElement.dataset.theme,'light','manual light applies immediately');
 equal(f.data.get('strata.themeMode'),'"light"','only validated mode is persisted');
 equal(f.data.get('strata.apikey'),'"untouched"','theme choice does not rewrite unrelated browser state');
 const before=f.previews();f.os(false);f.os(true);
 equal(f.doc.documentElement.dataset.theme,'light','OS changes cannot override manual light');
 equal(f.previews(),before,'ignored OS changes do not rebuild preview frames');
 f.ctx.tr=(key,fallback)=>'Translated '+fallback;f.apply();
 equal(f.mode(),'light','relocalization preserves the mode');
 equal(f.doc.documentElement.dataset.theme,'light','relocalization preserves the colour');
 ok(f.ctx.$('theme-btn').title.startsWith('Translated '),'theme-dependent accessible label is relocalized');
 const reload=fixture({values:Object.fromEntries(f.data),dark:true});
 equal(reload.painted.theme,'light','reload uses the same manual choice before paint');
 equal(reload.mode(),'light','reload retains manual mode');
 f.buttons[2].onclick();equal(f.doc.documentElement.dataset.theme,'dark','manual dark is selectable');
 equal(f.meta.content,'#040b08','browser theme colour follows manual dark');
 f.buttons[0].onclick();equal(f.mode(),'system','OS follow can be restored explicitly');
 f.os(false);equal(f.doc.documentElement.dataset.theme,'light','OS-follow mode responds to a new light preference');
 equal(f.meta.content,'#f4faf5','browser theme colour follows resolved light');
 equal(f.data.get('strata.themeMode'),'"system"','OS events do not turn system mode into a saved manual colour');
 ok(f.previews()>before&&f.diagrams()===f.previews(),'code previews and diagrams refresh together on real theme application');
}

// Radio keyboard navigation follows the existing segmented visual style.
{
 const f=fixture();
 const selected=()=>f.buttons.filter(button=>button.getAttribute('aria-checked')==='true');
 equal(selected().length,1,'exactly one radio is checked');
 equal(f.buttons.map(button=>button.tabIndex),[0,-1,-1],'only the selected choice is tabbable');
 ok(f.key(0,'ArrowRight'),'right arrow is handled');equal(f.mode(),'light','right arrow selects light');
 equal(f.doc.activeElement,f.buttons[1],'arrow navigation moves focus to the selected radio');
 equal(f.buttons.map(button=>button.tabIndex),[-1,0,-1],'roving tab stop follows the selection');
 ok(f.key(1,'End'),'End is handled');equal(f.mode(),'dark','End selects the last radio');
 ok(f.key(2,'ArrowDown'),'down arrow is handled');equal(f.mode(),'system','navigation wraps at the end');
 ok(f.key(0,'ArrowLeft'),'left arrow is handled');equal(f.mode(),'dark','left arrow wraps to dark');
 ok(f.key(2,'Home'),'Home is handled');equal(f.mode(),'system','Home selects OS follow');
 ok(f.key(0,'ArrowUp'),'up arrow is handled');equal(f.mode(),'dark','up arrow wraps backwards');
 ok(!f.key(2,'Tab'),'Tab keeps normal browser focus navigation');equal(selected().length,1,'keyboard actions never check two radios');
}
{
 const f=fixture({writeFails:true,dark:true});f.buttons[1].onclick();
 equal(f.doc.documentElement.dataset.theme,'light','blocked writes still permit a manual choice for this page');
 const legacy=fixture({legacyMedia:true});legacy.os(true);
 equal(legacy.doc.documentElement.dataset.theme,'dark','legacy media listeners continue to support system mode');
}

// Use the same three-segment component, with complete translations and no
// language handler that replaces the saved mode with a resolved colour.
ok(/class="seg seg--3" id="theme-options" role="radiogroup" aria-labelledby="theme-label"/.test(html),'existing segmented style and labelled radio group are reused');
equal((html.match(/data-i18n="theme\.(?:system|light|dark)"/g)||[]).length,3,'all three theme choices have translated labels');
ok(!app.includes('setTheme(document.documentElement.dataset.theme'),'language/start paths do not change mode to the resolved colour');
for(const language of ['en','ja','zh-CN','de','fr','es','pt-BR']){
 const catalog=JSON.parse(fs.readFileSync(path.join(root,'ui/web/locales',language+'.json'),'utf8'));
 for(const key of ['theme.system','theme.light','theme.dark'])ok(typeof catalog[key]==='string'&&catalog[key].length>0,language+' supplies '+key);
 ok(catalog['localTools.composerHint']!==undefined,language+' has the updated search-default guidance');
}
const ja=JSON.parse(fs.readFileSync(path.join(root,'ui/web/locales/ja.json'),'utf8'));
equal([ja['theme.system'],ja['theme.light'],ja['theme.dark']],['OSに従う','ライト','ダーク'],'Japanese labels are natural and concise');
ok(ja['localTools.composerHint'].includes('既定でオン'),'Japanese guidance reflects the enabled search default');
console.log(`PASS ${checks} theme mode checks; no browser/API/GPU execution`);
