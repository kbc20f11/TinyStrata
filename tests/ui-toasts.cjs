'use strict';
// Execute the production toast controller with deterministic time and a small
// DOM event fixture; real browser layout/interaction is checked separately.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/web/app.js'), 'utf8');
const controller = source.slice(source.indexOf('// ------------------------------------------------------------------ toasts'), source.indexOf('async function copyText'));

function fixture() {
  let now = 0, nextTimer = 1;
  const timers = new Map(), events = new Map(), windowEvents = new Map();
  const document = {hidden: false, activeElement: null,
    addEventListener: (name, fn) => { if (!events.has(name)) events.set(name, []); events.get(name).push(fn); }};
  class Element {
    constructor(tag='div') { this.tagName = tag.toUpperCase(); this.children = []; this.parentElement = null; this.listeners = new Map(); this.attrs = {}; this.className = ''; }
    get isConnected() { return this === body || !!this.parentElement?.isConnected; }
    set innerHTML(value) {
      this.html = value; this.children = [];
      if (value.includes('st-toast__body')) {
        const body = this.appendChild(new Element()); body.className = 'st-toast__body';
        for (const cls of ['st-toast__title', 't-text']) { const child=body.appendChild(new Element()); child.className=cls; }
      }
    }
    get innerHTML() { return this.html || ''; }
    appendChild(child) { if (child.parentElement) child.remove(); this.children.push(child); child.parentElement=this; return child; }
    contains(child) { return child === this || this.children.some(node => node.contains(child)); }
    querySelector(selector) { return this.children.flatMap(child => [child, ...child.all()]).find(node => node.className.split(' ').includes(selector.slice(1))) || null; }
    all() { return this.children.flatMap(child => [child, ...child.all()]); }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    getAttribute(name) { return this.attrs[name] ?? null; }
    addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, []); this.listeners.get(name).push(fn); }
    matches() { return this.hovered || this.contains(document.activeElement); }
    emit(name, extra={}) {
      if (name === 'pointerenter') this.hovered = true;
      if (name === 'pointerleave') this.hovered = false;
      const event={target:this, currentTarget:this, defaultPrevented:false, stopped:false,
        preventDefault(){this.defaultPrevented=true;},stopPropagation(){this.stopped=true;},...extra};
      let current=this;
      do {
        event.currentTarget=current;
        if (name === 'click') current.onclick?.(event);
        for (const fn of current.listeners.get(name) || []) fn(event);
        current=current.parentElement;
      } while (current && !event.stopped && ['click','keydown','focusin','focusout'].includes(name));
      return event;
    }
    click() { return this.emit('click'); }
    focus() {
      const old=document.activeElement;
      if (old === this) return;
      document.activeElement=this;
      old?.emit('focusout', {relatedTarget:this}); this.emit('focusin', {relatedTarget:old});
    }
    remove() {
      if (this.contains(document.activeElement)) document.activeElement=body;
      if (this.parentElement) this.parentElement.children=this.parentElement.children.filter(child=>child!==this);
      this.parentElement=null;
    }
  }
  const body=new Element('body'), viewport=body.appendChild(new Element()), trigger=body.appendChild(new Element('button'));
  document.body=body; document.activeElement=trigger; document.createElement=tag=>new Element(tag);
  const context=vm.createContext({document, window:{addEventListener:(name,fn)=>{if(!windowEvents.has(name))windowEvents.set(name,[]);windowEvents.get(name).push(fn);}},
    $:()=>viewport, tr:(_key,fallback)=>fallback, icon:name=>`<svg data-icon="${name}"></svg>`, performance:{now:()=>now},
    setTimeout:(fn,ms)=>{const id=nextTimer++;timers.set(id,{fn,at:now+ms});return id;},clearTimeout:id=>timers.delete(id)});
  vm.runInContext(controller,context);
  function advance(ms) {
    const target=now+ms;
    for (;;) {
      const due=[...timers].filter(([,entry])=>entry.at<=target).sort((a,b)=>a[1].at-b[1].at)[0];
      if(!due)break;
      now=due[1].at;timers.delete(due[0]);due[1].fn();
    }
    now=target;
  }
  const signal=(name)=>{for(const fn of windowEvents.get(name)||[])fn();};
  const hidden=(value)=>{document.hidden=value;for(const fn of events.get('visibilitychange')||[])fn();};
  return {toast:context.toast,viewport,trigger,body,document,advance,signal,hidden,timers};
}

let count=0;
function test(name, run) { run(fixture()); count++; console.log('PASS '+name); }
test('default duration and explicit short duration expire without focus theft', f=>{
  const notice=f.toast('info','Saved','Details'); assert.equal(f.document.activeElement,f.trigger);
  assert.equal(notice.querySelector('.st-toast__close').getAttribute('aria-label'),'Close');
  f.advance(6999); assert.equal(f.viewport.children.length,1); f.advance(1);assert.equal(f.viewport.children.length,0);
  f.toast('success','Copied','',1800);f.advance(1800);assert.equal(f.viewport.children.length,0);assert.equal(f.timers.size,0);
});
test('body and outside clicks retain selectable content; only close dismisses', f=>{
  const notice=f.toast('info','Notice','Select this text');notice.querySelector('.t-text').click();assert.equal(f.viewport.children.length,1);
  let clicks=0;f.trigger.onclick=()=>clicks++;f.trigger.click();assert.equal(clicks,1);assert.equal(f.viewport.children.length,1);
  notice.querySelector('.st-toast__close').click();assert.equal(f.viewport.children.length,0);assert.equal(f.timers.size,0);
});
test('focused Escape closes and returns focus without escaping into the page', f=>{
  const notice=f.toast('info','Notice'),close=notice.querySelector('.st-toast__close');close.focus();
  const event=close.emit('keydown',{key:'Escape'});assert.ok(event.defaultPrevented&&event.stopped);
  assert.equal(f.viewport.children.length,0);assert.equal(f.document.activeElement,f.trigger);
});
test('hover pauses remaining time rather than restarting or expiring it', f=>{
  const notice=f.toast('info','Notice');f.advance(2000);notice.emit('pointerenter');f.advance(20000);assert.equal(f.viewport.children.length,1);
  notice.emit('pointerleave');f.advance(4999);assert.equal(f.viewport.children.length,1);f.advance(1);assert.equal(f.viewport.children.length,0);
});
test('focus moving between action and close keeps expiry paused', f=>{
  const notice=f.toast('info','Deleted','',3500,{label:'Undo',run(){}}),action=notice.querySelector('.st-toast__action');
  f.advance(1000);action.focus();f.advance(20000);notice.querySelector('.st-toast__close').focus();f.advance(20000);
  assert.equal(f.viewport.children.length,1);f.trigger.focus();f.advance(8999);assert.equal(f.viewport.children.length,1);f.advance(1);assert.equal(f.viewport.children.length,0);
});
test('window blur and document hiding both pause remaining time', f=>{
  f.toast('info','Notice');f.advance(1000);f.signal('blur');f.advance(20000);f.signal('focus');
  f.advance(1000);f.hidden(true);f.advance(20000);assert.equal(f.viewport.children.length,1);
  f.hidden(false);f.advance(4999);assert.equal(f.viewport.children.length,1);f.advance(1);assert.equal(f.viewport.children.length,0);
});
test('identical passive notices coalesce without resetting expiry', f=>{
  const first=f.toast('warn','Same','Details');f.advance(6000);assert.equal(f.toast('warn','Same','Details'),first);
  assert.equal(f.viewport.children.length,1);f.advance(1000);assert.equal(f.viewport.children.length,0);
});
test('stack is bounded and notices being read are not replaced', f=>{
  const first=f.toast('info','First');first.emit('pointerenter');
  for(let n=2;n<=8;n++)f.toast('info','Notice'+n);
  assert.equal(f.viewport.children.length,4);assert.ok(first.isConnected);
});
test('queued notices get a full lifetime and action notices are preserved', f=>{
  let actions=0;
  for(let n=0;n<4;n++)f.toast('info','Undo'+n,'',1000,{label:'Undo',run:()=>actions++});
  const queued=f.toast('success','Queued');assert.equal(f.viewport.children.length,4);assert.equal(queued.isConnected,false);
  f.advance(10000);assert.equal(actions,0);assert.equal(f.viewport.children.length,1);assert.ok(queued.isConnected);
  f.advance(6999);assert.equal(f.viewport.children.length,1);f.advance(1);assert.equal(f.viewport.children.length,0);
});
test('action executes once even if the detached button is activated again', f=>{
  let actions=0;const notice=f.toast('info','Deleted','',1000,{label:'Undo',run:()=>actions++}),button=notice.querySelector('.st-toast__action');
  button.click();button.click();assert.equal(actions,1);assert.equal(f.viewport.children.length,0);assert.equal(f.timers.size,0);
});
console.log(`PASS ${count} toast behavior cases`);
