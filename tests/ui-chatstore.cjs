'use strict';
// Isolated browser-store protocol fixtures. No network, browser data or GPU.
const assert = require('node:assert/strict');
const {createChatStore} = require('../ui/web/chatstore.js');
const copy = x => JSON.parse(JSON.stringify(x));
const chat = (id='c1', text='hello') => ({id, title:'', time:1234, messages:[{role:'user', text}]});
function browser(initial=[], usable=true) {
  const rows = new Map(initial.map(c => [c.id, copy(c)]));
  return {rows, start:async()=>usable, all:async()=>[...rows.values()].map(copy),
    put:async(value, expected)=>{
      if (!usable) return false;
      if (expected === undefined || JSON.stringify(rows.get(value.id)) === JSON.stringify(expected)) rows.set(value.id,copy(value));
      return true;
    }, remove:async(id,expected)=>{
      if (expected === undefined || JSON.stringify(rows.get(id)) === JSON.stringify(expected)) rows.delete(id);
      return true;
    }};
}
function server(initial=[]) {
  const rows = new Map(initial.map(c=>[c.id,copy(c)])), requests=[], backups=[];
  let online=true, failImportAfterCommit=false, beforeImport=null;
  const reply=(status,value)=>({ok:status<400,status,json:async()=>copy(value)});
  async function fetch(url,options={}) {
    if(!online) throw Error('offline fixture');
    const method=options.method||'GET',body=options.body?JSON.parse(options.body):null;
    requests.push({url,method,body});
    const p=new URL(url,'http://fixture').pathname;
    if(method==='GET') return reply(200,{chats:[...rows.values()].filter(c=>!c._deleted),next:null});
    if(p.endsWith('/import')) {
      if(beforeImport) await beforeImport();
      backups.push(copy(body.chats));
      const mappings=[];
      for(const c of body.chats) {
        const old=rows.get(c.id); let id=c.id;
        if(old&&old._deleted) {mappings.push({from:id,id,revision:old._revision,deleted:true});continue;}
        if(old&&JSON.stringify({...old,_revision:undefined})!==JSON.stringify(c)) id=c.id+'conflict';
        if(!rows.has(id)) rows.set(id,{...copy(c),id,_revision:1});
        mappings.push({from:c.id,id,revision:rows.get(id)._revision});
      }
      if(failImportAfterCommit) throw Error('response lost');
      return reply(200,{confirmed:true,backup_id:'a'.repeat(64),mappings});
    }
    const id=decodeURIComponent(p.split('/').pop()),old=rows.get(id),rev=old?old._revision:0;
    if(body.expected_revision!==rev) return reply(409,{error:'conflict',revision:rev,current:old||null});
    if(method==='PUT') {const c={...copy(body.chat),_revision:rev+1};rows.set(id,c);return reply(200,{chat:c});}
    if(method==='DELETE') {rows.set(id,{id,_revision:rev+1,_deleted:true});return reply(200,{id,revision:rev+1,deleted:true});}
    throw Error('unexpected request');
  }
  return {rows,requests,backups,fetch,set online(v){online=v;},set failImportAfterCommit(v){failImportAfterCommit=v;},set beforeImport(v){beforeImport=v;}};
}
async function main() {
  let count=0;
  async function test(name,fn) {await fn();count++;console.log('PASS '+name);}
  await test('atomic migration preserves original until confirmed, attachments included',async()=>{
    const value=chat();value.messages[0].images=[{name:'x.png',url:'data:image/png;base64,YQ=='}];value.messages[0].files=[{name:'x.txt',text:'persist'}];
    const local=browser([value]),remote=server();let allow;const blocked=new Promise(resolve=>{allow=resolve;});
    remote.beforeImport=()=>blocked;
    const store=createChatStore({localStore:local,fetch:remote.fetch});const started=store.start();
    await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(local.rows.get('c1'),value);
    allow();await started;
    assert.equal(store.storage,'sqlite');assert.equal((await store.all())[0].messages[0].images[0].url,value.messages[0].images[0].url);
    assert.deepEqual(remote.backups[0],[value]);assert.equal(local.rows.get('c1')._revision,1);
  });
  await test('lost import response leaves IndexedDB original for idempotent retry',async()=>{
    const value=chat(),local=browser([value]),remote=server();remote.failImportAfterCommit=true;
    const first=createChatStore({localStore:local,fetch:remote.fetch});await first.start();assert.deepEqual(local.rows.get('c1'),value);
    remote.failImportAfterCommit=false;const second=createChatStore({localStore:local,fetch:remote.fetch});await second.start();
    assert.equal(remote.rows.size,1);assert.equal((await second.all()).length,1);
  });
  await test('queued snapshots serialize optimistic revisions and preserve attachments',async()=>{
    const local=browser(),remote=server(),store=createChatStore({localStore:local,fetch:remote.fetch});await store.start();
    const value=chat();const a=store.put(value);value.messages[0].text='changed after put';
    const b=store.put(chat('c1','second'));assert.equal(await a,true);assert.equal(await b,true);
    const puts=remote.requests.filter(r=>r.method==='PUT');assert.deepEqual(puts.map(r=>r.body.expected_revision),[0,1]);
    assert.equal(puts[0].body.chat.messages[0].text,'hello');assert.equal(remote.rows.get('c1')._revision,2);
  });
  await test('conflict never overwrites server and keeps pending browser copy',async()=>{
    const original={...chat(),_revision:1},local=browser([original]),remote=server([original]),problems=[];
    const store=createChatStore({localStore:local,fetch:remote.fetch,onProblem:x=>problems.push(x)});await store.start();
    remote.rows.set('c1',{...chat('c1','other tab'),_revision:2});assert.equal(await store.put(chat('c1','my edit')),false);
    assert.equal(remote.rows.get('c1').messages[0].text,'other tab');assert.equal(local.rows.get('c1').messages[0].text,'my edit');
    assert.equal(local.rows.get('c1')._pending,true);assert.deepEqual(problems,['conflict']);
    const reloaded=createChatStore({localStore:local,fetch:remote.fetch});await reloaded.start();assert.equal((await reloaded.all()).length,2);
  });
  await test('offline browser fallback retains changes',async()=>{
    const local=browser([chat()]),remote=server();remote.online=false;const store=createChatStore({localStore:local,fetch:remote.fetch});
    assert.equal(await store.start(),true);assert.equal(store.storage,'browser');await store.put(chat('c2','offline'));
    assert.equal((await store.all()).length,2);assert.equal(local.rows.get('c2')._pending,true);
  });
  await test('without browser or server memory remains usable for visit',async()=>{
    const remote=server();remote.online=false;const store=createChatStore({localStore:browser([],false),fetch:remote.fetch});
    assert.equal(await store.start(),false);await store.put(chat());assert.equal((await store.all()).length,1);assert.equal(store.storage,'memory');
  });
  await test('SQLite still persists when browser storage is denied',async()=>{
    const remote=server(),store=createChatStore({localStore:browser([],false),fetch:remote.fetch});
    assert.equal(await store.start(),true);assert.equal(await store.put(chat()),true);assert.equal(remote.rows.size,1);
  });
  await test('offline deletion replays without resurrecting stale browser chat',async()=>{
    const original={...chat(),_revision:1},local=browser([original]),remote=server([original]);remote.online=false;
    const offline=createChatStore({localStore:local,fetch:remote.fetch});await offline.start();await offline.remove('c1');assert.equal((await offline.all()).length,0);
    remote.online=true;const online=createChatStore({localStore:local,fetch:remote.fetch});await online.start();assert.equal((await online.all()).length,0);assert.equal(remote.rows.get('c1')._deleted,true);
  });
  await test('rejected deletion restores visible newer server record instead of hiding it forever',async()=>{
    const original={...chat(),_revision:1},local=browser([original]),remote=server([original]);
    const store=createChatStore({localStore:local,fetch:remote.fetch});await store.start();
    remote.rows.set('c1',{...chat('c1','newer tab'),_revision:2});assert.equal(await store.remove('c1'),false);
    assert.equal((await store.all())[0].messages[0].text,'newer tab');assert.equal(local.rows.get('c1')._pending,false);
  });
  console.log(`PASS ui-chatstore ${count} scenarios`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
