'use strict';
// Bounded UI tool protocol and accounting. No browser, network, model or GPU.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const grounding=fs.readFileSync(path.join(__dirname,'..','ui','web','grounding.js'),'utf8');
const source=fs.readFileSync(path.join(__dirname,'..','ui','web','local-tools.js'),'utf8');
let checks=0;
const ok=(value,label)=>{assert(value,label);checks++;};
function fixture(){
 const calls=[],executed=[];
 const ctx={console,URL,Map,Set,Promise,JSON,Number,Object,Array,Error,DOMException,
  StrataSandbox:{execute:async({code,signal})=>{executed.push(code);if(signal?.aborted)throw new DOMException('cancelled','AbortError');return {stdout:'42\n',stderr:'',result:'',exitstatus:0,timeMs:1,images:[{name:'x.png',mime:'image/png',size:3,data_url:'data:image/png;base64,AAAA'}],files:[]};}},
  fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>url.endsWith('/prepare')?{ready:true}:{result:'42',expression:JSON.parse(options.body).expression,query:JSON.parse(options.body).query,provider:'ddgs',results:[{title:'Source',url:'https://example.org/',snippet:'Result'}]}};}};
 vm.createContext(ctx);vm.runInContext(grounding,ctx);vm.runInContext(source,ctx);return {api:ctx.StrataLocalTools,ctx,calls,executed};
}
const call=(name,args,id='call_test')=>({id,name,arguments:JSON.stringify(args)});
async function main(){
 let f=fixture(),a=f.api;
 ok(a.definitions({}).map(t=>t.function.name).join(',')==='get_current_time,calculate,run_python','clock, calculator and Python work with search off');
 ok(a.definitions({codeExecution:false}).map(t=>t.function.name).join(',')==='get_current_time,calculate,run_python','retired saved code-off preference does not disable Python');
 ok(f.calls.length===0&&f.executed.length===0,'offering Python does not download or start its runtime');
 ok(a.definitions({webSearch:true,codeExecution:true}).map(t=>t.function.name).join(',')==='web_search,read_webpage,get_current_time,calculate,run_python','tools are model-independent');
 let lightweight=a.session({webSearch:false});
 let clock=await lightweight.run(call('get_current_time',{}));
 ok(clock.ok&&JSON.parse(clock.text).source==='computer_clock'&&f.calls.length===0,'clock reads browser time without network or Python');
 let calc=await lightweight.run(call('calculate',{expression:'6*7'}));
 ok(calc.ok&&JSON.parse(calc.text).result==='42'&&f.calls[0].url==='local-ui/calculate'&&f.executed.length===0,'calculator never bootstraps Python');
 ok(!(await lightweight.run(call('calculate',{expression:'1',code:'print(1)'}))).ok,'calculator rejects extra arguments before HTTP');
 ok(!(await lightweight.run(call('get_current_time',{code:'x'}))).ok,'clock rejects unrelated arguments');
 f.calls.length=0;
 const choice={webSearch:true}, snapshot=a.session(choice);choice.webSearch=false;
 ok((await snapshot.run(call('web_search',{query:'snapshot'}))).ok,'reply snapshots original search setting');
 let pages=a.session({webSearch:false});
 ok(!(await pages.run(call('read_webpage',{url:'https://example.org/'}))).ok,'page read cannot bypass search-off preference');
 pages=a.session({webSearch:true});
 ok((await pages.run(call('read_webpage',{url:'https://example.org/'}))).ok,'page lookup runs only when enabled');
 ok(!(await pages.run(call('read_webpage',{url:'https://example.org/#section'}))).ok,'same-page anchor cannot repeat lookup');
 ok((await pages.run(call('web_search',{query:'page context'}))).ok,'search shares page lookup budget');
 ok(!(await pages.run(call('web_search',{query:'fourth lookup'}))).ok,'combined web budget includes repeats');
 const cancelled=new AbortController();cancelled.abort();
 await assert.rejects(a.session({}).run(call('get_current_time',{}),cancelled.signal),e=>e.name==='AbortError');checks++;
 f.calls.length=0;
 const collected=new Map();
 a.mergeCalls(collected,[{index:0,id:'call_a',function:{name:'web_search',arguments:'{"que'}}]);
 const merged=a.mergeCalls(collected,[{index:0,function:{arguments:'ry":"hello"}'}}]);
 ok(a.parseCall(merged[0]).query==='hello','split streamed arguments reconstruct correctly');
 assert.throws(()=>a.mergeCalls(new Map(),[{index:999,function:{name:'web_search'}}]));checks++;
 assert.throws(()=>a.mergeCalls(new Map(),[{index:0,function:{arguments:'x'.repeat(110001)}}]));checks++;
 assert.throws(()=>a.parseCall(call('web_search',{},'')));checks++;
 let session=a.session({webSearch:true},{headers:()=>({'Content-Type':'application/json'})});
 let r=await session.run(call('web_search',{query:'Official docs'}));
 ok(r.ok&&r.sources.length===1&&f.calls.length===1,'search result and citations returned');
 r=await session.run(call('web_search',{query:'  official   DOCS '}));
 ok(!r.ok&&f.calls.length===1,'duplicate search does not issue another provider request');
 await session.run(call('web_search',{query:'second'}));
 r=await session.run(call('web_search',{query:'third'}));
 ok(!r.ok&&f.calls.length===2,'search budget includes failed/repeated attempts');
 session=a.session({webSearch:false,codeExecution:false});
 r=await session.run(call('web_search',{query:'secret'}));
 ok(!r.ok&&f.calls.length===2,'disabled tool cannot run');
 f=fixture();a=f.api;session=a.session({codeExecution:false});
 r=await session.run(call('run_python',{code:'print(6*7)'}));
 ok(r.ok&&r.images.length===1&&!r.text.includes('data:image'),'Python works with a retired false preference and binary exports stay separate from context');
 await session.run(call('run_python',{code:'print(1)'}));
 ok(f.calls.filter(c=>c.url.endsWith('/prepare')).length===1&&f.executed.length===2,'runtime preparation is cached for this page');
 await session.run(call('run_python',{code:'print(2)'}));
 r=await session.run(call('run_python',{code:'print(3)'}));
 ok(!r.ok&&f.executed.length===3,'Python loop has a per-reply budget');
 const controller=new AbortController();controller.abort();
 await assert.rejects(a.executePython('print(0)',{signal:controller.signal}),e=>e.name==='AbortError');checks++;
 const sum=a.combined([{usage:{prompt_tokens:100,completion_tokens:5,prompt_tokens_details:{cached_tokens:90}},timings:{prompt_n:10,prompt_ms:10,cache_n:90,predicted_n:5,predicted_ms:100}},
  {usage:{prompt_tokens:120,completion_tokens:15,prompt_tokens_details:{cached_tokens:100}},timings:{prompt_n:20,prompt_ms:20,cache_n:100,predicted_n:15,predicted_ms:400}}]);
 ok(sum.usage.completion_tokens===20&&sum.usage.prompt_tokens===220&&sum.timings.cache_n===190,'multi-round token/cache counts are cumulative');
 ok(sum.timings.predicted_per_second===40&&sum.timings.prompt_per_second===1000,'rates use pooled time, not per-round averages');
 const mcp=a.combined([{usage:{prompt_tokens:100,completion_tokens:50},timings:{prompt_n:100,prompt_ms:10,predicted_n:5,predicted_ms:100}}]);
 ok(mcp.usage.completion_tokens===50&&mcp.timings.predicted_per_second===undefined&&mcp.timings.predicted_n===undefined,'partial MCP native timing cannot label a cumulative reply rate');
 const missing=a.combined([{usage:{completion_tokens:5},timings:{predicted_n:5,predicted_ms:100}},{usage:{completion_tokens:20}}]);
 ok(missing.usage.completion_tokens===25&&!missing.timings,'a round without timings cannot be omitted from a pooled native rate');
 ok(a.safeURL('https://example.org/docs')==='https://example.org/docs'&&!a.safeURL('javascript:alert(1)')&&!a.safeURL('http://127.0.0.1/admin')&&!a.safeURL('http://user:password@example.org/'),'citations reject unsafe links');
 ok(a.safeImage('data:image/png;base64,AAAA')&&!a.safeImage('data:image/svg+xml;base64,AAAA'),'only raster data URLs can be displayed');
 let active=0,peak=0;
 await a.mapLimit([1,2,3,4],2,async()=>{active++;peak=Math.max(peak,active);await new Promise(r=>setTimeout(r,2));active--;});
 ok(peak===2&&active===0,'parallel execution obeys configured concurrency');
 console.log(`PASS ${checks} local-tool protocol checks`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
