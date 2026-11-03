'use strict';
// Bounded UI tool protocol and accounting. No browser, network, model or GPU.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'..','ui','web','local-tools.js'),'utf8');
let checks=0;
const ok=(value,label)=>{assert(value,label);checks++;};
function fixture(){
 const calls=[],executed=[];
 const ctx={console,URL,Map,Set,Promise,JSON,Number,Object,Array,Error,DOMException,
  StrataSandbox:{execute:async({code,signal})=>{executed.push(code);if(signal?.aborted)throw new DOMException('cancelled','AbortError');return {stdout:'42\n',stderr:'',result:'',exitstatus:0,timeMs:1,images:[{name:'x.png',mime:'image/png',size:3,data_url:'data:image/png;base64,AAAA'}],files:[]};}},
  fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>url.endsWith('/prepare')?{ready:true}:{query:JSON.parse(options.body).query,provider:'ddgs',results:[{title:'Source',url:'https://example.org/',snippet:'Result'}]}};}};
 vm.createContext(ctx);vm.runInContext(source,ctx);return {api:ctx.StrataLocalTools,ctx,calls,executed};
}
const call=(name,args,id='call_test')=>({id,name,arguments:JSON.stringify(args)});
async function main(){
 let f=fixture(),a=f.api;
 ok(a.definitions({}).length===0,'optional tools are absent until enabled');
 ok(a.definitions({webSearch:true,codeExecution:true}).map(t=>t.function.name).join(',')==='web_search,run_python','tools are model-independent');
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
 f=fixture();a=f.api;session=a.session({codeExecution:true});
 r=await session.run(call('run_python',{code:'print(6*7)'}));
 ok(r.ok&&r.images.length===1&&!r.text.includes('data:image'),'binary exports are separate from model context');
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
