"use strict";
// Exercise production ownership helpers and a gated SSE generation in Node.
// Real browser/SQLite navigation is verified separately with the private UI.
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm"), assert = require("node:assert/strict");
const H = require("../ui/web/history.js");
const app = fs.readFileSync(path.join(__dirname, "../ui/web/app.js"), "utf8");
const chatsSource = fs.readFileSync(path.join(__dirname, "../ui/web/chats-ui.js"), "utf8");
let checks = 0;
const equal = (a, b, label) => { assert.deepStrictEqual(a, b, label); checks++; };
const ok = (a, label) => { assert.ok(a, label); checks++; };
const plain = value => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };

function production(name, source = app) {
  const match = new RegExp(`(?:async )?function ${name}\\(`).exec(source);
  assert(match, "Missing production function: " + name);
  const end = source.indexOf("\n}\n", match.index);
  assert(end > match.index, "Missing function boundary: " + name);
  return source.slice(match.index, end + 3);
}

function fixture() {
  const A = {id: "A", title: "Origin A", time: 1, messages: [{role: "user", text: "A source"}]};
  const B = {id: "B", title: "Archive B", time: 2, messages: [{role: "user", text: "B source"}]};
  const values = new Map([["chatId", "B"]]), writes = [], removed = [], calls = [];
  const input = {value: "B unsent draft"}, elements = {input};
  const ctx = {console, H, Map, Set, Array, Object, String, JSON, Math, Date, Number, Promise,
    AbortController, DOMException, TextDecoder, performance,
    chat: B, chatId: B.id, chats: [A, B], messages: B.messages, homeChat: null, listOn: true,
    busy: null, pending: null, panel: null, modelSwitching: false, resourceOperation: null,
    settings: {autoCompact: true, whileBusy: "queue"}, health: {model: "qwen", images: false},
    attachments: [{kind: "file", name: "B.txt", text: "B attachment"}], composerDrafts: new Map(),
    store: {set: (key, value) => values.set(key, value)},
    chatStore: {put: value => writes.push(plain(value)), remove: id => removed.push(id)},
    $: id => elements[id] || (elements[id] = {}),
    tr: (key, fallback) => fallback, titleOf: c => c.title, stillWriting: () => calls.push("blocked"),
    renderSidebar: () => calls.push("sidebar"), renderAttachments: () => calls.push("attachments"),
    autosize: () => calls.push("autosize"), setBusy: value => calls.push(["busy", value]),
    activeModelNeedsDownload: () => false, canUseModelImages: () => true, clearUndo: () => {},
    toast: () => {}, saveComposerDraft: () => {}, closePanel: () => {},
    refreshTerminalStats: () => {}, updateAssistant: () => calls.push("paint"),
    terminalMessageElement: m => ctx.messages.includes(m) ? {message: m} : null,
  };
  vm.createContext(ctx);
  for (const name of ["generatingChat", "saveChat", "chatDraft", "deleteChat"]) vm.runInContext(production(name, chatsSource), ctx);
  for (const name of ["prepareContext", "settlePending", "pendingNow", "releasePending", "send"]) vm.runInContext(production(name), ctx);
  return {ctx, A, B, values, writes, removed, calls, input};
}

async function main() {
  {
    const {ctx, A, B, values, writes} = fixture();
    const before = JSON.stringify(B);
    A.messages.push({role: "assistant", text: "A completed"});
    ctx.saveChat(A);
    equal(writes.map(c => c.id), ["A"], "a hidden answer saves its own SQLite record");
    equal(values.get("chatId"), "B", "background persistence leaves the selected chat untouched");
    equal(ctx.messages, B.messages, "visible transcript identity stays B");
    equal(JSON.stringify(B), before, "background save does not rewrite B");
  }
  {
    const {ctx, A, B, writes} = fixture(), gate = deferred();
    const message = {role: "assistant", text: ""};
    A.messages.push(message);
    ctx.chat = A; ctx.chatId = "A"; ctx.messages = A.messages;
    ctx.StrataContextCompaction = {project: async rows => ({messages: rows, summary: ""}), prepare: () => gate.promise};
    const owner = {chat: A, messages: A.messages, settings: {autoCompact: true}};
    const run = ctx.prepareContext({model: "qwen"}, A.messages.slice(0, -1), message,
                                   new AbortController().signal, rows => rows, owner);
    await Promise.resolve();
    ctx.chat = B; ctx.chatId = "B"; ctx.messages = B.messages;
    gate.resolve({changed: true, checkpoint: {summary: "Only A source"}, beforeTokens: 500, afterTokens: 100,
                  request: {model: "qwen", messages: [{role: "assistant", content: "Only A source"}]}});
    const prepared = await run;
    equal(A.compaction.summary, "Only A source", "a count/summary that finishes off-screen commits to A");
    equal(B.compaction, undefined, "compaction never lands in the visible B record");
    equal(writes.map(c => c.id), ["A"], "compaction persistence retains its owner");
    equal(prepared.messages[0].content, "Only A source", "the prepared continuation keeps A's summary");
    equal(message.compacting, false, "the off-screen progress flag is cleared");
  }
  {
    const {ctx, A} = fixture(), gate = deferred(), message = {role: "assistant", text: ""};
    A.messages.push(message);
    const old = A.messages;
    ctx.StrataContextCompaction = {project: async rows => ({messages: rows, summary: ""}), prepare: () => gate.promise};
    const run = ctx.prepareContext({model: "qwen"}, old, message, new AbortController().signal,
      rows => rows, {chat: A, messages: old, settings: {autoCompact: true}});
    await Promise.resolve();
    A.messages = [{role: "user", text: "Replaced by another operation"}];
    gate.resolve({changed: true, checkpoint: {summary: "Stale"}});
    await assert.rejects(run, error => error.name === "AbortError"); checks++;
    equal(A.compaction, undefined, "replacing the origin transcript invalidates its pending checkpoint");
  }
  {
    const {ctx, A, B, calls} = fixture(), payload = {text: "queued A", images: [], files: [], mode: "queue", chat: A, chatId: "A"};
    ctx.pending = payload;
    ctx.send = (p, target) => calls.push({p, target});
    ctx.settlePending(false, true, A);
    equal(ctx.pending, null, "a completed origin releases its queued message");
    const sent = calls.find(c => c && c.target);
    equal(sent.target, A, "the automatic follow-up stays in origin A while B is visible");
    equal(sent.p, payload, "queued text and attachments survive unchanged");
    equal(ctx.messages, B.messages, "automatic follow-up does not switch the visible transcript");
    ctx.pending = {...payload, mode: "interject"};
    ctx.settlePending(false, true, A);
    ok(ctx.pending, "an interjection is not sent when the original did not acknowledge interruption");
    ctx.settlePending(true, false, A);
    equal(ctx.pending, null, "an acknowledged cut releases the origin interjection");
  }
  {
    const {ctx, A, B, input} = fixture();
    const beforeAttachments = JSON.stringify(ctx.attachments);
    ctx.pending = {text: "A parked", images: [], files: [{kind: "file", name: "A.txt"}], chat: A, chatId: "A"};
    ctx.releasePending();
    equal(input.value, "B unsent draft", "taking back a hidden pending message never changes B's composer");
    equal(JSON.stringify(ctx.attachments), beforeAttachments, "B attachments are not replaced by A's pending attachments");
    equal(ctx.chatDraft(A).text, "A parked", "the pending message returns to A's own draft");
    equal(ctx.chatDraft(A).attachments[0].name, "A.txt", "A's attachment returns with its text");
    equal(B.messages[0].text, "B source", "B's source message is intact");
  }
  {
    const {ctx, A, B, removed, input} = fixture();
    ctx.busy = {chat: A, chatId: "A", messages: A.messages, msg: {role: "assistant"}};
    const before = JSON.stringify(B);
    await ctx.send();
    equal(JSON.stringify(B), before, "submitting in B cannot join A's in-flight generation");
    equal(input.value, "B unsent draft", "a blocked cross-chat submit keeps the draft");
    ctx.deleteChat(A);
    equal(removed, [], "the hidden generating owner cannot be deleted from the sidebar");
    equal(ctx.chats.length, 2, "a blocked deletion leaves both records present");
  }
  {
    const f = fixture(), {ctx, A, B, writes, values, calls} = f;
    const stream = deferred(), requested = deferred();
    const packets = [
      'data: {"choices":[{"delta":{"content":"A prefix "}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"A final"},"finish_reason":"stop"}],"usage":{"completion_tokens":4,"prompt_tokens":3}}\n\ndata: [DONE]\n\n'];
    let reads = 0, frames = 0, request;
    Object.assign(ctx, {
      settings: {autoCompact: true}, globalTools: {max_concurrency: 1}, mcpInfo: {tools: 0},
      renderChat: () => {}, startTerminalClock: () => () => {}, paintLive: () => {},
      apiMessageSerializer: () => rows => rows.map(m => ({role: m.role, content: m.text})),
      thinkingPayload: () => ({}), samplingUntouched: () => true, projectionLoaded: () => false,
      headers: () => ({}), requestAnimationFrame: () => ++frames, cancelAnimationFrame: () => {},
      terminalBeginRound: () => {}, terminalFinalStats: value => value,
      terminalNumber: value => typeof value === "number" && Number.isFinite(value),
      fmt: String, scrollDown: () => calls.push("scroll"), settlePending: () => {},
      StrataLocalTools: {MAX_ROUNDS: 2, session: () => ({}), definitions: () => [],
        combined: rounds => rounds.length ? rounds[rounds.length - 1] : {},
        mergeCalls: () => [], mapLimit: async () => {}},
      prepareContext: async (body, rows) => ({...body, messages: rows.map(m => ({role: m.role, content: m.text}))}),
      fetch: async (_url, options) => {
        request = JSON.parse(options.body); requested.resolve();
        return {ok: true, body: {getReader: () => ({read: async () => {
          if (reads === 1) await stream.promise;
          if (reads >= packets.length) return {done: true};
          return {done: false, value: Buffer.from(packets[reads++])};
        }})}};
      },
    });
    vm.runInContext(production("generateReply"), ctx);
    ctx.chat = A; ctx.chatId = "A"; ctx.messages = A.messages;
    const generating = ctx.generateReply(A);
    await requested.promise;
    await new Promise(resolve => setImmediate(resolve));
    ctx.chat = B; ctx.chatId = "B"; ctx.messages = B.messages; values.set("chatId", "B");
    const visibleBefore = JSON.stringify(B);
    stream.resolve();
    await generating;
    equal(A.messages.at(-1).text, "A prefix A final", "a real streamed continuation completes in A after B is selected");
    equal(JSON.stringify(B), visibleBefore, "stream completion cannot mutate B's transcript");
    equal(values.get("chatId"), "B", "completion does not move the selected chat back to A");
    ok(writes.length >= 2 && writes.every(row => row.id === "A"), "initial and final records are persisted to A");
    equal(request.messages, [{role: "user", content: "A source"}], "the model request contains A's transcript only");
    equal(calls.filter(c => c === "scroll").length, 0, "background completion never scrolls the reader in B");
    equal(ctx.busy, null, "the completed stream releases the global generation slot");
  }
  console.log(`PASS background chat ownership: ${checks} checks`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
