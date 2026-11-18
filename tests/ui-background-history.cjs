"use strict";
// Exercise the production history controller with two independent transcripts.
// The fixture supplies DOM/storage boundaries; history transformations are real.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const H = require("../ui/web/history.js");
const source = fs.readFileSync(path.join(__dirname, "../ui/web/chats-ui.js"), "utf8");
const start = source.indexOf("const pendingForChat =");
const end = source.indexOf("// ------------------------------------------------------------------ wiring", start);
assert.ok(start >= 0 && end > start);
const copy = (value) => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
let checks = 0;
const equal = (actual, expected) => { assert.deepEqual(copy(actual), copy(expected)); checks++; };

class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.value = ""; }
  set innerHTML(value) { this.html = value; this.children = [];
    if (value.includes("<span")) this.appendChild(new Element("span"));
    if (value.includes("<button")) this.appendChild(new Element("button")); }
  get innerHTML() { return this.html || ""; }
  setAttribute(name, value) { this.attributes[name] = value; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { children.forEach((child) => this.appendChild(child)); }
  querySelector(selector) { return this.children.find((child) => child.tagName === selector) || null; }
  querySelectorAll() { return []; }
  remove() { this.removed = true; }
  before(child) { this.beforeElement = child; }
  focus() { this.focused = true; }
}

function fixture() {
  const a = {id: "a", title: "Running A", messages: [{role: "user", text: "Question A"},
    {role: "assistant", text: "Partial A", reasoning: "Thinking A"}]};
  const b = {id: "b", title: "Saved B", messages: [{role: "user", text: "Question B",
    files: [{name: "notes.txt", text: "Original source B"}]},
    {role: "assistant", text: "Old answer B", reasoning: "Old reasoning", tools: [{name: "lookup", result: "B source"}]},
    {role: "user", text: "Later B"}]};
  const originalA = copy(a);
  const elements = new Map();
  const element = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  let requests = 0;
  const saved = [];
  const ctx = {H, console, busy: {chat: a, chatId: a.id, msg: a.messages[1]},
    chat: b, chatId: b.id, messages: b.messages, chats: [a, b], listOn: true,
    pending: {chat: a, chatId: a.id, text: "Queued A"}, panel: null, undoState: null,
    modelSwitching: false, modelOp: null, resourceOperation: null, attachments: [],
    $: element, node: (tag, cls, text) => { const el = new Element(tag); el.className = cls; el.textContent = text; return el; },
    tr: (key, fallback) => typeof fallback === "string" ? fallback : key, icon: (name) => name,
    toast: () => {}, stillWriting: () => {}, renderChat: () => {}, setBusy: () => {},
    renderAttachments: () => {}, autosize: () => {}, copyText: () => {}, titleOf: (chat) => chat.title,
    saveChat: () => saved.push(copy(ctx.chat)),
    setMessages: (rows) => { ctx.messages = rows; ctx.chat.messages = rows; },
    showChat: (chat) => { ctx.chat = chat; ctx.chatId = chat.id; ctx.messages = chat.messages; },
    generatingChat: () => !!ctx.busy && ctx.busy.chatId === ctx.chatId,
    generateReply: () => { requests++; },
  };
  vm.createContext(ctx);
  vm.runInContext(source.slice(start, end), ctx);
  return {ctx, a, b, originalA, saved, element, requests: () => requests,
    localLocked: () => vm.runInContext("historyLocked()", ctx),
    requestLocked: () => vm.runInContext("historyRequestLocked()", ctx)};
}

{
  const f = fixture(), {ctx, b} = f;
  equal(f.localLocked(), false);
  equal(f.requestLocked(), true);
  const answerButtons = ctx.msgActions(b.messages[1], 1).children;
  equal(answerButtons.map((button) => button.disabled), [true, false, false]);
  const userButtons = ctx.msgActions(b.messages[0], 0).children;
  equal(userButtons.map((button) => button.disabled), [false, true]);
  equal(ctx.regenerateAt(1), false);
  equal(ctx.editUserAt(0), false);
  ctx.panel = {kind: "edit-user", index: 0, message: b.messages[0], el: new Element()};
  equal(ctx.sendUserEdit(0, "Changed B"), false);
  equal(b.messages[0].text, "Question B");
  equal(f.requests(), 0);
}
{
  const f = fixture(), {ctx, b, a} = f;
  const before = b.messages;
  ctx.panel = {kind: "edit", index: 1, el: new Element()};
  equal(ctx.saveAnswerEdit(1, "Corrected B"), true);
  equal(b.messages.map((row) => row.text), ["Question B", "Corrected B"]);
  equal(b.messages[0].files, [{name: "notes.txt", text: "Original source B"}]);
  equal(b.messages[1].reasoning, undefined);
  equal(b.messages[1].tools, undefined);
  equal(a, f.originalA);
  equal(f.saved.map((chat) => chat.id), ["b"]);
  equal(ctx.undoHistory(), true);
  assert.equal(b.messages, before); checks++;
  equal(a, f.originalA);
  equal(f.requests(), 0);
}
{
  const f = fixture(), {ctx, b, a} = f;
  const before = b.messages;
  ctx.panel = {kind: "rewind", index: 1, el: new Element()};
  equal(ctx.rewindHere(1), true);
  equal(b.messages.length, 2);
  equal(ctx.undoHistory(), true);
  assert.equal(b.messages, before); checks++;
  ctx.panel = {kind: "rewind", index: 1, el: new Element()};
  equal(ctx.branchHere(1), true);
  equal(ctx.chat.parent.id, b.id);
  equal(ctx.chat.messages.map((row) => row.text), ["Question B", "Old answer B"]);
  equal(b.messages, before);
  equal(a, f.originalA);
  equal(f.requests(), 0);
}
{
  const f = fixture(), {ctx, a} = f;
  ctx.chat = a; ctx.chatId = a.id; ctx.messages = a.messages;
  equal(f.localLocked(), true);
  equal(f.requestLocked(), true);
  equal(ctx.msgActions(a.messages[1], 1), null);
  ctx.panel = {kind: "edit", index: 1, el: new Element()};
  equal(ctx.saveAnswerEdit(1, "Corrupted A"), false);
  ctx.panel = {kind: "rewind", index: 0, el: new Element()};
  equal(ctx.rewindHere(0), false);
  equal(ctx.branchHere(0), false);
  equal(a, f.originalA);
}
for (const lock of ["modelSwitching", "modelOp", "resourceOperation"]) {
  const f = fixture();
  f.ctx[lock] = true;
  equal(f.localLocked(), true);
  equal(f.requestLocked(), true);
}
{
  const f = fixture();
  f.ctx.pending = {chat: f.b, chatId: f.b.id};
  equal(f.localLocked(), true);
  f.ctx.pending = {chat: f.a, chatId: f.a.id};
  equal(f.localLocked(), false);
}
console.log(JSON.stringify({checks, result: "PASS", model_requests: 0,
  coverage: "B local edit/rewind/branch/Undo while A generates; owner A and second-request actions stay locked"}));
