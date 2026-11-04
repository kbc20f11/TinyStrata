"use strict";
// Homepage/history presentation fixtures. No browser storage, SQLite or network.
const assert = require("node:assert/strict");
const H = require("../ui/web/history.js");
const make = (messages, extra = {}) => ({id: "cfixture", title: "", time: 1234, messages, ...extra});
const copy = (value) => JSON.parse(JSON.stringify(value));
let checks = 0;
function equal(actual, expected) { assert.deepEqual(actual, expected); checks++; }

for (const value of [null, undefined, {}, make([]), make([{role: "assistant", text: "", reasoning: "", tools: []}]),
  make([{role: "user", text: " \n\t "}]), make([{role: "assistant", stats: {tokens: 3}, meta: "Stopped"}]),
  make([{role: "system", text: "Internal instructions"}]), make([{role: "tool", text: "An orphan result"}])]) {
  equal(H.hasConversation(value), false);
  equal(H.chatTitle(value, "新しいチャット"), "新しいチャット");
}
for (const message of [
  {role: "user", text: "Actual question"}, {role: "assistant", text: "Actual answer"},
  {role: "assistant", reasoning: "Thinking while generating"}, {role: "assistant", error: "Interrupted"},
  {role: "assistant", error: {message: "Failed request"}},
  {role: "user", images: [{name: "picture.png"}]}, {role: "user", files: [{name: "notes.txt", text: ""}]},
  {role: "assistant", tools: [{name: "web_search", arguments: {query: "train"}}]}
]) equal(H.hasConversation(make([message])), true);

equal(H.chatTitle(make([{role: "user", text: "\n \n 最初の質問\r\n二行目"}])), "最初の質問");
equal(H.chatTitle(make([{role: "user", text: "Question"}]), "New", "Attachment"), "Question");
equal(H.chatTitle(make([{role: "user", text: "Question"}], {title: "My custom title"})), "My custom title");
equal(H.chatTitle(make([{role: "user", text: "x".repeat(100)}])).length, H.TITLE_MAX);
equal(H.chatTitle(make([{role: "user", text: " \n "}, {role: "assistant", text: ""},
  {role: "user", text: "First real question"}])), "First real question");
equal(H.chatTitle(make([{role: "user", images: [{name: "地図.png", url: "data:image/png;base64,AA=="}]},
  {role: "assistant", text: "A map"}, {role: "user", text: "Describe it further"}])), "地図.png");
equal(H.chatTitle(make([{role: "user", files: [{name: "report.txt", text: "report contents"}]}])), "report.txt");
equal(H.chatTitle(make([{role: "assistant", text: "\n Imported answer\nMore"}]), "New"), "Imported answer");
equal(H.chatTitle(make([{role: "assistant", reasoning: "Private reasoning"}]), "New"), "New");
equal(H.chatTitle(make([{role: "user", images: [{url: "data:image/png;base64,AA=="}]}]),
  "新しいチャット", "添付ファイルの会話"), "添付ファイルの会話");
equal(H.chatTitle(make([{role: "user", images: [{}]}, {role: "assistant", text: "An untitled attachment answer"}]),
  "New", "Attachment"), "Attachment");
equal(H.chatTitle(make([{role: "user", images: [{}]}])), "Attachment chat");
const anonymousAttachment = [{role: "user", images: [{url: "data:image/png;base64,AA=="}]}];
equal(H.chatTitle(make(anonymousAttachment), "新しいチャット", "添付ファイルの会話"), "添付ファイルの会話");
equal(H.chatTitle(make([...anonymousAttachment, {role: "assistant", text: "画像の説明です。"}]),
  "新しいチャット", "添付ファイルの会話"), "添付ファイルの会話");

const existing = [make([], {id: "coldempty", _revision: 7, title: "New chat"}),
  make([{role: "user", text: "Saved question", images: [{name: "x.png", url: "data:image/png;base64,AA=="}]}],
    {id: "csaved", _revision: 11})];
const before = copy(existing);
const visible = existing.filter(H.hasConversation);
equal(visible.map((c) => c.id), ["csaved"]);
equal(existing, before); // Filtering must not delete old rows or change attachment data/revisions.
equal(H.filterChats(existing, "", "New").length, 2); // The caller alone owns homepage filtering.

const current = existing[1];
const originalMessages = current.messages;
const plan = H.rewind(originalMessages, 0);
current.messages = plan.keep;
equal(H.hasConversation(current), false);
equal(current.id, "csaved");
equal(current._revision, 11);
equal(originalMessages.length, 1);
current.messages = originalMessages; // Undo restores the same chat identity rather than creating another row.
equal(H.hasConversation(current), true);
equal(H.chatTitle(current), "Saved question");

const branched = H.branch(current, [], 0, "Branch of saved", 1235, .5);
equal(H.hasConversation(branched), false);
equal(H.chatTitle(branched), "Branch of saved");
equal(branched.parent.id, "csaved");

console.log(JSON.stringify({checks, result: "PASS", storage_mutations: false, network: false}));
