// Local UI SQLite history, with an IndexedDB recovery copy. Original browser
// records survive until an atomic server import and its backup are confirmed.
// Independent implementation; no Open WebUI source is included.
"use strict";
(function (root) {
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const clean = (chat) => Object.fromEntries(Object.entries(clone(chat)).filter(([k]) =>
    !["_revision", "_pending", "_deleted", "_local_token", "_server_id"].includes(k)));
  const token = () => root.crypto && root.crypto.randomUUID ? root.crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  function browserStore(idb) {
    let db = null, failed = false;
    const ready = new Promise((resolve) => {
      let r;
      try { r = idb.open("strata", 1); } catch (_) { resolve(false); return; }
      r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains("chats")) r.result.createObjectStore("chats", {keyPath: "id"}); };
      r.onsuccess = () => {
        const opened = r.result;
        if (failed || !opened.objectStoreNames.contains("chats")) { opened.close(); resolve(false); return; }
        db = opened; db.onversionchange = () => { db.close(); db = null; }; resolve(true);
      };
      r.onerror = r.onblocked = () => { failed = true; resolve(false); };
    });
    const change = (id, next, expected) => new Promise((resolve) => {
      if (!db) { resolve(false); return; }
      try {
        const t = db.transaction("chats", "readwrite"), s = t.objectStore("chats");
        const write = () => next ? s.put(clone(next)) : s.delete(id);
        if (expected === undefined) write();
        else { const r = s.get(id); r.onsuccess = () => { if (JSON.stringify(r.result) === JSON.stringify(expected)) write(); }; }
        t.oncomplete = () => resolve(true); t.onerror = t.onabort = () => resolve(false);
      } catch (_) { resolve(false); }
    });
    return {start: () => ready,
      all: () => new Promise((resolve) => {
        if (!db) { resolve([]); return; }
        try {
          const t = db.transaction("chats", "readonly"), r = t.objectStore("chats").getAll();
          r.onsuccess = () => resolve(r.result || []); r.onerror = t.onabort = () => resolve([]);
        } catch (_) { resolve([]); }
      }), put: (v, expected) => change(v.id, v, expected), remove: (id, expected) => change(id, null, expected)};
  }
  function createChatStore({indexedDB = root.indexedDB, fetch: request = root.fetch && root.fetch.bind(root),
    localStore = null, onProblem = () => {}, endpoint = "/local-ui/chats"} = {}) {
    const local = localStore || browserStore(indexedDB), memory = new Map(), revisions = new Map(), blocked = new Set(), told = new Set();
    let server = false, localReady = false, started = null, queue = Promise.resolve();
    const problem = (kind) => { if (!told.has(kind)) { told.add(kind); onProblem(kind); } };
    const persist = async (v, expected) => {
      if (!localReady || !await local.put(v, expected)) { if (!server) problem("memory"); return false; }
      return true;
    };
    async function api(path = "", options = {}) {
      if (!request) throw new Error("Local server unavailable");
      const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = controller ? setTimeout(() => controller.abort(), 15000) : null;
      try {
        const r = await request(endpoint + path, {credentials: "same-origin", cache: "no-store", ...options,
          signal: controller ? controller.signal : undefined, headers: {"Content-Type": "application/json", ...(options.headers || {})}});
        const data = await r.json();
        if (!r.ok) { const e = new Error(data.error || "Local chat request failed"); e.status = r.status; e.data = data; throw e; }
        return data;
      } finally { if (timer !== null) clearTimeout(timer); }
    }
    async function remoteAll() {
      const chats = []; let after = "";
      for (let page = 0; page < 10001; page++) {
        const data = await api(after ? "?after=" + encodeURIComponent(after) : "");
        if (!Array.isArray(data.chats)) throw new Error("Invalid history response");
        chats.push(...data.chats);
        if (!data.next) return chats;
        if (data.next === after || typeof data.next !== "string") throw new Error("Invalid history cursor");
        after = data.next;
      }
      throw new Error("History page limit exceeded");
    }
    function accept(chats) {
      memory.clear();
      for (const c of chats) { memory.set(c.id, clone(c)); if (Number.isInteger(c._revision)) revisions.set(c.id, c._revision); }
    }
    async function rejectedDelete(record, error) {
      blocked.add(record.id); problem("conflict");
      const data = error.data || {}, restored = data.current ? {...data.current, _pending: false}
        : {id: record.id, _revision: data.revision || 0, _deleted: true, _pending: false};
      revisions.set(record.id, restored._revision);
      if (restored._deleted) memory.delete(record.id); else memory.set(record.id, clone(restored));
      await persist(restored, record);
    }
    async function importPending(records) {
      const candidates = records.filter((c) => !c._deleted && (!Number.isInteger(c._revision) || c._pending));
      if (candidates.length) {
        const receipt = await api("/import", {method: "POST", body: JSON.stringify({source: "indexeddb-strata-v1", chats: candidates.map(clean)})});
        if (receipt.confirmed !== true || typeof receipt.backup_id !== "string" || !Array.isArray(receipt.mappings)) throw new Error("Migration was not confirmed");
        const rows = await remoteAll(), byId = new Map(rows.map((c) => [c.id, c]));
        for (const mapping of receipt.mappings) {
          const original = candidates.find((c) => c.id === mapping.from), saved = byId.get(mapping.id);
          if (!original) continue;
          const replacement = mapping.deleted ? {id: original.id, _revision: mapping.revision, _deleted: true, _pending: false} : saved && {...saved, _pending: false};
          if (!replacement) throw new Error("Imported chat was not returned by server");
          if (mapping.id !== original.id) { await persist(replacement); await local.remove(original.id, original); }
          else await persist(replacement, original);
        }
      }
      for (const c of records.filter((c) => c._deleted && c._pending)) {
        try {
          const data = await api("/" + encodeURIComponent(c.id), {method: "DELETE", body: JSON.stringify({expected_revision: c._revision || 0})});
          await persist({...c, _revision: data.revision, _pending: false}, c);
        } catch (e) { if (e.status === 409) await rejectedDelete(c, e); else throw e; }
      }
    }
    async function start() {
      if (started) return started;
      started = (async () => {
        localReady = !!await local.start();
        const records = localReady ? await local.all() : []; accept(records.filter((c) => !c._deleted));
        let remote = [];
        try {
          remote = await remoteAll(); server = true; await importPending(records);
          remote = await remoteAll(); accept(remote);
          const remaining = localReady ? await local.all() : [];
          for (const c of remaining) if (c._pending || !Number.isInteger(c._revision)) {
            if (c._deleted) memory.delete(c.id); else memory.set(c.id, clone(c));
          }
        } catch (e) {
          for (const c of remote) if (!memory.has(c.id)) { memory.set(c.id, clone(c)); revisions.set(c.id, c._revision); }
          if (e.status === 409) problem("conflict");
          else if (e.status === 413 || e.status === 507) problem("quota");
          else { server = false; problem(localReady ? "server" : "memory"); }
        }
        return server || localReady;
      })(); return started;
    }
    function enqueue(work) {
      const next = queue.then(async () => { await start(); return work(); });
      queue = next.catch(() => { problem("failed"); return false; }); return queue;
    }
    function put(value) {
      let saved;
      try { saved = {...clean(value), _revision: revisions.get(value.id) || value._revision || 0, _pending: true, _local_token: token()}; }
      catch (_) { problem("failed"); return Promise.resolve(false); }
      return enqueue(async () => {
        memory.set(saved.id, clone(saved)); await persist(saved);
        if (!server || blocked.has(saved.id)) return false;
        try {
          const data = await api("/" + encodeURIComponent(saved.id), {method: "PUT", body: JSON.stringify({chat: clean(saved), expected_revision: revisions.get(saved.id) || saved._revision || 0})});
          const synced = {...data.chat, _pending: false}; revisions.set(saved.id, data.chat._revision);
          memory.set(saved.id, clone(synced)); await persist(synced, saved); return true;
        } catch (e) {
          if (e.status === 409) { blocked.add(saved.id); problem("conflict"); }
          else if (e.status === 413 || e.status === 507) problem("quota");
          else { server = false; problem("server"); }
          return false;
        }
      });
    }
    function remove(id) {
      return enqueue(async () => {
        const c = {id, _revision: revisions.get(id) || 0, _pending: true, _deleted: true, _local_token: token()};
        memory.delete(id); await persist(c);
        if (!server || blocked.has(id)) return false;
        try {
          const data = await api("/" + encodeURIComponent(id), {method: "DELETE", body: JSON.stringify({expected_revision: c._revision})});
          revisions.set(id, data.revision); await persist({...c, _revision: data.revision, _pending: false}, c); return true;
        } catch (e) { if (e.status === 409) await rejectedDelete(c, e); else { server = false; problem("server"); } return false; }
      });
    }
    return {start, put, remove, all: async () => { await start(); await queue; return [...memory.values()].map(clone); },
      flush: () => queue, get usable() { return server || localReady; }, get storage() { return server ? "sqlite" : localReady ? "browser" : "memory"; }};
  }
  const api = {createChatStore}; root.StrataChatStore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
