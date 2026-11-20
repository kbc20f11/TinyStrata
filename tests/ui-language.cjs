'use strict';
// Exercise the shipping catalog loader and language control together, including
// failed/reordered requests and reloads, without a browser or a model dependency.
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.join(__dirname, '..');
const i18n = fs.readFileSync(path.join(root, 'ui/web/i18n.js'), 'utf8');
const app = fs.readFileSync(path.join(root, 'ui/web/app.js'), 'utf8');
const controls = app.slice(app.indexOf('const urlLang ='), app.indexOf('// ------------------------------------------------------------------ start'));
const catalogs = Object.fromEntries(['en','ja','zh-CN','de','fr','es','pt-BR'].map(code =>
  [code, JSON.parse(fs.readFileSync(path.join(root, `ui/web/locales/${code}.json`), 'utf8'))]));
let cases = 0;
function element(key, english, mark = 'data-i18n') {
  return {dataset: {}, textContent: english, attributes: {[mark]: key},
    getAttribute(name) { return this.attributes[name] ?? null; },
    setAttribute(name, value) { this.attributes[name] = value; }};
}
function fixture({saved = 'ja', url = 'http://127.0.0.1:3000/#about', writeFails = false} = {}) {
  const location = new URL(url), storage = new Map([['lang', saved], ['themeMode', 'dark']]);
  const nodes = {lang: {value: '', innerHTML: ''}};
  const heading = element('tab.about', 'About');
  const input = element('composer.placeholder', '', 'data-i18n-placeholder');
  input.attributes.placeholder = 'Talk to TinyStrata';
  const requests = [], notices = [], writes = [];
  let fetcher = async code => ({ok: true, json: async () => catalogs[code]});
  const ctx = {URL, URLSearchParams, Intl, console, location,
    document: {documentElement: {lang: 'en'}, querySelectorAll(selector) {
      return [heading, input].filter(node => selector === `[${Object.keys(node.attributes)[0]}]`);
    }},
    fetch: async (url, options) => {
      const code = url.match(/locales\/([^/]+)\.json/)[1];
      requests.push({code, cache: options.cache});
      return fetcher(code);
    },
    history: {replaceState(_state, _unused, value) { location.href = new URL(value, location).href; }},
    store: {get: (key, fallback) => storage.get(key) ?? fallback,
      set(key, value) { if (writeFails) return false; storage.set(key, value); writes.push({key, value}); return true; }},
    $: id => nodes[id], esc: value => String(value), toast: (...args) => notices.push(args),
    sidebarOpen: true, busy: false, lastMetrics: null};
  for (const name of ['renderEmptyChat','applyTheme','renderMcp','setSidebar','syncThinkingControls',
    'outputs','renderChat','setBusy','renderModelDownload','renderModelDelete','renderLocalToolStatus']) ctx[name] = () => {};
  vm.createContext(ctx);
  vm.runInContext(i18n + '\n' + controls, ctx);
  return {ctx, storage, nodes, heading, input, requests, notices, writes, location,
    fetchWith(fn) { fetcher = fn; },
    start: () => vm.runInContext('changeLanguage(urlLang || store.get("lang", "ja"))', ctx),
    async choose(code) { nodes.lang.value = code; await nodes.lang.onchange(); }};
}
async function test(name, run) { await run(); cases++; console.log(`PASS ${name}`); }

(async () => {
  await test('all languages, translated attributes and English restoration', async () => {
    const f = fixture(); await f.start();
    for (const code of ['de','fr','es','pt-BR','zh-CN','ja','en']) {
      await f.choose(code);
      assert.equal(f.ctx.StrataI18n.lang, code);
      assert.equal(f.nodes.lang.value, code);
      assert.equal(f.ctx.document.documentElement.lang, code);
      assert.equal(f.heading.textContent, catalogs[code]['tab.about']);
      assert.equal(f.input.getAttribute('placeholder'), catalogs[code]['composer.placeholder']);
      assert.equal(f.storage.get('lang'), code);
      const reload = fixture({saved: f.storage.get('lang'), url: f.location.href});
      await reload.start(); assert.equal(reload.ctx.StrataI18n.lang, code);
    }
    assert.equal(f.storage.get('themeMode'), 'dark');
    assert.ok(f.requests.every(request => request.cache === 'no-store'));
  });
  await test('explicit choice supersedes a URL preview without losing other URL parts', async () => {
    const f = fixture({url: 'http://127.0.0.1:3000/?lang=ja&other=kept#about', saved: 'de'});
    await f.start(); assert.equal(f.ctx.StrataI18n.lang, 'ja');
    assert.equal(f.storage.get('lang'), 'de', 'URL previews do not overwrite the saved choice');
    await f.choose('en');
    assert.equal(f.location.search, '?other=kept'); assert.equal(f.location.hash, '#about');
    const reload = fixture({saved: f.storage.get('lang'), url: f.location.href});
    await reload.start(); assert.equal(reload.ctx.StrataI18n.lang, 'en');
  });
  for (const failure of ['http', 'network', 'json', 'invalid-catalog']) {
    await test(`${failure} failure preserves the display, selection and saved preference; retry succeeds`, async () => {
      const f = fixture({url: 'http://127.0.0.1:3000/?lang=ja#about'}); await f.start();
      const before = f.heading.textContent;
      f.fetchWith(async () => {
        if (failure === 'network') throw Error('disconnected');
        return {ok: failure !== 'http', json: async () => {
          if (failure === 'json') throw Error('invalid JSON');
          return [];
        }};
      });
      await f.choose('fr');
      assert.equal(f.ctx.StrataI18n.lang, 'ja'); assert.equal(f.nodes.lang.value, 'ja');
      assert.equal(f.heading.textContent, before); assert.equal(f.storage.get('lang'), 'ja');
      assert.equal(f.location.search, '?lang=ja');
      assert.equal(f.notices.length, 1); assert.equal(f.notices[0][0], 'error');
      assert.equal(f.notices[0][1], catalogs.ja['settings.languageFailed']);
      f.fetchWith(async code => ({ok: true, json: async () => catalogs[code]}));
      await f.choose('fr'); assert.equal(f.ctx.StrataI18n.lang, 'fr');
      assert.equal(f.storage.get('lang'), 'fr'); assert.equal(f.location.search, '');
    });
  }
  await test('failed startup remains usable and preserves the preferred language for retry', async () => {
    const f = fixture({saved: 'de'});
    f.fetchWith(async () => { throw Error('offline'); });
    assert.equal(await f.start(), false);
    assert.equal(f.ctx.StrataI18n.lang, 'en'); assert.equal(f.nodes.lang.value, 'en');
    assert.equal(f.storage.get('lang'), 'de'); assert.equal(f.notices.length, 1);
    f.fetchWith(async code => ({ok: true, json: async () => catalogs[code]}));
    await f.choose('de'); assert.equal(f.ctx.StrataI18n.lang, 'de');
  });
  for (const staleFails of [false, true]) {
    await test(`last choice wins when an older request ${staleFails ? 'fails' : 'finishes'} later`, async () => {
      const f = fixture(); await f.start();
      let release;
      f.fetchWith(code => code === 'de' ? new Promise(resolve => { release = resolve; }) :
        Promise.resolve({ok: true, json: async () => catalogs[code]}));
      const pending = f.choose('de');
      await f.choose('fr');
      release({ok: !staleFails, json: async () => catalogs.de});
      await pending;
      assert.equal(f.ctx.StrataI18n.lang, 'fr'); assert.equal(f.nodes.lang.value, 'fr');
      assert.equal(f.storage.get('lang'), 'fr'); assert.equal(f.notices.length, 0);
      assert.deepEqual(f.writes.map(write => write.value), ['fr']);
    });
  }
  await test('unwritable storage keeps this page usable and explains the persistence limit', async () => {
    const f = fixture({writeFails: true}); await f.start(); await f.choose('fr');
    assert.equal(f.ctx.StrataI18n.lang, 'fr'); assert.equal(f.nodes.lang.value, 'fr');
    assert.equal(f.storage.get('lang'), 'ja');
    assert.equal(f.notices[0][0], 'warn');
    assert.equal(f.notices[0][1], catalogs.fr['settings.languageUnsaved']);
  });
  await test('new notices exist with matching placeholders in every catalog', async () => {
    for (const key of ['settings.languageFailed','settings.languageRetry','settings.languageUnsaved']) {
      for (const [code, catalog] of Object.entries(catalogs)) {
        assert.ok(typeof catalog[key] === 'string' && catalog[key].length, `${code}: ${key}`);
        assert.deepEqual(catalog[key].match(/\{\w+\}/g), catalogs.en[key].match(/\{\w+\}/g));
      }
    }
  });
  console.log(`PASS ${cases} language behavior cases`);
})().catch(error => { console.error(error); process.exitCode = 1; });
