// Test doubles for the client app (owner: track T7).
//
// fakeShell(): the frozen `shell` object of C-§E.1 as DisPatch's loader
// hands it to mount(), with every slot recording what the add-on did, an
// i18n backed by the add-on's own English bundle, and the pane-ownership
// rule of A16 (settings.pane replaces the absent page; dispose re-applies it).
//
// withDom(): a jsdom window installed as the globals the client modules use
// (document, localStorage, fetch, navigator, …), restored afterwards.
// fakeFetch(): a router of JSON answers that records every request.

import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const HERE = dirname(fileURLToPath(import.meta.url));
export const STATIC = join(HERE, '..', '..', 'src', 'dispatch_voice', 'static');
export const LOCALES = join(STATIC, 'locales');

let jsdom = null;
try { jsdom = require('jsdom'); } catch { jsdom = null; }
export const HAS_DOM = !!jsdom;

export function loadBundle(lang = 'en') {
  return JSON.parse(readFileSync(join(LOCALES, `${lang}.json`), 'utf8'));
}

/** t() over a bundle: dotted keys, {name} interpolation, the key itself when missing. */
export function makeT(bundle, missing = []) {
  return (key, vars) => {
    const [ns, k] = String(key).split(/\.(.*)/s);
    let raw = bundle[ns] && bundle[ns][k];
    if (raw == null) { missing.push(key); raw = key; }
    return String(raw).replace(/\{(\w+)\}/g, (m, n) => (vars && vars[n] != null ? String(vars[n]) : m));
  };
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function railIcon(paths) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'rail-icon');
  svg.dataset.paths = String((paths || []).length);
  return svg;
}

/**
 * The shell. `opts`: { lang, locked, thread, bots, roster, extApi, perms }.
 * Returns { shell, log, emitLocked(), setLocked(v), absent, paneHost, header }.
 */
export function fakeShell(opts = {}) {
  const lang = opts.lang || 'en';
  const missing = [];
  const t = makeT(loadBundle(lang === 'en' ? 'en' : lang), missing);
  const log = { toasts: [], panes: [], rowExtras: [], tools: [], openPanes: [], settingsOpened: [], history: [], missing, actions: [] };
  let locked = !!opts.locked;
  const lockedFns = new Set();
  const threadFns = new Set();
  let thread = opts.thread || null;
  const bots = opts.bots || [];

  // The chat header with the thread menu button the slot inserts before.
  const header = el('div', { class: 'chat-header' }, [el('button', { id: 'thread-menu-btn', type: 'button' })]);
  document.body.append(header);
  // Settings → Voice: DisPatch's own absent page lives here until the add-on claims the pane.
  const paneHost = el('section', { class: 'settings-pane', id: 'spane-voice' });
  const absent = () => paneHost.replaceChildren(el('div', { class: 'voice-absent', text: 'absent page' }));
  absent();
  document.body.append(paneHost);
  const toolHost = el('div', { class: 'terminal-view', id: 'ext-tool-view', hidden: '' });
  document.body.append(toolHost);

  const shell = Object.freeze({
    EXT_API: opts.extApi == null ? 1 : opts.extApi,
    version: '2.2.0',
    el, railIcon,
    RAIL_ICONS: Object.freeze({ gear: ['M1 1'], close: ['M2 2'], forward: ['M3 3'], back: ['M4 4'], upload: ['M5 5'] }),
    loadStyle: () => Promise.resolve(),
    acquireInert: () => 'tok',
    releaseInert: () => {},
    isMobile: () => false,
    i18n: Object.freeze({
      t, tHtml: t, has: () => true, applyDom: () => {},
      list: (items) => (items || []).join(', '), n: (x) => String(x), lang: () => lang,
    }),
    onLocale: () => () => {},
    api: async () => ({}),
    onLocked(fn) { lockedFns.add(fn); return () => lockedFns.delete(fn); },
    onThread(fn) { threadFns.add(fn); return () => threadFns.delete(fn); },
    activeThread: () => thread,
    bots: () => bots,
    botById: (id) => bots.find((b) => b.id === id) || null,
    allBots: async () => opts.roster || bots,
    isLocked: () => locked,
    // DisPatch 3.0's shell.can(permission); absent unless the test passes `perms` (a 2.x shell).
    ...(opts.perms ? { can: (perm) => opts.perms.includes(perm) } : {}),
    language: () => lang,
    features: () => ({ extensions: 1 }),
    threadFace: (th, bot) => ({ src: '', letter: bot && bot.name ? bot.name[0] : '?' }),
    toast: (text, isError = false) => log.toasts.push({ text, isError }),
    confirm: async () => true,
    prompt: async (text, def) => def,
    navigate: () => {},
    chatHeader: Object.freeze({
      addAction(node, { before } = {}) {
        log.actions.push({ node, before });
        const ref = before ? header.querySelector(`#${before}`) : null;
        header.insertBefore(node, ref);
        return () => node.remove();
      },
    }),
    settings: Object.freeze({
      pane(id, spec) {
        const entry = { id, spec, disposed: false };
        log.panes.push(entry);
        paneHost.replaceChildren();          // A16: the add-on's pane replaces the absent page
        spec.render(paneHost);
        return () => { entry.disposed = true; absent(); };   // A16: dispose re-applies it
      },
      open(id) { log.settingsOpened.push(id); },
    }),
    bots: Object.freeze({
      rowExtra(fn) {
        const entry = { fn, disposed: false };
        log.rowExtras.push(entry);
        return () => { entry.disposed = true; };
      },
    }),
    tools: Object.freeze({
      register(tiles) {
        const entry = { tiles, disposed: false };
        log.tools.push(entry);
        return () => { entry.disposed = true; };
      },
      openPane(spec) {
        const entry = { spec, closed: false };
        log.openPanes.push(entry);
        toolHost.hidden = false;
        toolHost.replaceChildren();
        spec.render(toolHost);
        return () => { entry.closed = true; toolHost.replaceChildren(); toolHost.hidden = true; };
      },
    }),
    history: Object.freeze({
      push(tag, onBack) { const e = { tag, onBack, popped: false }; log.history.push(e); return () => { e.popped = true; }; },
    }),
  });
  return {
    shell, log, header, paneHost, toolHost,
    emitLocked() { locked = true; for (const fn of [...lockedFns]) fn(); },
    setLocked(v) { locked = !!v; },
    setThread(th) { thread = th; for (const fn of [...threadFns]) fn(th); },
    get lockedListeners() { return lockedFns.size; },
  };
}

/** A JSON router for fetch: routes = { 'GET /api/voice/status': body | (req) => body | {status, body} }. */
export function fakeFetch(routes = {}) {
  const calls = [];
  const fn = async (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    const u = new URL(String(url), 'http://localhost/');
    const path = u.pathname;
    let body = opts.body;
    if (typeof body === 'string') { try { body = JSON.parse(body); } catch { /* raw */ } }
    calls.push({ method, path, search: u.search, body });
    const key = `${method} ${path}`;
    let r = Object.prototype.hasOwnProperty.call(routes, key) ? routes[key] : undefined;
    if (typeof r === 'function') r = await r({ method, path, search: u.search, body });
    if (r === undefined) return new Response(JSON.stringify({ detail: 'not found' }), { status: 404, headers: { 'Content-Type': 'application/json' } });
    if (r && typeof r === 'object' && r.__status) {
      return new Response(JSON.stringify(r.body ?? {}), { status: r.__status, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response(JSON.stringify(r), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  fn.calls = calls;
  return fn;
}

const GLOBALS = ['document', 'window', 'navigator', 'localStorage', 'fetch', 'MutationObserver', 'HTMLElement',
  'Node', 'Event', 'KeyboardEvent', 'CustomEvent', 'location', 'history', 'isSecureContext', 'matchMedia',
  'requestAnimationFrame', 'cancelAnimationFrame', 'speechSynthesis', 'SpeechSynthesisUtterance', 'AudioContext',
  'WebSocket', 'Audio', 'MediaMetadata', 'AudioWorkletNode', 'addEventListener', 'removeEventListener', 'Blob', 'URL'];

/**
 * Run `fn(win)` with a jsdom window as the globals. `extra` are additional
 * globals to install (fake audio/socket classes, a fake fetch, …).
 */
export async function withDom(fn, extra = {}) {
  if (!jsdom) throw new Error('jsdom missing');
  const dom = new jsdom.JSDOM('<!doctype html><html lang="en"><body></body></html>', { url: 'https://localhost/', pretendToBeVisual: true });
  const win = dom.window;
  const saved = {};
  // Save every global this replaces — node's own (performance, crypto…) come back intact.
  for (const k of GLOBALS.concat(Object.keys(extra))) saved[k] = Object.getOwnPropertyDescriptor(globalThis, k);
  const set = (k, v) => Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  set('window', win);
  set('document', win.document);
  set('navigator', win.navigator);
  set('localStorage', win.localStorage);
  set('MutationObserver', win.MutationObserver);
  set('HTMLElement', win.HTMLElement);
  set('Node', win.Node);
  set('Event', win.Event);
  set('KeyboardEvent', win.KeyboardEvent);
  set('CustomEvent', win.CustomEvent);
  set('location', win.location);
  set('history', win.history);
  set('isSecureContext', true);
  set('matchMedia', () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  set('requestAnimationFrame', () => 0);
  set('cancelAnimationFrame', () => {});
  set('addEventListener', win.addEventListener.bind(win));
  set('removeEventListener', win.removeEventListener.bind(win));
  set('fetch', extra.fetch || fakeFetch());
  for (const [k, v] of Object.entries(extra)) if (k !== 'fetch') set(k, v);
  try {
    return await fn(win);
  } finally {
    for (const k of GLOBALS.concat(Object.keys(extra))) {
      if (saved[k]) Object.defineProperty(globalThis, k, saved[k]);
      else delete globalThis[k];
    }
    win.close();
  }
}

/** Let pending promise callbacks run (fetch answers, async renders). */
export async function settle(n = 5) {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
}
