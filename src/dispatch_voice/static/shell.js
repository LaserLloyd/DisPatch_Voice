// DisPatch Voice: the add-on's handle on DisPatch's frozen `shell` object.
// Owner: track T7. Contract: C-§E.1 of the v3 interface contracts.
//
// Every add-on module gets DOM helpers, translations and UI services from
// here, never from DisPatch's own files: the add-on cannot import
// /static/js/*.js (those paths are DisPatch internals and change between
// releases), and the one thing DisPatch promises is the `shell` object it
// hands to `mount(shell, opts)`. index.js binds it once; the functions below
// forward to it. Before a bind (unit tests that do not care, a module
// imported on its own) they fall back to small built-ins, so nothing throws.
//
// The fallbacks are deliberately plain: `t()` returns the key with its
// {placeholders} filled, `el()` is the same three-line builder DisPatch uses.
// Tests that check wording bind a fake shell built from the English bundle.

let bound = null;

/** Bind DisPatch's shell. Returns the previous one (tests restore it). */
export function bindShell(shell) {
  const prev = bound;
  bound = shell || null;
  return prev;
}

/** The bound shell, or null. */
export function getShell() { return bound; }

const SVG_NS = 'http://www.w3.org/2000/svg';

function fallbackEl(tag, attrs = {}, children = []) {
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

function fallbackIcon(paths) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  for (const [k, v] of Object.entries({
    viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.8',
    'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true',
  })) svg.setAttribute(k, v);
  svg.classList.add('rail-icon');
  for (const d of paths || []) {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('d', d);
    svg.append(p);
  }
  return svg;
}

/** The handful of shell icons the add-on draws, with line-art fallbacks. */
const ICON_FALLBACK = Object.freeze({
  gear: ['M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    'M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z'],
  close: ['M18 6 6 18', 'M6 6l12 12'],
  forward: ['M9 5l7 7-7 7'],
  back: ['M15 5 8 12l7 7'],
  upload: ['M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4', 'M17 8l-5-5-5 5', 'M12 3v12'],
});

/** Icon paths by shell name (RAIL_ICONS), falling back to the add-on's copy. */
export function iconPaths(name) {
  const set = bound && bound.RAIL_ICONS;
  return (set && set[name]) || ICON_FALLBACK[name] || [];
}

export function el(tag, attrs, children) {
  return bound && bound.el ? bound.el(tag, attrs, children) : fallbackEl(tag, attrs, children);
}

export function railIcon(paths) {
  return bound && bound.railIcon ? bound.railIcon(paths) : fallbackIcon(paths);
}

function interpolate(s, vars) {
  if (!vars) return s;
  return String(s).replace(/\{(\w+)\}/g, (m, k) => (vars[k] == null ? m : String(vars[k])));
}

/** Translate a key (the shell's i18n, with the add-on's bundle merged in). */
export function t(key, vars) {
  const i = bound && bound.i18n;
  if (i && typeof i.t === 'function') return i.t(key, vars);
  return interpolate(key, vars);
}

/** Does the shell (with the add-on's bundle merged in) have this key? Unbound: no. */
export function has(key) {
  const i = bound && bound.i18n;
  if (i && typeof i.has === 'function') { try { return !!i.has(key); } catch { return false; } }
  return false;
}

/** "a, b and c" in the UI language. */
export function listFormat(items) {
  const i = bound && bound.i18n;
  if (i && typeof i.list === 'function') return i.list(items);
  return (items || []).filter(Boolean).map(String).join(', ');
}

/**
 * Short facts about one thing, "Your own voice, added 3 Oct, 12 s" — a unit list
 * in the UI language, never a middle-dot meta string (SPEC §7.5).
 */
export function metaList(items) {
  const parts = (items || []).filter(Boolean).map(String);
  try {
    return new Intl.ListFormat(language(), { style: 'short', type: 'unit' }).format(parts);
  } catch { return parts.join(', '); }
}

/** The UI language code ("en", "de", …). */
export function language() {
  if (bound && typeof bound.language === 'function') return bound.language() || 'en';
  const i = bound && bound.i18n;
  if (i && typeof i.lang === 'function') return i.lang() || 'en';
  if (i && typeof i.lang === 'string') return i.lang || 'en';
  return (globalThis.document && document.documentElement && document.documentElement.lang) || 'en';
}

export function acquireInert() { return bound && bound.acquireInert ? bound.acquireInert() : null; }
export function releaseInert(token) { if (bound && bound.releaseInert) bound.releaseInert(token); }
export function isMobile() {
  if (bound && typeof bound.isMobile === 'function') return !!bound.isMobile();
  return !!(globalThis.matchMedia && globalThis.matchMedia('(max-width: 768px)').matches);
}

export function toast(text, isError = false) {
  if (bound && bound.toast) bound.toast(text, isError);
}
export function confirmDialog(text, opts) {
  if (bound && bound.confirm) return Promise.resolve(bound.confirm(text, opts));
  return Promise.resolve(!!(globalThis.confirm && globalThis.confirm(text)));
}
export function promptDialog(text, def) {
  if (bound && bound.prompt) return Promise.resolve(bound.prompt(text, def));
  return Promise.resolve(globalThis.prompt ? globalThis.prompt(text, def) : null);
}

/**
 * Tell DisPatch the person is using the app (DisPatch ≥ 2.2.0, `shell.activity`):
 * its idle auto-lock is re-armed as if they had touched the screen. A voice chat
 * makes no touches, so without this the shell locked itself mid-conversation
 * (FR-works 1). Feature-detected: a shell without the member ignores it.
 */
export function activity() {
  if (bound && typeof bound.activity === 'function') {
    try { bound.activity(); } catch { /* a shell that refuses is not fatal */ }
  }
}

/** Is the shell locked (Safe Mode / decoy)? Unbound = treated as locked. */
export function isLocked() {
  if (!bound || typeof bound.isLocked !== 'function') return true;
  return !!bound.isLocked();
}
