// Day-0 scaffold checks (owner: T0). Cheap guards every track benefits from:
// the extension entry's handshake shape, every static module parses as ESM,
// static/ holds only the asset types DisPatch's /ext/ route serves, the
// locale bundles share en's key set, and the shared fixtures load.

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const STATIC = join(ROOT, 'src', 'dispatch_voice', 'static');
const FIXTURES = join(ROOT, 'tests', 'fixtures');

// v3 contracts §B.1 rule 7: anything else answers 404 under /ext/.
const SERVED = new Set(['js', 'mjs', 'css', 'json', 'map', 'wasm', 'onnx', 'wav', 'mp3', 'ogg',
  'png', 'jpg', 'webp', 'woff2']);

function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

test('index.js: the entry exports EXT {id: voice, api: [1, 1]} and mount()', async () => {
  const mod = await import(pathToFileURL(join(STATIC, 'index.js')).href);
  assert.equal(mod.EXT.id, 'voice');
  assert.deepEqual(mod.EXT.api, [1, 1]);
  assert.equal(typeof mod.EXT.version, 'string');
  assert.equal(typeof mod.mount, 'function');
});

test('static/: every module parses as an ES module', () => {
  const files = walk(STATIC).filter((p) => p.endsWith('.js'));
  assert.ok(files.length >= 12, `expected the moved + new modules, got ${files.length}`);
  for (const f of files) {
    // package.json says "type": "module", so --check parses .js as ESM.
    execFileSync(process.execPath, ['--check', f], { stdio: 'pipe' });
  }
});

test('static/: only asset types /ext/ serves (no html/svg/xml; the Voices page is in standalone/)', () => {
  const bad = walk(STATIC)
    .filter((p) => !p.includes(`${'__pycache__'}`))
    .filter((p) => !SERVED.has(extname(p).slice(1).toLowerCase()))
    .map((p) => relative(STATIC, p));
  assert.deepEqual(bad, []);
});

test('locales: eight bundles with $meta and the same keys as en', () => {
  const dir = join(STATIC, 'locales');
  const langs = readdirSync(dir).filter((n) => n.endsWith('.json')).map((n) => n.slice(0, -5)).sort();
  assert.deepEqual(langs, ['ar', 'de', 'en', 'es', 'fr', 'ja', 'pt', 'zh']);
  const PLURAL = new Set(['zero', 'one', 'two', 'few', 'many', 'other']);
  const keys = (o, p = '') => Object.entries(o).flatMap(([k, v]) => {
    if (k === '$meta') return [];
    const isPlural = v && typeof v === 'object' && Object.keys(v).every((c) => PLURAL.has(c));
    return v && typeof v === 'object' && !isPlural ? keys(v, `${p}${k}.`) : [`${p}${k}`];
  });
  const load = (l) => JSON.parse(readFileSync(join(dir, `${l}.json`), 'utf8'));
  const en = keys(load('en')).sort();
  assert.ok(en.length > 100, `en has ${en.length} keys`);
  for (const l of langs) {
    const b = load(l);
    assert.ok(b.$meta && b.$meta.dir, `${l}: $meta`);
    assert.deepEqual(keys(b).sort(), en, `${l}: same keys as en`);
  }
});

test('fixtures: wire3.json and frames_b1a.json load', () => {
  const wire = JSON.parse(readFileSync(join(FIXTURES, 'wire3.json'), 'utf8'));
  for (const k of ['up', 'down', 'bad_up', 'bad_down']) assert.ok(Array.isArray(wire[k]), k);
  const frames = JSON.parse(readFileSync(join(FIXTURES, 'frames_b1a.json'), 'utf8'));
  const snap = frames.sequences.snapshot_mid_stream;
  assert.equal(snap[0].type, 'stream_start');
  assert.equal(snap[1].replace, true);
});
