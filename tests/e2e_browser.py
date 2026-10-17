#!/usr/bin/env python3
"""Headless Chromium end-to-end of the voice chat client on scratch instances (owner: T7).

The REAL client — the add-on's static modules mounted by the REAL DisPatch shell — in
headless Chromium (Playwright's own Chromium, or ``--chrome PATH`` / ``$DISPATCH_E2E_CHROME``), with a controllable fake
microphone (clips from ``tests/fixtures/clips`` played into a Web Audio stream that replaces
``getUserMedia``) and a fake ``speechSynthesis`` (headless Chromium ships no voices). Two
targets (SPEC §12, the "Headless Chromium e2e" row):

``--target dispatch``  a scratch DisPatch with the add-on and one API bot on the stub LLM;
                       real speech models (Parakeet, Smart Turn, the TTS). Proves the whole
                       loop as the server is TODAY: speak → transcript → the reply plays
                       through the player worklet with captions, tap-to-stop, Pause, Esc, plus
                       Settings › Voice (19 rows / 6 groups + Status, Connection), the Bots
                       popover and the Voice lab at 390 and 1366 px. v3-only assertions run
                       when the server says ``ready.proto >= 3`` and are reported as SKIP
                       (with the reason) while it speaks v2. On a box where other jobs load
                       models too, take a shared lock (any lock file, e.g. ``$VOICE_LOCK``):
                           flock "$VOICE_LOCK" uv run --offline --with playwright \\
                               python tests/e2e_browser.py --target dispatch
``--target ref``       the v3 wire: the page is the real DisPatch shell (every HTTP request
                       fulfilled from the scratch DisPatch by a Playwright route) opened on
                       the port of T6's shaper (``tests/shaper.py``), so ONLY the voice
                       WebSocket goes through the shaped link — to T6's reference v3 server
                       (``tests/transport/refserver.py``: the production transport modules,
                       an energy detector, fake STT, tone TTS) extended here with the server
                       half of SPEC §4.6 (``duck`` on speech while speaking, ``duck off`` +
                       ``resume {rewind}`` after a short sound, ``barge_in`` after 0.7 s of
                       voice or ``stop_playback``) and a log of every ``heard``/``link``/
                       ``duck_local`` frame it received. No speech models are loaded.
                       Asserts the §12 rows: the optimistic duck within 200 ms of an injected
                       onset (the player worklet's ``pause``), resume after a short cough,
                       "Wait, stop" → ``barge_in`` ≤ 1.0 s with a ``heard`` frame, captions
                       only for what played, on 3g-weak (PCM down) the fake synth speaks every
                       ``say`` once in order with ``link {mode: "text"}`` at the server and the
                       notice "using the phone's voice", on edge the first sentence ≤ 6 s
                       after ``eot``, and a blackout → offline → waiting → listening with
                       exactly one transcript for speech made while offline.

More scenarios (``--scenarios``; V2 verification):
``keys``      Space = the microphone (also with the focus on the picture while a reply plays:
              never the tap's interrupt), Tab trapped in the screen and in the quick sheet,
              Esc closes the sheet first and then hangs up, focus back to the mic button;
              All settings… opens Settings over the screen (mic paused, socket kept) and Esc
              closes only Settings.
``drop``      the socket drops mid-reply and the signal stays away > 60 s: offline → waiting
              → resume with ``after_partial``; no fully received sentence is spoken again,
              every sentence reaches the listener once in order (on the dispatch target: every
              sentence of the stored reply), no caption twice.
``cdp``       (both targets) a CDP-throttled link (``Network.emulateNetworkConditions``, 120 kbps / 450 ms,
              direct to the reference server): measured to reach the WebSocket, then the
              phone-voice failover and — when CDP lets go — back to the server's voice
              (dwell, healthy spell, probe); on the dispatch target the same against the real
              session. ``failback`` does the same through the shaper (3g-weak, then the live
              link turns healthy under the same socket).
``shots``     light (Daylight) and dark (Glacier) at 390 and 1366 px: Settings › Voice next
              to Settings › Device with their computed group/row/title/hint styles compared,
              the voice chat screen listening and speaking, the quick sheet; a literal-colour
              scan of ``voice.css`` (no literal colour, no radius DisPatch's CSS does not use,
              no font that is not a ``--font-*`` token).
``locales``   de, ar (RTL) and ja at 390 px: Settings › Voice and the quick sheet with no sideways
              scroll and no untranslated key, the English-only first-open line.
``undo``      (dispatch target) Undo by the button and by the U key on the real session:
              ``undo`` → ``undone``, the user's row leaves the chat, no reply is spoken or posted.
``bargein``   (dispatch target) talking over a reply on the real session: the optimistic duck
              ≤ 200 ms, the server's ``barge_in``, flush to the new gen, ``heard``, the
              interruption transcribed, no further caption from the cut reply.
``noise``     (dispatch target, 1.5.0) the default "normal" interrupt sensitivity on the real
              session: after two ordinary turns (the session learns the listener's level), a
              kitchen clatter and then a talker across the room (the ``medium`` clip 22 dB down)
              over a long reply neither pause nor cut it; the listener's own "Wait, stop…" at
              full level still cuts it with ``barge_in``.
``livelike``  (dispatch target) its own scratch DisPatch set up like the live box: the add-on
              by its entry point, voice on by ``voice.yaml``, the Kokoro plugin overlaid when
              it is installed (``$DISPATCH_VOICE_KOKORO_SRC``, default the sibling
              ``dispatch-voice-kokoro``), a PIN and a 60 s idle lock. A page nobody touches
              locks (the control); a voice chat left listening for 95 s is still open and
              unlocked; the next question is answered aloud; with a built-in default voice
              the acknowledgements are word lines, rendered for that voice.

Results: ``<out>/e2e_browser.json`` (every check with its detail) and screenshots in
``--shots`` (default ``.e2e-out/browser/shots`` in the repository, git-ignored). Exit 1 when a
check failed.
Ports: ``--port-base`` (default 19075): +0 DisPatch, +1 stub LLM, +2 reference server,
+3 shaper. Never the live app.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADDON = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "transport"))
sys.path.insert(0, str(ADDON / "src"))

#: The browser: Playwright's own Chromium unless ``--chrome`` / ``$DISPATCH_E2E_CHROME`` names one.
CHROME = os.environ.get("DISPATCH_E2E_CHROME", "").strip() or None
CLIPS = HERE / "fixtures" / "clips"
DEFAULT_BACKEND = Path.home() / "Projects/dispatch-chat/backend"
SPEECH_MODELS = Path.home() / ".local/share/dispatch-voice-models"
DEFAULT_OUT = ADDON / ".e2e-out" / "browser"
DEFAULT_SHOTS = DEFAULT_OUT / "shots"
EN = json.loads((ADDON / "src/dispatch_voice/static/locales/en.json").read_text())
# A light and a dark palette of DisPatch's theme package (the dark one is the app's default).
THEMES = {"dark": "glacier", "light": "daylight"}
CHROME_ARGS = [
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    # Chrome 151 Local Network Access: a page whose responses Playwright fulfilled has no
    # local address space, so its WebSocket to 127.0.0.1 is blocked (measured in the design prototype).
    "--disable-features=LocalNetworkAccessChecks",
]


# --------------------------------------------------------------------------- #
# the page instrumentation (one init script)
# --------------------------------------------------------------------------- #
INIT_JS = r"""
(() => {
  const now = () => performance.now();
  const E = window.__e2e = { ws: [], port: [], captions: [], states: [], urls: [], notices: [],
    mic: { injects: [], gum: 0 }, tts: { spoken: [], cancels: 0 } };
  const cap = (arr, n = 20000) => { if (arr.length > n) arr.splice(0, arr.length - n); };
  const b8 = (d) => { try { return d instanceof ArrayBuffer ? new Uint8Array(d) : new Uint8Array(d.buffer, d.byteOffset, d.byteLength); } catch { return new Uint8Array(0); } };
  // v3 downlink header: 0x03 | codec | kind | gen u16 | seq u32 | t u32 | count
  const down = (u) => (u.length >= 9 && u[0] === 3) ? { codec: u[1], kind: u[2], gen: u[3] | (u[4] << 8), seq: (u[5] | (u[6] << 8) | (u[7] << 16) | (u[8] << 24)) >>> 0 } : null;
  const up = (u) => (u.length >= 8 && u[0] === 3) ? { codec: u[1], flags: u[2], off: (u[3] | (u[4] << 8) | (u[5] << 16) | (u[6] << 24)) >>> 0 } : null;

  // ---- every voice-socket frame, both ways ----------------------------------
  const Orig = window.WebSocket;
  function W(url, protos) {
    const ws = protos === undefined ? new Orig(url) : new Orig(url, protos);
    const voice = /\/ws\/voice\//.test(String(url));
    if (!voice) return ws;
    E.urls.push(String(url));
    const sock = E.urls.length - 1;
    const send = ws.send.bind(ws);
    ws.send = (d) => {
      if (typeof d === 'string') { try { E.ws.push({ t: now(), dir: 'out', sock, m: JSON.parse(d) }); } catch {} }
      else { const u = b8(d); E.ws.push({ t: now(), dir: 'out', sock, bin: u.length, up: up(u), head: Array.from(u.slice(0, 8)) }); }
      cap(E.ws);
      return send(d);
    };
    ws.addEventListener('message', (e) => {
      if (typeof e.data === 'string') { try { E.ws.push({ t: now(), dir: 'in', sock, m: JSON.parse(e.data) }); } catch {} }
      else { const u = b8(e.data); E.ws.push({ t: now(), dir: 'in', sock, bin: u.length, down: down(u) }); }
      cap(E.ws);
    });
    ws.addEventListener('close', (e) => E.ws.push({ t: now(), dir: 'close', sock, code: e.code }));
    return ws;
  }
  W.prototype = Orig.prototype;
  W.CONNECTING = 0; W.OPEN = 1; W.CLOSING = 2; W.CLOSED = 3;
  window.WebSocket = W;

  // ---- what the main thread tells the worklets (player push/pause/resume/flush …) ----
  const pm = MessagePort.prototype.postMessage;
  MessagePort.prototype.postMessage = function (m, tr) {
    try {
      if (m && typeof m === 'object') {
        if (typeof m.type === 'string') {
          const e = { t: now(), type: m.type };
          for (const k of ['gen', 'seq', 'kind', 'rewind', 'ms', 'value']) if (k in m) e[k] = m[k];
          E.port.push(e); cap(E.port);
        } else if ('enabled' in m || 'gate' in m || 'hold' in m || 'localSpeaking' in m) {
          E.port.push({ t: now(), mic: JSON.parse(JSON.stringify(m)) }); cap(E.port);
        }
      }
    } catch {}
    return pm.call(this, m, tr);
  };

  // ---- what the worklets tell the main thread (played/idle/underrun; not level) ----
  E.portIn = [];
  const logIn = (e) => { try { const d = e && e.data; if (d && typeof d.type === 'string' && d.type !== 'level' && d.type !== 'played')
    { const x = { t: now(), type: d.type }; for (const k of ['gen', 'seq', 'ended', 'gapMs']) if (k in d) x[k] = d[k]; E.portIn.push(x); cap(E.portIn); } } catch {} };
  const desc = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
  if (desc && desc.set) {
    Object.defineProperty(MessagePort.prototype, 'onmessage', { configurable: true, enumerable: desc.enumerable,
      get() { return desc.get.call(this); },
      set(fn) { return desc.set.call(this, typeof fn === 'function' ? function (e) { logIn(e); return fn.call(this, e); } : fn); } });
  }

  // ---- captions, the ring's state, notices ------------------------------------
  const watch = () => {
    const mo = new MutationObserver((list) => {
      for (const r of list) {
        if (r.type === 'attributes' && r.target.classList && r.target.classList.contains('drive-overlay')) {
          const s = r.target.dataset.state; const last = E.states[E.states.length - 1];
          if (!last || last.s !== s) E.states.push({ t: now(), s });
        }
        if (r.type === 'childList') {
          for (const n of r.addedNodes) {
            if (n.nodeType !== 1) continue;
            if (n.classList.contains('drive-sentence')) E.captions.push({ t: now(), text: n.textContent.trim() });
            if (n.classList.contains('drive-overlay')) E.states.push({ t: now(), s: n.dataset.state });
          }
        }
        if (r.type === 'characterData' || r.type === 'childList') {
          const nt = document.querySelector('.drive-notice:not(.drive-link):not([hidden])');
          const txt = nt ? nt.textContent : '';
          if (txt && (!E.notices.length || E.notices[E.notices.length - 1].text !== txt)) E.notices.push({ t: now(), text: txt });
        }
      }
    });
    mo.observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['data-state', 'hidden'], characterData: true });
  };
  if (document.documentElement) watch(); else document.addEventListener('DOMContentLoaded', watch);

  // ---- WebCodecs off for a run that needs the PCM downlink ----------------------
  if (window.__E2E_NO_OPUS_DOWN && window.AudioDecoder) {
    window.AudioDecoder.isConfigSupported = async (c) => ({ supported: false, config: c });
  }

  // ---- the fake microphone: a Web Audio stream we can speak into -----------------
  const M = E.mic;
  const ensure = async () => {
    if (M.ctx) return;
    M.ctx = new AudioContext({ sampleRate: 48000 });
    M.dest = M.ctx.createMediaStreamDestination();
    const n = M.ctx.createBuffer(1, 96000, 48000);
    const d = n.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * 0.0015;   // ≈ −62 dBFS room
    const src = M.ctx.createBufferSource(); src.buffer = n; src.loop = true; src.connect(M.dest); src.start();
  };
  if (navigator.mediaDevices) {
    const orig = navigator.mediaDevices.getUserMedia ? navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices) : null;
    navigator.mediaDevices.getUserMedia = async (c) => {
      if (!c || !c.audio) return orig(c);
      await ensure(); try { await M.ctx.resume(); } catch {}
      M.gum += 1;
      return M.dest.stream;
    };
  }
  const play = async (buf, label, gain) => {
    await ensure(); try { await M.ctx.resume(); } catch {}
    const src = M.ctx.createBufferSource(); src.buffer = buf;
    const g = M.ctx.createGain(); g.gain.value = gain; src.connect(g); g.connect(M.dest);
    const rec = { label, at: now(), dur: buf.duration };
    src.start(); M.injects.push(rec); return rec;
  };
  M.say = async (url, gain = 1) => {
    await ensure();
    const buf = await M.ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    return play(buf, url, gain);
  };
  M.burst = async (seconds = 0.45, amp = 0.2) => {
    await ensure();
    const n = Math.round(48000 * seconds);
    const b = M.ctx.createBuffer(1, n, 48000); const d = b.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * amp * Math.min(1, i / 480, (n - i) / 480);
    return play(b, 'burst', 1);
  };

  // ---- a fake speechSynthesis (headless Chromium ships no voices) -----------------
  const T = E.tts;
  const voices = [{ voiceURI: 'e2e-en-US', name: 'E2E English', lang: 'en-US', localService: true, default: true }];
  let cur = null; let q = []; let timers = [];
  const clear = () => { timers.forEach(clearTimeout); timers = []; };
  const next = () => {
    if (cur || !q.length) { synth.speaking = !!cur; return; }
    cur = q.shift(); synth.speaking = true;
    const u = cur;
    timers.push(setTimeout(() => { if (cur !== u) return; T.spoken.push({ t: now(), text: u.text }); u.onstart && u.onstart({}); }, 40));
    timers.push(setTimeout(() => { if (cur !== u) return; cur = null; u.onend && u.onend({}); next(); }, 80 + u.text.length * 45 / (u.rate || 1)));
  };
  const synth = {
    speaking: false, pending: false, paused: false,
    getVoices: () => voices,
    addEventListener() {}, removeEventListener() {},
    speak(u) { q.push(u); next(); },
    cancel() { T.cancels += 1; const u = cur; cur = null; q = []; clear(); synth.speaking = false; if (u && u.onerror) u.onerror({ error: 'interrupted' }); },
    pause() {}, resume() {},
  };
  class U { constructor(text) { this.text = String(text); this.voice = null; this.lang = ''; this.rate = 1; this.pitch = 1; } }
  try { Object.defineProperty(window, 'speechSynthesis', { value: synth, configurable: true }); } catch {}
  window.SpeechSynthesisUtterance = U;
})();
"""


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def ok(self, name: str, cond, detail="") -> bool:
        good = bool(cond)
        self.rows.append({"name": name, "status": "PASS" if good else "FAIL", "detail": _short(detail)})
        print(("PASS " if good else "FAIL ") + name + (f"  [{_short(detail)}]" if detail not in ("", None) else ""), flush=True)
        return good

    def skip(self, name: str, why: str) -> None:
        self.rows.append({"name": name, "status": "SKIP", "detail": why})
        print(f"SKIP {name}  [{why}]", flush=True)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def _short(x, n: int = 400) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + "…"


def generated_clip(name: str) -> bytes | None:
    """Clips made on the fly (1.5.0 ``noise`` scenario): ``noise`` = 3 s of a busy room
    — low rumble, broadband hiss and a few sharp clatters (dishes, a door), peaking around
    −14 dBFS; nothing a recogniser can turn into words."""
    if name != "noise":
        return None
    import io
    rate = 16000
    rng = np.random.default_rng(20261017)
    n = 3 * rate
    hiss = rng.standard_normal(n) * 0.02
    rumble = np.convolve(rng.standard_normal(n), np.ones(200) / 200, mode="same") * 0.6
    x = hiss + rumble
    for at in (0.4, 1.1, 1.9, 2.5):
        i = int(at * rate)
        k = np.arange(int(0.12 * rate))
        x[i:i + len(k)] += rng.standard_normal(len(k)) * 0.25 * np.exp(-k / (0.025 * rate))
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def clip_onset_s(name: str, db: float = -40.0) -> float:
    """Seconds from the clip's start to its first 20 ms frame above `db` dBFS (the speech onset)."""
    with wave.open(str(CLIPS / f"{name}.wav")) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
    n = int(rate * 0.02)
    for i in range(0, len(x) - n, n):
        if 20 * np.log10(np.sqrt(np.mean(x[i:i + n] ** 2)) + 1e-9) > db:
            return i / rate
    return 0.0


# --------------------------------------------------------------------------- #
# servers
# --------------------------------------------------------------------------- #
class Bg:
    """One asyncio loop in a thread for the reference server and the shaper."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="e2e-bg", daemon=True).start()

    def run(self, coro, timeout: float = 30):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, fn, *a) -> None:
        self.loop.call_soon_threadsafe(fn, *a)


def start_scratch_dispatch(port: int, llm_port: int, out: Path, backend: Path, *, prewarm: bool,
                           reply: str = "long", forced: bool = True):
    """DisPatch from ``backend`` with the add-on (editable), voice forced on, one API bot
    ``voicetest`` on the stub LLM, the Voice cloning and Voice lab tools on. Speech models
    are read (only) from the live model dirs."""
    import shutil

    from e2e_voice_roundtrip import _wait_up
    from stub_llm import StubLLM
    llm = StubLLM(port=llm_port, delay_s=1.0, reply=reply, tokens_per_s=10.0).start()
    data = out / ("dispatch-data-" + secrets.token_hex(3))
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.yaml").write_text(json.dumps({"bots": [{
        "id": "voicetest", "name": "Voice Test", "emoji": "🎙", "order": 0, "visible": True, "safe": False,
        "api": {"provider": "openai", "base_url": llm.base_url, "model": "stub", "api_key": "not-a-key"},
    }]}, ensure_ascii=False))
    os.chmod(data / "config.yaml", 0o600)
    (data / "tools.yaml").write_text("tools:\n- {id: voice-cloning, kind: builtin, enabled: true}\n"
                                     "- {id: voice-lab, kind: builtin, enabled: true}\n")
    # A scratch run never holds the performance profile (BRIEF heat rules): that hold is
    # for the live box and is verified by e2e_voice_roundtrip.py, not here.
    vy: dict = {"rt": {"power_hold": "off"}}
    if not forced:                       # the Settings switch decides, starting OFF
        vy["enabled"] = False
    (data / "voice.yaml").write_text(json.dumps(vy) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPATCH_VOICE_MODELS", "DISPATCH_VOICE")}
    if forced:
        env["DISPATCH_VOICE"] = "1"
    env.update(DISPATCH_DATA_DIR=str(data), DISPATCH_EXTENSIONS="dispatch_voice:EXTENSION",
               DISPATCH_MIRROR="0", DISPATCH_GATEWAY_WS="0", OPENCLAW_BIN="/nonexistent/openclaw",
               DISPATCH_SPEECH_MODELS=str(SPEECH_MODELS), DISPATCH_VOICE_PREWARM="1" if prewarm else "0")
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    argv = [uv, "run", "--quiet", "--no-sync", "--offline", "--with-editable", str(ADDON), "uvicorn",
            "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--ws-max-size", "262144"]
    log = data / "server.log"
    with open(log, "wb") as f:
        proc = subprocess.Popen(argv, cwd=backend, env=env, stdout=f, stderr=subprocess.STDOUT)
    _wait_up(port, proc, log)
    return proc, llm, data


#: The Kokoro plugin checkout the like-live scenario overlays when it is there (it is a
#: separate GPL-3.0-or-later package: this harness only puts it on the scratch server's
#: command line, as the live unit does; it never imports it).
KOKORO_SRC = Path(os.environ.get("DISPATCH_VOICE_KOKORO_SRC", "").strip() or ADDON.parent / "dispatch-voice-kokoro")
#: The idle lock the like-live scenario sets (DisPatch's client never locks sooner than 30 s).
LIVELIKE_LOCK_S = 60


def start_livelike_dispatch(port: int, llm_port: int, out: Path, backend: Path, *, lock_s: int):
    """A scratch DisPatch set up the way SPEC §11 sets up the live box (FR-works coverage
    gap): the add-on found by its entry point (no ``DISPATCH_EXTENSIONS``), voice switched
    on by ``voice.yaml`` (not forced), the Kokoro plugin overlaid when its checkout and
    models are there (the default voice is then a built-in one, as on the live box), a PIN
    and a short idle lock. The PIN is made up here and never leaves the scratch data dir.
    Returns ``(proc, llm, data, pin, kokoro)``."""
    import shutil

    from stub_llm import StubLLM
    llm = StubLLM(port=llm_port, delay_s=1.0, reply="long", tokens_per_s=10.0).start()
    data = out / ("livelike-data-" + secrets.token_hex(3))
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.yaml").write_text(json.dumps({"bots": [{
        "id": "voicetest", "name": "Voice Test", "emoji": "🎙", "order": 0, "visible": True, "safe": False,
        "api": {"provider": "openai", "base_url": llm.base_url, "model": "stub", "api_key": "not-a-key"},
    }]}, ensure_ascii=False))
    os.chmod(data / "config.yaml", 0o600)
    (data / "tools.yaml").write_text("tools:\n- {id: voice-cloning, kind: builtin, enabled: true}\n"
                                     "- {id: voice-builtin, kind: builtin, enabled: true}\n"
                                     "- {id: voice-lab, kind: builtin, enabled: true}\n")
    (data / "voice.yaml").write_text(json.dumps({"enabled": True, "rt": {"power_hold": "off"}}) + "\n")
    pin = "".join(secrets.choice("123456789") for _ in range(6))
    (data / "security.yaml").write_text(f'pin: "{pin}"\nlock_timeout_seconds: {int(lock_s)}\n')
    os.chmod(data / "security.yaml", 0o600)
    kokoro = KOKORO_SRC if ((KOKORO_SRC / "pyproject.toml").is_file()
                            and (SPEECH_MODELS / "kokoro" / "voices-v1.0.bin").is_file()) else None
    drop = ("DISPATCH_VOICE_MODELS", "DISPATCH_VOICE", "DISPATCH_EXTENSIONS", "DISPATCH_VOICE_PKG",
            "DISPATCH_VOICE_EXTENSIONS", "DISPATCH_VOICE_TTS")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(DISPATCH_DATA_DIR=str(data), DISPATCH_MIRROR="0", DISPATCH_GATEWAY_WS="0",
               OPENCLAW_BIN="/nonexistent/openclaw", DISPATCH_SPEECH_MODELS=str(SPEECH_MODELS),
               DISPATCH_VOICE_PREWARM="1")
    overlays = ["--with-editable", str(ADDON)]
    if kokoro is not None:
        overlays += ["--with-editable", str(kokoro)]
        env["DISPATCH_VOICE_EXTENSIONS"] = "dispatch_voice_kokoro:setup"
        env["DISPATCH_VOICE_KOKORO_DIR"] = str(SPEECH_MODELS / "kokoro")
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    argv = [uv, "run", "--quiet", "--no-sync", "--offline", *overlays, "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--ws-ping-timeout", "60", "--ws-max-size", "262144"]
    log = data / "server.log"
    with open(log, "wb") as f:
        proc = subprocess.Popen(argv, cwd=backend, env=env, stdout=f, stderr=subprocess.STDOUT)
    _wait_ext_up(port, proc, log)
    return proc, llm, data, pin, kokoro


def _wait_ext_up(port: int, proc: subprocess.Popen, log: Path, timeout_s: float = 240) -> None:
    """Up = DisPatch answers ``/api/extensions`` (open to a loopback caller even with a
    PIN set, unlike ``/api/voice/*``) and lists the voice add-on."""
    import httpx
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        with contextlib.suppress(httpx.HTTPError, OSError, ValueError):
            r = httpx.get(f"http://127.0.0.1:{port}/api/extensions", timeout=1)
            if r.status_code == 200 and any(e.get("id") == "voice" for e in r.json().get("extensions", [])):
                return
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"server did not come up; see {log}")


def stop_proc(proc) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()


def new_thread(base: str) -> str:
    import httpx
    r = httpx.post(base + "/api/threads", json={"bot_id": "voicetest", "title": "Voice e2e"}, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def make_ref_app(state: Path):
    """T6's reference v3 server with the server half of SPEC §4.6 (a TEST DOUBLE)."""
    import refserver
    from starlette.responses import FileResponse, PlainTextResponse
    from starlette.routing import Route, WebSocketRoute

    voiced_db = refserver.VOICED_DB

    class BrowserRefSession(refserver.RefSession):
        # 0.5 s of voice while ducked → a real interruption. A stand-in for T3's word-gated
        # rule (SPEC §4.6: duck at 0.25 s of VAD speech, barge-in once words confirm it).
        BARGE_FRAMES = 25

        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            self.speaking = False
            self.release = asyncio.Event()
            self.release.set()
            self.ducked_at: float | None = None
            self.last_reply_seq = 0

        def note(self, **kw) -> None:
            refserver.WORLD.log.append({"sid": self.sid, "thread": self.th.id, "t": time.time(), **kw})

        async def speak_loop(self) -> None:
            while True:
                rep, start = await self.speak_q.get()
                gen = self.gen
                i = start
                while self.gen == gen:
                    if i < len(rep.sentences):
                        await self.speak(rep, rep.sentences[i])
                        i += 1
                        continue
                    if rep.done:
                        await self.send({"type": "say_end", "gen": self.gen, "message_id": rep.message_id,
                                         "last_seq": self.seq})
                        break
                    await rep.changed.wait()

        async def speak(self, rep, text: str) -> None:
            self.tr.at_sentence_boundary()
            self.seq += 1
            seq, gen = self.seq, self.gen
            if not self.speaking:
                self.speaking = True
                await self.send({"type": "state", "state": "speaking"})
            self.last_reply_seq = seq
            await self.tr.send_say({"gen": gen, "seq": seq, "text": text, "message_id": rep.message_id,
                                    "kind": "reply", "role": "character"})
            self.stats["say"] += 1
            if self.tr.link.text_mode:
                return
            total = int(len(text.split()) * refserver.WORD_S * refserver.SR_OUT)
            t = np.arange(total) / refserver.SR_OUT
            pcm = (0.25 * np.sin(2 * np.pi * (180 + 7 * (seq % 9)) * t) * 32767).astype(np.int16)
            step = int(refserver.TTS_CHUNK_S * refserver.SR_OUT)
            for n, i in enumerate(range(0, total, step)):
                await self.release.wait()          # held while ducked (§4.6 layer 2)
                if self.gen != gen:
                    return
                await asyncio.sleep(min(step, total - i) / refserver.SR_OUT * refserver.TTS_RTF)
                if self.gen != gen:
                    return
                await self.tr.send_audio(kind="reply", gen=gen, seq=seq, pcm16=pcm[i:i + step],
                                         t_ms=self.tr.now_ms(), chunk=n)
            await self.tr.end_sentence(kind="reply", gen=gen, seq=seq)

        async def barge_in(self, why: str) -> None:
            self.gen = (self.gen + 1) & 0xFFFF
            self.ducked_at = None
            self.release.set()
            self.speaking = False
            await self.tr.drop_older_gens(self.gen)
            while not self.speak_q.empty():
                self.speak_q.get_nowait()
            if self.speaker:
                self.speaker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self.speaker
            self.speaker = asyncio.create_task(self.speak_loop())
            await self.send({"type": "barge_in", "gen": self.gen})
            self.note(barge_in=why, gen=self.gen)

        async def on_audio(self, buf: bytes) -> None:
            if os.environ.get("E2E_DEBUG_RX"):
                with contextlib.suppress(Exception):
                    got = self.tr.decode_up(buf) if refserver.wire3.is_v3(buf) else None
                    if got is not None:
                        off, flags, pcm = got
                        db = 20 * np.log10(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2)) + 1e-9)
                        self.note(rx={"off": off, "n": len(pcm), "flags": flags, "db": round(float(db), 1),
                                      "pos": self.det.pos, "in_speech": self.det.in_speech})
            await super().on_audio(buf)
            if self.ducked_at is not None and self.det.in_speech:
                voiced = 0
                for f in self.det.frames:
                    rms = np.sqrt(np.mean((f.astype(np.float64) / 32768) ** 2)) + 1e-9
                    voiced += 20 * np.log10(rms) > voiced_db
                if voiced >= self.BARGE_FRAMES:
                    await self.barge_in("voice")

        async def on_event(self, ev: tuple) -> None:
            if ev[0] == "start" and self.speaking and self.ducked_at is None:
                self.ducked_at = time.monotonic()
                self.release.clear()
                await self.send({"type": "duck", "on": True, "gen": self.gen})
                self.note(duck=True)
                return
            if ev[0] == "end" and self.ducked_at is not None:
                held = time.monotonic() - self.ducked_at
                self.ducked_at = None
                await self.send({"type": "duck", "on": False, "gen": self.gen})
                await self.send({"type": "resume", "gen": self.gen, "rewind": held > 0.8})
                self.release.set()
                self.note(resume=True, rewind=held > 0.8, held_s=round(held, 2))
                return
            await super().on_event(ev)

        async def on_control(self, msg: dict) -> None:
            t = msg.get("type")
            if t in ("heard", "duck_local", "pause", "playback_done", "stop_playback", "cancel", "link", "probe",
                     "start", "quiet_check", "undo"):
                self.note(frame={k: v for k, v in msg.items() if k != "type"}, type=t)
            if t in ("stop_playback", "cancel"):
                await self.barge_in(t)
                return
            if t == "playback_done":
                if msg.get("gen") == self.gen and isinstance(msg.get("seq"), int) and msg["seq"] >= self.last_reply_seq \
                        and self.speaking:
                    self.speaking = False
                    await self.send({"type": "state", "state": "listening"})
                return
            await super().on_control(msg)

    refserver.RefSession = BrowserRefSession       # voice_ws builds sessions by this name
    app = refserver.make_app(state, "long", "fake")

    async def chat_ws(ws):
        """DisPatch's own chat socket (/ws) also reaches this origin: accept and idle."""
        await ws.accept()
        with contextlib.suppress(Exception):
            while True:
                await ws.receive_text()

    static = (ADDON / "src/dispatch_voice/static").resolve()

    async def ext_static(request):
        """AudioWorklet modules are fetched outside Playwright's routing, from this origin."""
        p = (static / request.path_params["path"]).resolve()
        if static not in p.parents or not p.is_file() or p.suffix not in (".js", ".css", ".json"):
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(p, media_type="text/javascript" if p.suffix == ".js" else None)

    app.router.routes.append(WebSocketRoute("/ws", chat_ws))
    app.router.routes.append(Route("/ext/voice/{path:path}", ext_static))
    return app, refserver


class RefTarget:
    def __init__(self, bg: Bg, port: int, shaper_port: int, out: Path) -> None:
        import uvicorn
        self.bg = bg
        self.port = port
        self.shaper_port = shaper_port
        self.app, self.mod = make_ref_app(out / "ref-state")
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off",
                             ws_max_size=262144, ws_ping_interval=20, ws_ping_timeout=60)
        self.server = uvicorn.Server(cfg)
        self.task = asyncio.run_coroutine_threadsafe(self.server.serve(), bg.loop)
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        self.proxy = None

    def shape(self, profile: str) -> None:
        import shaper
        if self.proxy is not None:
            self.bg.run(self.proxy.stop())
        self.proxy = shaper.make_proxy(profile, self.shaper_port, self.port)
        self.bg.run(self.proxy.start())

    def blackout(self, seconds: float) -> None:
        self.bg.call(self.proxy.blackout, seconds)

    def cut(self) -> None:
        """Drop every connection through the shaper now (RST both ways)."""
        self.bg.call(self.proxy.cut)

    def relink(self, profile: str) -> None:
        """Give the LIVE link another profile's rate and delay without dropping a
        connection: the radio got better (or worse) under an open socket."""
        import shaper
        p = shaper.PROFILES[profile]

        def apply() -> None:
            for link, kbps in ((self.proxy.up, p["up"]), (self.proxy.down, p["down"])):
                link.kbps = kbps
                link.delay_ms = p["delay_ms"]
                link.jitter_ms = p["jitter_ms"]
        self.bg.call(apply)

    @property
    def log(self) -> list[dict]:
        return list(self.mod.WORLD.log)

    def rows(self, tid: str) -> list[dict]:
        th = self.mod.WORLD.threads.get(tid)
        return list(th.rows) if th else []

    def stop(self) -> None:
        if self.proxy is not None:
            with contextlib.suppress(Exception):
                self.bg.run(self.proxy.stop())
        self.server.should_exit = True
        with contextlib.suppress(Exception):
            self.task.result(10)


# --------------------------------------------------------------------------- #
# the browser
# --------------------------------------------------------------------------- #
class Browser:
    def __init__(self, pw, *, width: int, height: int, direct: str, via: str | None = None,
                 no_opus_down: bool = False, mobile: bool = True, theme: str | None = None) -> None:
        self.b = pw.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
        # Service workers are blocked: their script and fetches bypass the routing below
        # (DisPatch's sw.js would be asked of the shaper origin), and the test wants the network.
        self.ctx = self.b.new_context(viewport={"width": width, "height": height}, device_scale_factor=1,
                                      is_mobile=mobile and width < 600, has_touch=mobile and width < 600,
                                      service_workers="block",
                                      color_scheme="light" if theme == "light" else "dark")
        if theme:
            # DisPatch's theme runtime reads its palette from this key before first paint
            # (index.html: data-storage-key). Dark = the app's default (Glacier).
            slug = THEMES[theme]
            self.ctx.add_init_script(f"try {{ localStorage.setItem('dispatch-palette', '{slug}'); }} catch (e) {{}}")
        self.direct = direct
        self.via = via or direct
        self.errors: list[str] = []
        api = self.ctx.request
        clip_re = re.compile(r"/__e2e/clip/([a-z0-9_]+)\.wav$")

        def route(r, request):
            m = clip_re.search(request.url.split("?")[0])
            gen = generated_clip(m.group(1)) if m else None
            if gen is not None:
                r.fulfill(status=200, headers={"Content-Type": "audio/wav"}, body=gen)
                return
            if m and (CLIPS / f"{m.group(1)}.wav").is_file():
                r.fulfill(status=200, headers={"Content-Type": "audio/wav"},
                          body=(CLIPS / f"{m.group(1)}.wav").read_bytes())
                return
            if via and request.url.startswith(via):
                url = direct + request.url[len(via):]
                hdr = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "origin", "referer")}
                resp = api.fetch(url, method=request.method, headers=hdr, data=request.post_data_buffer)
                r.fulfill(status=resp.status, headers=dict(resp.headers), body=resp.body())
                return
            r.continue_()

        self.ctx.route("**/*", route)
        self.page = self.ctx.new_page()
        if no_opus_down:
            self.page.add_init_script("window.__E2E_NO_OPUS_DOWN = true;")
        self.page.add_init_script(INIT_JS)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("console", lambda m: self.errors.append(f"console: {m.text}") if m.type == "error" else None)

    # ---- page helpers
    def ev(self, js: str, arg=None):
        return self.page.evaluate(js, arg)

    def frames(self, direction: str, typ: str | None = None, sock: int | None = None) -> list[dict]:
        return self.ev("""([d, t, s]) => window.__e2e.ws.filter((f) => f.dir === d && f.m && (!t || f.m.type === t)
            && (s === null || f.sock === s))""", [direction, typ, sock])

    def bins(self, direction: str) -> list[dict]:
        return self.ev("(d) => window.__e2e.ws.filter((f) => f.dir === d && f.bin !== undefined)", direction)

    def port(self, typ: str | None = None) -> list[dict]:
        return self.ev("(t) => window.__e2e.port.filter((p) => !t || p.type === t)", typ)

    def state(self) -> str | None:
        return self.ev("() => { const o = document.querySelector('.drive-overlay'); return o ? o.dataset.state : null; }")

    def word(self) -> str:
        return self.ev("() => { const n = document.querySelector('.drive-state'); return n ? n.textContent : ''; }")

    def now(self) -> float:
        return self.ev("() => performance.now()")

    def wait(self, js: str, timeout_s: float, arg=None, every: float = 0.1) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            with contextlib.suppress(Exception):
                if self.ev(js, arg):
                    return True
            time.sleep(every)
        return False

    def say(self, clip: str, gain: float = 1.0) -> dict:
        return self.ev("([u, g]) => window.__e2e.mic.say(u, g)", [f"/__e2e/clip/{clip}.wav", gain])

    def shot(self, shots: Path, name: str) -> None:
        shots.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            self.page.screenshot(path=str(shots / f"{name}.png"))

    def script_errors(self) -> list[str]:
        # Network noise, and the harness's own service-worker block (DisPatch's shell reads
        # navigator.serviceWorker, which throws in a context with workers blocked).
        noise = ("Failed to load resource", "WebSocket connection to", "net::ERR",
                 "Failed to read the 'serviceWorker' property")
        return [e for e in self.errors if not any(n in e for n in noise)]

    def summary(self) -> dict:
        """What happened on the voice socket (for a failed check's record)."""
        with contextlib.suppress(Exception):
            return self.ev("""() => { const E = window.__e2e; const n = {};
                for (const f of E.ws) { const k = f.dir + ':' + (f.m ? f.m.type : (f.bin !== undefined ? 'bin' : f.dir)); n[k] = (n[k] || 0) + 1; }
                return { frames: n, states: E.states.map((s) => [Math.round(s.t), s.s]), captions: E.captions.length,
                         spoken: E.tts.spoken.length, injects: E.mic.injects.map((i) => [Math.round(i.at), i.label, i.dur]),
                         gum: E.mic.gum, mic: E.port.filter((p) => p.mic).map((p) => [Math.round(p.t), p.mic]),
                         ins: E.ws.filter((f) => f.m && f.dir === 'in' && f.m.type !== 'pong').map((f) => [Math.round(f.t), f.m.type]).slice(0, 30),
                         outs: E.ws.filter((f) => f.dir === 'out' && f.m && f.m.type !== 'ping').map((f) => [Math.round(f.t), f.m.type]).slice(0, 30),
                         bins: E.ws.filter((f) => f.dir === 'out' && f.bin !== undefined).map((f) => Math.round(f.t)) }; }""")
        return {}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.b.close()


def open_voice_chat(br: Browser, tid: str, c: Checks, tag: str, *, ready_timeout: float = 120) -> bool:
    br.page.goto(f"{br.via}/?popout=1&bot=voicetest&thread={tid}")
    if not br.wait("() => { const b = document.getElementById('drive-btn'); return !!b && !b.hidden; }", 40):
        c.ok(f"{tag}: the mic button shows in the chat", False, br.errors[-3:])
        return False
    br.page.click("#drive-btn")
    br.page.wait_for_selector(".drive-overlay", timeout=10000)
    ok = br.wait("() => { const o = document.querySelector('.drive-overlay'); return o && ['listening', 'stopped'].includes(o.dataset.state); }",
                 ready_timeout)
    st = br.state()
    return c.ok(f"{tag}: the voice chat opens and reaches listening", ok and st == "listening",
                {"state": st, "notice": br.ev("() => (document.querySelector('.drive-notice:not([hidden])') || {}).textContent || ''")})


def caption_texts(br: Browser) -> list[str]:
    return br.ev("() => [...document.querySelectorAll('.drive-them .drive-sentence')].map((n) => n.textContent.trim())")


# --------------------------------------------------------------------------- #
# target: ref (the v3 wire through the shaper)
# --------------------------------------------------------------------------- #
def run_ref(pw, a, c: Checks, out: Path) -> dict:
    import httpx
    res: dict = {}
    bg = Bg()
    base = f"http://127.0.0.1:{a.port_base}"
    via = f"http://127.0.0.1:{a.port_base + 3}"
    proc = llm = None
    ref = None
    try:
        proc, llm, _data = start_scratch_dispatch(a.port_base, a.port_base + 1, out, a.backend, prewarm=False,
                                                  forced=False)
        ref = RefTarget(bg, a.port_base + 2, a.port_base + 3, out)
        # The Settings and Voice lab pages need no voice socket: checked here too, model-free.
        if "settings" in a.scenarios and a.target == "ref":
            for w, h in ((390, 844), (1366, 900)):
                res[f"settings_{w}"] = dispatch_settings(pw, a, c, base, w, h)
        if "lab" in a.scenarios and a.target == "ref":
            for w, h in ((390, 844), (1366, 900)):
                res[f"lab_{w}"] = dispatch_lab(pw, a, c, base, w, h)
        # Whatever the settings scenario did, voice is on from here.
        httpx.patch(base + "/api/voice/settings", json={"enabled": True}, timeout=30).raise_for_status()
        if "conversation" in a.scenarios:
            ref.shape("baseline")
            res["conversation"] = ref_conversation(pw, a, c, base, via, ref)
            res["conversation_1366"] = ref_conversation(pw, a, c, base, via, ref, width=1366, height=900)
        if "3g" in a.scenarios:
            ref.shape("3g-weak")
            res["3g"] = ref_phone_voice(pw, a, c, base, via, ref)
        if "edge" in a.scenarios:
            ref.shape("edge")
            res["edge"] = ref_edge(pw, a, c, base, via, ref)
        if "blackout" in a.scenarios:
            ref.shape("baseline")
            res["blackout"] = ref_blackout(pw, a, c, base, via, ref)
        if "keys" in a.scenarios:
            ref.shape("baseline")
            res["keys"] = ref_keys(pw, a, c, base, via, ref)
        if "drop" in a.scenarios:
            ref.shape("baseline")
            res["drop"] = ref_drop(pw, a, c, base, via, ref)
        if "cdp" in a.scenarios:
            res["cdp"] = ref_cdp(pw, a, c, base, ref)
        if "failback" in a.scenarios:
            ref.shape("3g-weak")
            res["failback"] = ref_failback(pw, a, c, base, via, ref)
        if "shots" in a.scenarios:
            ref.shape("baseline")
            res["shots"] = theme_shots(pw, a, c, base, via, ref)
        if "locales" in a.scenarios:
            ref.shape("baseline")
            res["locales"] = locale_pages(pw, a, c, base, via, ref)
    finally:
        if ref is not None:
            ref.stop()
        stop_proc(proc)
        if llm is not None:
            llm.stop()
    return res


def server_frames(ref: RefTarget, sid_set: set[str] | None, typ: str) -> list[dict]:
    return [x for x in ref.log if x.get("type") == typ and (sid_set is None or x.get("sid") in sid_set)]


def ref_conversation(pw, a, c: Checks, base: str, via: str, ref: RefTarget, width: int = 390,
                     height: int = 844) -> dict:
    tag = f"ref/v3/{width}"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=width, height=height, direct=base, via=via)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        url = br.ev("() => window.__e2e.urls[0]")
        q = dict(x.split("=", 1) for x in url.split("?", 1)[1].split("&"))
        out["url"] = url
        c.ok(f"{tag}: the socket negotiates v3 with a stream id and codec lists (C.1)",
             q.get("v") == "3" and re.fullmatch(r"[A-Za-z0-9-]{8,64}", q.get("s", "")) and "pcm16" in q.get("up", "")
             and "pcm16" in q.get("down", ""), q)
        ready = br.frames("in", "ready")[0]["m"]
        out["ready"] = {k: ready.get(k) for k in ("proto", "sid", "codec_in", "codec_out", "gate")}
        c.ok(f"{tag}: ready says proto 3 with a session id and the gate (C.4)",
             ready.get("proto") == 3 and ready.get("sid") and isinstance(ready.get("gate"), dict), out["ready"])
        c.ok(f"{tag}: Opus both ways in headless Chromium (WebCodecs) — or PCM16 named",
             ready.get("codec_in") in ("opus", "pcm16") and ready.get("codec_out") in ("opus", "pcm16"),
             {"in": ready.get("codec_in"), "out": ready.get("codec_out"), "url_up": q.get("up"), "url_down": q.get("down")})
        br.shot(a.shots, f"ref-{width}-listening")
        gate = [p for p in br.port() if p.get("mic") and "gate" in p["mic"]]
        c.ok(f"{tag}: the gate from ready reached the mic worklet (§5.6)",
             any(g["mic"]["gate"].get("on") for g in gate), gate[-1:] if gate else "none")

        # ---- one turn: speak, hear the reply, captions only as sentences play
        br.say("short")
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'transcript')", 20)
        c.ok(f"{tag}: the uplink carries the utterance (wire3 offsets) and a transcript comes back", got,
             {"up_frames": len(br.bins("out"))})
        ups = [b for b in br.bins("out") if b.get("up")]
        odd = [{"t": round(b["t"]), "bin": b["bin"], "head": b.get("head")} for b in br.bins("out") if not b.get("up")]
        c.ok(f"{tag}: every uplink message is wire3 (0x03, offsets)", ups and not odd,
             {"wire3": len(ups), "all": len(br.bins("out")), "codec": ups[0]["up"]["codec"] if ups else None, "odd": odd[:3],
              "ready_at": round(br.frames("in", "ready")[0]["t"]), "first_up": [round(b["t"]) for b in br.bins("out")[:3]]})
        br.wait("() => window.__e2e.captions.length >= 2", 25)
        says = [f for f in br.frames("in", "say") if f["m"].get("kind") == "reply"]
        caps = br.ev("() => window.__e2e.captions")
        downs = [b for b in br.bins("in") if b.get("down")]
        first_audio = {}
        for b in downs:
            first_audio.setdefault(b["down"]["seq"], b["t"])
        say_at = {f["m"]["seq"]: f["t"] for f in says}
        order_ok = all(say_at.get(s, 1e18) <= t for s, t in first_audio.items())
        c.ok(f"{tag}: every sentence's `say` arrives before its first audio (C.5 #1)", order_ok and first_audio,
             {"says": len(says), "audio_seqs": sorted(first_audio)})
        say_texts = [f["m"]["text"] for f in says]
        c.ok(f"{tag}: captions are the said sentences, in order, each appearing after its say",
             caps and all(cp["text"] in say_texts for cp in caps)
             and [cp["text"] for cp in caps] == say_texts[:len(caps)], {"captions": [x["text"] for x in caps][:4]})
        played_late = all(cp["t"] >= say_at.get(says[i]["m"]["seq"], 0) for i, cp in enumerate(caps[:len(says)]))
        c.ok(f"{tag}: a caption is added when its sentence starts playing, not when it is announced", played_late)
        time.sleep(2.2)
        heard = server_frames(ref, None, "heard")
        c.ok(f"{tag}: `heard` frames reach the server about once a second while the reply plays (C.3)",
             len(heard) >= 2 and all(h["frame"].get("source") == "server" for h in heard),
             heard[-2:] if heard else "none")
        br.shot(a.shots, f"ref-{width}-speaking")

        # ---- the optimistic duck, then "Wait, stop" → barge_in with heard
        br.wait("() => { const c = window.__e2e.captions; return c.length && performance.now() - c[0].t > 2000; }", 15)
        onset = clip_onset_s("interrupt")
        pauses_before = len(br.port("pause"))
        inj = br.say("interrupt")
        t_on = inj["at"] + onset * 1000
        br.wait("(n) => window.__e2e.port.filter((p) => p.type === 'pause').length > n", 3, pauses_before)
        pause = [p for p in br.port("pause")][pauses_before:]
        duck_ms = (pause[0]["t"] - t_on) if pause else None
        out["duck_ms"] = round(duck_ms) if duck_ms is not None else None
        c.ok(f"{tag}: the optimistic duck pauses the reply within 200 ms of the speech onset (§4.6 layer 1)",
             duck_ms is not None and duck_ms <= 200, {"onset_to_pause_ms": out["duck_ms"]})
        dl = [f for f in br.frames("out", "duck_local") if f["t"] >= inj["at"]]
        c.ok(f"{tag}: the client tells the server it ducked (duck_local on)", dl and dl[0]["m"].get("on") is True, dl[:1])
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'duck' && f.m.on)", 2)
        br.shot(a.shots, f"ref-{width}-hearing-duck")
        word = br.word()
        c.ok(f"{tag}: while the reply is held the screen says it is hearing you", word == EN["drive"]["state_hearing"], word)
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'barge_in')", 4)
        bi = br.frames("in", "barge_in")
        bi_ms = (bi[0]["t"] - t_on) if bi else None
        out["barge_ms"] = round(bi_ms) if bi_ms is not None else None
        c.ok(f"{tag}: \"Wait, stop\" → barge_in within 1.0 s of the onset", got and bi_ms is not None and bi_ms <= 1000,
             {"onset_to_barge_in_ms": out["barge_ms"]})
        if bi:
            gen = bi[0]["m"]["gen"]
            hd = [f for f in br.frames("out", "heard") if f["t"] >= bi[0]["t"]]
            c.ok(f"{tag}: right after barge_in the client sends `heard` with the NEW gen (C.3)",
                 hd and hd[0]["m"]["gen"] == gen and hd[0]["t"] - bi[0]["t"] < 200 and hd[0]["m"].get("played_ms", -1) >= 0,
                 hd[:1])
            time.sleep(0.5)
            srv = [h for h in server_frames(ref, None, "heard") if h["frame"].get("gen") == gen]
            c.ok(f"{tag}: the server received that `heard`", bool(srv), srv[:1])
            flush = [p for p in br.port("flush") if p["t"] >= bi[0]["t"] - 5]
            c.ok(f"{tag}: the player was flushed to the new gen", any(p.get("gen") == gen for p in flush), flush[:2])
            # Captions: nothing from the old reply after the cut.
            first_new = next((f["t"] for f in br.frames("in", "say") if f["m"].get("gen") == gen), None)
            br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'transcript' && f.t > t)",
                    12, bi[0]["t"])
            br.wait("(g) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'say' && f.m.gen === g)", 12, gen)
            first_new = next((f["t"] for f in br.frames("in", "say") if f["m"].get("gen") == gen), None)
            stray = [cp for cp in br.ev("() => window.__e2e.captions") if cp["t"] > bi[0]["t"] and (first_new is None or cp["t"] < first_new)]
            c.ok(f"{tag}: no caption from the cut reply appears after the barge-in (unheard text is not appended)",
                 not stray, stray[:2])

        # ---- a short cough while speaking: duck, then resume (with rewind) — no barge-in
        if br.wait("(g) => { const c = window.__e2e.captions; return c.length && performance.now() - c[c.length - 1].t > 1800; }", 20,
                   None):
            n_bi = len(br.frames("in", "barge_in"))
            n_res = len(br.port("resume"))
            br.ev("() => window.__e2e.mic.burst(0.45, 0.25)")
            got = br.wait("(n) => window.__e2e.port.filter((p) => p.type === 'resume').length > n", 4, n_res)
            res_msgs = br.port("resume")[n_res:]
            srv_resume = [f for f in br.frames("in", "resume")]
            c.ok(f"{tag}: after a short cough the reply resumes (server duck off + resume, rewind after > 0.8 s)",
                 got and srv_resume and len(br.frames("in", "barge_in")) == n_bi,
                 {"player_resume": res_msgs[-2:], "server_resume": srv_resume[-1:]})
        else:
            c.ok(f"{tag}: a reply sentence played long enough for the cough check", False)

        # ---- Pause = the mic only
        n_out = len(br.frames("out"))
        br.page.click(".drive-pause")
        time.sleep(0.3)
        sent = [f["m"]["type"] for f in br.frames("out")[n_out:] if f["m"]["type"] not in ("ping", "heard")]
        c.ok(f"{tag}: Pause sends only `pause` (never stop_playback)", sent == ["pause"], sent)
        mic_off = [p for p in br.port() if p.get("mic") and p["mic"].get("enabled") is False]
        c.ok(f"{tag}: Pause turns the mic worklet off", bool(mic_off))
        br.shot(a.shots, f"ref-{width}-paused")
        br.page.click(".drive-pause")
        br.page.keyboard.press("Escape")
        br.wait("() => !document.querySelector('.drive-overlay') && window.__e2e.ws.some((f) => f.dir === 'close')", 5)
        gone = br.ev("() => ({ overlay: !!document.querySelector('.drive-overlay'), closed: window.__e2e.ws.filter((f) => f.dir === 'close').map((f) => f.code), active: document.activeElement && document.activeElement.className })")
        c.ok(f"{tag}: Esc hangs up and the voice socket closes", not gone["overlay"] and gone["closed"], gone)
        # The whole conversation's uplink (the server's audio queue holds 64,000 B, C.2). The
        # pre-ready flush's pacing itself (T7b-2) is proven on PCM16 in drive-audio.test.js.
        ups = [b for b in br.bins("out") if b.get("up")]
        worst = max((sum(x["bin"] for x in ups if 0 <= x["t"] - b["t"] < 250) for b in ups), default=0)
        c.ok(f"{tag}: the uplink never bursts ≥ 64,000 B in 250 ms", bool(ups) and worst < 64000,
             {"worst_250ms_bytes": worst, "messages": len(ups)})
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def ref_phone_voice(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    tag = "ref/3g-weak (PCM down)"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via, no_opus_down=True)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        q = br.ev("() => window.__e2e.urls[0]")
        c.ok(f"{tag}: the client asked for PCM16 down (no decoder)", "down=pcm16" in q, q)
        br.say("short")
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'link' && f.m.mode === 'text')", 45)
        link = br.frames("out", "link")
        out["link"] = [f["m"] for f in link]
        c.ok(f"{tag}: the link monitor switches to text (`link {{mode: \"text\"}}`)", got, out["link"])
        time.sleep(1.0)
        srv_link = [x for x in server_frames(ref, None, "link") if x["frame"].get("mode") == "text"]
        c.ok(f"{tag}: the `link` frame reached the server", bool(srv_link), srv_link[:1])
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'say_end')", 60)
        br.wait("() => !window.speechSynthesis.speaking", 30)
        time.sleep(0.5)
        line = br.ev("() => { const n = document.querySelector('.drive-link'); return n && !n.hidden ? n.textContent : ''; }")
        c.ok(f"{tag}: the screen says \"{EN['drive']['link_phone']}\"", line == EN["drive"]["link_phone"], line)
        br.shot(a.shots, "ref-390-phone-voice")
        says = [f["m"] for f in br.frames("in", "say") if f["m"].get("kind") == "reply"]
        spoken = [x["text"] for x in br.ev("() => window.__e2e.tts.spoken")]
        pushed = {p["seq"] for p in br.port("push") if p.get("kind") == "reply"}
        phone_seqs = [s["seq"] for s in says if s["text"] in spoken]
        out.update({"says": len(says), "spoken": spoken, "server_seqs": sorted(pushed)})
        say_texts = [s["text"] for s in says]
        tail = say_texts[len(say_texts) - len(spoken):] if spoken else []
        c.ok(f"{tag}: the fake synth spoke every `say` after the switch exactly once, in order",
             spoken and spoken == tail and len(set(spoken)) == len(spoken), {"spoken": spoken[:3], "of": len(say_texts)})
        c.ok(f"{tag}: no audio was scheduled for a sentence the phone spoke (claims)",
             not (set(phone_seqs) & pushed), {"phone": phone_seqs, "server": sorted(pushed)})
        done = [f["m"] for f in br.frames("out", "playback_done")]
        c.ok(f"{tag}: the phone reports the reply's end with playback_done", bool(done), done[-1:])
        loc = [p for p in br.port() if p.get("mic") and "localSpeaking" in p["mic"]]
        c.ok(f"{tag}: local_speaking reached the mic worklet while the phone spoke",
             any(p["mic"]["localSpeaking"] for p in loc), len(loc))
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def ref_edge(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    tag = "ref/edge"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=40):
            return out
        br.say("short")
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'eot')", 30)
        eot = br.frames("in", "eot")
        got = br.wait("() => window.__e2e.captions.length > 0 || window.__e2e.tts.spoken.length > 0", 30)
        first = min([x["t"] for x in br.ev("() => window.__e2e.captions")] + [x["t"] for x in br.ev("() => window.__e2e.tts.spoken")]
                    or [None]) if got else None
        ms = (first - eot[0]["t"]) if (first is not None and eot) else None
        out["eot_to_first_sentence_ms"] = round(ms) if ms is not None else None
        out["path"] = "phone" if br.ev("() => window.__e2e.tts.spoken.length") else "server"
        if not c.ok(f"{tag}: the first spoken sentence starts ≤ 6 s after eot", ms is not None and ms <= 6000, out):
            out["diag"] = br.summary()
            out["links"] = [f["m"] for f in br.frames("out", "link")]
            out["server"] = [x for x in ref.log if x.get("thread") == tid and ("stats" in x or "open" in x or "rx" in x)]
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def ref_blackout(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    tag = "ref/blackout"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        rows0 = len(ref.rows(tid))
        t0 = time.monotonic()
        ref.blackout(a.blackout_s)
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'offline'", 20)
        c.ok(f"{tag}: a dead link shows offline", br.state() == "offline", br.state())
        got = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'waiting'", 75)
        c.ok(f"{tag}: after 60 s offline the screen says \"Waiting for signal\"",
             got and br.word() == EN["drive"]["state_waiting"], br.word())
        br.shot(a.shots, "ref-390-waiting")
        # Speak inside the last 20 s of the blackout: it waits in the ring.
        speak_at = a.blackout_s - 6.0
        while time.monotonic() - t0 < speak_at:
            time.sleep(0.2)
        br.say("short")
        br.wait("() => window.__e2e.notices.length > 0", 4)
        notes = [n["text"] for n in br.ev("() => window.__e2e.notices")]
        c.ok(f"{tag}: speech while offline gets \"{EN['drive']['offline_said']}\"", EN["drive"]["offline_said"] in notes, notes[-2:])
        got = br.wait("() => document.querySelector('.drive-overlay') && ['listening', 'speaking', 'thinking'].includes(document.querySelector('.drive-overlay').dataset.state) && window.__e2e.urls.length > 1", 60)
        states = [s["s"] for s in br.ev("() => window.__e2e.states")]
        walk = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
        out["walk"] = walk
        c.ok(f"{tag}: the state walk is offline → waiting → listening", got and _subseq(["listening", "offline", "waiting", "listening"], walk), walk)
        starts = [f["m"] for f in br.frames("out", "start")]
        cur = [s.get("cursor") for s in starts if s.get("cursor")]
        c.ok(f"{tag}: the reconnect `start` carries the stream cursor (C.5 #10)", bool(cur), cur[-1:])
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'cursor')", 10)
        # The first socket's pre-ready audio is paced through the replay pump too (T7b-2):
        # only a reconnect socket's replay-flagged messages are the offline ring replay.
        replay = [b for b in br.bins("out") if b.get("up") and b["up"]["flags"] & 2 and b.get("sock", 0) > 0]
        c.ok(f"{tag}: the ring replays what the mic heard while offline (replay flag)", bool(replay), len(replay))
        br.wait("(n) => false", 6, rows0)
        rows = ref.rows(tid)[rows0:]
        out["rows"] = [r["text"] for r in rows]
        c.ok(f"{tag}: exactly one transcript for the speech made while offline", len(rows) == 1, out["rows"])
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def ping_rtts(br: Browser, since: float = 0.0) -> list[float]:
    """Round trips of the client's own heartbeat (ping {t} → pong {t}) seen by the page."""
    outs = {f["m"].get("t"): f["t"] for f in br.frames("out", "ping") if f["t"] >= since}
    return [f["t"] - outs[f["m"]["t"]] for f in br.frames("in", "pong") if f["m"].get("t") in outs]


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 1) if xs else None


def link_line(br: Browser) -> str:
    return br.ev("() => { const n = document.querySelector('.drive-link'); return n && !n.hidden ? n.textContent : ''; }")


def _failover_and_back(br: Browser, a, c: Checks, tag: str, ref: RefTarget | None, *, heal, shot: str) -> dict:
    """The slow link is on when this starts. One reply → the phone's voice (every `say` once,
    in order, no server audio for a sentence the phone took, the notice); then ``heal()``
    makes the link healthy under the SAME socket → after the dwell, a healthy spell and a
    probe, `link {mode: "audio"}` → the next reply plays in the server's voice."""
    out: dict = {}
    t_ask = br.now()
    br.say("short")
    got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'link' && f.m.mode === 'text')", 45)
    c.ok(f"{tag}: the slow link switches to the phone voice (`link {{mode: \"text\"}}`)", got,
         [f["m"] for f in br.frames("out", "link")])
    br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'say_end')", 90)
    br.wait("() => !window.speechSynthesis.speaking", 40)
    br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 30)
    c.ok(f"{tag}: the notice says \"{EN['drive']['link_phone']}\" while degraded", link_line(br) == EN["drive"]["link_phone"],
         link_line(br))
    br.shot(a.shots, f"{shot}-phone-voice")
    says = [f["m"] for f in br.frames("in", "say") if f["t"] > t_ask and f["m"].get("kind") == "reply"]
    spoken = [x["text"] for x in br.ev("() => window.__e2e.tts.spoken")]
    pushed = {p["seq"] for p in br.port("push") if p.get("kind") == "reply"}
    tail = [s["text"] for s in says][len(says) - len(spoken):] if spoken else []
    out["phone"] = {"says": len(says), "spoken": len(spoken), "server_seqs": sorted(pushed)}
    c.ok(f"{tag}: the fake synth spoke every `say` after the switch exactly once, in order",
         spoken and spoken == tail and len(set(spoken)) == len(spoken), out["phone"])
    phone_seqs = {s["seq"] for s in says if s["text"] in spoken}
    c.ok(f"{tag}: no server audio was scheduled for a sentence the phone spoke (claims)", not (phone_seqs & pushed),
         {"phone": sorted(phone_seqs), "server": sorted(pushed)})
    c.ok(f"{tag}: the phone reports the reply's end (`playback_done`)", bool(br.frames("out", "playback_done")))
    n_spoken = len(spoken)
    heal()
    t0 = time.monotonic()
    got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'link' && f.m.mode === 'audio')", 200, every=0.5)
    back = [f["m"] for f in br.frames("out", "link") if f["m"].get("mode") == "audio"]
    out["recovered_after_s"] = round(time.monotonic() - t0, 1)
    probes = br.frames("out", "probe")
    c.ok(f"{tag}: once healthy, the client asks for the server's voice again (`link {{mode: \"audio\"}}`)",
         got and back, {"link": back[:1], "after_s": out["recovered_after_s"], "probes": len(probes)})
    c.ok(f"{tag}: … earned by a probe of the link (never on a guess)", bool(probes) and got,
         [f["m"] for f in probes][-1:])
    time.sleep(1.0)
    if ref is not None:
        srv = [x for x in server_frames(ref, None, "link") if x["frame"].get("mode") == "audio"
               and x["frame"].get("why") == "recovered"]
        c.ok(f"{tag}: the server received the switch back", bool(srv), srv[-1:])
    c.ok(f"{tag}: the low-signal notice is gone", link_line(br) == "", link_line(br))
    t_ask = br.now()
    n_push = len([p for p in br.port("push") if p.get("kind") == "reply"])
    br.say("short")
    br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'say_end' && f.t > t)", 60, t_ask)
    br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'playback_done' && f.t > t)", 60, t_ask)
    says = [f["m"] for f in br.frames("in", "say") if f["t"] > t_ask and f["m"].get("kind") == "reply"]
    pushed = {p["seq"] for p in br.port("push") if p.get("kind") == "reply" and p["t"] > t_ask}
    spoken = [x["text"] for x in br.ev("() => window.__e2e.tts.spoken")][n_spoken:]
    out["after"] = {"says": len(says), "pushed_seqs": sorted(pushed), "phone_spoke": spoken}
    c.ok(f"{tag}: the next reply plays in the server's voice (every sentence's audio reached the player)",
         says and {s["seq"] for s in says} <= pushed and len(br.port("push")) > n_push, out["after"])
    c.ok(f"{tag}: … and the phone voice speaks none of it", not spoken, spoken[:2])
    br.shot(a.shots, f"{shot}-recovered")
    return out


def ref_cdp(pw, a, c: Checks, base: str, ref: RefTarget) -> dict:
    """A CDP-throttled slow link: the page talks to the reference server DIRECTLY (no
    shaper) and `Network.emulateNetworkConditions` makes the link EDGE-like (120 kbps,
    450 ms). Measured first: does it reach the WebSocket? (design-lowbw #5 found it did
    not in an earlier run.) Then failover to the phone voice and back when CDP lets go."""
    tag = "ref/cdp-throttle (PCM down)"
    out: dict = {}
    tid = new_thread(base)
    direct_ref = f"http://127.0.0.1:{ref.port}"
    br = Browser(pw, width=390, height=844, direct=base, via=direct_ref, no_opus_down=True)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        br.wait("() => window.__e2e.ws.filter((f) => f.dir === 'in' && f.m && f.m.type === 'pong').length >= 3", 12)
        before = _median(ping_rtts(br))
        cdp = br.ctx.new_cdp_session(br.page)
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 450,
                                                      "downloadThroughput": 120_000 / 8, "uploadThroughput": 120_000 / 8})
        t_set = br.now()
        br.wait("(t) => window.__e2e.ws.filter((f) => f.dir === 'in' && f.m && f.m.type === 'pong' && f.t > t).length >= 2", 15, t_set)
        idle = _median(ping_rtts(br, t_set))
        out["rtt_ms"] = {"before": before, "cdp_idle": idle}
        # CDP's latency reaches a WebSocket only in part (178 ms idle for a 450 ms setting in
        # one run), its throughput cap fully (7 s RTT behind PCM16 audio): any clear rise counts.
        throttled = idle is not None and before is not None and idle >= max(50.0, 10 * before)
        c.ok(f"{tag}: CDP throttling reaches the voice WebSocket (heartbeat RTT before → under CDP)", throttled,
             out["rtt_ms"])
        if not throttled:
            c.skip(f"{tag}: CDP-throttled failover", "CDP did not slow the WebSocket in this Chromium; "
                   "ref/failback drives the same path through the shaping proxy")
            return out

        def heal() -> None:
            cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 0,
                                                          "downloadThroughput": -1, "uploadThroughput": -1})
        out.update(_failover_and_back(br, a, c, tag, ref, heal=heal, shot="v2-390-cdp"))
        out["rtt_ms"]["reply_under_cdp"] = _median([x for x in ping_rtts(br, t_set)])
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def ref_failback(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    """The same through the shaping proxy: 3g-weak, then the live link turns healthy."""
    tag = "ref/failback (shaper 3g-weak → healthy, PCM down)"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via, no_opus_down=True)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=40):
            return out
        out.update(_failover_and_back(br, a, c, tag, ref, heal=lambda: ref.relink("baseline"), shot="v2-390-shaper"))
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def _sentence_audio(br: Browser) -> list[dict]:
    """Every reply sentence whose audio reached the player, attributed to the socket whose
    `ready` came last before the push: [{sock, gen, seq, text, t}] in push order."""
    readies = sorted((f["t"], f["sock"]) for f in br.frames("in", "ready"))
    says = {(f["sock"], f["m"].get("gen"), f["m"].get("seq")): f["m"].get("text") for f in br.frames("in", "say")
            if f["m"].get("kind") == "reply"}
    seen: dict = {}
    for p in br.port("push"):
        if p.get("kind") != "reply":
            continue
        sock = max((s for t, s in readies if t <= p["t"]), default=0)
        k = (sock, p.get("gen"), p.get("seq"))
        if k not in seen:
            seen[k] = {"sock": sock, "gen": k[1], "seq": k[2], "text": says.get(k), "t": p["t"]}
    return sorted(seen.values(), key=lambda x: x["t"])


def _check_resume_anchor(c: Checks, tag: str, starts: list[dict], ap) -> None:
    """review-T7b H1: the session speaks only replies stored AFTER `start.after`, so the
    reconnect that resumes a cut reply must not name that reply as `after` (nor among
    `after_ids`); it names the user row the reply answers."""
    st = next((x for x in starts if x.get("after_partial")), None)
    mid = ap.get("message_id") if isinstance(ap, dict) else None
    ok = bool(st) and mid is not None and st.get("after") != mid and mid not in (st.get("after_ids") or [])
    c.ok(f"{tag}: the reconnect `start` anchors `after` before the cut reply (its user row), not on it",
         ok, {k: (st or {}).get(k) for k in ("after", "after_ids", "after_partial")})


def ref_drop(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    """The socket drops in the middle of a reply and the signal stays away past 60 s:
    offline → waiting → back, and the reply resumes where the listener was — no sentence
    whose audio had fully arrived is spoken again, and no caption appears twice."""
    tag = "ref/drop mid-reply"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        br.say("short")
        br.wait("() => window.__e2e.captions.length >= 2", 30)
        time.sleep(0.6)
        out["captions_at_cut"] = [x["text"] for x in br.ev("() => window.__e2e.captions")]
        ends0 = [(f["m"].get("seq")) for f in br.frames("in", "audio_end")]
        ref.blackout(a.drop_s)
        ref.cut()
        got = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'offline'", 15)
        c.ok(f"{tag}: the drop shows offline at once", got, br.state())
        got = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'waiting'", 75)
        c.ok(f"{tag}: after 60 s without signal the screen says \"{EN['drive']['state_waiting']}\"",
             got and br.word() == EN["drive"]["state_waiting"], br.word())
        hint = br.ev("() => (document.querySelector('.drive-hint') || {}).textContent || ''")
        c.ok(f"{tag}: … with its own hint, not the interrupt hint of a reply that went silent",
             hint == EN["drive"]["hint_waiting"], hint)
        br.shot(a.shots, "v2-390-drop-waiting")
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.sock > 0 && f.m && f.m.type === 'ready')", 60)
        c.ok(f"{tag}: the signal returns and a new socket reaches ready", got, br.state())
        starts = [f["m"] for f in br.frames("out", "start") if f["sock"] > 0]
        ap = next((s.get("after_partial") for s in starts if s.get("after_partial")), None)
        out["after_partial"] = ap
        out["audio_end_before_cut"] = ends0
        c.ok(f"{tag}: the reconnect `start` names the reply in flight and how many sentences were heard (after_partial)",
             isinstance(ap, dict) and ap.get("message_id") and isinstance(ap.get("sentences"), int) and ap["sentences"] >= 1,
             {"after_partial": ap, "audio_end_seqs_before_cut": ends0})
        _check_resume_anchor(c, tag, starts, ap)
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.sock > 0 && f.m && f.m.type === 'say_end')", 90)
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.sock > 0 && f.m && f.m.type === 'playback_done')", 90)
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 30)
        resume = [x["resume"] for x in ref.log if x.get("thread") == tid and "resume" in x]
        out["server_resume"] = resume
        c.ok(f"{tag}: the server resumed the reply from that sentence", resume and ap and resume[-1]["from_sentence"] == ap["sentences"],
             resume[-1:])
        audio = _sentence_audio(br)
        out["audio"] = [[x["sock"], x["seq"], (x["text"] or "")[:40]] for x in audio]
        first = [x["text"] for x in audio if x["sock"] == 0]
        second = [x["text"] for x in audio if x["sock"] > 0]
        complete = first[:ap["sentences"]] if ap else []
        again = [t for t in second if t in complete]
        c.ok(f"{tag}: no sentence whose audio had fully arrived is spoken again after the resume", not again and second,
             {"again": again, "first_socket": len(first), "second_socket": len(second)})
        replay_cut = [t for t in second if t in first and t not in complete]
        out["cut_sentence_restarted"] = replay_cut
        th = ref.mod.WORLD.threads.get(tid)
        rep = th.latest if th else None
        want = list(rep.sentences) if rep else []
        heard = []
        for t in first + second:
            if t not in heard:
                heard.append(t)
        c.ok(f"{tag}: every sentence of the reply reached the listener, in order", want and heard == want,
             {"want": len(want), "heard": len(heard)})
        caps = [x["text"] for x in br.ev("() => window.__e2e.captions")]
        dups = sorted({t for t in caps if caps.count(t) > 1})
        out["captions"] = len(caps)
        c.ok(f"{tag}: no caption appears twice (a sentence cut mid-way and restarted is captioned once)", not dups,
             {"dups": dups, "cut_restarted": replay_cut})
        shown = caption_texts(br)
        c.ok(f"{tag}: the reply on screen reads as one reply (no repeated sentence)", len(shown) == len(set(shown)),
             shown[-4:])
        br.shot(a.shots, "v2-390-drop-resumed")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def _active(br: Browser) -> dict:
    return br.ev("""() => { const a = document.activeElement; const o = document.querySelector('.drive-overlay');
        const s = document.querySelector('.drive-sheet');
        return { cls: a ? String(a.className) : '', id: a ? a.id : '', inOverlay: !!(o && a && o.contains(a)),
                 inSheet: !!(s && a && s.contains(a)) }; }""")


def ref_keys(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    """The keyboard: Space pauses/resumes the microphone (also while a reply plays, where
    it must not stop the reply), Esc closes the quick sheet first and then hangs up, Tab
    stays inside the screen (and inside the sheet), focus goes back to the mic button."""
    tag = "ref/keyboard (1366)"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=1366, height=900, direct=base, via=via, mobile=False)
    p = br.page
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=30):
            return out
        act = _active(br)
        c.ok(f"{tag}: focus moves into the voice chat screen (the picture)", "drive-ring" in act["cls"], act)
        trail = []
        for _ in range(12):
            p.keyboard.press("Tab")
            trail.append(_active(br))
        for _ in range(12):
            p.keyboard.press("Shift+Tab")
            trail.append(_active(br))
        c.ok(f"{tag}: Tab and Shift+Tab stay inside the screen (focus trap)", all(x["inOverlay"] for x in trail),
             sorted({x["cls"] for x in trail}))
        # Space with the focus where the screen put it (the picture) and while listening.
        br.ev("() => document.querySelector('.drive-ring').focus()")
        n = len(br.frames("out"))
        p.keyboard.press("Space")
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'paused'", 3)
        sent = [f["m"]["type"] for f in br.frames("out")[n:] if f["m"]["type"] not in ("ping", "heard")]
        c.ok(f"{tag}: Space pauses the microphone (sends only `pause`)", sent == ["pause"] and br.state() == "paused",
             {"sent": sent, "state": br.state()})
        n = len(br.frames("out"))
        p.keyboard.press("Space")
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 3)
        sent = [f["m"]["type"] for f in br.frames("out")[n:] if f["m"]["type"] not in ("ping", "heard")]
        c.ok(f"{tag}: Space again resumes listening", br.state() == "listening" and "start" in sent,
             {"sent": sent, "state": br.state()})
        # Space while the reply plays: the mic only (experience M7), the reply goes on.
        br.say("short")
        br.wait("() => window.__e2e.captions.length >= 1", 25)
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'speaking'", 5)
        br.ev("() => document.querySelector('.drive-ring').focus()")
        n = len(br.frames("out"))
        n_flush = len(br.port("flush"))
        p.keyboard.press("Space")
        time.sleep(0.4)
        sent = [f["m"]["type"] for f in br.frames("out")[n:] if f["m"]["type"] not in ("ping", "heard")]
        word = br.word()
        out["space_while_speaking"] = {"sent": sent, "word": word}
        c.ok(f"{tag}: Space while a reply plays pauses only the microphone — the reply keeps playing",
             sent == ["pause"] and len(br.port("flush")) == n_flush and word == EN["drive"]["paused_speaking"],
             out["space_while_speaking"])
        br.shot(a.shots, "v2-1366-paused-still-speaking")
        p.keyboard.press("Space")
        time.sleep(0.3)
        c.ok(f"{tag}: … and Space resumes the microphone", br.state() in ("speaking", "listening")
             and br.word() != EN["drive"]["paused_speaking"], br.word())
        # The quick sheet: focus inside, Tab trapped, Esc closes the sheet (not the call).
        p.click(".drive-gear")
        p.wait_for_selector(".drive-sheet", timeout=5000)
        p.wait_for_timeout(150)
        act = _active(br)
        trail = []
        for _ in range(10):
            p.keyboard.press("Tab")
            trail.append(_active(br)["inSheet"])
        c.ok(f"{tag}: the quick sheet takes the focus and keeps Tab inside", act["inSheet"] and all(trail),
             {"first": act, "trail": trail})
        p.keyboard.press("Escape")
        p.wait_for_timeout(200)
        act = _active(br)
        c.ok(f"{tag}: Esc closes the quick sheet only; focus returns to its gear",
             br.ev("() => !document.querySelector('.drive-sheet') && !!document.querySelector('.drive-overlay')")
             and "drive-gear" in act["cls"], act)
        # All settings… (SPEC §7.3): Settings opens OVER the screen, the microphone
        # pauses, nothing hangs up; closing Settings (Esc) leaves the voice chat open.
        p.click(".drive-gear")
        p.wait_for_selector(".drive-sheet", timeout=5000)
        n = len(br.frames("out"))
        n_close = len(br.ev("() => window.__e2e.ws.filter((f) => f.dir === 'close')"))
        p.click("#drive-sheet-all")
        got = br.wait("() => { const b = document.getElementById('botmanager-backdrop'); const v = document.getElementById('spane-voice');"
                      " return b && !b.classList.contains('hidden') && v && v.offsetParent !== null; }", 10)
        p.wait_for_timeout(400)
        over = br.ev("""() => { const m = document.querySelector('#botmanager-backdrop .modal') || document.getElementById('botmanager-backdrop');
            const r = m.getBoundingClientRect(); const x = r.left + r.width / 2, y = r.top + Math.min(r.height / 2, 120);
            const top = document.elementFromPoint(x, y);
            return { onTop: !!(top && document.getElementById('botmanager-backdrop').contains(top)),
                     overlay: !!document.querySelector('.drive-overlay'), state: document.querySelector('.drive-overlay')?.dataset.state,
                     inert: !!document.getElementById('botmanager-backdrop').closest('[inert]') }; }""")
        sent = [f["m"]["type"] for f in br.frames("out")[n:] if f.get("m") and f["m"]["type"] not in ("ping", "heard")]
        closes = len(br.ev("() => window.__e2e.ws.filter((f) => f.dir === 'close')")) - n_close
        out["all_settings"] = {**over, "sent": sent, "closes": closes}
        c.ok(f"{tag}: All settings… opens Settings › Voice over the screen (on top, usable), the microphone paused, the socket kept",
             got and over["onTop"] and not over["inert"] and over["overlay"] and over["state"] == "paused"
             and sent == ["pause"] and closes == 0, out["all_settings"])
        br.shot(a.shots, "v2-1366-all-settings-over")
        p.keyboard.press("Escape")
        p.wait_for_timeout(300)
        after = br.ev("""() => ({ modal: !document.getElementById('botmanager-backdrop').classList.contains('hidden'),
            overlay: !!document.querySelector('.drive-overlay'),
            focus: document.activeElement ? (document.activeElement.id || String(document.activeElement.className)) : null,
            focusInOverlay: !!(document.querySelector('.drive-overlay') && document.querySelector('.drive-overlay').contains(document.activeElement)) })""")
        out["after_settings"] = after
        closes = len(br.ev("() => window.__e2e.ws.filter((f) => f.dir === 'close')")) - n_close
        c.ok(f"{tag}: Esc closes Settings only; the voice chat stays open on the same socket",
             not after["modal"] and after["overlay"] and closes == 0, {**after, "closes": closes})
        br.ev("() => document.querySelector('.drive-ring') && document.querySelector('.drive-ring').focus()")
        p.keyboard.press("Escape")
        br.wait("() => !document.querySelector('.drive-overlay')", 5)
        act = br.ev("() => ({ id: document.activeElement && document.activeElement.id, overlay: !!document.querySelector('.drive-overlay') })")
        c.ok(f"{tag}: Esc hangs up and focus returns to the mic button", not act["overlay"] and act["id"] == "drive-btn", act)
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


# Computed styles that must match between the Voice pane and DisPatch's own Device pane.
STYLE_PROPS = ["fontFamily", "fontSize", "fontWeight", "color", "backgroundColor", "borderRadius",
               "borderTopColor", "borderTopWidth", "paddingTop", "paddingLeft", "lineHeight"]


def _pane_styles(br: Browser, pane: str) -> dict:
    return br.ev("""([pane, props]) => {
        const P = document.getElementById(pane); if (!P) return null;
        const pick = (sel) => { const n = P.querySelector(sel); if (!n) return null; const s = getComputedStyle(n);
            const o = {}; for (const k of props) o[k] = s[k]; return o; };
        return { lead: pick('.modal-hint'), title: pick('.dev-group-title'), body: pick('.dev-group-body'), row: pick('.dev-group-body > .dev-row'),
                 strong: pick('.dev-row-text > strong'), hint: pick('.dev-row-hint'), group: pick('.dev-group') }; }""",
                 [pane, STYLE_PROPS])


def theme_shots(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    """Light and dark, phone and desktop: Settings › Voice next to Settings › Device (the
    app's own pane of the same components), the voice chat screen listening, speaking with
    captions, and the quick sheet. Computed styles of the shared components must match."""
    out: dict = {}
    for theme in ("dark", "light"):
        for w, h in ((390, 844), (1366, 900)):
            tag = f"shots/{theme}/{w}"
            res: dict = {}
            br = Browser(pw, width=w, height=h, direct=base, mobile=w < 600, theme=theme)
            p = br.page
            try:
                p.goto(base + "/")
                br.wait("() => !!document.getElementById('drive-btn')", 40)
                got = br.ev("() => document.documentElement.getAttribute('data-theme')")
                c.ok(f"{tag}: the app is in its {theme} theme", (got == "light") == (theme == "light"), got)
                p.click("#manage-bots")
                p.wait_for_selector("#botmanager-backdrop:not(.hidden)", timeout=10000)
                p.click("#stab-device")
                p.wait_for_timeout(300)
                br.shot(a.shots, f"v2-{w}-{theme}-settings-device")
                dev = _pane_styles(br, "spane-device")
                p.click("#stab-voice")
                p.wait_for_selector("#drive-group-voices", timeout=20000)
                p.wait_for_timeout(400)
                br.shot(a.shots, f"v2-{w}-{theme}-settings-voice")
                voi = _pane_styles(br, "spane-voice")
                diffs = {}
                for part in ("lead", "title", "body", "row", "strong", "hint", "group"):
                    d, v = (dev or {}).get(part), (voi or {}).get(part)
                    if d is None or v is None:
                        diffs[part] = "missing" if (d is None) != (v is None) else None
                        continue
                    for k in STYLE_PROPS:
                        if d[k] != v[k]:
                            diffs[f"{part}.{k}"] = [d[k], v[k]]
                diffs = {k: v for k, v in diffs.items() if v}
                res["style_diffs"] = diffs
                c.ok(f"{tag}: Settings › Voice uses the Device pane's own group/row/title/hint styles (computed)", not diffs, diffs)
                sw = br.ev("() => { const p = document.getElementById('spane-voice'); return [p.scrollWidth, p.clientWidth]; }")
                c.ok(f"{tag}: no horizontal overflow on Settings › Voice", sw[0] <= sw[1] + 1, sw)
                br.ev("() => { const g = document.getElementById('drive-group-connection'); g && g.scrollIntoView({block: 'center'}); }")
                p.wait_for_timeout(200)
                br.shot(a.shots, f"v2-{w}-{theme}-settings-voice-lower")
                p.click("#bm-close")
            finally:
                br.close()
            # The voice chat screen (reference server, baseline link).
            tid = new_thread(base)
            br = Browser(pw, width=w, height=h, direct=base, via=via, mobile=w < 600, theme=theme)
            p = br.page
            try:
                if open_voice_chat(br, tid, c, tag, ready_timeout=30):
                    br.shot(a.shots, f"v2-{w}-{theme}-listening")
                    p.click(".drive-gear")
                    p.wait_for_selector(".drive-sheet", timeout=5000)
                    p.wait_for_timeout(300)
                    br.shot(a.shots, f"v2-{w}-{theme}-quick-sheet")
                    sheet = br.ev("""() => { const s = document.querySelector('.drive-sheet'); const r = s.getBoundingClientRect();
                        const rows = [...s.querySelectorAll('.drive-sheet-row, .dev-row')].map((n) => Math.round(n.getBoundingClientRect().height));
                        return { l: r.left, r: r.right, t: r.top, b: r.bottom, vw: innerWidth, vh: innerHeight, rows, sw: s.scrollWidth, cw: s.clientWidth }; }""")
                    res["sheet"] = sheet
                    c.ok(f"{tag}: the quick sheet fits the screen with no sideways scroll",
                         sheet["l"] >= -0.5 and sheet["r"] <= sheet["vw"] + 0.5 and sheet["sw"] <= sheet["cw"] + 1, sheet)
                    p.keyboard.press("Escape")
                    p.wait_for_timeout(150)
                    br.say("short")
                    br.wait("() => window.__e2e.captions.length >= 2", 25)
                    p.wait_for_timeout(400)
                    br.shot(a.shots, f"v2-{w}-{theme}-speaking")
                    p.keyboard.press("Escape")
                errs = br.script_errors()
                c.ok(f"{tag}: no script errors", not errs, errs[:3])
            finally:
                br.close()
            out[tag] = res
    # A source scan: the add-on's stylesheet names no colour of its own.
    css = (ADDON / "src/dispatch_voice/static/voice.css").read_text()
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    lit = re.findall(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgb|rgba|hsl|hsla|oklch|oklab)\(", body)
    named = re.findall(r"(?<![-\w])(?:white|black|red|green|blue|gray|grey)(?![-\w])", body)
    c.ok("voice.css: no literal colour (hex, rgb/hsl/oklch, named)", not lit and not named, {"lit": lit[:5], "named": named[:5]})
    # Radii and fonts: every literal radius is one DisPatch's own stylesheets already use
    # (pills 999px, avatars/inputs 10px, handles 2-3px, circles 50%); every other radius and
    # every font family is a theme token.
    static = a.backend.parent / "frontend/static"
    host_css = "\n".join(f.read_text() for f in [static / "app.css", *sorted((static / "ui-theme").glob("*.css"))]
                         if f.is_file())
    host_radii = set(re.findall(r"border-radius:\s*([0-9.]+(?:px|%))\s*[;}]", host_css))
    radii = re.findall(r"border-radius:\s*([^;}]+)", body)
    new_radii = sorted({v.strip() for r in radii for v in r.split() if re.fullmatch(r"[0-9.]+(?:px|%|rem|em)", v.strip())
                        and v.strip() not in host_radii and v.strip() != "0"})
    c.ok("voice.css: no new radius (literal radii are ones DisPatch's own CSS uses; the rest are --radius-* tokens)",
         not new_radii, {"new": new_radii, "host": sorted(host_radii)})
    fams = [f.strip() for f in re.findall(r"font-family:\s*([^;}]+)", body)]
    odd = [f for f in fams if not f.startswith("var(--font-") and f != "inherit"]
    c.ok("voice.css: no new font (every font-family is a --font-* token)", not odd, {"families": sorted(set(fams))})
    return out


RAW_KEY = re.compile(r"\b(?:drive|voice|voices|vtool|voice_tools|voice_absent)\.[a-z][a-z0-9_]+\b")


def locale_pages(pw, a, c: Checks, base: str, via: str, ref: RefTarget) -> dict:
    """The add-on in other UI languages at phone width: Settings › Voice and the voice chat
    screen with its quick sheet. No sideways scroll, no untranslated key showing, RTL for
    Arabic, the English-only first-open line for a non-English UI (SPEC H3)."""
    out: dict = {}
    for lang in ("de", "ar", "ja"):
        tag = f"locale/{lang}/390"
        br = Browser(pw, width=390, height=844, direct=base, via=via, mobile=True)
        br.ctx.add_init_script(f"try {{ localStorage.setItem('dispatch-lang', '{lang}'); }} catch (e) {{}}")
        p = br.page
        try:
            p.goto(base + "/")
            br.wait("() => !!document.getElementById('drive-btn')", 40)
            got = br.ev("() => [document.documentElement.lang, document.documentElement.dir]")
            c.ok(f"{tag}: the app runs in {lang}" + (" (rtl)" if lang == "ar" else ""),
                 got[0].startswith(lang) and (got[1] == "rtl") == (lang == "ar"), got)
            p.click("#manage-bots")
            p.wait_for_selector("#botmanager-backdrop:not(.hidden)", timeout=10000)
            p.click("#stab-voice")
            p.wait_for_selector("#drive-group-voices", timeout=20000)
            p.wait_for_timeout(400)
            txt = p.inner_text("#spane-voice")
            sw = br.ev("() => { const p = document.getElementById('spane-voice'); return [p.scrollWidth, p.clientWidth]; }")
            raw = sorted(set(RAW_KEY.findall(txt)))
            c.ok(f"{tag}: Settings › Voice has no sideways scroll and no untranslated key", sw[0] <= sw[1] + 1 and not raw,
                 {"scroll": sw, "raw": raw[:5]})
            br.shot(a.shots, f"v2-390-{lang}-settings-voice")
            br.ev("() => { const g = document.getElementById('drive-group-connection'); g && g.scrollIntoView({block: 'center'}); }")
            p.wait_for_timeout(200)
            br.shot(a.shots, f"v2-390-{lang}-settings-voice-lower")
            p.click("#bm-close")
        finally:
            br.close()
        tid = new_thread(base)
        br = Browser(pw, width=390, height=844, direct=base, via=via, mobile=True)
        br.ctx.add_init_script(f"try {{ localStorage.setItem('dispatch-lang', '{lang}'); }} catch (e) {{}}")
        p = br.page
        try:
            if open_voice_chat(br, tid, c, tag, ready_timeout=30):
                bundle = json.loads((ADDON / f"src/dispatch_voice/static/locales/{lang}.json").read_text())
                tip = br.ev("() => { const n = document.querySelector('.drive-tip'); return n && !n.hidden ? n.textContent : null; }")
                c.ok(f"{tag}: the first open says voice chat is English-only, in {lang}", tip == bundle["drive"]["english_only"], tip)
                br.shot(a.shots, f"v2-390-{lang}-listening")
                p.click(".drive-gear")
                p.wait_for_selector(".drive-sheet", timeout=5000)
                p.wait_for_timeout(300)
                sheet = br.ev("""() => { const s = document.querySelector('.drive-sheet'); const r = s.getBoundingClientRect();
                    return { l: r.left, r: r.right, vw: innerWidth, sw: s.scrollWidth, cw: s.clientWidth,
                             dsw: document.documentElement.scrollWidth, text: s.innerText }; }""")
                raw = sorted(set(RAW_KEY.findall(sheet.pop("text"))))
                c.ok(f"{tag}: the quick sheet fits with no sideways scroll and no untranslated key",
                     sheet["l"] >= -0.5 and sheet["r"] <= sheet["vw"] + 0.5 and sheet["sw"] <= sheet["cw"] + 1
                     and sheet["dsw"] <= sheet["vw"] + 1 and not raw, {**sheet, "raw": raw[:5]})
                br.shot(a.shots, f"v2-390-{lang}-quick-sheet")
                p.keyboard.press("Escape")
                p.keyboard.press("Escape")
            errs = br.script_errors()
            c.ok(f"{tag}: no script errors", not errs, errs[:3])
        finally:
            br.close()
    return out


def _subseq(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(x == y for y in it) for x in needle)


# --------------------------------------------------------------------------- #
# target: dispatch (the server as it is today, real speech models)
# --------------------------------------------------------------------------- #
def run_dispatch(pw, a, c: Checks, out: Path) -> dict:
    res: dict = {}
    base = f"http://127.0.0.1:{a.port_base}"
    if a.scenarios & {"settings", "lab", "conversation", "undo", "bargein", "noise", "cdp", "drop", "failback"}:
        res.update(_run_dispatch_main(pw, a, c, out, base))
    if "livelike" in a.scenarios:
        # Its own server (a PIN, entry points, Kokoro) on the same ports, after the first one stopped.
        res["livelike"] = dispatch_livelike(pw, a, c, out)
    return res


def _run_dispatch_main(pw, a, c: Checks, out: Path, base: str) -> dict:
    res: dict = {}
    proc = llm = None
    try:
        proc, llm, data = start_scratch_dispatch(a.port_base, a.port_base + 1, out, a.backend, prewarm=True)
        res["data"] = str(data)
        if "settings" in a.scenarios:
            for w, h in ((390, 844), (1366, 900)):
                res[f"settings_{w}"] = dispatch_settings(pw, a, c, base, w, h)
        if "lab" in a.scenarios:
            for w, h in ((390, 844), (1366, 900)):
                res[f"lab_{w}"] = dispatch_lab(pw, a, c, base, w, h)
        if "conversation" in a.scenarios:
            res["conversation"] = dispatch_conversation(pw, a, c, base, llm)
            res["overlay_1366"] = dispatch_overlay_desktop(pw, a, c, base)
        if "undo" in a.scenarios:
            res["undo"] = dispatch_undo(pw, a, c, base, llm)
        if "bargein" in a.scenarios:
            res["barge_in"] = dispatch_barge_in(pw, a, c, base)
        if "noise" in a.scenarios:
            res["noise"] = dispatch_noise(pw, a, c, base, llm)
        if "cdp" in a.scenarios:
            res["cdp"] = dispatch_cdp(pw, a, c, base)
        if a.scenarios & {"drop", "failback"}:
            sh = Shaper(Bg(), a.port_base + 3, a.port_base)
            via = f"http://127.0.0.1:{a.port_base + 3}"
            try:
                if "drop" in a.scenarios:
                    sh.shape("baseline")
                    res["drop"] = dispatch_drop(pw, a, c, base, via, sh)
                if "failback" in a.scenarios:
                    sh.shape("3g-weak")
                    res["failback"] = dispatch_failback(pw, a, c, base, via, sh)
            finally:
                sh.stop()
    finally:
        stop_proc(proc)
        if llm is not None:
            llm.stop()
    return res


def dispatch_settings(pw, a, c: Checks, base: str, width: int, height: int) -> dict:
    tag = f"dispatch/{width}"
    out: dict = {}
    br = Browser(pw, width=width, height=height, direct=base, mobile=False)
    p = br.page
    try:
        p.goto(base + "/")
        if not br.wait("() => !!document.getElementById('drive-btn')", 40):
            c.ok(f"{tag}: the add-on mounted (mic button exists)", False, br.errors[-3:])
            return out
        c.ok(f"{tag}: the add-on mounted (mic button exists)", True)
        p.click("#manage-bots")
        p.wait_for_selector("#botmanager-backdrop:not(.hidden)", timeout=10000)
        p.click("#stab-voice")
        p.wait_for_selector("#drive-group-switch", timeout=20000)
        flipped = False
        if br.ev("() => { const i = document.getElementById('drive-switch-input'); return !!i && !i.checked && !i.disabled; }"):
            # Voice off: switch it on here, then the Bots tab must show the pills in this
            # same Settings session (review T7a M1).
            p.click("#drive-switch-input")
            flipped = True
        p.wait_for_selector("#drive-group-voices", timeout=20000)
        p.wait_for_timeout(500)
        out["switched_on"] = flipped
        groups = br.ev("""() => [...document.querySelectorAll('#spane-voice section.dev-group, #spane-voice details.dev-group')]
            .map((g) => [g.id, [...g.querySelectorAll(':scope > .dev-group-body > .dev-row')].filter((r) => !r.hidden).length])""")
        out["groups"] = groups
        c.ok(f"{tag}: Settings › Voice: 6 groups + Status", len(groups) == 7 and groups[-1][0] == "drive-group-status", groups)
        # Conditional rows (§7.1): the narrator row is rendered but hidden outside the narrator
        # mode; the speed row exists only when built-in voices are installed (Kokoro plugin).
        all_rows = br.ev("""() => [...document.querySelectorAll('#spane-voice section.dev-group > .dev-group-body > .dev-row')].length""")
        builtins = br.ev("() => fetch('/api/voice/status').then((r) => r.json()).then((j) => (j.builtin_voices || []).length)")
        want = 19 if builtins else 18
        out["rows"] = {"rendered": all_rows, "builtin_voices": builtins}
        c.ok(f"{tag}: Settings › Voice: the six groups hold the §7.1 rows (19; 18 without built-in voices)",
             all_rows == want, out["rows"])
        conn = br.ev("""() => { const g = document.getElementById('drive-group-connection'); if (!g) return null;
            return { title: g.querySelector('.dev-group-title').textContent,
                     seg: [...g.querySelectorAll('#drive-slow .drive-seg-btn')].map((b) => b.textContent.trim()),
                     phone: !!g.querySelector('#drive-phone-voice') }; }""")
        c.ok(f"{tag}: the Connection group: four slow-link choices and the phone-voice row",
             conn and len(conn["seg"]) == 4 and conn["phone"], conn)
        sw = br.ev("() => { const p = document.getElementById('spane-voice'); return [p.scrollWidth, p.clientWidth]; }")
        c.ok(f"{tag}: no horizontal overflow on Settings › Voice", sw[0] <= sw[1] + 1, sw)
        c.ok(f"{tag}: no 'Drive' wording on the Voice page", "Drive" not in p.inner_text("#spane-voice"))
        br.shot(a.shots, f"dispatch-{width}-settings-voice")
        br.ev("() => { const g = document.getElementById('drive-group-connection'); g && g.scrollIntoView({block: 'center'}); }")
        p.wait_for_timeout(200)
        br.shot(a.shots, f"dispatch-{width}-settings-connection")
        # Bots tab: the pill and its popover, inside the pane (review T7a M1/M2)
        p.click("#stab-bots")
        br.wait("() => document.querySelectorAll('.bm-voice-pill').length > 0", 8)
        pills = p.locator(".bm-voice-pill").count()
        c.ok(f"{tag}: Voice pills on the Bots tab in the same Settings session"
             + (" (voice switched on just now)" if flipped else ""), pills > 0, pills)
        if pills:
            p.locator(".bm-voice-pill").first.click()
            p.wait_for_selector(".bm-voice-pop", timeout=5000)
            p.wait_for_timeout(150)
            r = br.ev("""() => { const a = document.querySelector('.bm-voice-pop').getBoundingClientRect();
                const b = document.getElementById('spane-bots').getBoundingClientRect();
                return { l: a.left, r: a.right, pl: b.left, pr: b.right, vw: innerWidth }; }""")
            c.ok(f"{tag}: the Bots popover stays inside the pane and the screen (left edge too)",
                 r["l"] >= max(0, r["pl"]) - 0.5 and r["r"] <= min(r["vw"], r["pr"]) + 0.5, r)
            br.shot(a.shots, f"dispatch-{width}-bots-popover")
            p.keyboard.press("Escape")
        p.click("#bm-close")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_lab(pw, a, c: Checks, base: str, width: int, height: int) -> dict:
    """The Voice lab in the real shell: its rows, no overflow, a screenshot."""
    tag = f"dispatch/{width}/lab"
    out: dict = {}
    br = Browser(pw, width=width, height=height, direct=base, mobile=False)
    p = br.page
    try:
        p.goto(base + "/")
        br.wait("() => !!document.getElementById('drive-btn')", 40)
        if not br.wait("() => !!document.querySelector('[data-tool=\"voice-lab\"]') || !!document.getElementById('tools-menu-btn')", 15):
            c.ok(f"{tag}: the Voice lab tool is listed", False)
            return out
        if p.locator('[data-tool="voice-lab"]:visible').count() == 0:
            p.click("#tools-menu-btn")
        p.locator('[data-tool="voice-lab"]').first.click()
        got = br.wait("() => !!document.getElementById('vtool-learned')", 20)
        ids = br.ev("""() => [...document.querySelectorAll('[id^="vtool-"]')].map((n) => n.id)""")
        out["rows"] = ids
        c.ok(f"{tag}: the Voice lab opens with its rows (learned pause, last voice chat, engines)",
             got and "vtool-learned" in ids and any(i.startswith("vtool-last") for i in ids), ids[:20])
        acks = br.ev("() => document.querySelectorAll('.vtool-acks-row').length")
        voices = br.ev("() => fetch('/api/voice/status').then((r) => r.json()).then((j) => Object.keys(j.acks || {}).length)")
        c.ok(f"{tag}: one acknowledgement row (with Rebuild) per voice the server reports", acks == voices,
             {"rows": acks, "status_voices": voices})
        root = br.ev("""() => { const r = document.querySelector('.voice-tool-root') || document.body;
            return [r.scrollWidth, r.clientWidth]; }""")
        c.ok(f"{tag}: no horizontal overflow in the Voice lab", root[0] <= root[1] + 1, root)
        br.shot(a.shots, f"dispatch-{width}-voice-lab")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_overlay_desktop(pw, a, c: Checks, base: str) -> dict:
    """The voice chat screen in the real shell at desktop width (1366 px)."""
    tag = "dispatch/1366"
    tid = new_thread(base)
    br = Browser(pw, width=1366, height=900, direct=base, mobile=False)
    try:
        br.page.goto(f"{base}/?popout=1&bot=voicetest&thread={tid}")
        got = br.wait(r"""() => performance.getEntriesByType('resource')
            .filter((e) => /voice-(player-)?worklet\.js/.test(e.name)).length >= 2""", 8)
        pre = br.ev(r"""() => performance.getEntriesByType('resource').filter((e) => /worklet\.js/.test(e.name))
            .map((e) => ({ name: e.name.split('/').pop(), start: Math.round(e.startTime) }))""")
        c.ok(f"{tag}: both audio worklets are prefetched before the first tap (warm cache for a slow link)", got, pre)
        if open_voice_chat(br, tid, c, tag, ready_timeout=120):
            col = br.ev("() => { const r = document.querySelector('.drive-col').getBoundingClientRect(); return [r.left, r.right, innerWidth]; }")
            c.ok(f"{tag}: the voice chat column is centred and fits", col[0] >= 0 and col[1] <= col[2], col)
            br.shot(a.shots, "dispatch-1366-listening")
            br.page.click(".drive-gear")
            br.page.wait_for_selector(".drive-sheet", timeout=5000)
            br.page.wait_for_timeout(200)
            br.shot(a.shots, "dispatch-1366-quick-sheet")
            br.page.keyboard.press("Escape")
            br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return {}


def dispatch_conversation(pw, a, c: Checks, base: str, llm) -> dict:
    tag = "dispatch/390"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        url = br.ev("() => window.__e2e.urls[0]")
        ready = br.frames("in", "ready")[0]["m"]
        proto = int(ready.get("proto") or 2)
        say_cap = proto >= 3 or ready.get("say") is True
        out.update({"url": url, "proto": proto, "say": say_cap})
        c.ok(f"{tag}: the client offers v3 in the URL (v=3, s, codec lists)", "v=3" in url and "s=" in url, url)
        br.shot(a.shots, "dispatch-390-listening")
        # ---- a real turn: Parakeet hears the clip; the reply plays through the player worklet
        br.say("short")
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'transcript' && f.m.text)", 40)
        tr = br.frames("in", "transcript")
        out["transcript"] = tr[0]["m"]["text"] if tr else None
        c.ok(f"{tag}: speech → transcript (real Parakeet)", got and "calendar" in (out["transcript"] or "").lower(), out["transcript"])
        got = br.wait("() => window.__e2e.captions.length > 0", 60)
        eot = br.frames("in", "eot")
        caps = br.ev("() => window.__e2e.captions")
        if got and eot:
            out["eot_to_first_caption_ms"] = round(caps[0]["t"] - eot[0]["t"])
        c.ok(f"{tag}: the reply plays through the player worklet and its caption appears as it plays", got,
             {"captions": [x["text"] for x in caps][:2], "eot_to_first_caption_ms": out.get("eot_to_first_caption_ms")})
        pushes = br.port("push")
        c.ok(f"{tag}: reply audio was pushed to the player worklet", len(pushes) > 0, len(pushes))
        br.shot(a.shots, "dispatch-390-speaking")
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'playback_done')", 120)
        c.ok(f"{tag}: playback_done goes out when the reply has played", got)
        got = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 60)
        c.ok(f"{tag}: back to listening after the reply", got, br.state())
        if say_cap:
            says = {f["m"]["seq"]: f["t"] for f in br.frames("in", "say") if f["m"].get("kind") == "reply"}
            hdrs = {}
            for f in br.frames("in", "audio"):
                hdrs.setdefault(f["m"].get("seq"), f["t"])
            first = {b["down"]["seq"]: b["t"] for b in br.bins("in") if b.get("down")} if proto >= 3 else hdrs
            c.ok(f"{tag}: the server announces each sentence with `say` before its audio (C.5 #1)",
                 says and all(says.get(q, 1e18) <= t for q, t in first.items() if q in says), {"says": len(says)})
            heard = br.frames("out", "heard")
            c.ok(f"{tag}: `heard` frames go out while the reply plays", bool(heard), heard[-1:])
        if proto >= 3:
            out["codecs"] = {"in": ready.get("codec_in"), "out": ready.get("codec_out")}
            ups = [b for b in br.bins("out") if b.get("up")]
            odd = [b for b in br.bins("out") if not b.get("up")]
            c.ok(f"{tag}: v3 server: every uplink message is wire3 with offsets, in the negotiated codec",
                 ups and not odd and ready.get("codec_in") in ("opus", "pcm16"),
                 {"wire3": len(ups), "other": len(odd), **out["codecs"]})
            dn = [b for b in br.bins("in") if b.get("down")]
            c.ok(f"{tag}: v3 server: reply audio arrives as wire3 downlink ({ready.get('codec_out')})",
                 dn and not [b for b in br.bins("in") if not b.get("down")], {"down": len(dn)})
            ends = br.frames("in", "audio_end")
            c.ok(f"{tag}: v3 server: each sentence's audio is closed by audio_end (T6-2)", bool(ends), len(ends))
        if proto < 3:
            c.skip(f"{tag}: v3 wire (wire3 Opus both ways, audio_end, link/probe, ring replay)",
                   "the server sends no ready.proto yet (T3 slice 4 not landed); the same client paths are "
                   "proven against the v3 reference server (--target ref)")
            raw = [b for b in br.bins("out") if b.get("bin") == 640 and not b.get("up")]
            c.ok(f"{tag}: v2 server: the uplink is raw 20 ms PCM16 frames", len(raw) > 10, len(raw))
        # ---- tap-to-stop during the next (long) reply
        n_caps0 = len(br.ev("() => window.__e2e.captions"))
        br.say("medium")
        br.wait("(n) => window.__e2e.captions.length > n", 90, n_caps0)
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'speaking'", 10)
        n_flush = len(br.port("flush"))
        time.sleep(1.0)
        out["before_tap"] = {"state": br.state(), "word": br.word(), "hint": br.ev("() => document.querySelector('.drive-hint').textContent"),
                             "captions": [[round(x["t"]), x["text"][:30]] for x in br.ev("() => window.__e2e.captions")],
                             "ins": [[round(f["t"]), f["m"]["type"], f["m"].get("state") or f["m"].get("seq")] for f in br.frames("in") if f["m"]["type"] not in ("pong",)][-25:],
                             "now": round(br.now())}
        br.page.click(".drive-ring")
        time.sleep(0.3)
        stop = br.frames("out", "stop_playback")
        c.ok(f"{tag}: the picture tap while speaking sends stop_playback", bool(stop), stop[-1:])
        if say_cap:
            c.ok(f"{tag}: … carrying what was heard (C.3a)", bool(stop) and isinstance(stop[-1]["m"].get("heard"), dict),
                 stop[-1:])
        c.ok(f"{tag}: … and flushes the player at once", len(br.port("flush")) > n_flush)
        n_caps = len(br.ev("() => window.__e2e.captions"))
        time.sleep(3.0)
        c.ok(f"{tag}: no further captions from the stopped reply", len(br.ev("() => window.__e2e.captions")) == n_caps)
        # ---- Pause, resume, hang up
        n_out = len(br.frames("out"))
        br.page.click(".drive-pause")
        time.sleep(0.3)
        sent = [f["m"]["type"] for f in br.frames("out")[n_out:] if f["m"]["type"] not in ("ping", "heard")]
        c.ok(f"{tag}: Pause sends only `pause`", sent == ["pause"], sent)
        br.shot(a.shots, "dispatch-390-paused")
        br.page.click(".drive-pause")
        br.page.keyboard.press("Escape")
        br.wait("() => !document.querySelector('.drive-overlay')", 5)
        c.ok(f"{tag}: Esc hangs up", br.ev("() => !document.querySelector('.drive-overlay')"))
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_barge_in(pw, a, c: Checks, base: str) -> dict:
    """Talking over a reply on the REAL session (Parakeet, Silero, the TTS): the optimistic
    duck, then the server's barge_in; the player is flushed to the new gen, `heard` goes out
    with it, nothing more of the cut reply is captioned, and the interruption is transcribed."""
    tag = "dispatch/390/barge-in"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base)
    # The optimistic duck runs only at the "high" interrupt sensitivity since 1.5.0.
    br.ctx.add_init_script("try { localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, interruptSensitivity: 'high' })); } catch (e) {}")
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        br.say("medium")
        br.wait("() => window.__e2e.captions.length >= 1", 90)
        br.wait("() => { const c = window.__e2e.captions; return c.length && performance.now() - c[0].t > 1500; }", 15)
        onset = clip_onset_s("interrupt")
        n_pause = len(br.port("pause"))
        inj = br.say("interrupt")
        t_on = inj["at"] + onset * 1000
        got = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'barge_in' && f.t > t)", 8, inj["at"])
        bi = [f for f in br.frames("in", "barge_in") if f["t"] > inj["at"]]
        pause = br.port("pause")[n_pause:]
        out["duck_ms"] = round(pause[0]["t"] - t_on) if pause else None
        out["barge_ms"] = round(bi[0]["t"] - t_on) if bi else None
        ready = br.frames("in", "ready")[0]["m"]
        out["proto"] = int(ready.get("proto") or 2)
        out["duck_local"] = bool([f for f in br.frames("out", "duck_local") if f["t"] > inj["at"]])
        if out["proto"] >= 3:
            c.ok(f"{tag}: the optimistic duck pauses the reply within 200 ms of the speech onset",
                 out["duck_ms"] is not None and out["duck_ms"] <= 200 and out["duck_local"], out)
        else:
            c.skip(f"{tag}: the optimistic duck ≤ 200 ms",
                   f"the server speaks proto {out['proto']}: the client ducks optimistically only on v3 "
                   f"(T7b-1); the server's own duck paused the reply after {out['duck_ms']} ms")
        c.ok(f"{tag}: the real session confirms the interruption with barge_in", got, out)
        if bi:
            gen = bi[0]["m"]["gen"]
            flush = [p for p in br.port("flush") if p["t"] >= bi[0]["t"] - 5]
            c.ok(f"{tag}: the player is flushed to the new gen", any(p.get("gen") == gen for p in flush), flush[:2])
            hd = [f["m"] for f in br.frames("out", "heard") if f["t"] >= bi[0]["t"] - 5]
            c.ok(f"{tag}: `heard` (what was heard of the cut reply) goes out with the barge-in", bool(hd), hd[:1])
            n_caps = len(br.ev("() => window.__e2e.captions"))
            got = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'transcript' && f.m.text && f.t > t)", 30, bi[0]["t"])
            tr = [f["m"]["text"] for f in br.frames("in", "transcript") if f["t"] > bi[0]["t"]]
            out["transcript"] = tr
            c.ok(f"{tag}: the interruption is transcribed as the next turn", got and any("stop" in x.lower() or "meant" in x.lower() for x in tr), tr)
            first_new = next((f["t"] for f in br.frames("in", "say") if f["m"].get("gen") == gen and f["m"].get("kind") == "reply"), None)
            stray = [cp for cp in br.ev("() => window.__e2e.captions")[n_caps:] if first_new is None or cp["t"] < first_new]
            c.ok(f"{tag}: no further caption from the cut reply", not stray, stray[:2])
        br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_noise(pw, a, c: Checks, base: str, llm) -> dict:
    """1.5.0: the default interrupt sensitivity ("normal") on the REAL session. Noise and a
    far-away talker over a long reply neither pause nor cut it; the listener's own voice does."""
    want = a.noise_sensitivity
    tag = f"dispatch/390/noise ({want})"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base)
    if want != "normal":      # the control run: --noise-sensitivity high is the 1.4.0 rules
        br.ctx.add_init_script("try { localStorage.setItem('dispatch-drive', JSON.stringify({ v: 3, interruptSensitivity: '%s' })); } catch (e) {}" % want)

    def since(t, d, typ):
        return [f for f in br.frames(d, typ) if f["t"] > t]

    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        sens = br.ev("() => JSON.parse(localStorage.getItem('dispatch-drive') || '{}').interruptSensitivity || 'normal'")
        start = [f["m"] for f in br.frames("out", "start")]
        c.ok(f"{tag}: the device runs the {want} sensitivity and sends it in start",
             sens == want and start and start[-1].get("interrupt_sensitivity") == want,
             {"stored": sens, "start": start[-1] if start else None})
        # Two ordinary turns, short replies played out: the session learns the listener's level.
        llm.configure(reply="short")
        for clip in ("short", "driver1b"):
            t0 = br.ev("() => performance.now()")
            br.say(clip)
            br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'say_end' && f.t > t)", 90, t0)
            br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 30)
            br.page.wait_for_timeout(1500)
        llm.configure(reply="long")
        t0 = br.ev("() => performance.now()")
        br.say("medium")
        br.wait("(t) => window.__e2e.captions.some((c) => c.t > t)", 90, t0)
        br.wait("() => { const c = window.__e2e.captions; return c.length && performance.now() - c[c.length - 1].t < 400; }", 15)
        br.page.wait_for_timeout(1800)
        n_pause = len(br.port("pause"))
        # 1) a busy room: rumble, hiss, clatters (≈ −14 dBFS peaks)
        inj = br.say("noise")
        br.page.wait_for_timeout(3600)
        out["noise"] = {"duck": [f["m"] for f in since(inj["at"], "in", "duck")],
                        "barge_in": len(since(inj["at"], "in", "barge_in")),
                        "duck_local": len(since(inj["at"], "out", "duck_local")),
                        "player_pauses": len(br.port("pause")) - n_pause}
        c.ok(f"{tag}: room noise over the reply never pauses or cuts it",
             not out["noise"]["duck"] and not out["noise"]["barge_in"] and not out["noise"]["duck_local"]
             and not out["noise"]["player_pauses"], out["noise"])
        # 2) a talker across the room: real words, 22 dB under the listener
        n_pause = len(br.port("pause"))
        inj = br.say("medium", 0.08)
        br.page.wait_for_timeout(8200)
        playing = br.ev("() => document.querySelector('.drive-overlay').dataset.state")
        out["far_talker"] = {"duck": [f["m"] for f in since(inj["at"], "in", "duck") if f["m"].get("on")],
                             "barge_in": len(since(inj["at"], "in", "barge_in")),
                             "transcripts": [f["m"].get("text") for f in since(inj["at"], "in", "transcript")],
                             "player_pauses": len(br.port("pause")) - n_pause, "state": playing}
        c.ok(f"{tag}: a talker across the room neither pauses nor cuts the reply, and is not a turn",
             not out["far_talker"]["duck"] and not out["far_talker"]["barge_in"]
             and not any(out["far_talker"]["transcripts"]), out["far_talker"])
        # 3) the listener, at full level: "Wait, stop, that's not what I meant."
        inj = br.say("interrupt")
        got = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'barge_in' && f.t > t)", 8, inj["at"])
        bi = since(inj["at"], "in", "barge_in")
        onset = clip_onset_s("interrupt")
        out["listener"] = {"barge_ms": round(bi[0]["t"] - (inj["at"] + onset * 1000)) if bi else None}
        c.ok(f"{tag}: the listener's own voice still cuts the reply (barge_in ≤ 1.5 s)",
             got and out["listener"]["barge_ms"] is not None and out["listener"]["barge_ms"] <= 1500, out["listener"])
        br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    except Exception as e:      # a harness failure is a failed check with the socket summary
        c.ok(f"{tag}: scenario ran", False, {"error": repr(e), "summary": br.summary()})
    finally:
        br.close()
    return out


class Shaper:
    """T6's shaping proxy alone, in front of the scratch DisPatch (the real session)."""

    def __init__(self, bg: Bg, listen: int, target: int) -> None:
        self.bg, self.listen, self.target, self.proxy = bg, listen, target, None

    def shape(self, profile: str) -> None:
        import shaper
        if self.proxy is not None:
            self.bg.run(self.proxy.stop())
        self.proxy = shaper.make_proxy(profile, self.listen, self.target)
        self.bg.run(self.proxy.start())

    blackout = RefTarget.blackout
    cut = RefTarget.cut
    relink = RefTarget.relink

    def stop(self) -> None:
        if self.proxy is not None:
            with contextlib.suppress(Exception):
                self.bg.run(self.proxy.stop())


def dispatch_drop(pw, a, c: Checks, base: str, via: str, sh: Shaper) -> dict:
    """The mid-reply drop (> 60 s) on the REAL session through the shaper: offline → waiting
    → resume; the reconnect names the reply (after_partial); no fully received sentence is
    played again; no caption twice; the chat holds the reply once."""
    tag = "dispatch/390/drop mid-reply"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        br.say("short")
        br.wait("() => window.__e2e.captions.length >= 2", 90)
        time.sleep(0.6)
        sh.blackout(a.drop_s)
        sh.cut()
        c.ok(f"{tag}: offline at once", br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'offline'", 15), br.state())
        got = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'waiting'", 75)
        hint = br.ev("() => (document.querySelector('.drive-hint') || {}).textContent || ''")
        c.ok(f"{tag}: \"{EN['drive']['state_waiting']}\" after 60 s, with its own hint", got and hint == EN["drive"]["hint_waiting"],
             {"word": br.word(), "hint": hint})
        got = br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.sock > 0 && f.m && f.m.type === 'ready')", 60)
        c.ok(f"{tag}: the signal returns and a new socket reaches ready", got, br.state())
        starts = [f["m"] for f in br.frames("out", "start") if f["sock"] > 0]
        ap = next((x.get("after_partial") for x in starts if x.get("after_partial")), None)
        out["after_partial"] = ap
        c.ok(f"{tag}: the reconnect `start` names the reply in flight (after_partial)",
             isinstance(ap, dict) and ap.get("message_id") and isinstance(ap.get("sentences"), int), ap)
        _check_resume_anchor(c, tag, starts, ap)
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'in' && f.sock > 0 && f.m && f.m.type === 'say_end')", 120)
        br.wait("() => window.__e2e.ws.some((f) => f.dir === 'out' && f.sock > 0 && f.m && f.m.type === 'playback_done')", 120)
        br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 30)
        audio = _sentence_audio(br)
        out["audio"] = [[x["sock"], x["seq"], (x["text"] or "")[:40]] for x in audio]
        first = [x["text"] for x in audio if x["sock"] == 0]
        second = [x["text"] for x in audio if x["sock"] > 0]
        complete = first[:ap["sentences"]] if isinstance(ap, dict) and isinstance(ap.get("sentences"), int) else []
        again = [t for t in second if t in complete]
        c.ok(f"{tag}: no sentence whose audio had fully arrived is played again after the resume", second and not again,
             {"again": again, "first": len(first), "second": len(second)})
        caps = [x["text"] for x in br.ev("() => window.__e2e.captions")]
        dups = sorted({t for t in caps if caps.count(t) > 1})
        c.ok(f"{tag}: no caption twice", not dups, dups)
        rows = [m for m in _messages(base, tid) if m.get("role") == "assistant"]
        c.ok(f"{tag}: the chat holds the reply once", len(rows) == 1, [(m.get("content") or "")[:60] for m in rows])
        if len(rows) == 1:
            # Every sentence of the STORED reply reached the listener once, in order, across
            # both sockets (the session's own canonical split of the final text).
            from dispatch_voice.host.session import split_reply
            norm = lambda x: " ".join((x or "").split())
            want = [norm(sp.text) for sp in split_reply(rows[0].get("content") or "")]
            heard: list[str] = []
            for x in audio:
                if norm(x["text"]) not in heard:
                    heard.append(norm(x["text"]))
            out["sentences"] = {"want": len(want), "heard": len(heard)}
            c.ok(f"{tag}: every sentence of the stored reply reached the listener, in order", want and heard == want,
                 {**out["sentences"], "missing": [w for w in want if w not in heard][:3],
                  "extra": [h for h in heard if h not in want][:3]})
        br.shot(a.shots, "v2-390-dispatch-drop-resumed")
        br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_failback(pw, a, c: Checks, base: str, via: str, sh: Shaper) -> dict:
    """The slow link → phone voice → back, on the REAL session through the shaper."""
    tag = "dispatch/390/failback (3g-weak → healthy, PCM down)"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, via=via, no_opus_down=True)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        out.update(_failover_and_back(br, a, c, tag, None, heal=lambda: sh.relink("baseline"), shot="v2-390-dispatch"))
        br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def dispatch_cdp(pw, a, c: Checks, base: str) -> dict:
    """A CDP-throttled slow link on the REAL session (no shaper: the page talks to the
    scratch DisPatch directly and `Network.emulateNetworkConditions` makes it EDGE-like):
    the phone-voice failover, then back to the server's voice when CDP lets go."""
    tag = "dispatch/390/cdp-throttle (PCM down)"
    out: dict = {}
    tid = new_thread(base)
    br = Browser(pw, width=390, height=844, direct=base, no_opus_down=True)
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        br.wait("() => window.__e2e.ws.filter((f) => f.dir === 'in' && f.m && f.m.type === 'pong').length >= 3", 12)
        before = _median(ping_rtts(br))
        cdp = br.ctx.new_cdp_session(br.page)
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 450,
                                                      "downloadThroughput": 120_000 / 8, "uploadThroughput": 120_000 / 8})
        t_set = br.now()
        br.wait("(t) => window.__e2e.ws.filter((f) => f.dir === 'in' && f.m && f.m.type === 'pong' && f.t > t).length >= 2", 15, t_set)
        idle = _median(ping_rtts(br, t_set))
        out["rtt_ms"] = {"before": before, "cdp_idle": idle}
        throttled = idle is not None and before is not None and idle >= max(50.0, 10 * before)
        c.ok(f"{tag}: CDP throttling reaches the voice WebSocket (heartbeat RTT before → under CDP)", throttled, out["rtt_ms"])
        if not throttled:
            c.skip(f"{tag}: CDP-throttled failover", "CDP did not slow the WebSocket in this Chromium")
            return out

        def heal() -> None:
            cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 0,
                                                          "downloadThroughput": -1, "uploadThroughput": -1})
        out.update(_failover_and_back(br, a, c, tag, None, heal=heal, shot="v2-390-dispatch-cdp"))
        br.page.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        br.close()
    return out


def _messages(base: str, tid: str) -> list[dict]:
    import httpx
    r = httpx.get(f"{base}/api/threads/{tid}/messages", timeout=30)
    r.raise_for_status()
    j = r.json()
    return j if isinstance(j, list) else (j.get("messages") or j.get("items") or [])


def dispatch_undo(pw, a, c: Checks, base: str, llm) -> dict:
    """Undo on the real session: the button right after a transcript, then the U key
    (with the focus where the screen puts it). The user's row leaves the chat and no reply
    follows. The stub LLM is slowed so the reply cannot start inside the Undo window. Then
    a new question is answered aloud (review-T7b H2: Undo must not mute the next reply)."""
    tag = "dispatch/390/undo"
    out: dict = {}
    tid = new_thread(base)
    llm.configure(delay_s=8.0)
    br = Browser(pw, width=390, height=844, direct=base)
    p = br.page
    try:
        if not open_voice_chat(br, tid, c, tag, ready_timeout=180):
            return out
        for how in ("button", "key"):
            t0 = br.now()
            br.say("short")
            got = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'transcript' && f.m.text && f.t > t)", 40, t0)
            tr = [f["m"] for f in br.frames("in", "transcript") if f["t"] > t0]
            mid = tr[-1].get("message_id") if tr else None
            shown = br.wait("() => { const b = document.querySelector('.drive-undo'); return b && !b.hidden; }", 3)
            label = br.ev("() => (document.querySelector('.drive-undo') || {}).textContent || ''")
            c.ok(f"{tag}/{how}: Undo is offered right after the transcript", got and shown and label.startswith("Undo"),
                 {"label": label, "message_id": mid})
            if how == "button":
                br.shot(a.shots, "v2-390-undo")
            user_rows = [m for m in _messages(base, tid) if m.get("role") == "user"]
            c.ok(f"{tag}/{how}: the transcript was posted to the chat as the user's message",
                 any(m.get("id") == mid for m in user_rows), [m.get("id") for m in user_rows])
            t1 = br.now()
            if how == "button":
                p.click(".drive-undo")
            else:
                br.ev("() => document.querySelector('.drive-ring').focus()")
                p.keyboard.press("u")
            got = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'in' && f.m && f.m.type === 'undone' && f.t > t)", 6, t1)
            sent = [f["m"] for f in br.frames("out", "undo") if f["t"] > t1]
            c.ok(f"{tag}/{how}: `undo` names the user's row and the server answers `undone`",
                 got and sent and sent[0].get("message_id") == mid, {"sent": sent[:1], "errors": [f["m"] for f in br.frames("in", "error") if f["t"] > t1]})
            br.wait("() => !!document.querySelector('.drive-notice:not(.drive-link):not([hidden])')", 2)
            note = br.ev("() => { const n = document.querySelector('.drive-notice:not(.drive-link):not([hidden])'); return n ? n.textContent : ''; }")
            ui = br.ev("() => ({ undo: !document.querySelector('.drive-undo').hidden, you: document.querySelector('.drive-you') ? document.querySelector('.drive-you').textContent : null, them: document.querySelectorAll('.drive-them .drive-sentence').length })")
            c.ok(f"{tag}/{how}: the screen says \"{EN['drive']['undone']}\" and clears the turn",
                 note == EN["drive"]["undone"] and not ui["undo"] and not ui["you"] and ui["them"] == 0, {"notice": note, **ui})
            # Past the stub's 8 s: nothing was said, nothing was stored.
            time.sleep(10)
            says = [f for f in br.frames("in", "say") if f["t"] > t1 and f["m"].get("kind") == "reply"]
            rows = _messages(base, tid)
            c.ok(f"{tag}/{how}: the user's message is gone from the chat and no reply was posted or spoken",
                 not any(m.get("id") == mid for m in rows) and not [m for m in rows if m.get("role") == "assistant"] and not says,
                 {"rows": [[m.get("role"), (m.get("content") or "")[:30]] for m in rows], "says": len(says)})
            c.ok(f"{tag}/{how}: back to listening", br.state() == "listening", br.state())
        # review-T7b H2: the next question after an Undo is answered ALOUD. The undo's
        # barge_in moved gen on; the answer comes at that same gen and must play.
        llm.configure(delay_s=1.0)
        t2 = br.now()
        n_caps = len(br.ev("() => window.__e2e.captions"))
        n_push = len(br.port("push"))
        br.say("short")
        capd = br.wait("(n) => window.__e2e.captions.length > n", 90, n_caps)
        done = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'playback_done' && f.t > t)", 120, t2)
        back = br.wait("() => document.querySelector('.drive-overlay').dataset.state === 'listening'", 60)
        says = [f["m"] for f in br.frames("in", "say") if f["t"] > t2 and f["m"].get("kind") == "reply"]
        flushes = [x.get("gen") for x in br.port("flush")]
        out["after_undo"] = {"says": len(says), "gen": says[0].get("gen") if says else None, "flushes": flushes,
                             "pushes": len(br.port("push")) - n_push}
        c.ok(f"{tag}: the next question after an Undo is answered aloud (played, captioned, playback_done, listening)",
             capd and done and back and says and len(br.port("push")) > n_push
             and all(g is None or g <= says[0].get("gen", 0) for g in flushes), out["after_undo"])
        p.keyboard.press("Escape")
        errs = br.script_errors()
        c.ok(f"{tag}: no script errors", not errs, errs[:3])
    finally:
        llm.configure(delay_s=1.0)
        br.close()
    return out


_LOCKED_JS = "() => { const s = document.getElementById('stab-voice'); return !!s && s.classList.contains('hidden'); }"


def _unlock(br: Browser, base: str, pin: str) -> bool:
    """Unlock the browser context with the PIN (the cookie lands in the context's jar)."""
    r = br.ctx.request.post(base + "/api/auth/unlock", data=json.dumps({"pin": pin}),
                            headers={"Content-Type": "application/json", "Origin": base})
    return r.status == 200


def _page_json(br: Browser, path: str) -> dict:
    """GET ``path`` from inside the page (its cookie, its origin); ``{"_status": n}`` on an error."""
    return br.ev("""async (u) => { const r = await fetch(u, {cache: 'no-store'});
        if (!r.ok) return {_status: r.status}; return await r.json(); }""", path)


def dispatch_livelike(pw, a, c: Checks, out: Path) -> dict:
    """The live box's set-up on a scratch DisPatch (FR-works findings 1, 3, 4 and the
    coverage gap it named): the add-on by its entry point, voice on by ``voice.yaml``, the
    Kokoro plugin overlaid when it is installed (so the default voice is a built-in one), a
    PIN and a 60 s idle lock.
      * control: an unlocked page that nobody touches locks after the idle window;
      * a voice chat left listening for 1.5 × that window, with no input event at all, is
        still open, still listening and still unlocked (the add-on reports activity to the
        shell, Amendment FINAL-FIX-3; the server slides the session on the socket);
      * a question asked after that stretch still gets a spoken reply, and with a built-in
        default voice every acknowledgement is a word line (no "Mm-hm" for Kokoro to garble)
        and the acknowledgements are rendered for the resolved built-in voice."""
    tag = "livelike/390"
    res: dict = {}
    base = f"http://127.0.0.1:{a.port_base}"
    proc = llm = None
    import httpx

    from dispatch_voice.text.acks import is_word_line
    try:
        proc, llm, data, pin, kokoro = start_livelike_dispatch(a.port_base, a.port_base + 1, out, a.backend,
                                                               lock_s=LIVELIKE_LOCK_S)
        res.update(data=str(data), kokoro=bool(kokoro))
        ext = httpx.get(base + "/api/extensions", timeout=10).json()
        row = next((e for e in ext.get("extensions", []) if e.get("id") == "voice"), {})
        c.ok(f"{tag}: the add-on is found by its entry point and loads, with a PIN set",
             ext.get("mode") == "entry_points" and row.get("state") == "ok",
             {"mode": ext.get("mode"), "state": row.get("state"), "source": row.get("source")})
        if kokoro is None:
            c.skip(f"{tag}: Kokoro as the default voice", f"no plugin checkout at {KOKORO_SRC} or no Kokoro models")
        br = Browser(pw, width=390, height=844, direct=base)
        # Reactions "Often" on this device: an acknowledgement is skipped on 10 % of the
        # turns instead of 35 %, so the word-line check below has lines to look at.
        br.ctx.add_init_script("try { localStorage.setItem('dispatch-drive', JSON.stringify({reactions: 'often'})); } catch (e) {}")
        p = br.page
        try:
            # ---- control: the idle lock is armed and short
            c.ok(f"{tag}: unlocked with the PIN", _unlock(br, base, pin))
            p.goto(base + "/")
            mounted = br.wait("() => !!document.getElementById('drive-btn')", 60)
            auth = _page_json(br, "/api/auth/status")
            c.ok(f"{tag}: the app boots unlocked with a {LIVELIKE_LOCK_S} s idle lock and the add-on mounts",
                 mounted and auth.get("authenticated") is True and auth.get("lock_timeout_seconds") == LIVELIKE_LOCK_S,
                 {"auth": {k: auth.get(k) for k in ("authenticated", "lock_timeout_seconds")}})
            t0 = time.monotonic()
            locked = br.wait(_LOCKED_JS, LIVELIKE_LOCK_S + 30, every=0.5)
            took = round(time.monotonic() - t0, 1)
            auth = _page_json(br, "/api/auth/status")
            res["control_locked_after_s"] = took
            c.ok(f"{tag}: control: an unlocked page nobody touches locks after the idle window",
                 locked and auth.get("authenticated") is False and took >= LIVELIKE_LOCK_S * 0.75,
                 {"after_s": took, "authenticated": auth.get("authenticated")})
            # ---- a voice chat through 1.5 × the idle window, no input event at all
            c.ok(f"{tag}: unlocked again", _unlock(br, base, pin))
            r = br.ctx.request.post(base + "/api/threads", data=json.dumps({"bot_id": "voicetest", "title": "Voice e2e"}),
                                    headers={"Content-Type": "application/json", "Origin": base})
            tid = r.json().get("id") if r.ok else None
            if not tid:
                c.ok(f"{tag}: a chat for the voice chat", False, {"status": r.status})
                return res
            if not open_voice_chat(br, tid, c, tag, ready_timeout=240):
                return res
            st = _page_json(br, "/api/voice/status")
            default = (st.get("default_voice") or "")
            ack_keys = sorted((st.get("acks") or {}).keys())
            res["voice"] = {"default_voice": default, "acks": ack_keys}
            if kokoro is not None:
                c.ok(f"{tag}: with Kokoro installed and no default chosen, the default resolves to a built-in "
                     "voice and the acknowledgements are rendered for it (FR-works 3/4)",
                     default.startswith("builtin:") and default in ack_keys, res["voice"])
            idle_s = LIVELIKE_LOCK_S * 1.5 + 5
            t0 = time.monotonic()
            while time.monotonic() - t0 < idle_s:
                time.sleep(1.0)
                if br.ev(_LOCKED_JS) or br.ev("() => !document.querySelector('.drive-overlay')"):
                    break
            held = round(time.monotonic() - t0, 1)
            state = br.state()
            auth = _page_json(br, "/api/auth/status")
            res["idle"] = {"held_s": held, "state": state, "authenticated": auth.get("authenticated")}
            c.ok(f"{tag}: a voice chat left listening for {idle_s:.0f} s (1.5 × the idle lock) is still open, "
                 "listening and unlocked (FR-works 1)",
                 held >= idle_s - 1 and state == "listening" and auth.get("authenticated") is True
                 and not br.ev(_LOCKED_JS), res["idle"])
            br.shot(a.shots, "livelike-390-after-idle")
            # ---- and it still works: questions get spoken replies (two turns, each long
            # enough for an acknowledgement to be due)
            llm.configure(delay_s=3.0)
            t1 = br.now()
            turns = []
            for clip in ("short", "medium"):
                t_turn = br.now()
                n_caps = len(br.ev("() => window.__e2e.captions"))
                br.say(clip)
                capd = br.wait("(n) => window.__e2e.captions.length > n", 120, n_caps)
                done = br.wait("(t) => window.__e2e.ws.some((f) => f.dir === 'out' && f.m && f.m.type === 'playback_done' && f.t > t)",
                               180, t_turn)
                back = br.wait("() => { const o = document.querySelector('.drive-overlay'); return o && o.dataset.state === 'listening'; }", 60)
                turns.append(bool(capd and done and back))
                time.sleep(4.5)                  # not a "fast turn" for the next one
            says = [f["m"] for f in br.frames("in", "say") if f["t"] > t1]
            acks = [s.get("text") or "" for s in says if s.get("kind") == "ack"]
            replies = [s for s in says if s.get("kind") == "reply"]
            res["turn"] = {"turns": turns, "replies": len(replies), "acks": acks}
            c.ok(f"{tag}: after the idle stretch spoken questions get spoken replies", all(turns) and replies,
                 res["turn"])
            if kokoro is not None and acks:
                c.ok(f"{tag}: with a built-in default voice every acknowledgement is a word line ({len(acks)} played)",
                     all(is_word_line(x) for x in acks), acks)
            elif kokoro is not None:
                c.skip(f"{tag}: with a built-in default voice every acknowledgement is a word line",
                       "no acknowledgement was due on these turns")
            p.keyboard.press("Escape")
            br.wait("() => !document.querySelector('.drive-overlay')", 5)
            errs = br.script_errors()
            c.ok(f"{tag}: no script errors", not errs, errs[:3])
        finally:
            br.close()
    finally:
        stop_proc(proc)
        if llm is not None:
            llm.stop()
    return res


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", choices=["ref", "dispatch", "both"], default="ref")
    ap.add_argument("--backend", type=Path, default=DEFAULT_BACKEND, help="DisPatch's backend/ (the scratch server)")
    ap.add_argument("--port-base", type=int, default=19075)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    ap.add_argument("--chrome", default=None,
                    help="Chromium executable (default: $DISPATCH_E2E_CHROME, else Playwright's own)")
    ap.add_argument("--scenarios", default="settings,lab,conversation,3g,edge,blackout,keys,drop,cdp,failback,shots,locales,undo,bargein,noise,livelike",
                    help="comma list: conversation, settings, lab (both targets); 3g, edge, blackout, keys, drop, "
                         "cdp, failback, shots, locales (ref); drop, cdp, failback, undo, bargein, noise, livelike (dispatch)")
    ap.add_argument("--noise-sensitivity", choices=["low", "normal", "high"], default="normal",
                    help="the interrupt sensitivity the noise scenario runs at (high = the 1.4.0 control)")
    ap.add_argument("--blackout-s", type=float, default=75.0)
    ap.add_argument("--drop-s", type=float, default=66.0, help="how long the signal stays away after the mid-reply drop")
    a = ap.parse_args(argv)
    a.scenarios = {x.strip() for x in a.scenarios.split(",") if x.strip()}
    if a.chrome:
        global CHROME
        CHROME = str(Path(a.chrome).expanduser())
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    c = Checks()
    results: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with sync_playwright() as pw:
        if a.target in ("ref", "both"):
            results["ref"] = run_ref(pw, a, c, a.out)
        if a.target in ("dispatch", "both"):
            results["dispatch"] = run_dispatch(pw, a, c, a.out)
    results["checks"] = c.rows
    n_pass = sum(r["status"] == "PASS" for r in c.rows)
    n_skip = sum(r["status"] == "SKIP" for r in c.rows)
    results["summary"] = {"pass": n_pass, "fail": len(c.failed), "skip": n_skip}
    (a.out / "e2e_browser.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\n{n_pass} passed, {len(c.failed)} failed, {n_skip} skipped  →  {a.out / 'e2e_browser.json'}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
