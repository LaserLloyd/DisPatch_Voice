#!/usr/bin/env python3
"""Real-browser verification of the T8 client audio modules (not collected by pytest).

Runs the shipped ES modules from ``src/dispatch_voice/static/`` in headless Chromium and
checks them against the server's own libopus (``dispatch_voice.transport.opus``) and
framing (``transport.wire3``):

* mic worklet at 48 kHz, 44.1 kHz and an 8 kHz (Bluetooth SCO) context: contiguous 16 kHz
  offsets and a waveform that matches the source clip;
* uplink: the browser's AudioEncoder packets, framed by createUplink, unpacked by
  ``unpack_up`` and decoded by the server's libopus, against the PCM the encoder was given
  (and, with ``--stt``, Parakeet's transcript of both, under the box-wide model lock);
* downlink: server-encoded Opus (24 kHz, 24 kbps) and PCM16 frames fed to createDownlink,
  decoded by WebCodecs and played by the real ``drive-player`` worklet, against the
  server's own 48 kHz decode;
* pause / resume / pause / resume{rewind} on the real worklet, sample for sample.

    cd ~/Projects/dispatch-voice
    .venv/bin/python tests/js/browser/audio_modules_check.py --port 19080 \\
        --out .e2e-out/audio-modules [--stt] [--chrome PATH]

Chromium: ``--chrome``, else ``$DISPATCH_E2E_CHROME``, else the first of chromium /
chromium-browser / google-chrome on PATH. Driven over CDP with the
``websockets`` client; no Playwright needed. Exit 0 = every check passed.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import socketserver
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from dispatch_voice.transport import opus, wire3  # noqa: E402

CLIP = ROOT / "tests" / "fixtures" / "clips" / "medium.wav"
TRANSCRIPTS = ROOT / "tests" / "fixtures" / "transcripts.json"


def lp(frames: list[bytes]) -> bytes:
    return b"".join(struct.pack("<I", len(f)) + f for f in frames)


def unlp(blob: bytes) -> list[bytes]:
    out, p = [], 0
    while p + 4 <= len(blob):
        (n,) = struct.unpack_from("<I", blob, p)
        out.append(blob[p + 4:p + 4 + n])
        p += 4 + n
    return out


def best_lag(ref: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (x behind ref) and the normalised correlation at it."""
    n = min(len(ref), len(x)) - 2 * max_lag
    a = ref[max_lag:max_lag + n].astype(np.float64)
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag + 1):
        b = x[max_lag + lag:max_lag + lag + n].astype(np.float64)
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best[1]:
            best = (lag, c)
    return best


class Handler(http.server.SimpleHTTPRequestHandler):
    run_dir: Path = Path(".")

    def log_message(self, *a):  # quiet
        pass

    def translate_path(self, path):
        p = urllib.parse.urlparse(path).path
        if p.startswith("/run/"):
            return str(self.run_dir / p[len("/run/"):])
        return super().translate_path(path)

    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = Path(q.get("name", ["upload.bin"])[0]).name
        n = int(self.headers.get("Content-Length", "0"))
        (self.run_dir / name).write_bytes(self.rfile.read(n))
        self.send_response(200)
        self.end_headers()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def prepare(run: Path) -> dict:
    """Server-side frames: the clip at 24 kHz through the server's libopus and as PCM16."""
    clip, sr = sf.read(CLIP, dtype="int16")
    assert sr == 16000
    pcm24 = np.clip(resample_poly(clip.astype(np.float64), 3, 2), -32768, 32767).astype(np.int16)
    enc = opus.Encoder(24000, 24000, application="audio")
    dec = opus.Decoder(48000)
    packets, ref48 = [], []
    for i in range(0, len(pcm24) - enc.frame + 1, enc.frame):
        pkt = enc.encode(pcm24[i:i + enc.frame])
        packets.append(pkt)
        ref48.append(dec.decode(pkt))
    frames = [wire3.pack_down(1, 0, 1, 1, i * 20, packets[i:i + 10]) for i in range(0, len(packets), 10)]
    (run / "down_opus.bin").write_bytes(lp(frames))
    pcm_frames = [wire3.pack_down(0, 0, 1, 1, i // 24, [pcm24[i:i + 960].tobytes()])
                  for i in range(0, len(pcm24) - 960 + 1, 960)]
    (run / "down_pcm.bin").write_bytes(lp(pcm_frames))
    ref = np.concatenate(ref48).astype(np.float32) / 32768
    np.save(run / "ref48.npy", ref)
    return {"clip": clip, "pcm24": pcm24, "ref48": ref, "packets": len(packets), "pcm_frames": len(pcm_frames)}


def _default_chrome() -> str | None:
    env = os.environ.get("DISPATCH_E2E_CHROME", "").strip()
    if env:
        return env
    for name in ("chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def cdp_run(chrome: str, port: int, url: str, profile: Path, timeout: float = 120) -> dict:
    from websockets.sync.client import connect

    proc = subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check", "--mute-audio",
         "--autoplay-policy=no-user-gesture-required", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as r:
                    pages = [t for t in json.load(r) if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except OSError:
                pass
            time.sleep(0.1)
        if not ws_url:
            raise RuntimeError("Chromium did not expose a page target")
        with connect(ws_url, max_size=None, open_timeout=10) as ws:
            seq = [0]

            def call(method, **params):
                seq[0] += 1
                ws.send(json.dumps({"id": seq[0], "method": method, "params": params}))
                while True:
                    m = json.loads(ws.recv(timeout=timeout))
                    if m.get("id") == seq[0]:
                        if "error" in m:
                            raise RuntimeError(f"{method}: {m['error']}")
                        return m.get("result", {})

            call("Page.enable")
            call("Page.navigate", url=url)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                r = call("Runtime.evaluate", expression="JSON.stringify(window.__result || null)", returnByValue=True)
                val = r.get("result", {}).get("value")
                if val and val != "null":
                    return json.loads(val)
                time.sleep(0.5)
            raise RuntimeError("the page did not finish")
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19080)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chrome", default=_default_chrome())
    ap.add_argument("--stt", action="store_true", help="also transcribe with Parakeet (hold the model lock)")
    a = ap.parse_args()
    if not a.chrome:
        ap.error("no Chromium found: pass --chrome PATH (or set DISPATCH_E2E_CHROME)")
    run = a.out
    run.mkdir(parents=True, exist_ok=True)
    prof = run / "chrome-profile"
    shutil.rmtree(prof, ignore_errors=True)

    prep = prepare(run)
    Handler.run_dir = run
    srv = Server(("127.0.0.1", a.port), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = cdp_run(a.chrome, a.port + 1, f"http://127.0.0.1:{a.port}/tests/js/browser/audio-modules.html", prof)
    finally:
        srv.shutdown()
        shutil.rmtree(prof, ignore_errors=True)
    (run / "page-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    check("page ran", "error" not in res, res.get("error", ""))
    if "error" in res:
        return report(checks, res, run)
    check("WebCodecs Opus up+down supported", res["support"] == {"up": True, "down": True}, json.dumps(res["support"]))

    clip = prep["clip"].astype(np.float32) / 32768
    mic16 = {}
    for rate in (48000, 44100, 8000):
        blob = (run / f"mic_{rate}.bin").read_bytes()
        n = len(blob) // 644
        offs = [struct.unpack_from("<I", blob, i * 644)[0] for i in range(n)]
        pcm = np.concatenate([np.frombuffer(blob[i * 644 + 4:(i + 1) * 644], np.int16) for i in range(n)]).astype(np.float32) / 32768
        mic16[rate] = pcm
        contiguous = offs == [i * 320 for i in range(n)]
        ref = clip if rate >= 16000 else resample_poly(resample_poly(clip, 1, 2), 2, 1).astype(np.float32)
        lag, corr = best_lag(ref, pcm, 400)
        need = 0.98 if rate >= 16000 else 0.95
        check(f"mic {rate} Hz: {n} frames, contiguous offsets, waveform = source", contiguous and corr >= need and n >= len(clip) // 320,
              f"frames {n} (clip {len(clip) // 320}), corr {corr:.4f} at lag {lag}")

    # Uplink: browser Opus → server unpack_up + libopus decode, against what the encoder got.
    src = mic16[48000]
    for kbps in (16, 12):
        msgs = unlp((run / f"up_opus{kbps}.bin").read_bytes())
        dec = opus.Decoder(16000)
        out = np.zeros(len(src) + 16000, np.float32)
        offs_ok, packets = True, 0
        expected = 0
        for m in msgs:
            codec, flags, off, pkts = wire3.unpack_up(m)
            offs_ok &= (codec == 1 and flags == 0 and off == expected and len(pkts) <= 5)
            for p in pkts:
                pcm = dec.decode(p).astype(np.float32) / 32768
                out[off:off + len(pcm)] = pcm
                off += len(pcm)
                packets += 1
            expected = off
        lag, corr = best_lag(src, out[:len(src)], 400)
        u = res["uplink"][str(kbps)]
        check(f"uplink Opus {kbps} kbps: browser packets decode on the server's libopus",
              # Waveform correlation of a perceptual codec is a sanity floor (the design prototype measured
              # 0.90/0.91 at 12/16 kbps); the transcript check (--stt) is the real criterion.
              offs_ok and packets == len(src) // 320 and corr >= 0.8 and u["codec"] == "opus" and not u["errors"],
              f"{u['messages']} messages, {packets} packets, {u['kbpsOnWire']} kbps on the wire, "
              f"{u['usPerFrame']} µs/frame, corr {corr:.4f} at lag {lag}")
        np.save(run / f"uplink_decoded_{kbps}.npy", out[:len(src) + 2000])

    # Replay ring: loaded through `?v=` like bridge.js does; replays the browser's own packets.
    live = [wire3.unpack_up(m) for m in unlp((run / "up_opus16.bin").read_bytes())]
    rep = [wire3.unpack_up(m) for m in unlp((run / "ring_replay16.bin").read_bytes())]
    live_pkts = [bytes(p) for _, _, _, pk in live for p in pk]
    rep_pkts = [bytes(p) for _, _, _, pk in rep for p in pk]
    rep_ok = all(c == 1 and f == wire3.FLAG_REPLAY and len(pk) <= 50 for c, f, _, pk in rep)
    r = res["ring"]
    check("replay ring: one opus-codec instance under ?v=, the browser's packets replayed with the replay flag",
          r["siblingQueried"] == 1 and rep_pkts == live_pkts and rep_ok and rep and rep[0][2] == live[0][2] and not r["cutOff"],
          f"{r['messages']} replay messages for {len(live)} live, {len(rep_pkts)} packets, "
          f"{r['seconds']:.2f} s held, sibling via query {r['siblingQueried']}")

    # Downlink + player against the server's own 48 kHz decode / the PCM source.
    ref48 = prep["ref48"]
    play = np.fromfile(run / "play_opus.f32", dtype=np.float32)
    lag, corr = best_lag(ref48, play, 200)
    n = min(len(ref48), len(play) - max(lag, 0)) - 400
    seg = play[max(lag, 0):max(lag, 0) + n]
    err = float(np.max(np.abs(ref48[max(-lag, 0):max(-lag, 0) + n] - seg)))
    po = res["playOpus"]
    check("downlink Opus: WebCodecs + drive-player = server libopus decode",
          corr >= 0.999 and abs(lag) <= 2 and po["underruns"] == 0 and po["idle"] and po["idle"][-1]["ended"],
          f"corr {corr:.6f} lag {lag} max|err| {err:.4f}, played msgs {po['played']}, idle {po['idle']}")
    ref_pcm48 = resample_poly(prep["pcm24"].astype(np.float64) / 32768, 2, 1).astype(np.float32)
    playp = np.fromfile(run / "play_pcm.f32", dtype=np.float32)
    lag, corr = best_lag(ref_pcm48, playp, 200)
    check("downlink PCM16 24 kHz: resampled to 48 kHz by the worklet",
          corr >= 0.999 and abs(lag) <= 2 and res["playPcm"]["underruns"] == 0, f"corr {corr:.6f} lag {lag}")

    # Pause / resume / rewind: compare windows of the paused render with the continuous one.
    pp = np.fromfile(run / "play_pause.f32", dtype=np.float32)
    q = 128
    t1, t2, t3, t4 = 192 * q, 384 * q, 576 * q, 768 * q
    ramp = 960
    a_ok = float(np.max(np.abs(pp[:t1] - playp[:t1])))
    stopped = t1 + ramp - 1        # the 20 ms ramp-down plays 959 more samples (the 960th is env 0)
    silent = float(np.max(np.abs(pp[t1 + ramp + q:t2])))
    seg2 = pp[t2 + ramp:t3]
    cont2 = playp[stopped + ramp:stopped + ramp + len(seg2)]
    b_ok = float(np.max(np.abs(seg2 - cont2)))
    seg3 = pp[t4 + ramp:t4 + ramp + 24000]
    c_ok = float(np.max(np.abs(seg3 - playp[ramp:ramp + 24000])))
    check("player pause → silence, resume → the next sample, resume{rewind} → the seq's first sample",
          a_ok < 1e-6 and silent < 1e-6 and b_ok < 1e-5 and c_ok < 1e-5,
          f"pre-pause {a_ok:.2e}, paused {silent:.2e}, resumed vs continuous {b_ok:.2e}, rewound vs start {c_ok:.2e}")
    check("headless speechSynthesis: 0 voices, waitForVoices gives up quietly",
          res["voices"]["count"] == 0 and res["voices"]["waitedMs"] < 1500, json.dumps(res["voices"]))

    if a.stt:
        stt(checks, run, src, mic16[8000])
    return report(checks, res, run)


def stt(checks, run: Path, src: np.ndarray, sco: np.ndarray) -> None:
    """Parakeet on the encoder input vs the server-decoded uplink (call under flock)."""
    from dispatch_voice.host.stt import ParakeetSTT
    import os
    models = Path(os.environ.get("DISPATCH_SPEECH_MODELS", str(Path.home() / ".local/share/dispatch-voice-models")))
    eng = ParakeetSTT(models / "parakeet-tdt-0.6b-v2", threads=4)
    base = eng.transcribe(src)
    want = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))["16k"]["medium"]
    out = {"pcm_via_mic_worklet": base}
    for kbps in (16, 12):
        out[f"opus{kbps}"] = eng.transcribe(np.load(run / f"uplink_decoded_{kbps}.npy"))
    norm = lambda s: " ".join("".join(c.lower() if c.isalnum() or c == " " else " " for c in s).split())  # noqa: E731
    ok = all(norm(v) == norm(want) for v in out.values())
    checks.append(("Parakeet: mic-worklet PCM and both Opus uplinks transcribe like the fixture", ok, json.dumps(out)))
    want8 = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))["8k"]["medium"]
    got8 = eng.transcribe(sco)
    checks.append(("Parakeet: the 8 kHz (SCO) mic-worklet path matches the 8 kHz fixture baseline",
                   norm(got8) == norm(want8), json.dumps({"sco_8k": got8})))


def report(checks, res, run: Path) -> int:
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    (run / "checks.json").write_text(json.dumps([{"name": n, "ok": o, "detail": d} for n, o, d in checks], indent=1), encoding="utf-8")
    print(f"{sum(c[1] for c in checks)}/{len(checks)} passed; ua {res.get('ua', '?')}")
    return 0 if all(c[1] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
