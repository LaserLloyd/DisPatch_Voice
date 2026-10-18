"""Day-0 scaffold tests (owner: T0): the layout, the shared fixtures and the doubles
behave as the v3 contracts say, so every track can build on them."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
from fakes import (
    assert_private,
    FIXTURES,
    FakeContext,
    ToolSpec,
    VoiceInstructions,
    clip_meta,
    frames_for,
    load_clip,
)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "dispatch_voice"

# --------------------------------------------------------------------------- #
# Layout: every track's owned paths exist (SPEC §14)
# --------------------------------------------------------------------------- #

OWNED = {
    "T3": ["src/dispatch_voice/__init__.py", "src/dispatch_voice/host/__init__.py",
           *[f"src/dispatch_voice/host/{n}.py" for n in (
               "ws", "session", "routes", "config", "vad_turn", "stt", "tts", "registry", "download",
               "power", "power_hold", "profiles_bridge")],
           "src/dispatch_voice/host/engines/chatterbox.py", "src/dispatch_voice/standalone/voices.html",
           "src/dispatch_voice/profiles.py", "src/dispatch_voice/speak.py", "src/dispatch_voice/router.py",
           "src/dispatch_voice/cli.py", "tests/session/conftest.py", "tests/host/conftest.py",
           "tests/contract/conftest.py", "tests/e2e_voice_roundtrip.py", "docs/voice-chat/index.md",
           "docs/voice-chat/host.md", "THIRD_PARTY_NOTICES.md", "README.md"],
    "T4": [*[f"src/dispatch_voice/rt/{n}.py" for n in (
               "__init__", "stt_stream", "endpoint", "speculate", "tts_stream", "backchannel", "heard")],
           "src/dispatch_voice/engine.py", "src/dispatch_voice/audio.py", "tests/rt/conftest.py",
           "tests/test_models_rt.py", "tests/latency_harness.py", "tests/fixtures/clips",
           "tests/fixtures/clips-8k", "docs/voice-chat/turn-taking.md"],
    "T5": [*[f"src/dispatch_voice/text/{n}.py" for n in ("__init__", "persona", "acks", "speech_text", "ack_lines")],
           "tests/text/conftest.py", "tests/manual/voice_meta_check.py", "docs/voice-chat/persona-acks-rp.md"],
    "T6": [*[f"src/dispatch_voice/transport/{n}.py" for n in ("__init__", "wire3", "opus", "offsets", "cursor", "link")],
           "tests/transport/conftest.py", "tests/fixtures/wire3.json", "tests/shaper.py", "tests/e2e_lowbw.py",
           "docs/voice-chat/low-bandwidth.md"],
    "T7": [*[f"src/dispatch_voice/static/{n}" for n in (
               "index.js", "drive.js", "drive-settings.js", "voices.js", "voice-tools.js", "voice.css")],
           *[f"src/dispatch_voice/static/locales/{lang}.json" for lang in ("ar", "de", "en", "es", "fr", "ja", "pt", "zh")],
           "package.json", "scripts/check-locales.py",
           *[f"tests/js/{n}.test.js" for n in ("drive", "voice-tab", "index", "settings", "absent-interplay")],
           "tests/e2e_browser.py", "docs/voice-chat/client.md"],
    "T8": [*[f"src/dispatch_voice/static/{n}.js" for n in (
               "voice-worklet", "voice-player-worklet", "opus-codec", "link-monitor", "local-voice",
               "uplink-ring", "seq-claims")],
           *[f"tests/js/{n}.test.js" for n in (
               "worklet", "player", "opus", "link-monitor", "local-voice", "uplink-ring", "seq-claims", "wire3")],
           "docs/voice-chat/audio-modules.md"],
    "T0": ["tests/conftest.py", "tests/fakes.py", "tests/stub_llm.py", "tests/load.py",
           "tests/fixtures/addendum.wav", "tests/fixtures/frames_b1a.json", "docs/voice-chat/protocol.md",
           ".github/workflows/ci.yml"],
}


@pytest.mark.parametrize("track", sorted(OWNED))
def test_owned_paths_exist(track):
    missing = [p for p in OWNED[track] if not (ROOT / p).exists()]
    assert missing == []


def test_docs_index_lists_the_eight_files():
    index = (ROOT / "docs/voice-chat/index.md").read_text(encoding="utf-8")
    files = ["index.md", "host.md", "turn-taking.md", "persona-acks-rp.md", "low-bandwidth.md",
             "client.md", "audio-modules.md", "protocol.md"]
    for f in files:
        assert (ROOT / "docs/voice-chat" / f).exists(), f
        assert f"]({f})" in index, f


def test_package_imports_light_and_declares_the_extension():
    # The top module is metadata only: importing it must not pull in numpy/onnxruntime.
    code = ("import sys, dispatch_voice as d; "
            "assert 'numpy' not in sys.modules and 'onnxruntime' not in sys.modules, sorted(sys.modules); "
            "print(d.EXTENSION['id'], d.EXTENSION['api'], d.EXTENSION['setup'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["voice", "[1,", "1]", "dispatch_voice.host:setup"]


def test_entry_point_is_registered():
    from importlib.metadata import entry_points

    eps = {e.name: e.value for e in entry_points(group="dispatch.extensions")}
    assert eps.get("voice") == "dispatch_voice:EXTENSION"


def test_every_package_module_imports():
    import importlib
    import pkgutil

    import dispatch_voice

    names = [m.name for m in pkgutil.walk_packages(dispatch_voice.__path__, "dispatch_voice.")]
    assert {"dispatch_voice.host.session", "dispatch_voice.rt.endpoint", "dispatch_voice.text.acks",
            "dispatch_voice.transport.wire3"} <= set(names)
    for n in names:
        importlib.import_module(n)


def test_standalone_page_moved_and_static_is_assets_only():
    assert not (PKG / "static" / "voices.html").exists()
    allowed = {".js", ".mjs", ".css", ".json", ".map", ".wasm", ".onnx", ".wav", ".mp3", ".ogg",
               ".png", ".jpg", ".webp", ".woff2"}
    bad = [p for p in (PKG / "static").rglob("*") if p.is_file() and p.suffix.lower() not in allowed]
    assert bad == []


def test_locale_bundles_pass_check_locales():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check-locales.py")],
                       check=False, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 errors" in r.stdout
    en = json.loads((PKG / "static/locales/en.json").read_text(encoding="utf-8"))
    # C-§E.3: chat.drive/chat.drive_aria moved into the add-on's namespace as
    # drive.button/drive.button_aria; settings.voice_hint is drive.settings_hint (T7).
    assert en["drive"]["title"] and en["drive"]["button"] and en["drive"]["settings_hint"]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _wav_info(p: Path) -> tuple[int, int, int, float]:
    with wave.open(str(p)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() / w.getframerate()


def test_clips_are_16k_mono_and_match_meta():
    meta = clip_meta()
    for name, m in meta.items():
        sr, ch, sw, secs = _wav_info(FIXTURES / "clips" / f"{name}.wav")
        assert (sr, ch, sw) == (16000, 1, 2), name
        assert abs(secs - m["seconds"]) < 0.02, name
        assert m["text"]
    assert {"short", "medium", "long", "pause", "backchannel", "interrupt", "driver1b", "driver2"} <= set(meta)
    assert abs(meta["backchannel"]["seconds"] - 0.87) < 0.01          # the 0.87 s "Uh-huh"


def test_addendum_has_two_seconds_of_silence_between_the_halves():
    info = json.loads((FIXTURES / "addendum.json").read_text(encoding="utf-8"))
    x = load_clip("addendum")
    a, b = info["first_end_sample"], info["second_start_sample"]
    assert abs((b - a) / 16000 - 2.0) < 0.002
    assert np.abs(x[a:b]).max() == 0.0
    assert np.abs(x[:a]).max() > 0.1 and np.abs(x[b:]).max() > 0.1


def test_clips_8k_are_band_limited():
    for name in ("short", "medium", "addendum"):
        wide, narrow = load_clip(name), load_clip(name, band="8k")
        assert len(narrow) >= len(wide)

        def hi_ratio(x):
            spec = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / 16000)
            return spec[f > 4200].sum() / spec.sum()

        assert hi_ratio(wide) > 1e-4, name          # the originals do carry energy above 4 kHz
        # linear upsampling (the worklet's) leaves some imaging above 4 kHz: ~10x less, not zero
        assert hi_ratio(narrow) < hi_ratio(wide) / 8, name


def test_wire3_placeholder_has_the_schema():
    w = json.loads((FIXTURES / "wire3.json").read_text(encoding="utf-8"))
    assert set(w["schema"]) == {"up", "down", "bad_up", "bad_down"}


# --------------------------------------------------------------------------- #
# FakeContext
# --------------------------------------------------------------------------- #


async def test_submit_turn_persists_and_broadcasts_before_the_reply(fake_ctx):
    seen = []
    unsub = await fake_ctx.subscribe_frames("t-voice-1", seen.append, token="tok")
    fake_ctx.reply_for = lambda text: "Answer one. The dentist is at three. Want a reminder?"
    rid = await fake_ctx.submit_turn("t-voice-1", " what's on today ", metadata={"voice": True},
                                     instructions=VoiceInstructions(system="Speak briefly."))
    await fake_ctx.drain()
    assert fake_ctx.db().get_message(rid).metadata == {"voice": True}
    types = [f["type"] for f in seen]
    assert types[0] == "message" and seen[0]["message"]["id"] == rid
    assert types[1:4] == ["thinking", "stream_start", "stream_chunk"]
    assert types[-2:] == ["stream_done", "thinking"]
    done = seen[-2]
    assert done["provisional_id"].startswith("run:") and done["message"]["role"] == "assistant"
    assert "".join(f["text"] for f in seen if f["type"] == "stream_chunk") == done["message"]["content"]
    (_, kw), = fake_ctx.calls_to("submit_turn")
    assert kw["instructions"].system == "Speak briefly."
    assert await fake_ctx.submit_turn("t-voice-1", "   ") is None
    await unsub()


async def test_subscribe_replays_the_recorded_snapshot_first(fake_ctx, b1a_frames):
    seen = []
    snap = frames_for("snapshot_mid_stream", "t-voice-1")
    await fake_ctx.subscribe_frames("t-voice-1", seen.append, token="tok", replay=snap)
    for f in frames_for("after_snapshot", "t-voice-1"):
        fake_ctx.emit(f)
    await fake_ctx.drain()
    assert [f["type"] for f in seen] == ["stream_start", "stream_chunk", "stream_chunk", "stream_chunk",
                                         "stream_done", "thinking"]
    assert seen[1]["replace"] is True
    spoken = seen[1]["text"] + seen[2]["text"] + seen[3]["text"]
    assert spoken == b1a_frames["reply"] == seen[4]["message"]["content"]


async def test_subscription_queue_is_bounded_and_drops_the_oldest(tmp_path):
    ctx = FakeContext(tmp_path, queue_max=4)
    ctx.add_thread("t", "sage")
    seen = []
    await ctx.subscribe_frames("t", seen.append, token=None)
    for i in range(10):
        ctx.emit({"type": "progress", "thread_id": "t", "item": {"kind": "text", "n": i}})
    await ctx.drain()
    assert [f["item"]["n"] for f in seen] == [6, 7, 8, 9]
    assert ctx.dropped_frames == 6


async def test_speculation_adoption_and_staleness(fake_ctx):
    fake_ctx.speculation_reply = "Answer one."
    run = await fake_ctx.speculate_turn("t-voice-1", "what is on my calendar today")
    assert run.head_id == fake_ctx.db().last_message_id("t-voice-1") is None
    second = await fake_ctx.speculate_turn("t-voice-1", "what is on my calendar today please")
    assert run.discarded and not second.discarded           # a second call discards the first
    fake_ctx.db().add_message("t-voice-1", "user", "typed in meanwhile")
    await fake_ctx.submit_turn("t-voice-1", "what is on my calendar today please", prefetched=second)
    assert second.discarded and fake_ctx.calls_to("speculation_stale")
    fake_ctx.add_thread("t-gw", "nova")
    assert await fake_ctx.speculate_turn("t-gw", "a gateway bot never speculates") is None


async def test_abort_note_heard_and_retract(fake_ctx):
    fake_ctx.partial_on_abort = "You have two things this afternoon. A dentist appointment at three,"
    rid = await fake_ctx.abort_turn("t-voice-1", keep_partial=True)
    row = fake_ctx.db().get_message(rid)
    assert row.metadata == {"voice_heard": {"interrupted": True}}
    assert await fake_ctx.abort_turn("t-voice-1", keep_partial=False) is None
    assert await fake_ctx.note_heard("t-voice-1", "run:advisor-x", heard_upto=3, heard_text="a",
                                     unheard_text="b") is False
    ok = await fake_ctx.note_heard("t-voice-1", rid, heard_upto=45, heard_text="x" * 300,
                                   unheard_text="y" * 300)
    vh = fake_ctx.db().get_message(rid).metadata["voice_heard"]
    assert ok and vh["interrupted"] is True and vh["upto"] == 45
    assert len(vh["heard"]) == 160 and len(vh["unheard"]) == 240
    uid = await fake_ctx.submit_turn("t-voice-1", "hello", metadata={"voice": True})
    assert await fake_ctx.replies_after("t-voice-1", rid) == []
    assert await fake_ctx.replies_after("t-voice-1", "nope") is None
    await fake_ctx.retract_message("t-voice-1", rid)              # no `voice` key: kept
    await fake_ctx.retract_message("t-voice-1", uid)              # voice row: removed
    assert fake_ctx.db().get_message(rid) and fake_ctx.db().get_message(uid) is None


def test_include_router_rules(fake_ctx):
    from fastapi import APIRouter
    from fastapi.testclient import TestClient

    r = APIRouter()

    @r.get("/status")
    def status():
        return {"ok": True}

    with pytest.raises(ValueError):
        fake_ctx.include_router(r, prefix="/api/other")
    ws = APIRouter()

    @ws.websocket("/ws")
    async def sock(websocket):  # pragma: no cover - never mounted
        pass

    with pytest.raises(ValueError):
        fake_ctx.include_router(ws, prefix="/api/voice")
    fake_ctx.include_router(r, prefix="/api/voice", body_limits={"/api/voice/lab/test": 320_000})
    c = TestClient(fake_ctx.app)
    assert c.get("/api/voice/status").json() == {"ok": True}
    fake_ctx.operator = False
    assert c.get("/api/voice/status").status_code == 403


def test_serve_static_hashes_the_real_static_dir(fake_ctx):
    v1 = fake_ctx.serve_static(PKG / "static", entry="index.js", css=["voice.css"], locales=True)
    assert len(v1) == 16 and fake_ctx.static["not_served"] == []
    assert fake_ctx.serve_static(PKG / "static", entry="index.js") == v1


async def test_thread_websocket_gate_and_unavailable(fake_ctx, fake_engines):
    seen = []

    async def handler(ws, sock):
        seen.append(sock.thread.id)
        sock.touch()
        await sock.close_unavailable("Voice isn't set up on this server")

    fake_ctx.thread_websocket("voice", handler, max_total=3)
    ws = fake_engines.FakeWS()
    await fake_ctx.connect("voice", "t-voice-1", ws)
    assert ws.accepted and ws.closed_with == 1013 and fake_ctx.touched == ["tok-unlocked"]
    # 1.3.1 (DisPatch 2.3.1): a `safe` bot's thread reaches the handler too — voice
    # is unlocked-only, the tier is the device's, not the bot's — and gets the
    # add-on's own refusal, not the gate's.
    fake_ctx.add_thread("t-safe", "alpha")
    ws2 = fake_engines.FakeWS()
    await fake_ctx.connect("voice", "t-safe", ws2)
    assert ws2.accepted and ws2.closed_with == 1013
    # A thread that does not exist is still refused before accept.
    ws3 = fake_engines.FakeWS()
    await fake_ctx.connect("voice", "t-nope", ws3)
    assert not ws3.accepted and ws3.closed_with == 1008               # pre-accept refusal
    assert seen == ["t-voice-1", "t-safe"]
    assert fake_ctx.socket_count("voice") == (0, 3)


def test_validators_tools_lifecycle(fake_ctx):
    def check(v):
        if v not in ("", "stock-host"):
            raise ValueError("unknown voice")

    fake_ctx.set_bot_field_validator("voice", check)
    fake_ctx.validate_bot_field("voice", "stock-host")
    with pytest.raises(ValueError):
        fake_ctx.validate_bot_field("voice", "nope")
    fake_ctx.register_tools([ToolSpec("voice-lab", "voice_lab", "Voice lab", "🧪", True, lambda: True)])
    fake_ctx.on_startup(lambda: None, thread=True)
    fake_ctx.on_shutdown(lambda: None)
    assert [t.id for t in fake_ctx.tools] == ["voice-lab"] and fake_ctx.startup[0][1] is True
    assert fake_ctx.bot_backend("sage") == "advisor" and fake_ctx.bot_backend("nova") == "gateway"
    assert fake_ctx.paths.state_dir.is_dir()
    assert_private(fake_ctx.paths.state_dir, directory=True)


def test_fake_engines_drive_the_moved_detector_types(fake_engines):
    from dispatch_voice.host.vad_turn import TurnEnd

    ev = fake_engines.detector.feed(np.zeros(320, np.int16))
    assert isinstance(ev[0], TurnEnd)
    chunks = list(fake_engines.tts.stream("Hello there."))
    assert len(chunks) == 3 and fake_engines.tts.said == ["Hello there."]
    assert fake_engines.stt.transcribe(np.zeros(16000, np.float32)) == "what's the weather"


# --------------------------------------------------------------------------- #
# stub LLM
# --------------------------------------------------------------------------- #


def test_stub_llm_varies_replies_and_streams():
    from stub_llm import StubLLM

    with StubLLM(delay_s=0.2, tokens_per_s=400) as llm:
        body = {"model": "stub", "messages": [{"role": "system", "content": "Be brief."},
                                               {"role": "user", "content": "hi"}]}
        r1 = httpx.post(f"{llm.base_url}/chat/completions", json=body, timeout=10).json()
        r2 = httpx.post(f"{llm.base_url}/chat/completions", json=body, timeout=10).json()
        a = r1["choices"][0]["message"]["content"]
        b = r2["choices"][0]["message"]["content"]
        # DisPatch drops a reply sharing a 60-char opening with a recent one: differ early.
        assert a != b and a[:20] != b[:20]
        t0 = time.monotonic()
        first = None
        text = ""
        with httpx.stream("POST", f"{llm.base_url}/chat/completions", json={**body, "stream": True},
                          timeout=10) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                piece = json.loads(line[5:])["choices"][0]["delta"].get("content")
                if piece:
                    first = first or time.monotonic() - t0
                    text += piece
        assert 0.18 <= first < 1.0 and text.startswith("Answer 3.")
        log = llm.chat_requests()
        assert [r["last_user"] for r in log] == ["hi"] * 3 and log[0]["system"] == "Be brief."
        llm.configure(reply="long")
        long = httpx.post(f"{llm.base_url}/chat/completions", json=body, timeout=10).json()
        assert long["choices"][0]["message"]["content"].count(".") >= 12


def test_stub_llm_embeddings_delay_and_determinism():
    from stub_llm import StubLLM

    with StubLLM(delay_s=0, embed_delay_s=0.3) as llm:
        t0 = time.monotonic()
        r = httpx.post(f"{llm.base_url}/embeddings", json={"model": "e", "input": ["a", "b"]}, timeout=5).json()
        assert time.monotonic() - t0 >= 0.3
        v = [d["embedding"] for d in r["data"]]
        assert len(v) == 2 and abs(sum(x * x for x in v[0]) - 1) < 1e-9 and v[0] != v[1]
        r2 = httpx.post(f"{llm.base_url}/embeddings", json={"model": "e", "input": "a"}, timeout=5).json()
        assert r2["data"][0]["embedding"] == v[0]


def test_stub_llm_records_a_discarded_stream():
    from stub_llm import StubLLM

    with StubLLM(delay_s=0, tokens_per_s=20, reply="long") as llm:
        body = {"model": "stub", "stream": True, "messages": [{"role": "user", "content": "go"}]}
        with httpx.stream("POST", f"{llm.base_url}/chat/completions", json=body, timeout=10) as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    break                      # close after the first delta (a discard)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not llm.chat_requests()[0]["aborted"]:
            time.sleep(0.05)
        assert llm.chat_requests()[0]["aborted"] is True


def test_load_module_runs_briefly():
    from load import cpu_load

    with cpu_load(2) as procs:
        time.sleep(0.2)
        assert all(p.is_alive() for p in procs)
    assert not any(p.is_alive() for p in procs)

