"""The add-on's HTTP surface under ``/api/voice`` (contracts §D), the model
lifecycle (prewarm, the live switch, release) and the per-voice renders
(acknowledgements, spoken notices, previews).

Everything here is mounted by ``host.setup`` through DisPatch's
``ctx.include_router(router, prefix="/api/voice", body_limits=...)``: every
route gets DisPatch's operator dependency (unlocked tier only, Safe Mode
403, never the machine tier) and the prefix joins the Safe-Mode block list.
The router built here therefore carries no gate of its own, and no
``WebSocketRoute`` (contracts §B.1 rule 2: the thread socket is DisPatch's,
see ``host/ws.py``).

Routes: ``GET /status`` (what the Settings page and the mic button draw
from), ``GET/PATCH /settings`` (the switch and the server preferences,
voice.yaml, applied live), ``GET /preview`` (a WAV sample of a voice),
``GET /lab`` + ``POST /lab/test`` (the Voice lab), and the profile router
at ``/profiles`` (``profiles_bridge``, when the Chatterbox models are fetched).

The opt-in voice TOOLS (Voice cloning, Built-in voices, Voice lab) are
``tools.yaml`` rows the add-on registers with DisPatch (``host/tools.py``);
their switch decides whether the Tools tab shows a tile. The routes
themselves are reachable to any operator (the security boundary is the
operator gate, which DisPatch applies); when a DisPatch exposes a tool-switch
query on the context (``ctx.tool_enabled``) the lab routes honour it too.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import threading
import wave
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import Response

from . import context
from .config import (
    BOT_KEYS,
    GATEWAY_TRAILERS,
    HARD_SILENCE_MAX,
    HARD_SILENCE_MIN,
    MAX_SESSIONS_MAX,
    MAX_SESSIONS_MIN,
    MERGE_WINDOW_MAX,
    MERGE_WINDOW_MIN,
    RP_MODES,
    SPEED_MAX,
    SPEED_MIN,
    THINKING_LEVELS,
    VoiceSettings,
    load_settings,
)

log = logging.getLogger("dispatch.voice")

#: Close code for "the server can't do voice right now" (feature off/missing).
UNAVAILABLE = 1013

#: Close reasons for a voice socket refused (or ended) because voice is off.
#: The client shows its own localised words for them (drive.js matches the
#: start); they are also what a log line or a non-browser client reads.
OFF_SETTINGS = "Voice is turned off in Settings › Voice"
OFF_SERVER = "Voice is turned off on this server (DISPATCH_VOICE)"
MODELS_MISSING = "voice models missing"

#: The ``assets_version`` DisPatch computed for ``/ext/voice/`` when ``setup``
#: ran. DisPatch recomputes its own hash whenever a static file changes (an
#: editable checkout), so the client keys its cache on ``opts.assetsVersion``
#: from ``/api/extensions``, not on this; ``status`` reports it as
#: ``assets_version_at_setup`` so nobody mistakes it for the live value.
ASSETS_VERSION: str | None = None


async def warm_models(settings: VoiceSettings):
    """Load (once per process) and return (stt_engine, tts_engine)."""
    from . import stt, tts
    return await asyncio.gather(asyncio.to_thread(stt.shared_engine, settings),
                                asyncio.to_thread(tts.shared_engine, settings))


# --------------------------------------------------------------------------- #
# Per-voice renders: acknowledgements and spoken notices
# --------------------------------------------------------------------------- #

def _cache_dir(settings: VoiceSettings, tts_engine, profile_id: str | None):
    """Where this voice's renders are kept across restarts, or None.

    Disk caching is opt-in per engine (``cache_tag``): only an engine whose
    output for (voice, text) is reusable returns a tag — the cloned-voice
    engine (seeded lines) and the optional Kokoro plugin's built-in voices."""
    from . import tts as _tts
    tag = _tts.cache_tag(tts_engine, profile_id)
    if not tag or not _CACHE_TAG_RE.match(tag):
        return None
    return context.paths().cache_dir / tag


#: A plugin's cache tag becomes a directory name: plain characters only.
_CACHE_TAG_RE = __import__("re").compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
#: Tag directories kept under <data>/voice-cache (each speed step or voice
#: change makes a new one; the least recently written go first).
CACHE_KEEP = 8


def _prune_cache(root) -> None:
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return
    import shutil
    for d in dirs[CACHE_KEEP:]:
        shutil.rmtree(d, ignore_errors=True)


def _render_cached(settings, tts_engine, profile_id, text: str, *, wait_idle: bool):
    """int16 PCM of ``text`` in this voice: disk cache → synthesis. The cache
    key is the one ``text/acks.RenderQueue`` writes, so a line either path
    rendered serves both."""
    from . import tts as _tts
    d = _cache_dir(settings, tts_engine, profile_id)
    f = None
    if d is not None:
        f = d / (hashlib.sha256(text.encode()).hexdigest()[:24] + ".npy")
        with contextlib.suppress(Exception):
            return np.load(f).astype("<i2")
    if wait_idle:
        _tts.REPLY_ACTIVITY.wait_idle()
    pcm, _ = tts_engine.synth(text, profile_id)
    pcm = np.asarray(pcm, "<i2")
    if f is not None and len(pcm):
        with contextlib.suppress(Exception):
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                _prune_cache(d.parent)
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, pcm)
            tmp.replace(f)
    return pcm


def voice_for_bot(settings: VoiceSettings, tts_engine, bot_id: str | None,
                  locked: bool = False) -> str | None:
    """The profile id the TTS engine speaks with for this bot (None = the
    engine's default voice). The acknowledgement lines of that voice are the
    render queue's (``host/acks.py``): the old per-profile fillers and the
    per-process ack list are gone (SPEC §6.2)."""
    from . import registry
    if locked:
        # A locked (Safe-Mode) device hears the bot's "voice on locked devices" when one
        # is set and still exists; otherwise the standard voice. Decided by the caller
        # from the socket's locked state, never from anything the client sent.
        want = settings.locked_voice(bot_id)
        if want and classify_voice(settings, want) in ("profile", "builtin"):
            return registry.resolve_profile(bot_id, voice=want)
    return registry.resolve_profile(bot_id)


_NOTICES: OrderedDict[tuple, np.ndarray] = OrderedDict()
_NOTICES_LOCK = threading.Lock()


def notice_pcm(tts_engine, profile_id: str | None, text: str, *, wait_idle: bool = False):
    """A spoken error line ("Sorry, that didn't go through.") in the
    session's voice: memory → disk (built-in voice) → synthesis."""
    settings = load_settings()
    key = (id(tts_engine), profile_id or "", str(_cache_dir(settings, tts_engine, profile_id)), text)
    with _NOTICES_LOCK:
        if key in _NOTICES:
            return _NOTICES[key]
    pcm = _render_cached(settings, tts_engine, profile_id, text, wait_idle=wait_idle)
    with _NOTICES_LOCK:
        _NOTICES[key] = pcm
        while len(_NOTICES) > 64:
            _NOTICES.popitem(last=False)
    return pcm


# --------------------------------------------------------------------------- #
# Start-up prewarm (review H5)
# --------------------------------------------------------------------------- #

_PREWARM: dict[str, object] = {"state": "idle"}
#: Bumped by release_models(). A prewarm that started before a release drops
#: what it loaded when it finishes, so "off" stays off.
_GEN = [0]


def _profile_to_warm(registry, tts) -> str | None:
    """A cloned voice some bot will actually speak with: the default profile,
    else the first bot's own profile. None = only built-in voices in use."""
    pid = registry.resolve_profile(None)
    if pid and not tts.builtin_id(pid):
        return pid
    try:
        for bot in context.bots_all():
            pid = registry.resolve_profile(bot.id)
            if pid and not tts.builtin_id(pid):
                return pid
    except Exception:
        log.exception("voice: looking for a voice profile to warm failed")
    return None


def _speaks_stock(eng) -> bool:
    """True when the cloned-voice engine is the last resort of this chain."""
    try:
        from .engines.chatterbox import ChatterboxTTS
    except Exception:
        return False
    return any(isinstance(e, ChatterboxTTS) and not e.has_fallback for e in _walk(eng))


def _walk(eng):
    from . import tts
    return tts.walk(eng)


def prewarm(settings: VoiceSettings | None = None) -> bool:
    """Load every speech model and run one throwaway synthesis, so the first
    voice chat after a restart accepts fast and its first reply does not pay
    the engine load (~4.5 s for the cloned voice). Blocking: DisPatch runs it
    in a background thread at start-up (``on_startup(thread=True)``). Returns
    False when voice is off."""
    settings = settings or load_settings()
    if not settings.enabled or not settings.prewarm or settings.missing():
        _PREWARM["state"] = "skipped"
        return False
    import time
    t0 = time.monotonic()
    gen = _GEN[0]
    _PREWARM["state"] = "running"
    try:
        from . import registry, stt, tts, vad_turn
        stt_engine = stt.shared_engine(settings)
        eng = tts.shared_engine(settings)
        vad_turn._shared_smart_turn(settings)
        vad_turn.build_detector(settings)          # Silero + ORT kernels
        stt_engine.transcribe(np.zeros(16000, np.float32))
        for e in tts.walk(eng):
            warm = getattr(e, "warm", None)
            if callable(warm):
                try:
                    warm()
                except Exception:
                    log.exception("voice: %s warm-up failed", type(e).__name__)
        pid = _profile_to_warm(registry, tts)
        if pid or _speaks_stock(eng):
            # A cloned voice some bot uses, or the stock voice that speaks
            # when no built-in voice plugin is installed (this also builds
            # the stock profile on an install that predates it).
            try:
                from . import profiles_bridge
                profiles_bridge.store().engine.warm()
                eng.synth("Okay.", pid)
            except Exception:
                log.exception("voice: cloned-voice warm-up failed")
        if gen != _GEN[0]:
            # Switched off (release_models ran) while we were loading: give
            # back what this run loaded after the release.
            release_models()
            return False
        from . import acks
        try:
            acks.ensure_all(settings, eng, context.bots_all(), default_profile=registry.resolve_profile(None))
        except Exception:
            log.exception("voice: acknowledgement renders could not be queued")
        _PREWARM["state"] = "done"
        _PREWARM["seconds"] = round(time.monotonic() - t0, 1)
        log.info("voice: speech models warm in %.1f s", time.monotonic() - t0)
        return True
    except Exception:
        _PREWARM["state"] = "failed"
        log.exception("voice: prewarm failed; models will load on the first session")
        return False


def prewarm_start() -> None:
    """The ``on_startup`` hook (run by DisPatch in a thread): record the
    switch baseline a later hand edit of voice.yaml is compared against, and
    warm the models when voice is on."""
    try:
        st = load_settings()
        _APPLIED["on"] = st.switched_on
        if not (st.prewarm and st.enabled):
            return
    except Exception:
        log.exception("voice: prewarm could not start")
        return
    prewarm(st)


def shutdown() -> None:
    """The ``on_shutdown`` hook: never leave a power-profile hold behind, and
    give the plugins their release call."""
    from . import power
    power.release_all()
    with contextlib.suppress(Exception):
        release_models(why="shutdown")


# --------------------------------------------------------------------------- #
# The switch, applied live
# --------------------------------------------------------------------------- #

def release_models(*, why: str = "voice switched off") -> None:
    """Drop every loaded speech model and render cache, so the memory goes
    back when voice is switched off (or the process stops: ``why`` names the
    reason in the journal). The next session (or prewarm) loads them again.
    What a plugin keeps in its own module globals is the plugin's."""
    import gc

    from . import stt, tts, vad_turn
    _GEN[0] += 1
    with stt._ENGINES_LOCK:
        stt._ENGINES.clear()
    with tts._ENGINES_LOCK:
        tts._ENGINES.clear()
    with vad_turn._SMART_TURN_LOCK:
        vad_turn._SMART_TURN.clear()
    from . import acks
    acks.release()
    with _NOTICES_LOCK:
        _NOTICES.clear()
    with _PREVIEWS_LOCK:
        _PREVIEWS.clear()
    try:
        from .engines import chatterbox
        chatterbox.release()
    except Exception:
        log.debug("voice: no cloned-voice engine to release", exc_info=True)
    from . import registry
    registry.run_release_hooks()          # plugins drop what they cache themselves
    _PREWARM.clear()
    _PREWARM["state"] = "idle"
    gc.collect()
    _trim_heap()
    log.info("voice: speech models released (%s)", why)


def _trim_heap() -> None:
    """Hand freed heap back to the OS. glibc keeps what the ONNX sessions
    freed in its arenas, so without this the process stayed ~700 MB fatter
    after a release and grew by a full model set on every off/on cycle
    (measured 2026-10-10). A no-op where there is no glibc."""
    try:
        import ctypes
        import ctypes.util
        name = ctypes.util.find_library("c")
        if name:
            ctypes.CDLL(name).malloc_trim(0)
    except Exception:
        log.debug("voice: malloc_trim unavailable", exc_info=True)


async def end_all_sessions(code: int = UNAVAILABLE, reason: str = OFF_SETTINGS) -> int:
    """End every open voice chat (voice was switched off). Returns how many."""
    from . import ws
    return await ws.end_all_sessions(code, reason)


async def apply_switch(was_on: bool, settings: VoiceSettings) -> None:
    """The switch moved (or a PATCH saved it): make the running process match.

    on → off: open voice chats end with 1013 and the speech models are
    released (no restart). off → on: the start-up prewarm runs now, in the
    background, so the first chat is fast."""
    now_on = settings.switched_on
    if was_on and not now_on:
        n = await end_all_sessions(UNAVAILABLE, OFF_SETTINGS)
        if n:
            log.info("voice: switched off; ended %d voice chat(s)", n)
        await asyncio.to_thread(release_models)
    elif now_on and not was_on and settings.enabled and settings.prewarm:
        threading.Thread(target=prewarm, args=(settings,), name="voice-prewarm",
                         daemon=True).start()


#: The switch as this process last applied it (None = not looked yet).
_APPLIED: dict[str, bool | None] = {"on": None}
_WATCH: dict[str, object] = {}
#: How often a running server re-reads voice.yaml for a hand-edited switch.
SWITCH_WATCH_S = 5.0


async def sync_switch(settings: VoiceSettings | None = None) -> None:
    """Apply a switch change that did not come through PATCH (someone edited
    ``enabled`` in voice.yaml by hand): the same live effects as the app's
    switch — off ends chats and releases the models, on prewarms. Cheap:
    voice.yaml is re-read only when its mtime moved (config.load_prefs)."""
    settings = settings or load_settings()
    now = settings.switched_on
    was = _APPLIED["on"]
    _APPLIED["on"] = now
    if was is None or was == now:
        return
    log.info("voice: the switch changed in voice.yaml (%s → %s); applying it",
             "on" if was else "off", "on" if now else "off")
    await apply_switch(was, settings)


def ensure_switch_watch() -> None:
    """Start (once, on the running loop) the background check that makes a
    hand edit of voice.yaml apply within SWITCH_WATCH_S, open chats
    included. Called from the status route and the voice socket."""
    task = _WATCH.get("task")
    if task is not None and not task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def watch():
        while True:
            await asyncio.sleep(SWITCH_WATCH_S)
            try:
                await sync_switch(await asyncio.to_thread(load_settings))
            except Exception:
                log.exception("voice: applying a voice.yaml switch change failed")

    _WATCH["task"] = loop.create_task(watch(), name="voice-switch-watch")


def _tool_enabled(tool_id: str) -> bool:
    """Is this opt-in tool switched on? Only a DisPatch that exposes
    ``ctx.tool_enabled`` can say; otherwise the routes are open to operators
    (the tool switch then only governs the tile)."""
    ctx = context.get_context()
    fn = getattr(ctx, "tool_enabled", None) if ctx is not None else None
    if fn is None:
        return True
    try:
        return bool(fn(tool_id))
    except Exception:
        return False


def _require_tool(tool_id: str) -> None:
    if not _tool_enabled(tool_id):
        raise HTTPException(404, "Not found")


# --------------------------------------------------------------------------- #
# Bot field validators (contracts §D ``PUT /api/bots/order``; §F.5)
# --------------------------------------------------------------------------- #

VOICE_RP_MODES = ("", "all", "skip_actions", "dialogue", "dialogue_narrator")


def validate_bot_voice(voice: str) -> None:
    """ValueError (DisPatch answers 422) unless ``voice`` is "" (use the
    default), a voice profile id, or a known ``builtin:<id>`` from an
    installed built-in voice provider."""
    if voice == "":
        return
    try:
        kind = classify_voice(load_settings(), voice)
    except Exception:
        log.exception("voice: bot voice check failed")
        kind = None
    if kind not in ("profile", "builtin"):
        raise ValueError(f"Unknown voice: {str(voice)[:80]}")


def validate_bot_voice_rp(mode: str) -> None:
    """``voice_rp``: "" (server default) or one of the four role-play modes."""
    if not isinstance(mode, str) or mode not in VOICE_RP_MODES:
        raise ValueError("voice_rp must be one of: " + ", ".join(m or '""' for m in VOICE_RP_MODES))


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

#: Body caps DisPatch enforces before the body is read (``include_router(body_limits=)``).
PROFILE_UPLOAD_LIMIT = 20 * 1024 * 1024 + 64 * 1024     # audio.MAX_UPLOAD_BYTES + multipart slack


def body_limits() -> dict[str, int]:
    return {"/api/voice/profiles": PROFILE_UPLOAD_LIMIT,
            "/api/voice/profiles/analyse": PROFILE_UPLOAD_LIMIT,
            "/api/voice/lab/test": LAB_TEST_MAX_BYTES}


def build_router() -> APIRouter:
    """The ``/api/voice`` routes (relative paths: DisPatch adds the prefix and
    the operator gate when ``setup`` hands this router to ``ctx.include_router``)."""
    r = APIRouter()

    @r.get("/status")
    async def voice_status():
        ensure_switch_watch()
        settings = await asyncio.to_thread(load_settings)
        await sync_switch(settings)
        return await status_payload(settings)

    @r.get("/settings")
    async def get_voice_settings():
        settings = load_settings()
        ch = await asyncio.to_thread(_voice_choices, settings)
        return _settings_out(settings, ch)

    @r.patch("/settings")
    async def patch_voice_settings(body: dict = Body(...)):
        from . import config as _vc
        settings = load_settings()
        unknown = set(body) - SETTING_KEYS
        if unknown:
            raise HTTPException(422, f"Unknown setting: {sorted(unknown)[0]}")
        update: dict = {}
        if "enabled" in body:
            v = body["enabled"]
            if not isinstance(v, bool):
                raise HTTPException(422, "enabled must be true or false")
            if settings.forced is not None and v != settings.switched_on:
                raise HTTPException(409, "DISPATCH_VOICE is set on the server, so the "
                                         "switch can't change it")
            if settings.forced is None:
                update["enabled"] = v
        if "speed" in body:
            update["speed"] = _ranged(body["speed"], "speed", SPEED_MIN, SPEED_MAX)
        if "merge_window_s" in body:
            update["merge_window_s"] = _ranged(body["merge_window_s"], "merge_window_s",
                                               MERGE_WINDOW_MIN, MERGE_WINDOW_MAX)
        if "hard_silence_s" in body:
            update["hard_silence_s"] = _ranged(body["hard_silence_s"], "hard_silence_s",
                                               HARD_SILENCE_MIN, HARD_SILENCE_MAX)
        if "voice_turn_thinking" in body:
            v = body["voice_turn_thinking"]
            if v not in THINKING_LEVELS:
                raise HTTPException(422, f"voice_turn_thinking must be one of {', '.join(THINKING_LEVELS)}")
            update["voice_turn_thinking"] = v
        if "max_sessions" in body:
            v = body["max_sessions"]
            if isinstance(v, bool) or not isinstance(v, int) or not MAX_SESSIONS_MIN <= v <= MAX_SESSIONS_MAX:
                raise HTTPException(422, f"max_sessions must be a whole number between "
                                         f"{MAX_SESSIONS_MIN} and {MAX_SESSIONS_MAX}")
            update["max_sessions"] = v
        if "rt" in body:
            update_rt = _validate_rt(body["rt"])
            if update_rt:
                update["rt"] = update_rt
        if "rp_speech" in body:
            v = body["rp_speech"]
            if v not in RP_MODES:
                raise HTTPException(422, f"rp_speech must be one of {', '.join(RP_MODES)}")
            update["rp_speech"] = v
        if "expressive_tags" in body:
            if not isinstance(body["expressive_tags"], bool):
                raise HTTPException(422, "expressive_tags must be true or false")
            update["expressive_tags"] = body["expressive_tags"]
        if "gateway_trailer" in body:
            v = body["gateway_trailer"]
            if v not in GATEWAY_TRAILERS:
                raise HTTPException(422, f"gateway_trailer must be one of {', '.join(GATEWAY_TRAILERS)}")
            update["gateway_trailer"] = v
        if "opus" in body:
            v = body["opus"]
            if v not in _vc.OPUS_MODES:
                raise HTTPException(422, f"opus must be one of {', '.join(_vc.OPUS_MODES)}")
            update["opus"] = v
        for key, rungs in (("opus_down_kbps", _vc.OPUS_DOWN_KBPS), ("opus_up_kbps", _vc.OPUS_UP_KBPS)):
            if key in body:
                v = body[key]
                if isinstance(v, bool) or v not in rungs:
                    raise HTTPException(422, f"{key} must be one of {', '.join(map(str, rungs))}")
                update[key] = int(v)
        if "narrator_voice" in body:
            v = body["narrator_voice"]
            if v in ("", None):
                update["narrator_voice"] = ""
            else:
                kind = await asyncio.to_thread(classify_voice, settings, v)
                if kind is None or _vc.clean_voice_id(v) is None:
                    raise HTTPException(422, "Unknown voice")
                update["narrator_voice"] = v
        if "bots" in body:
            update["bots"] = _validate_bots(body["bots"])
        if "rig" in body:
            update["rig"] = _validate_rig(body["rig"])
        reset: list[str] = []
        if "reset" in body:
            v = body["reset"]
            if (not isinstance(v, list) or not all(isinstance(k, str) for k in v)
                    or not set(v) <= RESETTABLE):
                raise HTTPException(422, "reset takes a list of: " + ", ".join(sorted(RESETTABLE)))
            reset = [k for k in v if k not in update]
        profile_default = _NOCHANGE
        if "default_voice" in body:
            v = body["default_voice"]
            kind = await asyncio.to_thread(classify_voice, settings, v)
            if kind is None or v == "":
                raise HTTPException(422, "Unknown voice")
            if kind == "builtin":
                from .tts import builtin_id
                update["builtin_voice"] = builtin_id(v)
                profile_default = None              # the built-in voice speaks
            else:
                profile_default = v
        from . import tts as _tts
        if profile_default is not _NOCHANGE:
            try:
                pb = _dv()
                if pb is not None:
                    await asyncio.to_thread(pb.set_default_profile, profile_default)
            except Exception:
                log.exception("voice: could not set the default voice profile")
                raise HTTPException(500, "Could not save the default voice") from None
        was_on = settings.switched_on
        if update:
            await asyncio.to_thread(_vc.save_prefs, update)
        if reset:
            await asyncio.to_thread(_vc.forget_prefs, reset)
        fresh = load_settings()
        _tts.apply_live(fresh)
        await apply_switch(was_on, fresh)
        _APPLIED["on"] = fresh.switched_on
        ch = await asyncio.to_thread(_voice_choices, fresh)
        return _settings_out(fresh, ch)

    @r.get("/preview")
    async def voice_preview(voice: str, text: str | None = None):
        settings = load_settings()
        if text is not None and len(text) > PREVIEW_MAX_CHARS:
            raise HTTPException(422, f"Keep the sample under {PREVIEW_MAX_CHARS} characters")
        line = (text or "").strip() or PREVIEW_TEXT
        kind = await asyncio.to_thread(classify_voice, settings, voice)
        if kind is None or not voice:
            raise HTTPException(422, "Unknown voice")
        if not settings.enabled or await asyncio.to_thread(settings.missing):
            raise HTTPException(503, "Voice isn't set up on this server")
        try:
            wav = await asyncio.to_thread(_preview_wav, settings, voice, line)
        except PreviewBusy:
            raise HTTPException(503, "The voice engine is busy; try the sample again") from None
        except Exception as e:
            # A provider's renderer failing is the engine's problem, not a
            # server crash: say so in a JSON 503 like the rest of the API.
            log.exception("voice: preview of %s failed", voice)
            raise HTTPException(503, f"Couldn't play a sample of that voice ({type(e).__name__})") from None
        return Response(wav, media_type="audio/wav",
                        headers={"Cache-Control": "private, max-age=300"})

    @r.get("/lab")
    async def voice_lab():
        _require_tool("voice-lab")
        settings = load_settings()
        return await asyncio.to_thread(_lab_info, settings)

    @r.post("/lab/test")
    async def voice_lab_test(request: Request):
        _require_tool("voice-lab")
        settings = load_settings()
        if not settings.enabled or settings.missing():
            raise HTTPException(503, "Turn voice on (and finish setting it up) to run a test")
        size = request.headers.get("content-length")
        if size is not None and (not size.isdigit() or int(size) > LAB_TEST_MAX_BYTES):
            raise HTTPException(413, f"Keep the test under {LAB_TEST_MAX_S} seconds")
        raw = await request.body()
        if len(raw) > LAB_TEST_MAX_BYTES:
            raise HTTPException(413, f"Keep the test under {LAB_TEST_MAX_S} seconds")
        if len(raw) % 2 or len(raw) < LAB_TEST_MIN_BYTES:
            raise HTTPException(422, "Send at least half a second of 16 kHz mono PCM16")
        if not _LAB_LOCK.acquire(blocking=False):
            raise HTTPException(409, "A test is already running")
        try:
            return await _lab_test(settings, raw)
        finally:
            _LAB_LOCK.release()

    @r.post("/acks/rebuild")
    async def voice_acks_rebuild(voice: str = ""):
        """Re-render one voice's acknowledgement lines (the Voice lab's Rebuild)."""
        from . import acks
        key = voice.strip()
        if key and _vc_mod().clean_voice_id(key) is None:
            raise HTTPException(422, "Unknown voice")
        if not await asyncio.to_thread(acks.rebuild, key):
            raise HTTPException(404, "No acknowledgement lines are kept for that voice")
        return {"ok": True, "acks": await asyncio.to_thread(acks.status)}

    return r


def _vc_mod():
    from . import config as _vc
    return _vc


def _validate_bots(raw) -> dict:
    """``bots: {<id>: {backchannel?, follow_up?, locked_voice?}}`` for PATCH: a known
    bot (when the context lists bots), boolean flags, and ``locked_voice`` (a voice
    id that exists, or "" for the same as standard), nothing else."""
    from . import config as _vc
    from .config import clean_bots
    if not isinstance(raw, dict):
        raise HTTPException(422, "bots takes an object of {bot id: {backchannel, follow_up}}")
    known = {getattr(b, "id", None): b for b in context.bots_all()}
    for bot_id, conf in raw.items():
        if not isinstance(bot_id, str) or not bot_id:
            raise HTTPException(422, "bots takes an object of {bot id: {backchannel, follow_up}}")
        if known and bot_id not in known:
            raise HTTPException(422, f"Unknown bot: {bot_id[:40]}")
        if not isinstance(conf, dict):
            raise HTTPException(422, f"bots.{bot_id[:40]} takes an object")
        extra = set(conf) - set(BOT_KEYS) - {"locked_voice"}
        if extra:
            raise HTTPException(422, f"Unknown setting: bots.{bot_id[:40]}.{min(extra)}")
        if "locked_voice" in conf:
            lv = conf["locked_voice"]
            if lv is None or lv == "":
                conf["locked_voice"] = ""
            elif (not isinstance(lv, str) or _vc.clean_voice_id(lv) is None
                  or classify_voice(load_settings(), lv) not in ("profile", "builtin")):
                raise HTTPException(422, f"Unknown voice: {str(lv)[:80]}")
        for k, v in conf.items():
            if k == "locked_voice":
                continue
            if not isinstance(v, bool):
                raise HTTPException(422, f"bots.{bot_id[:40]}.{k} must be true or false")
    return clean_bots(raw)


def _validate_rt(raw) -> dict:
    """The ``rt`` tree of a settings PATCH (contracts §F.1): every key named
    with its range, a wrong type or an unknown key → 422 with the reason."""
    from dispatch_voice.rt.settings import POWER_HOLDS, RANGES, SPECULATE_MODES, STT_ENGINES, RtSettings

    from . import config as _vc
    if not isinstance(raw, dict):
        raise HTTPException(422, "rt takes an object with: " + ", ".join(_vc.RT_KEYS))
    unknown = set(raw) - set(_vc.RT_KEYS)
    if unknown:
        raise HTTPException(422, f"Unknown setting: rt.{sorted(unknown)[0]}")
    defaults = RtSettings()
    choices = {"power_hold": POWER_HOLDS, "speculate": SPECULATE_MODES, "stt_engine": STT_ENGINES}
    for k, v in raw.items():
        if k in choices:
            if v not in choices[k]:
                raise HTTPException(422, f"rt.{k} must be one of {', '.join(choices[k])}")
            continue
        d = getattr(defaults, k)
        if isinstance(d, bool):
            if not isinstance(v, bool):
                raise HTTPException(422, f"rt.{k} must be true or false")
        elif isinstance(d, (int, float)):
            lo, hi = RANGES.get(k, (None, None))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (lo is not None and not lo <= v <= hi):
                raise HTTPException(422, f"rt.{k} must be a number between {lo} and {hi}")
            if isinstance(d, int) and int(v) != v:
                raise HTTPException(422, f"rt.{k} must be a whole number between {lo} and {hi}")
    return _vc.clean_rt(raw)


def _validate_rig(raw) -> dict:
    """The ``rig`` tree of a settings PATCH (host/rig.py): ``enabled``, ``url`` (a
    ws:// or wss:// address, or "" to clear it), ``send_reference_clip`` and the
    deadlines, each named with its range. The token is never a setting: it stays in
    the server's environment."""
    from . import rig as _rig
    keys = ("enabled", "url", "send_reference_clip", *_rig.RIG_RANGES)
    if not isinstance(raw, dict):
        raise HTTPException(422, "rig takes an object with: " + ", ".join(keys))
    unknown = set(raw) - set(keys)
    if unknown:
        raise HTTPException(422, f"Unknown setting: rig.{sorted(unknown)[0]}")
    for k in ("enabled", "send_reference_clip"):
        if k in raw and not isinstance(raw[k], bool):
            raise HTTPException(422, f"rig.{k} must be true or false")
    if "url" in raw:
        v = raw["url"]
        if not isinstance(v, str) or (v.strip() and not _rig.valid_url(v.strip())):
            raise HTTPException(422, "rig.url must be a ws:// or wss:// address with no "
                                     "credentials or query (the token stays in the environment)")
    for k, (lo, hi) in _rig.RIG_RANGES.items():
        if k in raw:
            v = raw[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise HTTPException(422, f"rig.{k} must be a whole number between {lo} and {hi}")
    out = _rig.clean_rig({**raw, **({"url": raw["url"].strip()} if isinstance(raw.get("url"), str) else {})})
    return out


def effective_bots(settings: VoiceSettings) -> dict:
    """``{bot id: {backchannel, follow_up}}`` for every bot the context lists
    (the per-bot defaults the Bots-tab popover shows). Since 1.3.1 a ``safe``
    bot is one of them: which bots voice serves is not narrowed by the
    Safe-Mode flag (since 1.3.3 a locked device uses it on those)."""
    out = {}
    for b in context.bots_all():
        bid = getattr(b, "id", None)
        if not bid:
            continue
        out[bid] = settings.bot(bid, backend=context.bot_backend(bid))
    for bid in settings.bots:            # a bot the file names but the context does not (yet)
        if bid not in out and isinstance(bid, str):
            out[bid] = settings.bot(bid)
    for bid, row in out.items():         # only when set, so an untouched bot's row is unchanged
        lv = settings.locked_voice(bid)
        if lv:
            row["locked_voice"] = lv
    return out


def locked_status() -> dict:
    """What a LOCKED device may know (DisPatch ≥ 2.3.4 serves it in this
    add-on's ``/api/extensions`` row): is voice on, and can it run. Read fresh
    each time, like ``/api/voice/status``; ``missing()`` is a few stats."""
    settings = load_settings()
    ready = not settings.missing()
    forced = settings.forced
    enabled = (forced == "on") if forced is not None else bool(settings.switch and ready)
    return {"enabled": bool(enabled), "ready": bool(ready)}


async def status_payload(settings: VoiceSettings) -> dict:
    """``GET /api/voice/status``: today's fields plus the v3 additions that
    exist at this slice (contracts §D)."""
    from dispatch_voice import EXTENSION, __version__

    from . import power
    st = await asyncio.to_thread(settings.status)
    st["extensions"] = list(_LOADED)
    st["voices_ui"] = None                 # the standalone page is `dispatch-voice serve` only
    st.update(await asyncio.to_thread(_voice_choices, settings))
    st["warm"] = _PREWARM.get("state") == "done"
    st["stock_voice"] = _stock_id()
    st["plugins"] = await asyncio.to_thread(_plugins, settings)
    st["langs"] = ["en"]
    # The wire (contracts §D): protocol 3 is spoken to a client that asks for it;
    # `opus` = this server can and may negotiate Opus (libopus present, not `opus: off`).
    from dispatch_voice.transport import PROTO
    from dispatch_voice.transport import opus as _opus
    st["proto"] = PROTO
    st["opus"] = settings.opus != "off" and await asyncio.to_thread(_opus.available)
    st["sessions"] = _sessions_count(settings)
    st["power_hold"] = power.state(settings)
    st["extension"] = {"name": EXTENSION["name"], "version": __version__}
    st["assets_version_at_setup"] = ASSETS_VERSION
    from . import acks
    st["acks"] = await asyncio.to_thread(acks.status)
    st["bots"] = effective_bots(settings)
    from . import rig as _rig
    st["rig"] = await asyncio.to_thread(_rig.status, settings)
    return st


def _sessions_count(settings: VoiceSettings) -> dict:
    ctx = context.get_context()
    fn = getattr(ctx, "socket_count", None) if ctx is not None else None
    if fn is not None:
        try:
            open_, max_ = fn("voice")
            return {"open": int(open_), "max": int(max_)}
        except Exception:
            log.exception("voice: socket_count failed")
    from . import ws
    return {"open": ws.open_count(), "max": settings.max_sessions}


#: Every key PATCH /api/voice/settings accepts.
SETTING_KEYS = frozenset({"enabled", "default_voice", "speed", "voice_turn_thinking",
                          "merge_window_s", "hard_silence_s", "reset", "max_sessions", "rt",
                          "rp_speech", "narrator_voice", "expressive_tags", "gateway_trailer", "bots",
                          "opus", "opus_down_kbps", "opus_up_kbps", "rig"})
#: Keys ``reset`` may put back to the server's default (the Voice lab's).
RESETTABLE = frozenset({"voice_turn_thinking", "merge_window_s", "hard_silence_s"})

LAB_TEST_MAX_S = 10
LAB_TEST_MAX_BYTES = 16000 * 2 * LAB_TEST_MAX_S
LAB_TEST_MIN_BYTES = 16000            # half a second
LAB_TEST_FALLBACK = "I didn't catch anything that time. Try again a little closer to the microphone."
_LAB_LOCK = threading.Lock()


def _ranged(v, name: str, lo: float, hi: float) -> float:
    if (not isinstance(v, (int, float)) or isinstance(v, bool)
            or not lo <= float(v) <= hi):
        raise HTTPException(422, f"{name} must be between {lo} and {hi}")
    return round(float(v), 3)


def _settings_out(settings: VoiceSettings, ch: dict) -> dict:
    # `max_sessions` is read once, at setup, for the gate's `max_total` (the
    # context is sealed after setup): a changed value is saved but takes effect
    # at the next restart, and the response says so for the Settings pane.
    live_max = _sessions_count(settings)["max"]
    return {"enabled": settings.switched_on, "forced": settings.forced,
            "default_voice": ch["default_voice"], "speed": settings.speed,
            "voice_turn_thinking": settings.voice_turn_thinking or "off",
            "merge_window_s": settings.merge_window_s,
            "hard_silence_s": settings.turn_hard_silence_s,
            "max_sessions": settings.max_sessions,
            "max_sessions_live": live_max,
            "restart_required": settings.max_sessions != live_max,
            "rt": settings.rt_settings(),
            "rp_speech": settings.rp_speech,
            "narrator_voice": settings.narrator_voice,
            "expressive_tags": settings.expressive_tags,
            "gateway_trailer": settings.gateway_trailer,
            "opus": settings.opus, "opus_down_kbps": settings.opus_down_kbps,
            "opus_up_kbps": settings.opus_up_kbps,
            "bots": effective_bots(settings),
            "rig": settings.rig.to_dict()}


def _stock_id() -> str | None:
    if "dispatch-voice" not in _LOADED:
        return None
    try:
        from . import profiles_bridge
        return profiles_bridge.stock_id()
    except Exception:
        return None


#: What the Built-in voices tool suggests installing while NO provider is
#: registered: the documented optional plugin. Once a provider registers, it
#: describes itself (registry ``info``) and this is not used.
SUGGESTED_BUILTIN = {
    "name": "dispatch-voice-kokoro",
    "module": "dispatch_voice_kokoro",
    "licence": "GPL-3.0-or-later",
    "install": [
        {"where": "plugin", "cmd": "uv run dispatch-voice-kokoro fetch-models"},
        {"where": "env", "cmd": "DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup"},
        {"where": "backend", "cmd": "uv run --offline --with-editable /path/to/dispatch-voice "
                                    "--with-editable /path/to/dispatch-voice-kokoro uvicorn app.main:app"},
    ],
    "enable": [{"where": "env", "cmd": "DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup"}],
}


def _plugins(settings: VoiceSettings) -> dict:
    """The optional pieces, for Settings and the voice tools: installed?
    ready? what is missing? No paths, no keys. A registered built-in voice
    provider names itself (``registry.builtin_info``)."""
    import importlib.util

    from . import registry
    from .engines import chatterbox
    mounted = "dispatch-voice" in _LOADED
    builtin_missing: list[str] = []
    for spec in list(registry._engines.values()):
        if spec.name != "chatterbox":
            builtin_missing += spec.missing_for(settings)
    has = registry.has_builtin_voices()
    info = registry.builtin_info()
    sug = SUGGESTED_BUILTIN
    try:
        sug_present = importlib.util.find_spec(sug["module"]) is not None
    except Exception:
        sug_present = False
    return {
        "cloning": {"installed": True, "ready": mounted,
                    "missing": [] if mounted else chatterbox.missing()},
        "builtin": {"installed": has,
                    # Not registered, but the suggested plugin is importable:
                    # it only needs enabling.
                    "package": has or sug_present,
                    "ready": has and not builtin_missing,
                    "missing": builtin_missing,
                    "name": info.get("name") if has else None,
                    "licence": info.get("licence") if has else None,
                    "models": info.get("models", []) if has else [],
                    "suggested": {k: sug[k] for k in ("name", "licence", "install", "enable")}},
    }


def _lab_info(settings: VoiceSettings) -> dict:
    """Everything the Voice lab shows. Safe for an unlocked client: no key
    values, no file paths; the HQ address is reduced to its host."""
    import os
    from urllib.parse import urlsplit

    from . import config as _vc
    from . import power, registry, stt, tts
    try:
        resolved = (tts._auto_engine(settings, registry)
                    if settings.tts_engine == "auto" else settings.tts_engine)
    except Exception:
        resolved = None
    fb = None
    if resolved and registry.engine_spec(resolved) is not None:
        spec = registry.fallback_for(settings, resolved, registry.engine_spec(resolved).sample_rate)
        fb = spec.name if spec else None
    with stt._ENGINES_LOCK:
        stt_loaded = bool(stt._ENGINES)
    with tts._ENGINES_LOCK:
        tts_loaded = bool(tts._ENGINES)
    hq_host = ""
    if settings.hq_url:
        try:
            hq_host = urlsplit(settings.hq_url).netloc
        except ValueError:
            hq_host = ""
    miss = settings.missing()
    return {
        "switch": settings.switched_on, "forced": settings.forced,
        "enabled": settings.enabled, "ready": not miss, "missing": miss,
        "stt": settings.stt_engine, "smart_turn": settings.smart_turn_path.exists(),
        "tts": settings.tts_engine, "tts_resolved": resolved, "fallback": fb,
        "engines": registry.engine_names(),
        "loaded": stt_loaded or tts_loaded,
        "warm": _PREWARM.get("state", "idle"), "warm_seconds": _PREWARM.get("seconds"),
        "minimax": {"selected": resolved == "minimax",
                    "key_set": bool(os.environ.get("MINIMAX_API_KEY")),
                    "model": settings.minimax_model, "voice": settings.minimax_voice},
        "hq": {"configured": bool(settings.hq_url), "host": hq_host,
               "model": settings.hq_model, "timeout_s": settings.hq_timeout_s},
        "thinking": settings.voice_turn_thinking or "off",
        "thinking_levels": list(THINKING_LEVELS),
        "timings": {
            "merge_window_s": {"value": settings.merge_window_s,
                               "default": _vc._float("VOICE_MERGE_WINDOW", 1.8),
                               "min": MERGE_WINDOW_MIN, "max": MERGE_WINDOW_MAX},
            "hard_silence_s": {"value": settings.turn_hard_silence_s,
                               "default": _vc._float("VOICE_HARD_SILENCE", 1.4),
                               "min": HARD_SILENCE_MIN, "max": HARD_SILENCE_MAX},
        },
        "power_hold": power.state(settings),
        "platform": power.platform_facts(),
        "acks": __import__("dispatch_voice.host.acks", fromlist=["status"]).status(),
        "rig": _rig_lab(settings),
    }


def _rig_lab(settings: VoiceSettings) -> dict:
    """The Voice lab's GPU rig fields: the status plus the deadlines (no token, the
    address reduced to its host)."""
    from . import rig as _rig
    out = _rig.status(settings)
    rs = settings.rig
    out["deadlines"] = {"stt_final_ms": rs.stt_final_deadline_ms, "tts_start_ms": rs.tts_start_deadline_ms,
                        "ahead_ms": rs.ahead_ms, "first_chunk_tokens": rs.first_chunk_tokens}
    return out


async def _lab_test(settings: VoiceSettings, raw: bytes) -> dict:
    """Hear ``raw`` (16 kHz mono PCM16), then say it back: the same engines a
    voice chat uses, timed. Returns the transcript, the spoken audio as a
    base64 WAV and the latency of each step."""
    import base64
    import time

    from . import registry
    from .tts import REPLY_ACTIVITY, speakable
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    t0 = time.monotonic()
    stt_engine, tts_engine = await warm_models(settings)
    load_ms = round((time.monotonic() - t0) * 1000)
    t1 = time.monotonic()
    text = (await asyncio.to_thread(stt_engine.transcribe, audio) or "").strip()
    stt_ms = round((time.monotonic() - t1) * 1000)
    say = speakable(text) or LAB_TEST_FALLBACK
    pid = await asyncio.to_thread(registry.resolve_profile, None)

    def speak_timed():
        # Never ahead of someone's reply sentence on the engine (review L1).
        REPLY_ACTIVITY.wait_idle(timeout=20)
        t = time.monotonic()
        first = None
        parts = []
        for chunk in tts_engine.stream(say, pid):
            if first is None:
                first = time.monotonic() - t
            parts.append(np.asarray(chunk, "<i2"))
        pcm = np.concatenate(parts) if parts else np.zeros(0, "<i2")
        return pcm, first, time.monotonic() - t

    pcm, first_s, total_s = await asyncio.to_thread(speak_timed)
    rate = int(getattr(tts_engine, "sample_rate", 24000) or 24000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return {
        "transcript": text, "said": say,
        "heard_ms": round(len(audio) / 16),
        "load_ms": load_ms, "stt_ms": stt_ms,
        "tts_first_ms": round(first_s * 1000) if first_s is not None else None,
        "tts_total_ms": round(total_s * 1000),
        "spoken_ms": round(len(pcm) * 1000 / rate),
        "audio_wav_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


#: What mounted: "dispatch-voice" when the cloned-voice engine and the
#: profile router are up, then every DISPATCH_VOICE_EXTENSIONS target that loaded.
_LOADED: list[str] = []
_NOCHANGE = object()

PREVIEW_TEXT = "Hi, this is how I'll sound in voice chat."
PREVIEW_MAX_CHARS = 120
_PREVIEWS: OrderedDict[tuple, bytes] = OrderedDict()
_PREVIEWS_LOCK = threading.Lock()


def _dv():
    """The profile bridge when the cloned-voice engine is mounted, else None."""
    if "dispatch-voice" not in _LOADED:
        return None
    try:
        from . import profiles_bridge
        from .engines import chatterbox
        return profiles_bridge if chatterbox.available() else None
    except Exception:
        return None


def classify_voice(settings: VoiceSettings, voice) -> str | None:
    """"builtin" | "profile" for a known voice id, None for anything else.
    "" is "use the default" (only meaningful per bot)."""
    from .tts import builtin_id, builtin_voices
    if not isinstance(voice, str) or len(voice) > 80:
        return None
    if voice == "":
        return "default"
    b = builtin_id(voice)
    if b:
        return "builtin" if any(v["id"] == voice for v in builtin_voices(settings)) else None
    pb = _dv()
    if pb is not None and pb.profile_exists(voice):
        return "profile"
    return None


def default_builtin(settings: VoiceSettings, builtins: list[dict]) -> str | None:
    """The built-in voice that speaks when no profile applies: the saved
    choice when the provider still lists it, else the provider's default,
    else its first voice. None without a provider."""
    if not builtins:
        return None
    from .tts import BUILTIN
    want = f"{BUILTIN}{settings.builtin_voice}" if settings.builtin_voice else ""
    ids = [v["id"] for v in builtins]
    if want in ids:
        return want
    return next((v["id"] for v in builtins if v.get("default")), ids[0])


def effective_voice(settings: VoiceSettings, profile_id: str | None) -> str | None:
    """The voice that actually speaks for a session whose resolved profile is
    ``profile_id`` (``registry.resolve_profile``'s answer, where None means "the
    engine's default voice").

    None is resolved the way ``/api/voice/status.default_voice`` is: the built-in
    voice a plugin speaks by default (a box with Kokoro and no cloned default
    — the live box), else the stock voice the cloned engine falls back to, else
    None. The acknowledgement kit, its first-chunk prior and the expressive-tag
    decision key on this, so a Kokoro default is never handed non-word lines or
    told it may write ``[laugh]`` (FR-works 3/4)."""
    if profile_id:
        return profile_id
    from .tts import builtin_voices
    try:
        b = default_builtin(settings, builtin_voices(settings))
    except Exception:
        log.debug("voice: built-in voice list unavailable", exc_info=True)
        b = None
    if b:
        return b
    pb = _dv()
    if pb is not None:
        try:
            return pb.default_profile() or pb.stock_id() or None
        except Exception:
            log.debug("voice: profile store unavailable", exc_info=True)
    return None


def _voice_choices(settings: VoiceSettings) -> dict:
    """What the Settings rows render from: default voice, built-ins, profiles.

    ``default_voice`` is the voice a bot with no voice of its own speaks in:
    the default cloned profile, else the plugin's built-in voice, else (no
    built-in voice plugin) the stock synthetic voice, else ""."""
    from .tts import builtin_voices
    pb = _dv()
    builtins = builtin_voices(settings)
    profiles: list[dict] = []
    default = None
    if pb is not None:
        try:
            if not builtins:
                pb.stock_profile(build=False)       # list it once it can exist
            profiles = pb.profiles()
            default = pb.default_profile()
        except Exception:
            log.exception("voice: listing voice profiles failed")
    if not default:
        default = default_builtin(settings, builtins)
    if not default and pb is not None:
        try:
            default = pb.stock_profile(build=False)
        except Exception:
            default = None
    return {"default_voice": default or "",
            "builtin_voices": builtins,
            "profiles": profiles}


class PreviewBusy(Exception):
    """Another sample is rendering, or a chat kept the engine busy too long."""


#: One sample renders at a time, and only between reply sentences: a preview
#: tapped during someone's chat must not delay their next sentence by a
#: whole synth (review L1).
_PREVIEW_SEM = threading.BoundedSemaphore(1)
PREVIEW_WAIT_S = 20.0


def _preview_wav(settings: VoiceSettings, voice: str, text: str) -> bytes:
    from . import tts
    key = (voice, round(settings.speed, 3), text)
    with _PREVIEWS_LOCK:
        if key in _PREVIEWS:
            _PREVIEWS.move_to_end(key)
            return _PREVIEWS[key]
    if not _PREVIEW_SEM.acquire(timeout=PREVIEW_WAIT_S):
        raise PreviewBusy()
    try:
        tts.REPLY_ACTIVITY.wait_idle(timeout=PREVIEW_WAIT_S)
        return _render_preview(settings, voice, text, key)
    finally:
        _PREVIEW_SEM.release()


def _render_preview(settings: VoiceSettings, voice: str, text: str, key: tuple) -> bytes:
    from . import registry, tts
    preview = registry.builtin_preview() if tts.builtin_id(voice) else None
    if preview is not None:
        # The provider renders its own voices, whatever engine is primary.
        pcm, rate = preview(settings, voice, text)
        eng = None
    else:
        eng = tts.shared_engine(settings)
        pcm, rate = eng.synth(text, voice)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate or (eng.sample_rate if eng is not None else 24000)))
        w.writeframes(np.asarray(pcm, "<i2").tobytes())
    data = buf.getvalue()
    with _PREVIEWS_LOCK:
        _PREVIEWS[key] = data
        while len(_PREVIEWS) > 32:
            _PREVIEWS.popitem(last=False)
    return data
