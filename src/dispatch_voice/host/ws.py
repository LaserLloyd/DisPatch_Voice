"""``serve(ws, sock)``: the body of ``/ws/voice/{thread_id}`` AFTER DisPatch's
gate (contracts §B.4; SPEC §3.3).

DisPatch owns the gate and the slots: Origin == Host, a live unlocked
session, the thread exists and its bot is not ``safe`` (all 1008 pre-accept),
the per-session slot (4001 "replaced") and the global ``max_total`` (1013
"busy"). What reaches this module is a GATED but NOT yet accepted socket and
a :class:`ThreadSocket` carrying the thread, the bot view, the session token
and three callables (``session_live``, ``touch``, ``close_unavailable``).

This module decides the add-on's own two refusals — the Settings › Voice
switch is off, or the models are missing — with ``sock.close_unavailable``
(accept-then-1013, so a browser reads the reason instead of a bare 1006),
accepts the socket, loads the models (instant after the start-up prewarm),
and runs one :class:`VoiceSession` wired to the host through
:class:`VoiceHost`: ``submit_turn`` (with T5's spoken-reply instructions
built per turn: the bot's backend, its ``voice.yaml`` defaults, the device's
style/follow-up choices and, once, the heard/unheard note of an interruption)
/ ``abort_turn(keep_partial)`` / ``retract_message`` / ``replies_after`` /
``note_heard`` on the context, ``session_live`` and ``touch`` from the
socket, and the reply frames through ``ctx.subscribe_frames`` (unsubscribed
when the session ends). The session also gets its acknowledgement kit (the
pools and the shared render queue, SPEC §6.2), the bot's role-play mode and
the narrator voice (SPEC §6.3). The power-profile hold is taken on a thread
beside the session, never on the way to ``ready``: the hold only makes the
reply faster, so a slow or wedged power daemon must not delay the ready tone
(``ready.power_hold`` may therefore read ``off`` for the first ~0.3 s; the
status route catches up). A socket the gate evicts (4001 "replaced") ends
its session at once through ``sock.on_replaced``.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from . import context, routes
from .config import RP_MODES, VoiceSettings, load_settings
from .routes import MODELS_MISSING, OFF_SERVER, OFF_SETTINGS, UNAVAILABLE

log = logging.getLogger("dispatch.voice")

#: Every open VoiceSession in this process (the switch ends them; status counts them).
_SESSIONS: dict[int, object] = {}
#: The open session of each voice chat, keyed by its commit-cursor key (the device or
#: session-token slot, and the random stream id ``s`` of the chat): a socket carrying the
#: same key IS the same voice chat reconnecting, so it replaces the older session at once
#: (review-T3s4 #4: a half-open predecessor ran on for up to 30 s, synthesising the reply
#: the new one was resuming).
_BY_STREAM: dict[tuple, object] = {}
#: Learned endpointing per device (``?d=`` of the socket; SPEC §4.3): the
#: server's value wins over the device blob's. LRU of LEARNED_MAX devices.
_LEARNED: dict[str, object] = {}
LEARNED_MAX = 64


def learned_for(device: str | None):
    """The stored ``Learned`` of a device, or None."""
    if not device:
        return None
    v = _LEARNED.get(device)
    if v is not None:
        _LEARNED[device] = _LEARNED.pop(device)    # most recently used last
    return v


def remember_learned(device: str | None, learned) -> None:
    if not device or learned is None:
        return
    _LEARNED.pop(device, None)
    _LEARNED[device] = learned
    while len(_LEARNED) > LEARNED_MAX:
        _LEARNED.pop(next(iter(_LEARNED)))


#: One commit cursor per process (``<state_dir>/cursor.sqlite``, contracts §B.3) and one
#: Tailscale relay indicator (its ``tailscale status`` cache is shared by every chat).
_CURSOR: tuple[Path, object] | None = None
_RELAY: object | None = None
_WIRE_LOCK = threading.Lock()


def _cursor_store(state_dir: Path):
    """The process's :class:`CommitCursor` at ``state_dir`` (re-opened when the dir changes:
    tests set up a new context per case)."""
    global _CURSOR
    from dispatch_voice.transport import CommitCursor
    state_dir = Path(state_dir)
    with _WIRE_LOCK:
        if _CURSOR is None or _CURSOR[0] != state_dir:
            if _CURSOR is not None:
                with contextlib.suppress(Exception):
                    _CURSOR[1].close()
            _CURSOR = (state_dir, CommitCursor(state_dir))
        return _CURSOR[1]


def _relay_indicator():
    global _RELAY
    from dispatch_voice.transport import RelayIndicator
    with _WIRE_LOCK:
        if _RELAY is None:
            _RELAY = RelayIndicator()
        return _RELAY


def _query(ws) -> dict:
    try:
        q = ws.query_params
        return {k: q.get(k) for k in q}
    except Exception:
        return {}


def wire_options(ws, sock, settings: VoiceSettings, tts_engine) -> dict:
    """The session's wire (slice 4, SPEC §5): the :class:`Transport` negotiated from the
    socket's query (v2 exactly as before without ``v=3``), the commit cursor and its key
    — ``(device id, or a hash of the unlocked session, stream id)``, none without a valid
    ``s`` (review-T6 M2: never a shared key) — and the relay lookup for ``metrics.link``.
    The session token itself is never stored: the cursor file holds a hash of it."""
    from dispatch_voice.transport import Transport, peer_address
    tr = Transport.negotiate(ws, _query(ws), opus_mode=settings.opus,
                             opus_down_kbps=settings.opus_down_kbps, opus_up_kbps=settings.opus_up_kbps,
                             rate_out=int(getattr(tts_engine, "sample_rate", 24000) or 24000),
                             hard_silence_s=settings.turn_hard_silence_s, hold_max_s=settings.rt.hold_max_s)
    out: dict = {"transport": tr}
    if tr.v3 and tr.stream is not None:
        slot = tr.device
        if slot is None:
            token = str(getattr(sock, "token", "") or "")
            slot = "s-" + hashlib.sha256(token.encode()).hexdigest()[:24] if token else None
        if slot is not None:
            ctx = context.get_context()
            state_dir = getattr(getattr(ctx, "paths", None), "state_dir", None) or context.paths().state_dir
            try:
                out["cursor"] = _cursor_store(state_dir)
                out["cursor_key"] = (slot, tr.stream)
            except Exception:
                log.exception("voice: commit cursor unavailable; reconnects may replay one utterance")
    if tr.v3:
        try:
            client = getattr(ws, "client", None)
            addr = peer_address(getattr(client, "host", None), getattr(ws, "headers", None))
        except Exception:
            addr = None
        relay = _relay_indicator()

        async def lookup() -> dict | None:
            if not addr:
                return None
            try:
                return await relay.lookup(addr)
            except Exception:
                return None
        out["relay"] = lookup
    return out


def _device_of(ws) -> str | None:
    try:
        d = ws.query_params.get("d")
    except Exception:
        return None
    return d if isinstance(d, str) and 8 <= len(d) <= 64 else None


class _Reservation:
    """A chat being built (``serve`` between accept and the session): it counts as
    "a chat is open" for the acknowledgement render queue, never as a session, and
    a switch-off that lands meanwhile ends the session the moment it exists."""

    def __init__(self) -> None:
        self.ended: tuple[int, str] | None = None

    def cancel(self, code: int, reason: str) -> None:
        self.ended = (code, reason)


def open_count() -> int:
    return sum(1 for s in _SESSIONS.values() if not isinstance(s, _Reservation) and hasattr(s, "end"))


async def end_all_sessions(code: int = UNAVAILABLE, reason: str = OFF_SETTINGS) -> int:
    """End every open voice chat (voice was switched off). Returns how many
    sessions were ended (a chat still being built is stopped too, but is not
    counted — review-T3s2 #7)."""
    n = 0
    for session in list(_SESSIONS.values()):
        if isinstance(session, _Reservation):
            session.cancel(code, reason)
            continue
        n += 1
        with contextlib.suppress(Exception):
            session.end(code, reason)
    return n


async def serve(ws, sock, *, settings: VoiceSettings | None = None, engines: tuple | None = None,
                detector=None, voice_for_bot: Callable | None = None) -> None:
    """The handler DisPatch calls for a gated ``/ws/voice/{thread_id}`` socket.
    ``settings``/``engines``/``detector``/``voice_for_bot`` are test seams."""
    ctx = context.get_context()
    if ctx is None:
        raise RuntimeError("voice: serve() needs the DisPatch extension context (host.setup)")
    if settings is None:
        routes.ensure_switch_watch()
        settings = await asyncio.to_thread(load_settings)
        await routes.sync_switch(settings)
    thread_id = str(getattr(sock.thread, "id", ""))
    if not settings.switched_on:
        reason = OFF_SERVER if settings.forced == "off" else OFF_SETTINGS
        log.info("voice: refused %s (%s)", thread_id, reason)
        await sock.close_unavailable(reason)
        return
    if await asyncio.to_thread(settings.missing):      # stat + find_spec: off the loop
        log.info("voice: refused %s (%s)", thread_id, MODELS_MISSING)
        await sock.close_unavailable(MODELS_MISSING)
        return

    # Accept FIRST (the gate passed): the client gets an open socket at once
    # and buffers the mic until `ready`. With the start-up prewarm the loads
    # below are cache hits; on a cold process they take a few seconds.
    await ws.accept()
    if engines is None:
        try:
            engines = await routes.warm_models(settings)
        except Exception:
            log.exception("voice: model load failed")
            with contextlib.suppress(Exception):
                await ws.close(code=UNAVAILABLE, reason="voice models failed to load")
            return
    stt_engine, tts_engine = engines
    if detector is None:
        try:
            from .vad_turn import build_detector
            detector = await asyncio.to_thread(build_detector, settings)
        except Exception:
            log.exception("voice: turn detector load failed")
            with contextlib.suppress(Exception):
                await ws.close(code=UNAVAILABLE, reason="voice models failed to load")
            return

    bot_id = getattr(sock.bot, "id", None)
    # Reserve the chat's place in _SESSIONS NOW: the acknowledgement render
    # queue renders only while no chat is open, and the kit below queues this
    # voice's lines — without the reservation the render thread could start a
    # line (≈ 3 s of the engine lock) in the gap before the session exists,
    # and the first turn would wait for it (measured: +3 s to first audio).
    reservation = _Reservation()
    _SESSIONS[id(reservation)] = reservation
    try:
        try:
            profile_id = await asyncio.to_thread(
                voice_for_bot or (lambda b: routes.voice_for_bot(settings, tts_engine, b)), bot_id)
        except Exception:
            log.exception("voice: voice profile lookup failed; using the default voice")
            profile_id = None
        conv = await asyncio.to_thread(session_options, settings, tts_engine, sock.bot, profile_id)
        rt_pieces = await asyncio.to_thread(realtime_options, settings, stt_engine, tts_engine,
                                            profile_id, conv.get("narrator_id"), _device_of(ws))
        wire = await asyncio.to_thread(wire_options, ws, sock, settings, tts_engine)

        host = build_host(ctx, sock, bot_id=bot_id, profile_id=profile_id, tts_engine=tts_engine)
        from . import power
        from .session import VoiceSession
        session = VoiceSession(ws, thread_id=thread_id, settings=settings,
                               detector=detector, stt_engine=stt_engine,
                               tts_engine=tts_engine, host=host, profile_id=profile_id,
                               notice=lambda text: routes.notice_pcm(tts_engine, profile_id, text),
                               bot_id=bot_id, power_state=lambda: power.state(settings), **conv,
                               **rt_pieces, **wire)
        _SESSIONS[id(session)] = session
        if reservation.ended is not None:
            session.end(*reservation.ended)     # switched off while the chat was being built
        stream_key = wire.get("cursor_key")
        if stream_key is not None:
            older = _BY_STREAM.get(stream_key)
            _BY_STREAM[stream_key] = session
            if older is not None and older is not session:
                log.info("voice: %s reconnected; the older socket of this voice chat is replaced",
                         thread_id)
                with contextlib.suppress(Exception):
                    older.end(REPLACED, "replaced")
    finally:
        _SESSIONS.pop(id(reservation), None)
    # Evicted by a newer socket of the same session (DisPatch's slot, 4001):
    # end now instead of waiting for a half-open phone's close handshake
    # (contracts amendment T1-1; the callback is optional on the socket).
    on_replaced = getattr(sock, "on_replaced", None)
    if callable(on_replaced):
        on_replaced(lambda: session.end(REPLACED, "replaced"))
    unsubscribe = None
    hold = _Hold(settings)
    try:
        # Everything that must be undone is inside the try: a failure anywhere
        # from here (thread exhaustion, a raising context) still runs the
        # clean-up, so no subscription, session entry or hold is left behind.
        unsubscribe = await ctx.subscribe_frames(thread_id, session.on_app_frame, token=sock.token)
        hold.start()                       # beside the session, off the ready path
        await session.run()
    except Exception as e:
        if type(e).__name__ != "WebSocketDisconnect":
            log.exception("voice: session error")
    finally:
        _SESSIONS.pop(id(session), None)
        key = getattr(session, "cursor_key", None)
        if key is not None and _BY_STREAM.get(key) is session:
            _BY_STREAM.pop(key, None)
        # The clean-up must finish even when THIS task is cancelled mid-way (a
        # server shutting down, a test client tearing the handler down): a
        # cancelled await here would leave the power profile held and the
        # frame subscription open, so it runs shielded, as its own task.
        cleanup = asyncio.ensure_future(_cleanup(ws, session, unsubscribe, hold))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            if cleanup.done():
                raise
            cleanup.add_done_callback(lambda _t: None)   # carries on in the loop


def session_options(settings: VoiceSettings, tts_engine, bot, profile_id: str | None) -> dict:
    """The conversation pieces of one session (SPEC §6.2/§6.3; blocking, so
    ``serve`` runs it off the loop): the acknowledgement kit (pools bound to
    the bots, the shared render queue with this voice's missing lines queued),
    the role-play mode (the bot's ``voice_rp``, else the server's
    ``rp_speech``), the narrator voice when it exists, whether the voice
    renders expressive tags, and the bot's follow-up default. Every piece is
    fail-soft: a broken ack queue leaves the earcon, a missing narrator voice
    leaves the character voice."""
    from . import acks, persona
    bot_id = getattr(bot, "id", None)
    backend = context.bot_backend(bot_id)
    # The voice that really speaks (None resolved to the box's default voice):
    # the kit's lines, its ETA prior and the tag decision all follow it.
    voice = acks.resolved_voice(settings, profile_id)
    try:
        kit = acks.kit_for(settings, tts_engine, bot_id=bot_id, profile_id=voice,
                           name=_voice_name(voice))
    except Exception:
        log.exception("voice: acknowledgements unavailable for this session")
        kit = acks.AckKit.silent()
    rp = (getattr(bot, "voice_rp", "") or "").strip() or settings.rp_speech
    narrator = None
    if rp == "dialogue_narrator" and settings.narrator_voice:
        narrator = settings.narrator_voice if _voice_exists(settings, settings.narrator_voice) else None
        if narrator is None:
            log.warning("voice: narrator voice %r is not installed; the character voice narrates",
                        settings.narrator_voice)
    return {"acks": kit, "rp_mode": rp if rp in RP_MODES else "all", "narrator_id": narrator,
            "tags_ok": persona.tags_capable(voice, settings, tts_engine),
            "bot_follow_up": persona.follow_up_default(bot_id, settings, backend),
            "bot_backchannel": bool(settings.bot(bot_id, backend=backend).get("backchannel", False)),
            "bot_backend": backend}


def realtime_options(settings: VoiceSettings, stt_engine, tts_engine, profile_id: str | None,
                     narrator_id: str | None, device: str | None) -> dict:
    """The realtime pieces of one session (SPEC §4; blocking, off the loop):
    the streaming STT over the shared recogniser (``rt.stt_engine``; Moonshine
    falls back to Parakeet tail mode when absent), the turn policy with the
    device's learned endpointing (the server's value wins over the blob's), the
    ``on_learned`` sink, and the voice preparation the session runs in the
    background at ``start`` (Amendment T4-2). Every piece is fail-soft: without
    a streaming STT the 2.1 whole-utterance path runs."""
    from dispatch_voice.rt.endpoint import TurnPolicy
    from dispatch_voice.rt.stt_stream import build_streaming_stt
    out: dict = {"rt": settings.rt}
    try:
        out["sstt"] = build_streaming_stt(settings, stt_engine)
    except Exception:
        log.exception("voice: streaming STT unavailable; whole-utterance transcription")
        return out
    learned = learned_for(device)
    out["policy"] = TurnPolicy(settings.rt, learned)
    out["learned_from_server"] = learned is not None
    out["on_learned"] = lambda value: remember_learned(device, value)
    prepare_fns = [getattr(e, "prepare", None) for e in _walk(tts_engine)]
    prepare_fns = [fn for fn in prepare_fns if callable(fn)]
    if prepare_fns:
        def prepare() -> None:
            pids: list = []
            for pid in (profile_id, narrator_id):      # None = the engine's default voice
                if pid not in pids:
                    pids.append(pid)
            for pid in pids:
                for fn in prepare_fns:
                    try:
                        fn(pid)
                        break
                    except LookupError:
                        continue            # not this engine's voice
        out["prepare"] = prepare
    return out


def _walk(engine):
    from .tts import walk
    return walk(engine)


def _voice_exists(settings: VoiceSettings, voice: str) -> bool:
    try:
        return routes.classify_voice(settings, voice) is not None
    except Exception:
        return False


def _voice_name(profile_id: str | None) -> str | None:
    """The voice's display name for the Voice lab's acknowledgement row."""
    if not profile_id:
        return None
    try:
        from .tts import builtin_id
        if builtin_id(profile_id):
            return builtin_id(profile_id)
        from . import profiles_bridge
        for p in profiles_bridge.profiles():
            if p["id"] == profile_id:
                return p["name"]
    except Exception:
        return None
    return None


class _Hold:
    """One session's share of the power-profile hold (``host/power.py``),
    taken on a thread so the ready tone never waits for the power daemon,
    and given back exactly once: ``release`` first waits for the acquire
    (whatever became of it) so the two can never cross and leave the child
    holding the profile for a session that already ended."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._thread = threading.Thread(target=self._acquire, name="voice-power-acquire", daemon=True)
        self.started = False

    def _acquire(self) -> None:
        from . import power
        try:
            power.acquire(self._settings)
        except Exception:
            log.exception("voice: power hold acquire failed")

    def start(self) -> None:
        try:
            self._thread.start()
        except RuntimeError:               # can't start a thread: no hold, nothing to release
            log.warning("voice: could not start the power-hold thread; voice chat runs at the "
                        "box's current profile")
            return
        self.started = True

    def release(self) -> None:
        """Blocking (the acquire's start timeout at most): call it off the loop."""
        if not self.started:
            return
        from . import power
        self._thread.join(power.START_TIMEOUT_S + STOP_SLACK_S)
        power.release()


#: Grace beyond the holder's start timeout before `release` stops waiting for the acquire.
STOP_SLACK_S = 2.0
#: DisPatch's "a newer socket of the same session took the slot" close code.
REPLACED = 4001


async def _cleanup(ws, session, unsubscribe, hold: _Hold) -> None:
    try:
        await asyncio.to_thread(hold.release)
    except Exception:
        log.exception("voice: power hold release failed")
    if unsubscribe is not None:
        with contextlib.suppress(Exception):
            await unsubscribe()
    code, reason = session.close_code
    with contextlib.suppress(Exception):
        await ws.close(code=code, reason=reason or None)


def build_host(ctx, sock, *, bot_id: str | None, profile_id: str | None = None, tts_engine=None):
    """A :class:`VoiceHost` over the extension context for one socket."""
    from . import persona, registry
    from .session import VoiceHost
    thread_id = str(getattr(sock.thread, "id", ""))

    async def _instructions(opts: dict):
        pid = profile_id
        if pid is None:
            try:
                pid = await asyncio.to_thread(registry.resolve_profile, bot_id)
            except Exception:
                pid = None
        settings = load_settings()
        instructions = persona.instructions_for(
            bot_id, pid, style=opts.get("reply_style", "normal"),
            follow_up=opts.get("follow_up"), interrupted=opts.get("interrupted"),
            settings=settings, tts_engine=tts_engine)
        return instructions, (settings.voice_turn_thinking or None)

    async def submit(tid: str, text: str, opts: dict | None = None) -> str | None:
        opts = opts or {}
        instructions, thinking = await _instructions(opts)
        kw = {"instructions": instructions, "thinking": thinking, "metadata": {"voice": True}}
        if opts.get("prefetched") is not None:
            kw["prefetched"] = opts["prefetched"]       # adopt the speculative run (B.1 rule 4)
        return await ctx.submit_turn(tid, text, **kw)

    speculate_turn = getattr(ctx, "speculate_turn", None)

    async def speculate(tid: str, text: str, opts: dict | None = None):
        """The speculative start (SPEC §4.4): the SAME instructions the real turn
        would get, through ``ctx.speculate_turn`` (advisor/API bots only; the
        host answers None for gateway bots or while a turn holds the thread)."""
        instructions, thinking = await _instructions(opts or {})
        return await speculate_turn(tid, text, instructions=instructions, thinking=thinking)

    async def abort(tid: str, keep_partial: bool = False) -> str | None:
        return await ctx.abort_turn(tid, keep_partial=keep_partial)

    async def retract(tid: str, message_id: str) -> None:
        await ctx.retract_message(tid, message_id, only_metadata_key="voice")

    async def replies_after(tid: str, after_id: str):
        return await ctx.replies_after(tid, after_id)

    async def note_heard(tid: str, message_id: str, *, heard_upto: int, heard_text: str,
                         unheard_text: str) -> bool:
        fn = getattr(ctx, "note_heard", None)
        if fn is None:
            return False
        return bool(await fn(tid, message_id, heard_upto=int(heard_upto),
                             heard_text=heard_text, unheard_text=unheard_text))

    def session_live() -> bool:
        try:
            return bool(sock.session_live())
        except Exception:
            return False

    def touch() -> None:
        fn = getattr(sock, "touch", None)
        if fn is not None:
            fn()

    assert thread_id
    return VoiceHost(submit=submit, session_live=session_live, abort=abort,
                     retract=retract, replies_after=replies_after, touch=touch,
                     note_heard=note_heard,
                     speculate=speculate if callable(speculate_turn) else None)
