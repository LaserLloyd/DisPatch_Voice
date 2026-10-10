"""Acknowledgement lines in the host: the pools, the background render queue
and the per-session kit the voice session plays from (SPEC §6.2; T5's
``text/acks.py`` is the pure machinery, this module wires it to the engines,
the cache directory and the open sessions).

* ``<data>/voice-acks.yaml`` (``ctx.paths.acks_yaml``) is hot-reloaded through
  :class:`~dispatch_voice.text.acks.PoolsFile` and bound to DisPatch's bot
  list, so a bot whose id or display name is a style key gets that style.
* One :class:`~dispatch_voice.text.acks.RenderQueue` per process renders
  every line a voice may say — the pools in render priority order, then the
  spoken notices and the tool status lines — one engine call at a time, with
  a fixed seed per line, only while no reply is being synthesised and NO voice
  chat is open (see :func:`_busy`), into ``<data>/voice-cache/<engine tag>/``
  (the engine's ``cache_tag``; an engine without one renders into memory only). Lines no pool names are pruned after the first ``ensure`` of a
  voice (never before: Amendment T5-4).
* :class:`AckKit` is what a session gets: the selector (Reactions mode from
  ``start.acks``), the queue, the ETA estimator and the voice/bot keys, plus
  ``pcm``/``clip_s`` for the timing gate (only RENDERED lines are ever picked;
  the ``ack`` earcon covers a voice with none).

The old per-profile ``fillers/<n>.wav`` are no longer played by the host
(SPEC §6.2); the profile store still renders them for the standalone page.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..text import acks as _acks
from ..text.acks import AckSelector, EtaEstimator, Pools, PoolsFile, RenderQueue
from . import context

log = logging.getLogger("dispatch.voice")

#: Prior for "transcript → first reply audio" until the session has two samples
#: (SPEC §6.2: the lab's measured first chunk + 1.2 s TTFT): the cloned voice's
#: first chunk ≈ 1.2 s at 2 GHz; a built-in voice's ≈ 0.4 s.
FIRST_CHUNK_PRIOR_S = {"cloned": 1.2, "builtin": 0.4}

#: Spoken lines that are not acknowledgements but are rendered the same way
#: (and kept by prune): the error notices and the tool status lines.
TOOL_LINES = ("Looking that up", "Searching", "Checking your cards", "Checking your calendar",
              "Checking your email", "Checking my notes", "Checking the weather",
              "Working on the picture", "Checking with the team", "Working on it")
NOTICE_TEXTS = ("Sorry, that didn't go through.", "Nothing to read out. It's in the chat.",
                "Still working on that. It'll be in the chat.", "Still there?")

_LOCK = threading.Lock()
_POOLS: PoolsFile | None = None
_POOLS_PATH: Path | None = None
_RQ: RenderQueue | None = None
_RQ_ENGINE: int | None = None          # id() of the engine the queue renders with
_NAMES: dict[str, str] = {}            # voice key → display name (status rows)
_ETA: dict[str, float] = {}            # voice key → last first-chunk EMA (next session's prior)


def voice_key(profile_id: str | None) -> str:
    """The render queue's voice key: the profile id, "" for the default voice."""
    return str(profile_id or "")


def pools() -> Pools:
    """The current pools (hot-reloaded), bound to the context's bots."""
    global _POOLS, _POOLS_PATH
    path = context.paths().acks_yaml
    with _LOCK:
        if _POOLS is None or _POOLS_PATH != path:
            _POOLS = PoolsFile(path)
            _POOLS_PATH = path
            _POOLS.bind(context.bots_all())
        pf = _POOLS
    return pf.get()


def rebind() -> None:
    """Re-read the bot list (a bot was renamed or added)."""
    with _LOCK:
        pf = _POOLS
    if pf is not None:
        pf.bind(context.bots_all())


def _render_with(tts_engine):
    """``render(voice, text, seed) -> (int16, rate)`` over the engine chain:
    ``render_line`` (seeded, the pure one) when an engine in the chain has it,
    else ``synth``."""
    from . import tts as _tts

    def render(voice: str, text: str, seed: int):
        pid = voice or None
        for e in _tts.walk(tts_engine):
            fn = getattr(e, "render_line", None)
            if callable(fn):
                try:
                    pcm = fn(text, pid, seed=seed)
                    return np.asarray(pcm, "<i2"), int(getattr(e, "sample_rate", tts_engine.sample_rate))
                except LookupError:
                    continue            # not this engine's voice: the next in the chain
        pcm, rate = tts_engine.synth(text, pid)
        return np.asarray(pcm, "<i2"), int(rate)

    return render


def _busy() -> bool:
    """A reply is being synthesised, or ANY voice chat is open: the queue waits.

    Review M2 (six renders held the engine lock for ~20 s and starved the first
    reply) wanted "idle and no session thinking/speaking"; measured on this
    box, that still left a hole — a render that starts while the listener is
    silent holds the cloned-voice engine's lock for the whole line (≈ 2.5–3 s),
    and a turn that ends inside it waits that long for its first audio. A
    synthesis cannot be interrupted once started, so the only safe rule is: no
    render while a chat is open. Lines render at prewarm and after hang-ups;
    a brand-new voice's first chat gets the earcon, every later one the lines.
    ``DISPATCH_VOICE_ACK_RENDER=0`` switches background rendering off entirely."""
    from . import ws
    from .tts import REPLY_ACTIVITY
    if _render_disabled():
        return True
    if REPLY_ACTIVITY.busy:
        return True
    return bool(ws._SESSIONS)


def _render_disabled() -> bool:
    import os
    return os.environ.get("DISPATCH_VOICE_ACK_RENDER", "1").strip().lower() in ("0", "false", "off", "no")


def queue(settings, tts_engine) -> RenderQueue:
    """The process-wide render queue for this engine (rebuilt when the engine
    object changes: a release + reload)."""
    global _RQ, _RQ_ENGINE
    from . import routes
    with _LOCK:
        if _RQ is not None and _RQ_ENGINE == id(tts_engine):
            return _RQ
        old = _RQ

        def cache_dir(voice: str):
            return routes._cache_dir(settings, tts_engine, voice or None)

        rq = RenderQueue(_render_with(tts_engine), busy=_busy, cache_dir=cache_dir,
                         keep=lambda: (*NOTICE_TEXTS, *TOOL_LINES),
                         default_rate=int(getattr(tts_engine, "sample_rate", 24000)))
        _RQ, _RQ_ENGINE = rq, id(tts_engine)
    if old is not None:
        old.stop(timeout=0.1)
    return rq


def current_queue() -> RenderQueue | None:
    with _LOCK:
        return _RQ


def lines_for(bot_id: str | None, profile_id: str | None) -> list[str]:
    """Everything this bot may say in this voice: the pools (priority order), the
    notices and the tool lines."""
    p = pools()
    out = p.render_lines(bot_id=bot_id, voice=profile_id)
    seen = set(out)
    for text in (*NOTICE_TEXTS, *TOOL_LINES, *p.tool_texts()):
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def ensure_voice(settings, tts_engine, *, bot_id: str | None, profile_id: str | None,
                 name: str | None = None, prune: bool = True) -> RenderQueue:
    """Queue the missing lines of this voice (background), then prune the stale
    ones — ``prune`` after ``ensure``, never before (Amendment T5-4)."""
    rq = queue(settings, tts_engine)
    key = voice_key(profile_id)
    if name:
        _NAMES[key] = name
    rq.ensure(key, lines_for(bot_id, profile_id))
    if prune:
        try:
            rq.prune(key)
        except Exception:
            log.exception("voice: ack cache prune failed for %s", key or "default")
    return rq


def status() -> dict:
    """``{voice: {ready, total, failed, name}}`` for ``/api/voice/status.acks``
    and the Voice lab row; the default voice is keyed ``default``."""
    rq = current_queue()
    if rq is None:
        return {}
    out = {}
    for voice, st in rq.status().items():
        key = voice or "default"
        row = dict(st)
        name = _NAMES.get(voice)
        if name:
            row["name"] = name
        out[key] = row
    return out


def rebuild(voice: str) -> bool:
    """Re-render one voice's lines (``POST /api/voice/acks/rebuild?voice=``);
    False when nothing was ever ensured for it."""
    rq = current_queue()
    key = "" if voice in ("", "default") else voice
    if rq is None or key not in rq.status():
        return False
    rq.rebuild(key)
    return True


def release() -> None:
    """Models released / voice switched off: stop the render thread and drop
    the in-memory lines (the disk cache stays)."""
    global _RQ, _RQ_ENGINE
    with _LOCK:
        rq, _RQ, _RQ_ENGINE = _RQ, None, None
    if rq is not None:
        rq.stop(timeout=2.0)
        rq.clear()


def reset() -> None:
    """Tests only."""
    global _POOLS, _POOLS_PATH
    release()
    with _LOCK:
        _POOLS = None
        _POOLS_PATH = None
        _NAMES.clear()
        _ETA.clear()


# --------------------------------------------------------------------------- #
# The per-session kit
# --------------------------------------------------------------------------- #

def _first_chunk_prior(voice: str | None) -> float:
    """``voice`` is the RESOLVED voice (``routes.effective_voice``): a box whose
    default is a built-in voice gets the built-in prior, not the cloned one."""
    from .tts import builtin_id
    key = voice_key(voice)
    if key in _ETA:
        return _ETA[key]
    return FIRST_CHUNK_PRIOR_S["builtin" if builtin_id(voice) else "cloned"]


def resolved_voice(settings, profile_id: str | None) -> str | None:
    """The voice that actually speaks ``profile_id`` (None = the engine's
    default): ``routes.effective_voice``, fail-soft. Every key in this module
    — the render queue's voice, the selector's word-line filter, the ETA prior —
    is this voice, never the bare None (FR-works 4: the live box's Kokoro
    default was keyed "", so the Kokoro word-line rule never applied)."""
    if profile_id:
        return profile_id
    try:
        from . import routes
        return routes.effective_voice(settings, None)
    except Exception:
        log.debug("voice: default voice not resolvable", exc_info=True)
        return None


@dataclass
class AckKit:
    """What one voice session needs to acknowledge (SPEC §6.2)."""
    selector: AckSelector
    queue: RenderQueue | None
    voice: str                     # render-queue key of the character voice
    bot_id: str | None
    estimator: EtaEstimator
    rate: int = 24000
    said: list[str] = field(default_factory=list)   # what this kit played (tests, lab)

    @classmethod
    def silent(cls, *, mode: str = "sometimes") -> AckKit:
        """No pools, no queue: the earcon is all a turn can get."""
        return cls(AckSelector(Pools(default={}), mode=mode), None, "", None,
                   EtaEstimator(first_chunk_s=FIRST_CHUNK_PRIOR_S["cloned"]))

    @property
    def mode(self) -> str:
        return self.selector.mode

    def set_mode(self, mode: str) -> None:
        self.selector.set_mode(mode)

    def pcm(self, text: str) -> np.ndarray | None:
        """The rendered line (int16), or None when it is not rendered (yet)."""
        if self.queue is None:
            return None
        try:
            pcm = self.queue.get(self.voice, text)
        except Exception:
            log.exception("voice: ack lookup failed")
            return None
        return None if pcm is None else np.asarray(pcm, "<i2")

    def clip_s(self, text: str) -> float | None:
        if self.queue is None:
            return None
        try:
            return self.queue.duration_s(self.voice, text)
        except Exception:
            return None

    def has_rendered(self) -> bool:
        """At least one line of this voice is ready (else the earcon stands in)."""
        if self.queue is None:
            return False
        try:
            st = self.queue.status().get(self.voice) or {}
        except Exception:
            return False
        return int(st.get("ready", 0)) > 0

    def ready_count(self) -> int:
        if self.queue is None:
            return 0
        try:
            return int((self.queue.status().get(self.voice) or {}).get("ready", 0))
        except Exception:
            return 0

    def observe(self, *, ttft_s: float | None = None, first_chunk_s: float | None = None) -> None:
        self.estimator.observe(ttft_s=ttft_s, first_chunk_s=first_chunk_s)
        ema = self.estimator.ema.get("chunk")
        if first_chunk_s is not None and ema is not None and self.estimator.n["chunk"] >= 2:
            _ETA[self.voice] = float(ema)      # the next session starts warm


def kit_for(settings, tts_engine, *, bot_id: str | None, profile_id: str | None,
            mode: str = "sometimes", name: str | None = None) -> AckKit:
    """Build a session's kit: the pools bound to the bots, the shared queue (its
    missing lines queued now), the selector in the device's Reactions mode."""
    voice = resolved_voice(settings, profile_id)
    try:
        rq = ensure_voice(settings, tts_engine, bot_id=bot_id, profile_id=voice, name=name)
    except Exception:
        log.exception("voice: ack render queue unavailable; the earcon stands in")
        rq = None
    try:
        p = pools()
    except Exception:
        log.exception("voice: ack pools unavailable")
        p = Pools(default={})
    return AckKit(AckSelector(p, mode=mode), rq, voice_key(voice), bot_id,
                  EtaEstimator(first_chunk_s=_first_chunk_prior(voice)),
                  rate=int(getattr(tts_engine, "sample_rate", 24000)))


def intent_of(text: str, *, after_barge_in: bool = False) -> str:
    return _acks.intent_of(text, after_barge_in=after_barge_in)


def ensure_all(settings, tts_engine, bots: Iterable | None = None, default_profile: str | None = None) -> None:
    """Start-up prewarm: the default voice's lines, then every bot's lines in
    the voice that bot actually speaks (its flavour lines too, so a bot with a
    character flavour in ``voice-acks.yaml`` has them from its first chat; one
    queue, priority order; nothing blocks). Every voice is the RESOLVED one."""
    from . import registry
    default = resolved_voice(settings, default_profile)
    ensure_voice(settings, tts_engine, bot_id=None, profile_id=default)
    for b in bots or ():
        if getattr(b, "safe", False):
            continue
        bid = getattr(b, "id", None)
        try:
            pid = registry.resolve_profile(bid)
        except Exception:
            pid = None
        ensure_voice(settings, tts_engine, bot_id=bid, profile_id=resolved_voice(settings, pid),
                     prune=False)
