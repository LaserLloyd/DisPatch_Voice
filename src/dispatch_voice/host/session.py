"""One voice chat: mic audio in, spoken reply out.

    browser ──PCM16 16 kHz──▶ TurnDetector ──Pause──▶ Smart Turn ‖ StreamingSTT.at_pause ──▶ TurnPolicy
                                   │ SpeechStart        └─ end/hold ─▶ Speculator.start ─▶ final() ─▶ host.submit
                                   ▼ (barge-in / merge)                               (prefetched=run)
    browser ◀── say + PCM chunks ── TTS ◀── speech text ◀── SentenceSplitter ◀── reply frames
                                   (normalise + role-play)   (source spans)    (stream_chunk /
                                                                                stream_done /
                                                                                message / progress)

Everything the session needs from the app comes through :class:`VoiceHost`, so
the module never imports DisPatch and the tests drive it with fakes.

The realtime turn (SPEC §4.1–4.7; slice 3)
-------------------------------------------
With a streaming STT (``sstt``) the detector runs in ``probe_events`` mode: at
every pause ≥ the (learned) probe silence it emits a ``Pause`` and the session
runs Smart Turn and ``at_pause()`` in parallel, then asks ``TurnPolicy``:

* ``end`` — the turn ends now; ``hold`` — ``eot_soft {p, hold_ms}``, the turn
  ends now but its reply is HELD for the learned hold (speech inside it
  merges); ``reprobe`` — Smart Turn again after +0.4 s of silence (end at
  1.0 s); ``listen`` — the hard silence ends it, the bot may backchannel;
* on an ``end``/``hold`` row with text the ``Speculator`` starts the host's
  speculative run (advisor by default, ``rt.speculate``) while ``final()``
  finishes (≤ 300 ms); the run is adopted (``submit(prefetched=run)``) when
  the final text matches, else discarded. A merge or addendum discards;
* the "oh, and…" addendum: a second utterance within ``rt.addendum_s`` ×
  patience that STARTED while nothing of the reply was audible merges with the
  first (retract + abort(keep_partial=False) + one combined submit); once a
  reply sentence was audible it is an interruption. Turn ends run off the audio
  worker, serialised; speech during one joins that turn;
* speech over a reply ducks it (0.25 s of it over the server's audio, 0.6 s over
  the phone's voice, where the duck is informational: VAD alone never cancels),
  and its words decide (``classify_interruption`` on a 0.5 s tick, at the first
  pause, or at the end of the utterance): echo → ``echo_ignored`` + ``resume``;
  backchannel/noise → ``resume``; stop (or a short tick carrying a command word
  the bot is not saying) → ``barge_in`` at once, the utterance goes on and only
  more words make it a turn; real → ``barge_in`` + ``abort_turn(keep_partial=
  True)`` + ``note_heard``, THEN the words become the next turn (Speculator.start
  after the abort; the next submit waits for the note). Words over a reply nobody
  heard yet replace it (no partial, no note). ``resume {gen, rewind}`` rewinds
  the sentence when the pause lasted > 0.8 s (never over the phone's voice); a
  duck with no verdict after ``rt.false_interrupt_timeout_s`` resumes;
* learned endpointing (``TurnPolicy.observe``): a false end (speech back within
  1 s of ``eot``) raises the hold floor and the probe silence; clean turns decay
  them; the values travel in ``ready.rt.learned`` (the server's win over the
  device blob's ``start.rt.learned``; ``start.rt.learned_reset`` forgets them).

What this hub does per reply (SPEC §4.1, §5.2, §6.2–6.3, §4.8)
-----------------------------------------------------------------
* every sentence gets its ``seq`` AT ENQUEUE and a ``say`` frame the instant
  its hold releases — before (or instead of) its first audio; ``say_end``
  closes the reply. In text mode (``link {mode: text}``, or ``start.slow_link:
  always_phone``) the engine is never called: the phone reads the ``say``;
* the reply text is T5's speech text (markdown → words, numbers read out,
  tags filtered) rendered per the bot's role-play mode: a narrator segment is
  its own ``seq`` in the narrator voice, ``role: narrator``;
* acknowledgements are T5's timing gate (``TurnAcks``): one rendered line at
  +0.6 s only when the reply is really going to take a while, the 5 s / 15 s
  wait lines, and a spoken status per gateway tool call ("Searching"),
  rendered through the host's queue — never synthesised at play time. In
  phone-voice mode the ack is the ``ack`` earcon (never a spoken "Mm.");
* heard tracking: each spoken ``seq`` carries its source span; the client's
  ``heard {gen, seq, played_ms, source}`` locates the cut, and a real
  interruption runs abort(keep_partial=True) → wait ≤ 1.5 s for the exact
  post-``barge_in`` ``heard`` → ``note_heard`` (deferred to ``stream_done``
  while the row has no id yet). The heard/unheard note feeds the NEXT turn's
  instructions once; merges, undo and echoes write nothing.

Turn merging (the road-noise simulation's main finding)
-------------------------------------------------------
Smart Turn scored mid-sentence pauses as "finished" in 4 of 10 runs. So after
a turn ends we KEEP LISTENING for ``merge_window`` seconds (× patience):

* reply speech is HELD (not muted) during the window — never speak a reply to
  an utterance that may still be growing. The LLM takes 1–3 s anyway, so the
  hold rarely costs audible time;
* if the driver resumes and the new piece transcribes to real words, the
  in-flight turn is aborted, the first user row is retracted, and the merged
  utterance is submitted as ONE message;
* a cough or an empty transcript just releases the hold.

Wire protocol on ``/ws/voice/{thread_id}``: ``docs/voice-chat/protocol.md``
(contracts §C) is normative. In short — client → server: binary PCM16LE mono
16 kHz, and JSON ``start`` (re-sendable settings: patience, barge_in,
reply_style, follow_up (null = the bot's default), acks, slow_link, after,
after_ids), ``pause``, ``stop_playback``, ``playback_done {gen, seq?}``,
``cancel``, ``undo``, ``ping``, ``heard``, ``link {mode}``. Server → client:
``ready`` (with ``sid``, ``say``, ``acks``, ``langs``, ``power_hold``),
``state``, ``eot``, ``transcript``, ``echo_ignored``, ``ack`` (earcon),
``say``/``say_end``, ``audio`` + one binary frame, ``duck``, ``barge_in``,
``turn_error``, ``undone``, ``pong``, ``metrics``, ``locked``, ``error``.

Close codes the session ends with: 1008 locked, 4001 replaced (a newer socket
of the same session took over), 1001 heartbeat lost, 1011 internal error.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from ..rt.backchannel import (
    COMMAND_WORDS,
    FILLERS,
    BackchannelGate,
    classify_interruption,
)
from ..rt.backchannel import tokens as _bc_tokens
from ..rt.endpoint import REPROBE_AFTER_S, Learned, Signals, TurnPolicy
from ..rt.heard import HeardMap, HeardPosition, Span
from ..rt.settings import RtSettings
from ..rt.speculate import Speculator
from ..text.acks import TurnAcks
from ..text.speech_text import RpRenderer, SentenceSplitter, normalise
from ..transport import GapTooLarge, Transport, wire3
from .acks import AckKit, intent_of
from .config import PATIENCE, REPLY_STYLES, RP_MODES
from .tts import REPLY_ACTIVITY
from .vad_turn import FRAME as VAD_FRAME
from .vad_turn import Discarded, Pause, Resumed, SpeechStart, TurnDetector, TurnEnd

log = logging.getLogger("dispatch.voice")


@dataclass
class VoiceHost:
    """What the app lends a voice session. ``ws.build_host`` builds one per socket."""
    #: Persist ``text`` as a user message in the thread and run the normal
    #: agent turn (voice instructions, no-thinking). Returns the new message id.
    #: May take a third argument: ``{"reply_style", "follow_up" (bool | None =
    #: the bot's default), "interrupted" ({heard, unheard} | None), "prefetched"
    #: (a SpeculativeRun to adopt, or None)}``.
    submit: Callable[..., Awaitable[str | None]]
    #: ``speculate(thread_id, text, opts) -> SpeculativeRun | None``: start the
    #: model call before the turn is committed (contracts §B.1 ``speculate_turn``,
    #: same instructions as ``submit`` would build). None = no speculation.
    speculate: Callable[..., Awaitable[object | None]] | None = None
    #: Is the unlocked session that opened this socket still valid?
    session_live: Callable[[], bool] = lambda: True
    #: Stop the voice turns this thread is waiting on: every one that has not
    #: started yet is cancelled, the running one is aborted. May take
    #: ``keep_partial`` (contracts §B.1: True = an interruption, the spoken
    #: prefix is persisted and its row id returned; False = a merge, nothing).
    abort: Callable[..., Awaitable[str | None]] | None = None
    #: Remove a user row this session submitted (turn merge, undo).
    retract: Callable[[str, str], Awaitable[None]] | None = None
    #: Assistant replies newer than a message id: [(id, content), ...].
    replies_after: Callable[[str, str], Awaitable[list[tuple[str, str]]]] | None = None
    #: Slide the opening session's idle window (auth.touch_session). Called on
    #: real activity — mic audio and control frames, never a ping — at most
    #: once per ``TOUCH_EVERY_S``, so a drive longer than the idle lock does
    #: not lock the session mid-sentence.
    touch: Callable[[], None] | None = None
    #: ``note_heard(thread_id, message_id, *, heard_upto, heard_text, unheard_text) -> bool``
    #: (contracts §B.1): stamp how much of a reply was heard; False = unknown id.
    note_heard: Callable[..., Awaitable[bool]] | None = None


#: Minimum seconds between two VoiceHost.touch calls.
TOUCH_EVERY_S = 30.0
#: A merged utterance (driver kept resuming) never grows past this many chars.
MAX_UTTERANCE_CHARS = 4000
#: A reply stream with no frame for this long no longer counts as "still
#: coming" (a turn that died without a stream_done must not pin `speaking`).
REPLY_STALE_S = 30.0
#: `undo` works this long after the transcript it takes back.
UNDO_WINDOW_S = 8.0
#: A turn with nothing spoken this long after it was sent is given up on.
TURN_TIMEOUT_S = 120.0
#: After the agent says it stopped, wait this long for late frames before
#: deciding the reply had nothing to say out loud.
NOTHING_SPOKEN_GRACE_S = 1.0
#: The unlocked session is re-checked at least this often, even when the
#: socket is silent (paused) — review M7.
LIVE_CHECK_S = 15.0
#: A client that pings and then goes silent this long is gone (half-open).
DEAD_AFTER_S = 30.0
#: A duck (possible barge-in) that never turns into a verdict is undone (SPEC §4.6).
DUCK_MAX_S = 4.0
#: A reply paused this long resumes from the start of its sentence (``resume {rewind}``).
REWIND_AFTER_S = 0.8
#: Speech back within this long of an ``eot`` is a false end (learned endpointing).
FALSE_END_S = 1.0
#: ``partial`` frames go out at most this often (C.4: ≤ 2/s).
PARTIAL_MIN_GAP_S = 0.5
#: ``quiet_check`` is answered at most once per this long (C.3).
QUIET_CHECK_EVERY_S = 60.0
#: Idle-touch policy (SPEC §5.6): an attached chat with the mic enabled and a
#: client frame (pings count) younger than this slides the idle window.
IDLE_TOUCH_FRESH_S = 10.0
#: Close code for a voice chat the server ended because nothing was said either way
#: for ``rt.quiet_max_s`` (FR-hardening L3). Not a network drop: the client stops and
#: does not reconnect (Amendment FINAL-FIX-2).
QUIET_CLOSE = 4002
QUIET_REASON = "quiet"
#: A backchannel line plays at −6 dB (SPEC §4.7).
BACKCHANNEL_GAIN = 0.5
#: The echo verdict applies only while audio is actually playing (Amendment T4-3):
#: speaking, or within this long of the last reply audio stopping.
ECHO_PLAYING_GRACE_S = 1.0
#: A tick partial (speech still going over a reply) confirms a REAL interruption
#: only with at least this many words; shorter ones wait for the pause verdict.
TICK_MIN_WORDS = 3
#: While speech runs over a reply the tail is re-read this often (captions alone:
#: the STT's own 1 s). Natural interruptions rarely pause ≥ 0.25 s in their first
#: second ("Wait, stop, that's…" has one 0.13 s gap), so the word verdict comes
#: from these ticks, not from a probe: onset → `barge_in` ≤ 1.0 s (SPEC §12).
TICK_OVER_REPLY_S = 0.5
#: Sustained speech before the server ducks a reply, by output path (SPEC §4.6
#: layer 2, §13 C5): the server's own audio is reversible (paused, resumed), so
#: 0.25 s; the phone's voice has no echo reference over a car kit and cannot be
#: paused, so 0.6 s (and the duck is only informational there).
SERVER_AUDIO_BARGE_IN_S = 0.25
PHONE_VOICE_BARGE_IN_S = 0.6
#: Silero margin while our own voice may reach the mic (SPEC §4.6 layer 2).
SPEAKING_VAD_THRESHOLD = 0.65
#: Per-seq bookkeeping (heard map, seq → reply id/text) kept for this many pieces;
#: hours of chat must not grow it without bound (review-T3s2 #10).
SEQ_KEEP = 256
#: Reply streams remembered (sentence state, provisional → row ids).
REPLIES_KEEP = 64
#: A stored row whose content starts with a streaming reply's first characters is that reply.
HEAD_CHARS = 80
#: The uplink audio queue is bounded in BYTES (contracts §C.2, A11): 2 s of 16 kHz PCM.
AUDIO_QUEUE_BYTES = 64_000
#: While replayed uplink audio (the client's ring after a reconnect, flags bit1) is queued
#: the queue's bound is this instead (review-T3s4 #8): a replay is a burst by design and a
#: dropped piece of it is a lost utterance; the client's ring is ≤ 320 KB (10 s of PCM16,
#: 30 s of Opus). Live audio is dropped at once otherwise (the CPU is behind).
REPLAY_QUEUE_BYTES = 512_000
#: Control-frame budget (L9): a bucket of RATE_BURST frames refilled at RATE_PER_S.
RATE_BURST = 40
RATE_PER_S = 10.0
#: `ping` has its own bucket (review-T3s4 #9): a burst of control frames must never cost
#: the heartbeat its `pong` (the client counts unanswered pings toward "the link is dead");
#: the client pings every 2 s, so 10 then 2/s still bounds a flood.
PING_BURST = 10
PING_PER_S = 2.0
#: `start.after_partial` entries kept (review-T3s4 #9) and the longest message id accepted.
AFTER_PARTIAL_KEEP = 8
ID_MAX_CHARS = 64
#: A JSON control frame longer than this is not parsed (a `start` is < 2 KB): it is a
#: rate-limited `bad_frame`, so a flood of big frames never costs parse time.
CONTROL_MAX_CHARS = 16_384
#: A `typed` frame's text (C.3, 1.2.0): the same bound as a spoken utterance.
TYPED_MAX_CHARS = MAX_UTTERANCE_CHARS
#: After a `barge_in` the client sends one exact `heard` for the new gen at
#: once; the session waits at most this long for it before `note_heard`
#: (SPEC §4.8, hardening L7); past that the last periodic `heard` is used.
HEARD_WAIT_S = 1.5
#: A periodic `heard` older than this is not trusted as "where they were".
HEARD_STALE_S = 3.0
#: A deferred note (the row had no id yet) is dropped after this long.
DEFERRED_NOTE_S = 30.0
#: `replay_last` / "say that again" (SPEC §4.6 item 4, C.3): the last reply (and the
#: last cut) stays replayable this long after its last piece was said.
REPLAY_KEEP_S = 120.0
#: The last reply's synthesised audio kept for an instant replay, at most this many
#: seconds of it (≈ 4.3 MB at 24 kHz); pieces past the cap are synthesised again.
REPLAY_PCM_MAX_S = 90.0
#: A reply's text is tracked (for a replay) up to this many characters.
REPLY_TEXT_MAX = 32_000
#: Reply texts remembered for `replay_last` (the cut reply and the last one said).
REPLY_TEXTS_KEEP = 16
#: Cached audio goes out in pieces of this many seconds (the hold/duck is honoured between).
REPLAY_CHUNK_S = 0.5
#: A listener at least this far into the cut piece heard it: a continue starts at the next.
REPLAY_HEARD_FRACTION = 0.95
#: The whole utterance, fillers aside, is one of these: say the last reply again
#: without a turn (SPEC §4.6 item 4; "while listening" — never over a reply).
REPLAY_COMMANDS = frozenset({
    "say that again", "say it again", "say again", "say that one more time",
    "repeat that", "repeat it", "repeat", "repeat that again",
    "can you repeat that", "could you repeat that", "can you say that again",
    "could you say that again", "come again", "what did you say", "pardon", "pardon me",
    "one more time",
})
#: ... or only its last sentence (review-T3s4 #11: "the last part" replayed the whole reply).
REPLAY_LAST_COMMANDS = frozenset({
    "repeat the last part", "repeat the last bit", "repeat the last sentence",
    "say the last part again", "say the last bit again", "say the last sentence again",
    "the last part again", "what was the last part", "can you repeat the last part",
    "could you repeat the last part",
})
#: Words a replay command may carry around it ("sorry, say that again please").
REPLAY_FILLERS = frozenset({"um", "uh", "erm", "er", "hmm", "oh", "please", "sorry", "hey",
                            "so", "just", "okay", "ok"})
#: Send-ahead pacing on a v3 socket (T3s4): reply audio goes out at most this many
#: seconds ahead of real-time playback. The engine makes speech 3-10x faster than it
#: plays; unpaced, a whole reply's audio sat in the socket's buffers, so on a slow link
#: everything sent after it — the phone voice's `say` frames after a switch to text,
#: `pong`, `barge_in` — waited behind seconds of audio the client would drop
#: (measured: 3g-weak with a PCM downlink, 400 KB queued, pongs 9-12 s late, the
#: client's dead-socket rule fired mid-reply).
SEND_AHEAD_S = 4.0
#: The lead on a PCM16 downlink (review-T3s4 #7; a client without WebCodecs Opus): the
#: lead's BYTES are what queues pongs and control frames behind audio on a slow link.
PCM_SEND_AHEAD_S = 1.25
#: Paced audio goes out in slices of at most this many seconds (an engine that yields a
#: whole sentence as one chunk — Kokoro — would otherwise put all of it in flight at once).
PACE_SLICE_S = 0.25
#: A merge hold with no utterance open and no turn end in progress is stuck after this
#: long (the speech it waited for is gone): released by the watchdog (review-T3s3 #1(d);
#: the longest real hold is the addendum window, 3 s × patience, while speech is open).
HOLD_STUCK_S = 4.0
#: The detector is fed at most this much audio per call (1 s of 16 kHz PCM16).
FEED_MAX_BYTES = 32_000
#: A coalesced uplink batch bigger than this (messages or bytes) is decoded in a worker
#: thread (review-T3s4 #10); live audio (one 100 ms message at a time) stays inline.
DECODE_INLINE_ITEMS = 4
DECODE_INLINE_BYTES = 8_000
#: Frames of the next utterance kept while a turn end reads the stream (≈ 60 s of 32 ms
#: frames; a turn end takes ≲ 5 s even with a slow gateway abort).
SINK_BUF_MAX = 2000
#: Reactions modes (`start.acks`, contracts §C.3).
ACK_MODES = ("off", "sometimes", "often")
#: `start.slow_link` values the server reads (`always_phone` = text mode for the session).
SLOW_LINK_MODES = ("phone", "wait", "text", "always_phone")

#: What the driver hears when a turn ends with nothing to say (H1), and the
#: "Hang up when quiet" check (experience M9).
NOTICE_LINES = {
    "agent_failed": "Sorry, that didn't go through.",
    "no_speech_in_reply": "Nothing to read out. It's in the chat.",
    "timeout": "Still working on that. It'll be in the chat.",
    "quiet_check": "Still there?",
}


@dataclass
class _Piece:
    """One spoken piece of a reply, in the order the reply was split: its seq (None = not
    said here: heard before a reconnect, or before the cut a continue resumed from), its
    source span in the reply's text, the words, and the queued item while it has one."""
    seq: int | None
    start: int
    end: int
    text: str
    say: _Say | None = None


@dataclass
class _Reply:
    """One reply stream (provisional or final id) and its sentence bookkeeping.

    The text is handed to the splitter canonically (:func:`feed_canonical`), so the pieces
    depend on the text alone, never on how the deltas happened to be cut: a resume that
    splits the stored row or a snapshot whole gets exactly the pieces the listener heard
    (review-T3s4 #1)."""
    splitter: SentenceSplitter = field(default_factory=SentenceSplitter)
    renderer: RpRenderer = field(default_factory=RpRenderer)
    consumed: int = 0               # = len(text)
    done: bool = False
    muted: bool = False
    updated: float = 0.0
    mid: str | None = None          # the persisted row id, once known
    last_seq: int | None = None     # highest seq queued for this reply
    pieces: int = 0                 # spoken pieces queued
    end_sent: bool = False          # say_end went out
    skipped: int = 0                # leading pieces not said here (heard on an earlier socket / before a cut)
    late: bool = False              # joined mid-stream without its stream_start (C.5 #11)
    head: str = ""                  # the first characters received (a stored row is the same reply)
    log: list[_Piece] = field(default_factory=list)   # every piece, in order (said, queued or skipped)
    continued: bool = False         # rebuilt by `replay_last` after a cut (continue, not a replay)
    text: str = ""                  # the reply's text as received (what the pieces come from)
    pending: str = ""               # received text not handed to the splitter yet (no whitespace after it)
    #: A continue (`replay_last`): pieces whose source span ends at or before this offset were
    #: heard before the cut and are skipped — by source offset, never by count (review-T3s4 #1).
    skip_upto: int | None = None
    #: After a rewrite: words the listener already has that the new text has not confirmed
    #: yet (its splitter needs the whitespace after a sentence): a new piece made of the next
    #: of these words IS heard already (never said twice); any other piece ends the wait.
    expect: list[str] = field(default_factory=list)

    @property
    def seqs(self) -> list[int]:
        """The seq of every piece said or queued here, in order."""
        return [p.seq for p in self.log if p.seq is not None]


@dataclass
class _ReplyText:
    """A reply's text as this session knows it, kept even while the reply is muted (a
    tap-stop silences the rest of a run that goes on into the chat): `replay_last`
    continues a cut reply from it, or says a finished one again (SPEC §4.6 item 4)."""
    text: str = ""
    done: bool = False              # the final content (stream_done / message)
    mid: str | None = None          # the persisted row id, once known
    late: bool = False              # deltas of a stream seen without its start: only the final text counts
    at: float = 0.0


@dataclass
class _Say:
    """One queued spoken piece; ``seq`` is assigned when it is queued (SPEC §5.2)."""
    gen: int
    seq: int
    text: str                       # exactly what the TTS gets / the say frame carries
    message_id: str | None
    kind: str = "reply"             # reply | ack | notice
    role: str = "character"         # character | narrator
    voice: str | None = None        # the engine's profile id for this piece
    rid: str | None = None          # the reply stream it belongs to
    intent: str | None = None       # ack lines: the intent they answer
    pcm: np.ndarray | None = None   # pre-rendered (acks, notices): no engine call
    dropped: bool = False           # a rewrite replaced it before its `say` went out (review-T3s4 #5)
    say_sent: bool = False          # its `say` is going out / went out: the listener hears it


@dataclass
class _Turn:
    """One spoken message this session submitted, from submit to its answer."""
    msg_id: str | None
    text: str
    at: float                      # clock() when the transcript was sent
    echoed: bool = False           # its user row came back through the tap
    started: bool = False          # the agent run for it began (thinking/started)
    stopped: bool = False          # ... and ended (thinking/stopped)
    spoke: bool = False            # at least one speakable sentence queued
    said: bool = False             # a reply sentence's `say` went out: the listener heard it start
    closed: bool = False           # answered, failed or abandoned: no more checks
    abandoned: bool = False        # cancelled/merged/undone: its reply stays silent
    failed: bool = False           # turn_error went out for it
    abort_seq: int | None = None   # frame counter when the host abort ran
    first_text_at: float | None = None   # when its reply's first text arrived
    said_at: float | None = None   # when its reply first became audible
    speculated: bool = False       # a speculative run was started for it
    adopted: bool = False          # ... and the submit adopted it


@dataclass
class _Heard:
    gen: int
    seq: int
    played_ms: int
    source: str
    at: float


@dataclass
class _Utt:
    """One utterance's own state, from its SpeechStart to its end. A turn end that is
    still running (or queued behind the previous one) when the NEXT utterance begins
    keeps reading ITS utterance — never the new one's freshly reset flags (review-T3s3
    #1(c): a queued end lost its stop-cut abort and abandoned the cut run without its
    spoken prefix)."""
    uid: int = 0
    started_at: float | None = None
    #: began while our voice was playing (None = no SpeechStart seen): only such an
    #: utterance can be an echo (Amendment T4-3)
    while_playing: bool | None = None
    interrupted: bool = False       # already cut the reply (a real verdict)
    stop_cut: bool = False          # a tick's stop word cut the reply (the run goes on)
    stop_cut_aborted: bool = False  # ... and that run was aborted since
    hold_onset: float | None = None  # the onset of the speech a merge hold is for


def _log_task_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("voice: a background step failed", exc_info=exc)


def _platform_facts() -> dict:
    """C.4 ``metrics.platform_profile`` / ``max_mhz`` (SPEC §4.11: every timing is read
    beside the power profile it ran in); never raises."""
    try:
        from .power import platform_facts
        return {k: v for k, v in platform_facts().items() if v is not None}
    except Exception:  # noqa: BLE001 - a metrics nicety; the turn never depends on it
        return {}


def _norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _is_replay(buf: bytes) -> bool:
    """A v3 uplink message from the client's ring (flags bit1 ``replay``, C.2)."""
    return len(buf) >= 3 and wire3.is_v3(buf) and bool(buf[2] & wire3.FLAG_REPLAY)


def _patience_value(v, default: float) -> float:
    if isinstance(v, str) and v in PATIENCE:
        return PATIENCE[v]
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.3 <= float(v) <= 3.0:
        return float(v)
    return default


def _takes_opts(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3 or any(p.kind == p.VAR_POSITIONAL for p in params)


def _accepts_kw(fn, name: str) -> bool:
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _int_field(msg: dict, key: str, lo: int = 0) -> int | None:
    v = msg.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        v = int(v)
    except (OverflowError, ValueError):
        return None
    return v if v >= lo else None


_WS_CHAR = re.compile(r"\s")


def feed_canonical(splitter: SentenceSplitter, pending: str, new: str, *,
                   final: bool) -> tuple[list[Span], str]:
    """Hand ``pending + new`` to ``splitter`` in a CANONICAL order: one push per whitespace
    character (each push ends right after it), the tail without whitespace kept back as the
    new ``pending``; ``final`` pushes the tail and flushes. The splitter therefore sees the
    same sequence of prefixes however the text arrived — deltas of any size, a snapshot,
    the stored row — and its pieces depend on the text alone (review-T3s4 #1: its
    first-sentence clause cut fires on a prefix without the full stop, so a reply streamed
    as deltas and the same reply split whole came out as different pieces, and a resume
    that skips ``after_partial.sentences`` pieces lost a sentence). A sentence end is only
    visible with whitespace after it, so nothing is spoken later than before. Returns
    ``(spans, pending)``."""
    buf = pending + new
    out: list[Span] = []
    i = 0
    for m in _WS_CHAR.finditer(buf):
        j = m.end()
        out += splitter.push(buf[i:j])
        i = j
    rest = buf[i:]
    if final:
        if rest:
            out += splitter.push(rest)
        out += splitter.flush()
        rest = ""
    return out, rest


def split_reply(text: str) -> list[Span]:
    """A whole reply's sentence spans exactly as a voice session splits it (streamed or
    not): the harnesses compare what was heard against this."""
    spans, _ = feed_canonical(SentenceSplitter(), "", text or "", final=True)
    return spans


def subspans(span: Span, pieces: list[tuple[str, str]]) -> list[Span]:
    """The source span shared out over the role-play pieces of one sentence,
    proportionally to their text, so every spoken piece keeps a source range
    (Amendment T5-2: ``text`` = the words the engine said, ``source`` = the
    markup). One piece = the whole span with its text replaced."""
    n = len(span)
    if len(pieces) <= 1:
        text = pieces[0][1] if pieces else span.text
        return [Span(text, span.start, span.end, span.source)]
    total = sum(len(t) for _, t in pieces) or 1
    out: list[Span] = []
    acc = 0
    a = span.start
    for i, (_, text) in enumerate(pieces):
        acc += len(text)
        b = span.end if i == len(pieces) - 1 else span.start + round(acc / total * n)
        b = max(a, min(span.end, b))
        src = span.source[a - span.start:b - span.start] if span.source else ""
        out.append(Span(text, a, b, src))
        a = b
    return out


class VoiceSession:
    def __init__(self, ws, *, thread_id: str, settings, detector: TurnDetector,
                 stt_engine, tts_engine, host: VoiceHost,
                 profile_id: str | None = None, acks: AckKit | None = None,
                 notice: Callable[[str], object] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 bot_id: str | None = None, narrator_id: str | None = None,
                 rp_mode: str = "all", tags_ok: bool = False,
                 bot_follow_up: bool = True,
                 power_state: Callable[[], str] | None = None,
                 heard_wait_s: float = HEARD_WAIT_S,
                 sstt=None, policy: TurnPolicy | None = None,
                 speculator: Speculator | None = None, rt: RtSettings | None = None,
                 bot_backchannel: bool = False, bot_backend: str | None = None,
                 prepare: Callable[[], object] | None = None,
                 on_learned: Callable[[Learned], None] | None = None,
                 learned_from_server: bool = False,
                 transport: Transport | None = None, cursor=None, cursor_key: tuple | None = None,
                 relay: Callable[[], Awaitable[dict | None]] | None = None,
                 send_ahead_s: float | None = SEND_AHEAD_S, rig=None) -> None:
        self.ws = ws
        self.thread_id = thread_id
        self.settings = settings
        self.detector = detector
        self.stt = stt_engine
        self.tts = tts_engine
        self.host = host
        self.profile_id = profile_id      # opaque voice id for the TTS engine
        self.bot_id = bot_id
        # ---- the wire (slice 4, SPEC §5): every frame goes out through the transport
        # (v2 exactly as before without `v=3`); the commit cursor makes a reconnect
        # replay-safe (C.5 #8/#10). Without a transport the session speaks v2.
        self.tr = transport if transport is not None else Transport(
            ws, rate_out=int(getattr(tts_engine, "sample_rate", 24000) or 24000))
        self.cursor = cursor if (cursor is not None and cursor_key is not None) else None
        self.cursor_key = cursor_key if self.cursor is not None else None
        self._relay = relay
        self._relay_info: dict | None = None
        #: Send-ahead pacing (v3 only; None = off): playback is assumed to run in real
        #: time from the first audio after the client drained, minus the time held.
        self.send_ahead_s = send_ahead_s if self.tr.v3 else None
        self._pace_t0: float | None = None
        self._pace_sent = 0.0
        self._pos = 0                     # uplink offset (u32) after the last sample fed
        self._local_speaking = False      # uplink flag: the phone's own voice is playing
        self._audio_q_bytes = 0
        self._replay_q_bytes = 0          # bytes of replayed (ring) audio in the queue
        self._dropped_audio = 0
        self._last_cursor: object = object()        # start.cursor, re-read only when it changes
        self._last_after_partial: object = object()
        self._after_partial: dict[str, int] = {}    # reply id -> leading pieces already heard
        self._eot_upto: int | None = None           # `upto` of the last eot (the cursor's commit)
        self._streams_started: deque[str] = deque(maxlen=64)   # stream_start ids seen (C.5 #11)
        #: The end offset of the highest REPLAYED uplink audio fed (flags bit1): an utterance
        #: that ends inside it at or below the stream's committed offset was submitted by an
        #: earlier socket — never submitted twice (C.5 #8, the belt to the cursor answer).
        self._replay_hi: int | None = None
        # ---- `replay_last` / "say that again" (SPEC §4.6 item 4): the reply texts, the
        # last reply said, its audio for an instant replay, and the last cut.
        self._texts: OrderedDict[str, _ReplyText] = OrderedDict()
        self._last_said_rid: str | None = None
        self._last_said_at: float | None = None
        self._pcm_keep: dict[tuple[str, str, str | None], np.ndarray] = {}
        self._pcm_rid: str | None = None
        self._pcm_samples = 0
        self._cut_at: float | None = None           # when the last cut happened
        self._cut_turn: _Turn | None = None         # the turn whose reply it cut
        self._cut_heard: _Heard | None = None       # where the listener was (the best `heard` of the cut)
        #: The reply a continue is playing (stream id): when its last piece drains, its row is
        #: noted as heard in full (the cut's note no longer holds).
        self._continued: str | None = None
        # ---- the realtime pieces (slice 3; every one optional: without a
        # streaming STT the 2.1 whole-utterance path runs unchanged) ----
        self.rt = rt if rt is not None else (getattr(settings, "rt", None) or RtSettings())
        if sstt is not None and not hasattr(detector, "frame_sink"):
            # The streaming STT hears only what the detector's frame sink hands it; a
            # detector without one (a scripted double, an older build) would leave it
            # empty and every turn would transcribe to "". Whole-utterance path instead.
            log.info("voice: the turn detector has no frame sink; whole-utterance transcription")
            sstt = None
        self.sstt = sstt                  # StreamingSTT (C-§B.3), or None
        self.policy = policy or TurnPolicy(self.rt)
        self.bot_backend = bot_backend
        self.bot_backchannel = bool(bot_backchannel)
        self.bc_gate = BackchannelGate()
        if speculator is not None:
            self.spec: Speculator | None = speculator
        elif host.speculate is not None and self.rt.speculate != "off":
            self.spec = Speculator(self._spec_start, max_per_turn=self.rt.speculate_max_per_turn,
                                   min_gap_s=self.rt.speculate_min_gap_s, clock=clock)
        else:
            self.spec = None
        self._prepare_fn = prepare
        self._prepared = False
        self._on_learned = on_learned
        self._learned_from_server = bool(learned_from_server)
        self._blob_applied = False
        # start.rt (C.3): the device's realtime switches
        self.rt_partials = True
        self.rt_speculate = True
        self.rt_backchannels = False
        self.rt_optimistic_duck = True
        # the probe flow
        self._pause_id = 0                # bumps when a probed pause is stale (speech resumed, turn ended)
        self._pause_task: asyncio.Task | None = None
        self._tick_task: asyncio.Task | None = None
        #: The open (or last) utterance's own state; the properties below read and
        #: write it (`_utt_started_at`, `_interrupted_utt`, `_utt_while_playing`,
        #: `_stop_cut`, `_stop_cut_aborted`, `_hold_onset`).
        self._utt = _Utt()
        self._utt_seq = 0
        self._prev_verdict: tuple[str, float] | None = None
        self._duck_verdict = False        # the current duck got its word verdict
        self._resume_gap: float | None = None   # SpeechStart − last eot (false-end learning)
        self._last_hold_s = 0.0
        self._last_p: float | None = None
        self._last_partial_at = float("-inf")
        self._last_quiet_check = float("-inf")
        self._audio_stopped_at = float("-inf")
        self._spec_run = None             # the run handed to the last submit (stale check)
        #: Thread head as the frames show it (hardening M1): bumps on every persisted
        #: row (`message`, `stream_done`) so a speculation whose thread moved under it
        #: is dropped before DisPatch has to refuse it.
        self._head_id: str | None = None
        self._head_moves = 0
        #: When the last real interruption cut a reply, when its note settled, and when the
        #: last submit went to the host (metrics.cut_settle_ms / interrupt_to_submit_ms).
        self._interrupt_at: float | None = None
        self._cut_settled_at: float | None = None
        self._submit_at: float | None = None
        self._tick_default_s: float | None = None   # the STT's own tick cadence (captions)
        #: Turn ends in progress (final() + the submit run off the audio worker) and
        #: whether speech started during one (it then joins that turn).
        self._ending = 0
        self._resumed_during_end = False
        self._end_lock = asyncio.Lock()
        #: The narrator's voice for `dialogue_narrator` (None = the character voice).
        self.narrator_id = narrator_id
        self.rp_mode = rp_mode if rp_mode in RP_MODES else "all"
        self.tags_ok = bool(tags_ok)
        self.bot_follow_up = bool(bot_follow_up)
        self.acks = acks if acks is not None else AckKit.silent()
        #: text -> int16 PCM at the engine rate (spoken notices, rendered on
        #: demand); None = synthesise with the session engine.
        self._notice_fn = notice
        self._power_state = power_state
        self.clock = clock
        self.heard_wait_s = float(heard_wait_s)
        #: Random per VoiceSession: the client keys its seq claims on it (C.5 #9).
        self.sid = secrets.token_hex(8)

        self.listening = False
        self.gen = 0                      # bumps on every barge-in
        self.seq = 0                      # assigned at enqueue (SPEC §5.2)
        self.speaking = False             # audio (or a say) sent and not yet confirmed played
        self.patience = settings.patience
        self.barge_in_enabled = True      # start.barge_in
        self.reply_style = "normal"       # start.reply_style
        self.follow_up: bool | None = None   # start.follow_up; None = the bot's default
        self.slow_link = "phone"          # start.slow_link (informational, except always_phone)
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()     # byte-bounded: AUDIO_QUEUE_BYTES
        self._say_q: asyncio.Queue[_Say] = asyncio.Queue()
        self._frames_q: asyncio.Queue[tuple[int, dict]] = asyncio.Queue(maxsize=2000)
        self._frame_seq = 0
        #: Reply frames wait for the client's first `start`: it carries `after_partial`, and
        #: the subscribe snapshot of a reply still streaming is queued BEFORE the receive
        #: loop reads that frame (the client sends it on open). Spoken any earlier, a
        #: reconnect said the cut reply from its start (measured, T3s4: advisor, stall-5s).
        self._started = asyncio.Event()
        self._replies: dict[str, _Reply] = {}
        #: Reply texts that did not extend what was received (`_apply_rewrite`), in order.
        self._rewrites: list[tuple[str, str, bool, str | None]] = []
        self._final_ids: dict[str, str] = {}      # provisional id -> persisted row id
        self._spoken_ids: deque[str] = deque(maxlen=64)
        self._resumed_ids: deque[str] = deque(maxlen=16)   # rows spoken from `replies_after`
        self._recent_said: deque[str] = deque(maxlen=6)
        self._stop_tts = threading.Event()
        self._eot_at: float | None = None
        self._first_audio_logged = True
        self._awaiting_reply = False
        self._last_live_check = 0.0
        self._closed = False
        self._last_touch = float("-inf")
        #: The last utterance or reply sentence (the quiet backstop, FR-hardening L3).
        self._last_exchange = clock()
        self._tts_busy = False            # the TTS worker is on a sentence
        self._client_drained = True       # last playback_done matched our gen
        self._fatal = asyncio.Event()     # run() must end (worker died, lapsed, evicted)
        self.close_code: tuple[int, str] = (1000, "")
        self._base = (detector.ep.cfg.min_speech_s, detector.ep.cfg.turn_hard_silence_s,
                      detector.ep.cfg.no_smart_turn_silence_s)
        self._base_threshold = float(getattr(detector.ep.cfg, "threshold", 0.5))
        #: The detector annotates its events with uplink positions (`feed(..., end=)`).
        self._feed_takes_end = _accepts_kw(getattr(detector, "feed", None), "end")
        #: The streaming STT is one stream for the whole session. The detector marks the
        #: end of every utterance in its frame sink (a ``None`` frame, under its lock, in
        #: order with the frames): from that mark until the turn end that owns it has read
        #: the stream (``final()``) and reset it, the next utterance's frames wait here and
        #: then go into the fresh stream — never lost, never filed under the previous turn
        #: (review-T3s3 #2). ``None`` items in the buffer are later utterances' end marks.
        self._sink_lock = threading.Lock()
        self._sink_pending = 0
        self._sink_buf: deque = deque()
        if self.sstt is not None:
            # Probe mode: the detector hands every utterance frame to the
            # streaming STT and emits Pause/Resumed at the (learned) marks; the
            # session decides (SPEC §4.1). A detector without the attributes
            # (a scripted double) simply keeps emitting TurnEnd.
            cfg = detector.ep.cfg
            if hasattr(cfg, "probe_events"):
                cfg.probe_events = True
            cfg.turn_probe_silence_s = self.policy.probe_silence_s
            if hasattr(detector, "frame_sink"):
                detector.frame_sink = self._on_frame
        self._last_state: str | None = None
        self._deaf = False                # mic ignored while speaking (barge_in off)
        self._last_after: object = object()
        self._last_reply_seq = 0
        # the link: audio or text (phone voice); applied at the next sentence (C.5 #6)
        self._mode_next = "audio"
        self._said_upto = 0               # highest reply seq whose `say` went out
        self._streaming_seq: int | None = None   # the reply piece being sent right now
        self._drained_seq: int | None = None     # last playback_done seq
        # turn merge
        self._merge_until = 0.0           # replies held until this clock time
        self._hold = False                # held while the driver is mid-resume
        self._last_submit: tuple[str, str | None] | None = None   # (text, msg id)
        self._await_user_id: str | None = None   # ignore replies until this row echoes
        # soft barge-in (H3)
        self._duck = False
        self._duck_at = 0.0
        #: The duck is informational (the phone's own voice is the output: it cannot
        #: pause and keeps speaking, so the server neither holds `say` frames nor
        #: rewinds; feasibility H4).
        self._duck_informational = False
        # turn tracking (H1, M3, M4, undo, cancel)
        self._turns: deque[_Turn] = deque(maxlen=16)   # submitted, not started yet
        self._owner: _Turn | str | None = None   # whose run the reply frames are from
        self._current: _Turn | None = None
        self._turn_tasks: set[asyncio.Task] = set()
        self._echoed_ids: deque[str] = deque(maxlen=16)
        # acknowledgements (SPEC §6.2)
        self._turn_acks: TurnAcks | None = None
        self._ack_task: asyncio.Task | None = None
        self._ack_earcon_done = False
        self._line_until = 0.0            # an ack/notice clip plays until about then
        self._last_turn_at: float | None = None
        self._last_acked = False
        self._ack_metrics: dict = {}
        # heard tracking (SPEC §4.8)
        self.heard = HeardMap()
        #: The GPU rig's reply engine (host/engines/rig_tts.py) when one is in play, else
        #: None: cancel{gen} before the host abort, pause/resume with the duck, credit
        #: from `heard`, and its word timings for the heard map (ADDON-ENGINE §4).
        self.rig = rig
        if rig is not None and callable(getattr(rig, "words_for", None)):
            self.heard.words_of = rig.words_for
        self._last_heard: _Heard | None = None
        self._heard_fut: asyncio.Future | None = None
        self._heard_gen: int | None = None
        #: The exact post-barge_in `heard` may land after the fallback position was
        #: used: kept reachable until the refine window closes (review-T3s2 #4).
        self._refine_fut: asyncio.Future | None = None
        #: The gen and reply stream id of the reply the last interruption cut: a
        #: periodic `heard` counts as its position only when it names that reply
        #: (review-T3s2 #2 — never the previous reply's words).
        self._cut_gen: int | None = None
        self._cut_rid: str | None = None
        self._cut_at_seq: int | None = None
        self._cut_row: str | None = None       # the row an abort after a stop cut persisted
        self._note_task: asyncio.Task | None = None   # the last cut's abort + heard + note
        self._seq_rid: dict[int, str] = {}       # seq -> reply stream id (heard map key)
        self._said_text: dict[int, tuple[int, str]] = {}   # seq -> (gen, the words its `say` carried)
        self._seqs_noted: deque[int] = deque()   # pruning order for the three maps above
        self._pending_note: dict | None = None          # feeds the next turn's instructions
        self._deferred_note: tuple[str, HeardPosition, float] | None = None
        self._after_barge_in = False
        # heartbeat + rate limit
        self._pinged = False
        self._last_rx = clock()
        self._tokens = float(RATE_BURST)
        self._tokens_at = clock()
        self._ping_tokens = float(PING_BURST)
        self._ping_tokens_at = clock()
        self._last_limited: dict[str, float] = {}
        self.watch_tick_s = 1.0
        self._submit_opts = _takes_opts(host.submit)
        self._abort_kw = _accepts_kw(host.abort, "keep_partial")

    # ------------------------------------------------------------------ io #
    async def send(self, obj: dict) -> None:
        """Any JSON frame, through the transport (the only send path: a v2 audio
        header and its PCM stay adjacent under the transport's lock)."""
        if self._closed:
            return
        await self.tr.send(obj)

    async def _send_audio(self, *, kind: str, gen: int, seq: int, pcm: np.ndarray, chunk: int,
                          text: str | None, message_id: str | None, role: str,
                          intent: str | None = None, last: bool = False) -> None:
        """One chunk of a sentence: v2 = the JSON ``audio`` header + raw PCM (adjacent);
        v3 = Opus/PCM binary messages (the sentence's text travelled in its ``say``)."""
        if self._closed:
            return
        await self.tr.send_audio(kind=kind, gen=gen, seq=seq, pcm16=pcm, text=text,
                                 message_id=message_id, role=role, chunk=chunk, intent=intent, last=last)

    async def state(self, s: str) -> None:
        self._last_state = s
        if s == "hearing":
            self._last_exchange = self.clock()    # the listener is talking
        await self.send({"type": "state", "state": s})

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error", "code": code, "message": message})

    def _idle_state(self) -> str:
        """The state to report when nothing just changed: a re-sent ``start``
        (amendment B4), an empty transcript, an echo. A turn still waiting on
        its answer stays ``thinking`` — reporting ``listening`` there made the
        client drop its Thinking counter and cancel affordance mid-turn."""
        if self.speaking:
            return "speaking"
        if self.listening and (self._awaiting_reply or self._talking()):
            return "thinking"
        return "listening" if self.listening else "paused"

    def end(self, code: int, reason: str = "") -> None:
        """Ask run() to finish; the route closes the socket with ``code``."""
        self.close_code = (code, reason)
        self._fatal.set()

    @property
    def mode(self) -> str:
        """The link mode the NEXT sentence will take (audio | text)."""
        return self._mode_next

    @property
    def _hold(self) -> bool:
        """Reply speech is held: the driver may still be adding to the turn."""
        return self.__dict__.get("_hold_v", False)

    @_hold.setter
    def _hold(self, on: bool) -> None:
        on = bool(on)
        if on and not self.__dict__.get("_hold_v", False):
            self._hold_since = self.clock()      # the stuck-hold check (watchdog) reads it
        self._hold_v = on

    # The open utterance's own state (`_Utt`), under the names the flow has always used.
    @property
    def _utt_started_at(self) -> float | None:
        return self._utt.started_at

    @_utt_started_at.setter
    def _utt_started_at(self, v: float | None) -> None:
        self._utt.started_at = v

    @property
    def _interrupted_utt(self) -> bool:
        return self._utt.interrupted

    @_interrupted_utt.setter
    def _interrupted_utt(self, v: bool) -> None:
        self._utt.interrupted = bool(v)

    @property
    def _utt_while_playing(self) -> bool | None:
        return self._utt.while_playing

    @_utt_while_playing.setter
    def _utt_while_playing(self, v: bool | None) -> None:
        self._utt.while_playing = v

    @property
    def _stop_cut(self) -> bool:
        return self._utt.stop_cut

    @_stop_cut.setter
    def _stop_cut(self, v: bool) -> None:
        self._utt.stop_cut = bool(v)

    @property
    def _stop_cut_aborted(self) -> bool:
        return self._utt.stop_cut_aborted

    @_stop_cut_aborted.setter
    def _stop_cut_aborted(self, v: bool) -> None:
        self._utt.stop_cut_aborted = bool(v)

    @property
    def _hold_onset(self) -> float | None:
        return self._utt.hold_onset

    @_hold_onset.setter
    def _hold_onset(self, v: float | None) -> None:
        self._utt.hold_onset = v

    # ------------------------------------------------------- app frames #
    def on_app_frame(self, frame: dict) -> None:
        """Called (synchronously) for every frame the app broadcasts."""
        if frame.get("thread_id") != self.thread_id:
            return
        if frame.get("type") not in ("stream_start", "stream_chunk", "stream_done", "message",
                                     "thinking", "error", "progress"):
            return
        self._frame_seq += 1
        with contextlib.suppress(asyncio.QueueFull):
            self._frames_q.put_nowait((self._frame_seq, frame))

    def _new_reply(self) -> _Reply:
        return _Reply(renderer=RpRenderer(self.rp_mode))

    def _reply_text_update(self, rid: str, full: str | None, delta: str | None,
                           final: bool, message_id: str | None = None) -> list[tuple[Span, str | None, str]]:
        r = self._replies.get(rid)
        if r is None:
            r = self._replies[rid] = self._new_reply()
        if message_id:
            r.mid = message_id
            if message_id != rid:
                self._final_ids[rid] = message_id
        if r.muted or r.done:
            return []
        r.updated = self.clock()
        if full is not None:
            if full.startswith(r.text):
                new = full[len(r.text):]
            elif r.text.startswith(full) and not r.text[len(full):].strip():
                new = ""                # the same text without its trailing whitespace (a stored row)
            else:
                # Not an extension of what this reply received: a rewrite (the advisor's
                # provider fallback, a sanitised text, a gateway buffer that changed what
                # it sent; the final content differing from the deltas). Speaking
                # `full[consumed:]` spliced two texts mid-word (review-T3s4 #5): the frames
                # worker re-splits it and compares the spoken pieces (`_apply_rewrite`).
                self._rewrites.append((rid, full, final, message_id))
                return []
        else:
            new = delta or ""
        r.text += new
        r.consumed = len(r.text)
        if len(r.head) < HEAD_CHARS and new:
            r.head = r.text[:HEAD_CHARS]
        spans, r.pending = feed_canonical(r.splitter, r.pending, new, final=final)
        if final:
            r.done = True
        return [(s, message_id or rid, rid) for s in spans]

    def _mute(self, rid: str | None) -> None:
        if rid:
            self._replies.setdefault(rid, self._new_reply()).muted = True

    def _mute_frame(self, f: dict) -> None:
        msg = f.get("message") or {}
        if msg.get("role") == "user":
            return
        for rid in (f.get("message_id"), f.get("provisional_id"), msg.get("id")):
            self._mute(rid)
        if f.get("type") == "message" and msg.get("id"):
            self._spoken_ids.append(msg["id"])

    def handle_reply_frame(self, f: dict) -> list[tuple[Span, str | None, str]]:
        """Reply frame → sentence spans to say, each with (message id, reply id).
        Pure bookkeeping (testable)."""
        t = f.get("type")
        msg = f.get("message") or {}
        if self._await_user_id is not None:
            # After an abandoned turn, everything until the next user row we
            # sent echoes back belongs to the abandoned turn.
            if t == "message" and msg.get("role") == "user" and msg.get("id") == self._await_user_id:
                self._await_user_id = None
            else:
                self._mute_frame(f)
            return []
        if t == "stream_chunk":
            mid = f.get("message_id")
            if not mid:
                return []
            if f.get("replace"):
                # The subscribe snapshot (A5): the full text so far. A reply that was
                # waiting as "late" is caught up by it, so its next deltas count again.
                late = self._replies.get(mid)
                if late is not None and late.late:
                    late.late = False
                return self._reply_text_update(mid, f.get("text") or "", None, False)
            r = self._replies.get(mid)
            if r is None and mid not in self._streams_started:
                # A delta of a stream this session never saw start, and no snapshot came
                # (C.5 #11, the late-joiner belt): its deltas are dropped — splicing them
                # would speak a tail from mid-sentence — and it is spoken from the
                # stream_done content, minus what `start.after_partial` named.
                r = self._replies[mid] = self._new_reply()
                r.late = True
                return []
            if r is not None and r.late:
                return []
            return self._reply_text_update(mid, None, f.get("text") or "", False)
        if msg.get("role") != "assistant" or (msg.get("metadata") or {}).get("sub"):
            return []
        mid = msg.get("id")
        if (msg.get("metadata") or {}).get("voice_heard"):
            # The H3 partial row of a reply this session cut (the spoken prefix,
            # persisted by abort_turn): it was heard already, never re-spoken.
            self._mute_frame(f)
            return []
        if t == "stream_done":
            rid = f.get("provisional_id") or f.get("message_id") or mid
            if mid and rid != mid and mid in self._resumed_ids:
                # This reply was already spoken from its stored row (a reconnect's
                # `replies_after` found it stored): the stream's tail stays silent.
                self._mute(rid)
                self._mute(mid)
                return []
            out = self._reply_text_update(rid, msg.get("content") or "", None, True, mid)
            if rid != mid and self._replies.get(rid, _Reply()).muted:
                self._mute(mid)
            for i in (rid, mid):
                if i:
                    self._spoken_ids.append(i)
            return out
        # plain "message"
        if not mid or mid in self._spoken_ids:
            return []
        self._spoken_ids.append(mid)
        return self._reply_text_update(mid, msg.get("content") or "", None, True, mid)

    # ---------------------------------------------------- turn tracking #
    def _on_user_echo(self, mid: str | None) -> None:
        if not mid:
            return
        self._echoed_ids.append(mid)       # may echo before submit() returns
        if mid == self._await_user_id:
            self._await_user_id = None
        for t in self._turns:
            if t.msg_id == mid:
                t.echoed = True

    def _on_started(self, seq: int) -> None:
        """A run began on this thread. It belongs to the oldest voice turn of
        ours whose row has echoed and that has not started — except a turn we
        abandoned and had the host cancel: a `started` that arrived AFTER that
        cancel cannot be its run (review M3/M4: no double answers)."""
        while self._turns:
            t = self._turns[0]
            if not t.echoed:
                break
            if t.abandoned and t.abort_seq is not None and seq > t.abort_seq:
                self._turns.popleft()          # cancelled before it ever ran
                continue
            self._turns.popleft()
            t.started = True
            self._owner = t
            return
        self._owner = "other"                  # a typed turn, another device…

    async def _on_stopped(self) -> None:
        t = self._owner
        if not isinstance(t, _Turn) or t.closed:
            return
        t.stopped = True
        if t.spoke:
            t.closed = True
            return
        task = asyncio.create_task(self._nothing_spoken_later(t))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _nothing_spoken_later(self, t: _Turn) -> None:
        await asyncio.sleep(NOTHING_SPOKEN_GRACE_S)
        if not t.closed and not t.spoke and not t.abandoned:
            await self._turn_failed(t, "no_speech_in_reply")

    async def _on_agent_error(self, f: dict) -> None:
        t = self._owner
        if not isinstance(t, _Turn) or t.closed or t.abandoned:
            return
        detail = f.get("message") if isinstance(f.get("message"), str) else None
        await self._turn_failed(t, "agent_failed", detail)

    async def _turn_failed(self, t: _Turn | None, reason: str, detail: str | None = None) -> None:
        """turn_error + a short spoken line + back to listening (H1)."""
        if t is not None:
            if t.closed:
                return
            t.closed = True
            t.failed = True
        self._awaiting_reply = False
        if self._turn_acks is not None:
            self._turn_acks.on_error()
            self._turn_acks = None
        self._cancel_ack_task()
        frame = {"type": "turn_error", "message_id": t.msg_id if t else None, "reason": reason}
        if detail:
            frame["detail"] = detail[:200]
        await self.send(frame)
        line = NOTICE_LINES.get(reason)
        if line:
            await self._say_notice(line)
        if not self.speaking:
            await self.state("listening" if self.listening else "paused")

    async def _say_notice(self, text: str) -> None:
        """A spoken notice line (errors, "Still there?"): rendered through the
        ack queue when it is, else on demand; a `say` alone in text mode."""
        await self._say_line("notice", text)

    async def _abandon(self, t: _Turn | None, *, abort: bool, keep_partial: bool = False) -> str | None:
        """Give up on a voice turn: its reply (now or late) is never spoken,
        and with ``abort`` the host stops it (cancelled if not started yet).
        Returns the row id the abort persisted (an interruption), else None."""
        if t is None or t.abandoned:
            return None
        t.abandoned = True
        t.closed = True
        self._awaiting_reply = False
        self._turn_acks = None
        self._cancel_ack_task()
        if self._owner is t:
            # Its run is the one producing frames right now: mute them.
            for r in self._replies.values():
                if not r.done:
                    r.muted = True
        if abort and self.host.abort:
            t.abort_seq = self._frame_seq
            return await self._host_abort(keep_partial=keep_partial)
        return None

    async def _host_abort(self, *, keep_partial: bool) -> str | None:
        fn = self.host.abort
        if fn is None:
            return None
        try:
            if self._abort_kw:
                row = await fn(self.thread_id, keep_partial=keep_partial)
            else:
                row = await fn(self.thread_id)
        except Exception:
            log.exception("voice: abort failed")
            return None
        return row if isinstance(row, str) and row else None

    def _owner_is_abandoned(self) -> bool:
        return isinstance(self._owner, _Turn) and self._owner.abandoned

    def _text_row(self, rid: str) -> _ReplyText:
        rt = self._texts.get(rid)
        if rt is None:
            rt = self._texts[rid] = _ReplyText()
            while len(self._texts) > REPLY_TEXTS_KEEP:
                self._texts.popitem(last=False)
        else:
            self._texts.move_to_end(rid)
        return rt

    def _track_text(self, f: dict) -> None:
        """Keep every reply's text as it streams — muted or not (a tap-stop silences the
        rest of a run that goes on into the chat), so `replay_last` can continue a cut
        reply or say a finished one again. Deltas of a stream seen without its
        `stream_start` (or snapshot) are a tail and are not kept (C.5 #11): such a reply
        is known from its final content only."""
        t = f.get("type")
        now = self.clock()
        if t == "stream_chunk":
            rid = f.get("message_id")
            piece = f.get("text")
            if not isinstance(rid, str) or not rid or not isinstance(piece, str):
                return
            known = rid in self._texts
            rt = self._text_row(rid)
            if f.get("replace"):
                rt.text, rt.late = piece[:REPLY_TEXT_MAX], False
            elif not known and rid not in self._streams_started:
                rt.late = True
            elif not rt.late and not rt.done and len(rt.text) < REPLY_TEXT_MAX:
                rt.text = (rt.text + piece)[:REPLY_TEXT_MAX]
            rt.at = now
            return
        if t not in ("stream_done", "message"):
            return
        msg = f.get("message") or {}
        meta = msg.get("metadata") or {}
        if msg.get("role") != "assistant" or meta.get("sub") or not isinstance(msg.get("content"), str):
            return
        mid = msg.get("id") if isinstance(msg.get("id"), str) and msg.get("id") else None
        rid = (f.get("provisional_id") or f.get("message_id") or mid) if t == "stream_done" else mid
        if not isinstance(rid, str) or not rid:
            return
        rt = self._text_row(rid)
        rt.text, rt.done, rt.late, rt.at = msg["content"][:REPLY_TEXT_MAX], True, False, now
        if mid and not mid.startswith("run:"):
            rt.mid = mid
            if mid != rid:
                self._texts[mid] = rt                # the row id names the same reply
                self._texts.move_to_end(mid)

    async def _frames_sync(self, timeout_s: float = 2.0) -> None:
        """Wait until the frames worker has handled every frame queued so far."""
        ev = asyncio.Event()
        self._frame_seq += 1
        try:
            self._frames_q.put_nowait((self._frame_seq, {"type": "_sync", "event": ev}))
        except asyncio.QueueFull:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(ev.wait(), timeout_s)

    async def _frames_worker(self) -> None:
        while True:
            seq, f = await self._frames_q.get()
            if not self._started.is_set():
                await self._started.wait()
            t = f.get("type")
            if t == "_sync":
                f["event"].set()
                continue
            if t == "_replay_last":
                # Serialised with the reply frames: the rebuild reads the text the frames
                # gave so far and the frames after it continue exactly where it stopped.
                await self._do_replay(str(f.get("mode") or "continue"))
                continue
            self._track_text(f)
            if t == "thinking":
                if f.get("status") == "started":
                    self._on_started(seq)
                elif f.get("status") == "stopped":
                    await self._on_stopped()
                continue
            if t == "error":
                await self._on_agent_error(f)
                continue
            if t == "progress":
                self._on_progress(f)
                continue
            if t == "stream_start":
                if isinstance(f.get("message_id"), str):
                    self._streams_started.append(f["message_id"])
                continue
            msg = f.get("message") or {}
            if (t in ("message", "stream_done") and isinstance(msg.get("id"), str) and msg["id"]
                    and msg["id"] != self._head_id):
                # A persisted row: the thread head moved (a typed message, a reply
                # landing). A speculative prompt built before it is stale (M1).
                self._head_id = msg["id"]
                self._head_moves += 1
            if t == "message" and msg.get("role") == "user":
                self._on_user_echo(msg.get("id"))
            await self._resolve_deferred_note(f)
            if self._owner_is_abandoned():
                self._mute_frame(f)
                continue
            owner = self._owner
            rids: list[str] = []
            for span, mid, rid in self.handle_reply_frame(f):
                if rid not in rids:
                    rids.append(rid)
                queued = await self._enqueue_reply(span, mid, rid)
                if queued and isinstance(owner, _Turn):
                    owner.spoke = True
                    if owner.stopped:
                        owner.closed = True
            if self._rewrites:
                for rid, *_rest in list(self._rewrites):
                    if rid not in rids:
                        rids.append(rid)
                if await self._apply_rewrites() and isinstance(owner, _Turn):
                    owner.spoke = True
                    if owner.stopped:
                        owner.closed = True
            self._note_reply_text(f, owner)
            for rid in self._done_rids(f, rids):
                await self._maybe_say_end(rid)
            await self._maybe_finish_speaking()

    def _done_rids(self, f: dict, rids: list[str]) -> list[str]:
        """Reply ids that just ended with this frame (say_end candidates)."""
        out = list(rids)
        msg = f.get("message") or {}
        for rid in (f.get("provisional_id"), f.get("message_id"), msg.get("id")):
            if rid and rid in self._replies and self._replies[rid].done and rid not in out:
                out.append(rid)
        return out

    def _note_reply_text(self, f: dict, owner) -> None:
        """The reply's first text for the ack gate (its first word is avoided)
        and the ETA estimator (transcript → first token)."""
        t = f.get("type")
        text = f.get("text") if t == "stream_chunk" else ((f.get("message") or {}).get("content")
                                                           if t in ("stream_done", "message") else None)
        if not text or (f.get("message") or {}).get("role") == "user":
            return
        if self._turn_acks is not None:
            self._turn_acks.on_reply_text(text)
            self._poll_acks_soon()
        if isinstance(owner, _Turn) and owner.first_text_at is None and owner is self._current:
            owner.first_text_at = self.clock()
            self.acks.observe(ttft_s=max(0.0, owner.first_text_at - owner.at))

    def _on_progress(self, f: dict) -> None:
        """A gateway tool call (`progress {item.kind: "tool"}`): one spoken
        status per distinct tool, ≤ 2 per turn (SPEC §6.2, experience M10)."""
        item = f.get("item") or {}
        if not isinstance(item, dict) or item.get("kind") != "tool":
            return
        if self._turn_acks is None or not self._awaiting_reply:
            return
        name = item.get("name")
        self._turn_acks.on_tool(name if isinstance(name, str) else None)
        self._poll_acks_soon()

    def _reply_pending(self) -> bool:
        """A reply is still streaming in (more sentences may follow)."""
        now = self.clock()
        return any(not r.done and not r.muted and now - r.updated < REPLY_STALE_S
                   for r in self._replies.values())

    async def _maybe_finish_speaking(self) -> None:
        """Leave the speaking state only when the whole reply is out: the
        client drained its queue for this generation, nothing is queued or
        being synthesised, and no reply stream is still open. Clearing it any
        earlier (the next sentence still synthesising) made our own voice look
        like a barge-in and aborted the agent run."""
        if (self.speaking and self._client_drained and self._say_q.empty()
                and not self._tts_busy and not self._reply_pending()):
            self._set_speaking(False)
            await self.state("listening" if self.listening else "paused")

    def _render_pieces(self, r: _Reply, span: Span) -> list[tuple[Span, str, str]]:
        """A sentence span → its spoken pieces ``(source sub-span, role, words)``: T5's speech
        text, then the role-play renderer (stateful per reply: call in order). Empty when the
        sentence has nothing to say."""
        spoken = normalise(span.text, tags_ok=self.tags_ok)
        if not spoken:
            return []
        try:
            pieces = [(role, text) for role, text in r.renderer.render(spoken) if text]
        except Exception:
            log.exception("voice: role-play rendering failed; speaking the sentence whole")
            pieces = [("character", spoken)]
        if not pieces:
            return []
        return [(ps, role, text) for ps, (role, text) in zip(subspans(span, pieces), pieces, strict=True)]

    async def _enqueue_reply(self, span: Span, message_id: str | None, rid: str) -> bool:
        """Speech text + role-play rendering → one queued piece per segment,
        each with its seq and source sub-span (heard tracking). False when the
        sentence has nothing to say."""
        r = self._replies.get(rid)
        if r is None:
            r = self._replies[rid] = self._new_reply()
        return await self._enqueue_pieces(r, self._render_pieces(r, span), message_id, rid)

    async def _enqueue_pieces(self, r: _Reply, pieces: list[tuple[Span, str, str]],
                              message_id: str | None, rid: str) -> bool:
        if not pieces:
            return False
        if self._awaiting_reply:
            self._awaiting_reply = False
        if self._turn_acks is not None:
            self._turn_acks.on_reply_queued()
            self._poll_acks_soon()
        skip_key = next((k for k in (rid, message_id, r.mid) if k and self._after_partial.get(k)), None)
        for piece_span, role, text in pieces:
            if r.expect:
                ws = _norm_words(text)
                if ws and ws == r.expect[:len(ws)]:
                    # A rewrite's re-split confirmed words the listener already has: heard.
                    del r.expect[:len(ws)]
                    r.skipped += 1
                    r.log.append(_Piece(None, piece_span.start, piece_span.end, text))
                    continue
                r.expect = []                          # the text moved on: they were stale
            if skip_key is not None and self._after_partial.get(skip_key, 0) > 0:
                # heard on the socket before a reconnect (C.5 #11): never twice
                self._after_partial[skip_key] -= 1
                if not self._after_partial[skip_key]:
                    self._after_partial.pop(skip_key, None)
                skip = True
            else:
                # a continue: heard before the cut (by source offset, review-T3s4 #1)
                skip = r.skip_upto is not None and piece_span.end <= r.skip_upto
            if skip:
                r.skipped += 1
                r.log.append(_Piece(None, piece_span.start, piece_span.end, text))
                self._recent_said.append(text)
                continue
            self.seq += 1
            seq = self.seq
            # Every piece of one reply is noted under its STREAM id, so the heard map
            # sees the whole reply even when the stream's provisional id turns into
            # the row id mid-reply (review-T3s2 #3); `_persisted_id` translates at use.
            self.heard.note(seq, message_id=rid, span=piece_span, audio_ms=0)
            self._remember_seq(seq, rid)
            r.last_seq = seq
            r.pieces += 1
            voice = self.narrator_id if (role == "narrator" and self.narrator_id) else self.profile_id
            # A piece said before (`replay_last`) goes out from its kept audio, no engine call.
            kept = self._pcm_keep.get((rid, text, voice)) if rid == self._pcm_rid else None
            item = _Say(self.gen, seq, text, message_id, "reply", role, voice, rid, pcm=kept)
            r.log.append(_Piece(seq, piece_span.start, piece_span.end, text, item))
            await self._say_q.put(item)
        return True

    async def _apply_rewrites(self) -> int:
        """The rewrites `_reply_text_update` met (in order), see :meth:`_apply_rewrite`.
        Returns how many pieces they queued."""
        queued = 0
        while self._rewrites:
            rid, full, final, message_id = self._rewrites.pop(0)
            try:
                queued += await self._apply_rewrite(rid, full, final, message_id)
            except Exception:
                log.exception("voice: a reply rewrite failed; the rest of the reply is skipped")
                r = self._replies.get(rid)
                if r is not None:
                    r.muted = True
        return queued

    async def _apply_rewrite(self, rid: str, full: str, final: bool, message_id: str | None) -> int:
        """A reply whose text was REWRITTEN (`stream_chunk {replace}` or the final content
        not extending what was received; review-T3s4 #5). The new text is split from scratch,
        exactly as a fresh stream of it would be, and lined up with what this reply already
        has by its SPOKEN WORDS — a change of markup or whitespace (a directive, a link's URL,
        a stripped row) changes nothing audible, even where it moves a sentence boundary:

        * the old pieces inside the words both texts share are kept (said or queued), and the
          new pieces inside them are not said again; their source spans move to the new text;
        * after them, an old piece whose `say` has not gone out is dropped; the words of one
          the listener already has are EXPECTED: new pieces made of them are heard already
          (the new text's splitter may still need the whitespace after a sentence), any other
          piece ends the wait;
        * every other new piece is queued: the listener hears the corrected text from the
          first changed sentence on — never a splice; at most the sentence that changed while
          it was being said is heard twice.

        Returns the number of pieces queued."""
        r = self._replies.get(rid)
        if r is None or r.muted or r.done:
            return 0
        if message_id:
            r.mid = message_id
        fresh = self._new_reply()
        spans, pending = feed_canonical(fresh.splitter, "", full, final=final)
        new: list[tuple[Span, str, str]] = []
        for span in spans:
            new.extend(self._render_pieces(fresh, span))
        old = r.log
        old_words = [_norm_words(p.text) for p in old]
        new_words = [_norm_words(t) for _s, _r, t in new]
        flat_old = [w for ws in old_words for w in ws] + list(r.expect)
        flat_new = [w for ws in new_words for w in ws]
        common = 0
        while common < min(len(flat_old), len(flat_new)) and flat_old[common] == flat_new[common]:
            common += 1
        # The old pieces whose words all lie inside the shared words: kept.
        kept, covered = 0, 0
        for ws in old_words:
            if covered + len(ws) > common:
                break
            covered += len(ws)
            kept += 1

        def first_new(cov: int) -> tuple[int, int]:
            """The first new piece not inside the first ``cov`` words, and its first word."""
            acc = 0
            for j, ws in enumerate(new_words):
                if acc + len(ws) > cov:
                    return j, acc
                acc += len(ws)
            return len(new), acc

        first, first_word = first_new(covered)
        while kept and first_word < covered:
            # The first new piece to say straddles the kept words (the new text moved a
            # sentence boundary): an UNSAID kept piece it also says is given to it.
            last = old[kept - 1]
            if last.say is None or last.say.say_sent:
                break                                       # heard: it stays heard
            kept -= 1
            covered -= len(old_words[kept])
            first, first_word = first_new(covered)
        # Their spans in the new text: the new pieces covering the same words.
        bounds, acc = [], 0
        for (ps, _role, _text), ws in zip(new, new_words, strict=True):
            bounds.append((acc, acc + len(ws), ps))
            acc += len(ws)
        acc = 0
        for p, ws in zip(old[:kept], old_words[:kept], strict=True):
            a, b = acc, acc + len(ws)
            acc = b
            cover = [(x, y, ps) for x, y, ps in bounds if x < b and y > a]
            if not cover:
                continue
            x0, y0, ps0 = cover[0]
            if len(cover) == 1 and (x0, y0) == (a, b):
                span = ps0                                  # the same piece: exact
            elif len(cover) == 1 and y0 > x0:
                # Several old pieces inside one new piece: their share of its source range,
                # by word position (the heard map needs only the right region).
                n = len(ps0)
                st = ps0.start + round((a - x0) / (y0 - x0) * n)
                en = ps0.start + round((b - x0) / (y0 - x0) * n)
                span = Span(p.text, st, max(st, en), full[st:max(st, en)])
            else:
                st, en = cover[0][2].start, cover[-1][2].end
                span = Span(p.text, st, en, full[st:en])
            p.start, p.end = span.start, span.end
            if p.seq is not None:
                self.heard.respan(p.seq, span)
        expect: list[str] = []
        dropped: list[_Piece] = []
        for p, ws in zip(old[kept:], old_words[kept:], strict=True):
            if p.say is not None and not p.say.say_sent:
                p.say.dropped = True                        # never said: replaced
                dropped.append(p)
            else:
                expect.extend(ws)                           # heard: confirmed or stale later
        self._forget_pieces(old[kept:])
        # (the new pieces before `first` lie inside the kept words: not said again)
        (log.info if kept < len(old) or r.expect else log.debug)(
            "voice: reply %s rewritten: %d piece(s) kept, %d dropped unsaid, %d new",
            rid, kept, len(dropped), max(0, len(new) - first))
        r.splitter, r.renderer, r.pending = fresh.splitter, fresh.renderer, pending
        r.text, r.consumed, r.head = full, len(full), full[:HEAD_CHARS]
        r.log = old[:kept]
        r.expect = expect + list(r.expect)           # (words heard after the kept pieces, in order)
        r.last_seq = next((p.seq for p in reversed(r.log) if p.seq is not None), None)
        r.pieces = sum(1 for p in r.log if p.seq is not None)
        r.updated = self.clock()
        before = r.pieces
        await self._enqueue_pieces(r, new[first:], message_id or r.mid or rid, rid)
        if final:
            r.done = True
            r.expect = []
        return r.pieces - before

    def _forget_pieces(self, pieces: list[_Piece]) -> None:
        """Pieces whose source offsets no longer describe the reply's text: out of the heard
        map (a cut there writes no misleading note)."""
        for p in pieces:
            if p.seq is not None:
                self.heard.forget(p.seq)
                self._seq_rid.pop(p.seq, None)

    def _remember_seq(self, seq: int, rid: str | None) -> None:
        """Per-seq bookkeeping, bounded (SEQ_KEEP pieces; review-T3s2 #10)."""
        if rid is not None:
            self._seq_rid[seq] = rid
        self._seqs_noted.append(seq)
        while len(self._seqs_noted) > SEQ_KEEP:
            old = self._seqs_noted.popleft()
            self.heard.forget(old)
            self._seq_rid.pop(old, None)
            self._said_text.pop(old, None)
        if len(self._replies) > REPLIES_KEEP:
            # Forget the oldest FINISHED reply streams (a live one is never dropped).
            for rid_old in list(self._replies)[: len(self._replies) - REPLIES_KEEP]:
                if self._replies[rid_old].done or self._replies[rid_old].muted:
                    self._replies.pop(rid_old, None)
                    self._final_ids.pop(rid_old, None)

    async def _maybe_say_end(self, rid: str) -> None:
        """`say_end {gen, message_id, last_seq}` once the reply is complete and
        its last sentence's `say` went out (C.4) — and, with server audio, after
        that sentence's last audio message (Amendment T8-1: the player's `end` and
        the phone voice's `final` rely on that order). A reply whose last piece is
        still being sent gets it from `_stream_sentence` when the piece ends."""
        r = self._replies.get(rid)
        if r is None or r.end_sent or not r.done or r.muted:
            return
        if r.last_seq is not None and self._streaming_seq == r.last_seq:
            return
        if r.last_seq is None:
            if not r.skipped:
                return
            last = self._said_upto           # every piece was heard before a reconnect
        elif self._said_upto < r.last_seq:
            return
        else:
            last = r.last_seq
        r.end_sent = True
        done_fn = getattr(self.tts, "reply_done", None)
        if done_fn is not None:
            with contextlib.suppress(Exception):
                done_fn(rid)
        await self.send({"type": "say_end", "gen": self.gen, "message_id": r.mid or rid,
                         "last_seq": last})

    # -------------------------------------------------------------- tts #
    def _held(self) -> bool:
        """Reply speech waits: the driver may still be adding to the turn (the
        policy's hold, the addendum), or someone is talking over the server's
        audio (the duck). A duck over the PHONE's voice holds nothing: the phone
        keeps speaking what it has, and starving it of the next `say` would only
        put gaps into its own echo (feasibility H4)."""
        return ((self._hold or (self._duck and not self._duck_informational))
                or self.clock() < self._merge_until)

    async def _wait_hold(self, gen: int) -> bool:
        """Wait out a hold; False when the piece was dropped meanwhile."""
        t0 = time.monotonic()
        while self._held() and gen == self.gen and not self._closed:
            await asyncio.sleep(0.03)
        if self._pace_t0 is not None:
            self._pace_t0 += time.monotonic() - t0      # a held player does not play
        return gen == self.gen and not self._closed

    async def _paced(self, gen: int, seconds: float) -> bool:
        """Send-ahead pacing (v3): wait while the audio already sent runs more than
        ``send_ahead_s`` ahead of real-time playback, then count ``seconds`` more. The
        origin is the first audio after the client drained its queue (``playback_done``);
        a hold moves it (see :meth:`_wait_hold`). Real time, not the session clock: the
        listener's speaker plays in real time. False when the piece was dropped meanwhile."""
        if self.send_ahead_s is None:
            return gen == self.gen and not self._closed
        now = time.monotonic()
        if self._client_drained or self._pace_t0 is None:
            self._pace_t0, self._pace_sent = now, 0.0
        lead = self._send_ahead_now()
        while gen == self.gen and not self._closed and self._pace_t0 is not None:
            ahead = self._pace_sent - (time.monotonic() - self._pace_t0)
            if ahead <= lead:
                break
            # (short naps: a cut must free the worker for the next piece at once)
            await asyncio.sleep(min(0.05, ahead - lead))
        if gen != self.gen or self._closed:
            # Dropped while waiting (a barge-in, the hang-up): this piece is never sent, so
            # it never counts as audio the client is playing (review-T3s4 #2).
            return False
        if self._pace_t0 is None:                 # a barge-in reset the clock meanwhile
            self._pace_t0, self._pace_sent = time.monotonic(), 0.0
        self._pace_sent += max(0.0, float(seconds))
        return True

    def _send_ahead_now(self) -> float:
        """The send-ahead lead for the next piece: ``send_ahead_s`` (4 s), but at most
        PCM_SEND_AHEAD_S on a PCM16 downlink (review-T3s4 #7): 384 kbps of PCM keeps the
        bytes of the whole lead in flight, and on an EDGE-class link 4 s of it is ~13 s of
        airtime — every `pong`, `say` and `barge_in` queued behind it and the client's
        dead-socket rule fired mid-reply. 1.25 s (+ one 0.25 s slice) is ≈ 72 KB."""
        lead = float(self.send_ahead_s or 0.0)
        if self.tr.v3 and self.tr.codec_out == "pcm16":
            lead = min(lead, PCM_SEND_AHEAD_S)
        return lead

    def _say_frame(self, item: _Say) -> dict:
        # v3: `t` is the transport's clock, the same one the binary frames carry (the
        # link monitor compares them); v2 keeps the session clock as before.
        t = self.tr.now_ms() if self.tr.v3 else round(self.clock() * 1000)
        frame = {"type": "say", "gen": item.gen, "seq": item.seq,
                 "text": item.text, "message_id": item.message_id, "kind": item.kind, "role": item.role,
                 "t": t}
        if item.intent:
            frame["intent"] = item.intent
        return frame

    async def _emit_say(self, item: _Say) -> None:
        """The `say` itself: v3 through ``send_say`` (it ends the previous sentence's
        audio first, so no sentence's audio is split by the next `say`); v2 as JSON."""
        frame = self._say_frame(item)
        if self.tr.v3:
            await self.tr.send_say(frame)
        else:
            await self.send(frame)

    async def _send_say(self, item: _Say) -> None:
        item.say_sent = True                  # from here a rewrite can no longer drop it
        await self._emit_say(item)
        if item.kind == "reply":
            self._last_exchange = self.clock()    # the bot is talking (not acks, not notices)
            self._said_upto = max(self._said_upto, item.seq)
            self._last_reply_seq = item.seq
            if item.rid:
                self._last_said_rid, self._last_said_at = item.rid, self.clock()
        else:
            self._remember_seq(item.seq, None)
        self._said_text[item.seq] = (item.gen, item.text)
        self._recent_said.append(item.text)   # its echo must never become a turn (C.5 #5)

    def _note_reply_said(self) -> None:
        """The listener can hear a reply sentence: its first audio went out (server
        audio — the `say` precedes it by the engine's first-chunk time, during which
        nothing is audible), or its `say` went out (the phone's voice reads it at
        once). From here the turn can no longer be merged with an addendum (SPEC
        §4.3: once a sentence plays, more speech is an interruption), and it was a
        clean turn for the learned endpointing (a false end would have merged before)."""
        t = self._owner if isinstance(self._owner, _Turn) else self._current
        if t is None or t.said:
            return
        t.said = True
        t.said_at = self.clock()
        self._observe(false_end=False, resume_gap_s=None)

    def _observe(self, *, false_end: bool, resume_gap_s: float | None) -> None:
        """Feed the policy's learned endpointing and publish the values."""
        if self.sstt is None:
            return                       # the 2.1 path has no learned hold
        try:
            self.policy.observe(false_end=false_end, resume_gap_s=resume_gap_s)
            self._learned_from_server = True     # this session's own values win from now on
            self.detector.ep.cfg.turn_probe_silence_s = self.policy.probe_silence_s
            if self._on_learned is not None:
                self._on_learned(self.policy.learned)
        except Exception:
            log.exception("voice: learned endpointing update failed")

    async def _stream_sentence(self, item: _Say) -> bool:
        """Synthesise ``item.text`` and ship every chunk as it is produced.

        Synthesis starts AT ONCE, even while playback is held (merge window,
        duck): only the sending waits (review M1); the ``say`` frame goes out
        the instant the hold releases (SPEC §5.2), before the first audio.
        Returns False when the engine failed before any audio."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._stop_tts.clear()
        stop = self._stop_tts
        gen, words, mid, kind = item.gen, item.text, item.message_id, item.kind

        # The engine finished the piece before anything stopped it (its audio is whole).
        complete = [False]

        reply_fn = getattr(self.tts, "stream_reply", None)

        def produce():
            REPLY_ACTIVITY.enter()
            try:
                # With a rig in play the engine needs the reply's identity (one reply,
                # one engine; one rig context per reply).
                chunks = (reply_fn(words, item.voice, stop, gen=gen, seq=item.seq, rid=item.rid, kind=kind)
                          if callable(reply_fn) else self.tts.stream(words, item.voice, stop))
                for chunk in chunks:
                    loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
                    if stop.is_set():
                        break
                else:
                    complete[0] = not stop.is_set()
            except Exception as e:      # surfaced on the loop side
                loop.call_soon_threadsafe(q.put_nowait, ("error", e))
            finally:
                REPLY_ACTIVITY.exit()
            loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        t0 = self.clock()
        fut = loop.run_in_executor(None, produce)
        pending: list = []
        n = 0
        ok = True
        said = False
        # A reply piece's audio is kept for an instant `replay_last` — all the engine made of
        # it, sent or not: a tap cuts a piece mid-way (its slices are paced), and "Tap to
        # continue" then starts at that very piece (whole pieces only: the engine completed it).
        parts: list[np.ndarray] | None = [] if kind == "reply" and item.rid else None
        self._streaming_seq = item.seq
        try:
            # The hold first: whatever the engine produced meanwhile waits in
            # `pending`; a failure that arrives before the hold releases means
            # no `say` at all for this piece.
            while self._held() and gen == self.gen and not self._closed:
                while not q.empty():
                    pending.append(q.get_nowait())
                await asyncio.sleep(0.03)
            if gen != self.gen or self._closed:
                stop.set()
                return True
            if item.dropped:
                stop.set()                       # replaced by a rewrite before it was said
                return True
            if not (pending and pending[0][0] == "error"):
                await self._send_say(item)
                said = True
            rate = max(1, int(getattr(self.tts, "sample_rate", 24000) or 24000))
            # v3 (paced): a chunk goes out in slices, each paced and each after the hold, so a
            # whole-sentence chunk (Kokoro) never puts all of its bytes in flight at once.
            step = max(1, int(PACE_SLICE_S * rate)) if self.send_ahead_s is not None else 0
            while True:
                what, chunk = pending.pop(0) if pending else await q.get()
                if what == "done":
                    break
                if what == "error":
                    log.error("voice: TTS failed: %s", chunk)
                    parts = None
                    ok = n > 0
                    if not ok:
                        await self.send({"type": "turn_error", "message_id": mid,
                                         "reason": "tts_failed", "detail": str(chunk)[:200]})
                    break
                pcm = chunk if isinstance(chunk, np.ndarray) else np.frombuffer(bytes(chunk), "<i2")
                pcm = pcm.reshape(-1)
                if not len(pcm):
                    continue
                if parts is not None:
                    parts.append(np.array(pcm, copy=True))
                # The piece's length grows by what the engine made, not by what went out yet: a
                # `heard {played_ms}` is a share of the whole sentence (a whole-sentence chunk is
                # paced out in slices; counted per slice, a tap 0.3 s in read as "heard it all").
                self.heard.add_audio(item.seq, round(len(pcm) / rate * 1000))
                for i in range(0, len(pcm), step or len(pcm)):
                    piece = pcm[i:i + step] if step else pcm
                    if not await self._wait_hold(gen) or not await self._paced(gen, len(piece) / rate):
                        stop.set()
                        break
                    if not self.speaking:
                        self._set_speaking(True)
                        await self.state("speaking")
                    if n == 0 and kind == "reply":
                        self._note_reply_said()
                    self._client_drained = False
                    await self._send_audio(kind=kind, gen=gen, seq=item.seq, pcm=piece, chunk=n,
                                           text=words if n == 0 else None, message_id=mid, role=item.role)
                    if n == 0:
                        await self._first_audio(t0, kind)
                    n += 1
        finally:
            stop.set()
            self._streaming_seq = None
            with contextlib.suppress(Exception):
                await fut
            if parts is not None and not item.dropped:
                # What the engine made that the loop never read (a cut or the hold's end came
                # first) is still this piece's audio — in order: held items, then the queue.
                for what, chunk in [*pending, *[q.get_nowait() for _ in range(q.qsize())]]:
                    if what == "error":
                        parts = None
                        break
                    if what == "chunk":
                        a = chunk if isinstance(chunk, np.ndarray) else np.frombuffer(bytes(chunk), "<i2")
                        parts.append(np.array(a.reshape(-1), copy=True))
                if parts and complete[0]:
                    self._keep_pcm(item, parts)
        if n:
            # v3: the sentence's padded tail + `audio_end` (Amendment T6-2), before the
            # next sentence's `say`; a no-op on v2 and for a gen a barge-in dropped.
            await self.tr.end_sentence(kind=kind, gen=gen, seq=item.seq)
        if said and kind == "reply" and item.rid:
            await self._maybe_say_end(item.rid)
        return ok

    def _take_kept(self, item: _Say) -> bool:
        """A reply piece whose audio was kept since it was queued (the cut piece of a "Tap to
        continue" finishes winding down after the continue is rebuilt): use it, no engine call."""
        if item.kind != "reply" or not item.rid or item.rid != self._pcm_rid:
            return False
        kept = self._pcm_keep.get((item.rid, item.text, item.voice))
        if kept is None or not len(kept):
            return False
        item.pcm = kept
        return True

    def _keep_pcm(self, item: _Say, parts: list[np.ndarray]) -> None:
        """Keep a whole reply piece's audio for an instant `replay_last` — the last reply
        only (a piece of another reply starts the cache afresh), at most
        REPLAY_PCM_MAX_S of it."""
        try:
            pcm = np.concatenate(parts).astype("<i2", copy=False).reshape(-1)
        except ValueError:
            return
        if item.rid != self._pcm_rid:
            self._pcm_keep.clear()
            self._pcm_rid, self._pcm_samples = item.rid, 0
        key = (item.rid, item.text, item.voice)
        if key in self._pcm_keep:
            return
        rate = max(1, int(getattr(self.tts, "sample_rate", 24000) or 24000))
        if self._pcm_samples + len(pcm) > REPLAY_PCM_MAX_S * rate:
            return
        self._pcm_keep[key] = pcm
        self._pcm_samples += len(pcm)

    async def _send_cached(self, item: _Say) -> bool:
        """A reply piece from the replay cache: the `say` the instant the hold releases,
        then its audio in REPLAY_CHUNK_S pieces (a duck holds it between them, a
        `barge_in` drops the rest) and `audio_end` — the engine is never called."""
        gen = item.gen
        if not await self._wait_hold(gen) or item.dropped:
            return True
        pcm = item.pcm
        rate = max(1, int(getattr(self.tts, "sample_rate", 24000) or 24000))
        step = max(1, int((PACE_SLICE_S if self.send_ahead_s is not None else REPLAY_CHUNK_S) * rate))
        self._streaming_seq = item.seq
        sent = 0
        try:
            await self._send_say(item)
            # the piece's whole length at once (a `heard` is a share of the sentence, not of
            # the slices sent so far)
            self.heard.add_audio(item.seq, round(len(pcm) / rate * 1000))
            for i in range(0, len(pcm), step):
                if not await self._wait_hold(gen) or not await self._paced(gen, min(step, len(pcm) - i) / rate):
                    break
                if not self.speaking:
                    self._set_speaking(True)
                    await self.state("speaking")
                if sent == 0:
                    self._note_reply_said()
                self._client_drained = False
                piece = pcm[i:i + step]
                await self._send_audio(kind=item.kind, gen=gen, seq=item.seq, pcm=piece, chunk=sent,
                                       text=item.text if sent == 0 else None, message_id=item.message_id,
                                       role=item.role)
                sent += 1
        finally:
            self._streaming_seq = None
        if sent:
            await self.tr.end_sentence(kind=item.kind, gen=gen, seq=item.seq)
        if item.rid:
            await self._maybe_say_end(item.rid)
        return True

    async def _first_audio(self, t0: float, kind: str) -> None:
        first_s = self.clock() - t0
        if kind != "reply":
            return
        self.acks.observe(first_chunk_s=first_s)
        if self._turn_acks is not None:
            self._turn_acks.on_reply_audio()
        if not self._first_audio_logged and self._eot_at is not None:
            self._first_audio_logged = True
            ms = round((self.clock() - self._eot_at) * 1000)
            log.info("voice: eot→first reply audio %d ms (first chunk %d ms)", ms, round(first_s * 1000))
            await self.send({"type": "metrics", "eot_to_first_audio_ms": ms,
                             "first_chunk_ms": round(first_s * 1000), **self._ack_metrics,
                             **self._speculation_metrics(), **self._link_metrics(),
                             **_platform_facts()})

    async def _lookup_relay(self) -> None:
        """The Tailscale path to the phone, once per socket, in the background (a
        ``tailscale status`` call must never sit between two audio chunks)."""
        try:
            self._relay_info = await asyncio.wait_for(self._relay(), 5.0)
        except Exception:
            self._relay_info = None

    def _link_metrics(self) -> dict:
        """``metrics.link`` (C.4) on a v3 socket: the server's view of the link (mode,
        rungs, why, degraded time, the slowest write) and the Tailscale path when the
        cached ``tailscale status`` knows the phone (never an address)."""
        if not self.tr.v3:
            return {}
        relay = self._relay_info
        try:
            link = self.tr.link_metrics(relay)
        except Exception:
            return {}
        link["mode"] = self._mode_next if self.slow_link != "always_phone" else "text"
        if relay and isinstance(relay.get("path"), str):
            link["path"] = relay["path"]
        return {"link": link}

    async def _turn_step(self, what: str, coro) -> None:
        """A pause decision or a turn end, off the audio worker: a failure is logged and
        the session goes back to listening (the turn is lost, the chat is not)."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice: %s failed; back to listening", what)
            self._hold = False
            with contextlib.suppress(Exception):
                await self.error("stt_failed", "Couldn't make that out. Try again.")
                await self.state(self._idle_state())

    def _interrupt_metrics(self) -> dict:
        """What an interruption cost (review-T3s2 #9, re-measured per review-T3s3 #8):
        ``cut_settle_ms`` = the `barge_in` that cut a reply → its note settled (the abort,
        ≤ 1.5 s on a gateway bot, then the wait for the exact `heard`, ≤ 1.5 s), and
        ``interrupt_to_submit_ms`` = the end of the words that cut it (`eot`) → their
        submit, i.e. the wait the listener felt after they stopped talking (the cut came
        mid-utterance, so cut → submit was mostly their own speech). Once per interruption."""
        at, self._interrupt_at = self._interrupt_at, None
        if at is None:
            return {}
        out: dict = {}
        if self._cut_settled_at is not None and self._cut_settled_at >= at:
            out["cut_settle_ms"] = round((self._cut_settled_at - at) * 1000)
        if self._eot_at is not None and self._submit_at is not None and self._submit_at >= self._eot_at:
            out["interrupt_to_submit_ms"] = round((self._submit_at - self._eot_at) * 1000)
        return out

    def _speculation_metrics(self) -> dict:
        """``speculated`` / ``speculation_hit`` for the turn being answered and the
        Speculator's running counters (C.4 ``metrics.speculation``)."""
        t = self._owner if isinstance(self._owner, _Turn) else self._current
        out = {"speculated": bool(t is not None and t.speculated),
               "speculation_hit": bool(t is not None and t.adopted)}
        if self.spec is not None:
            out["speculation"] = dict(self.spec.stats)
        return out

    async def _say_text_only(self, item: _Say) -> bool:
        """Text mode: the `say` alone (the phone reads it); the engine stays idle."""
        if not await self._wait_hold(item.gen) or item.dropped:
            return True
        self._streaming_seq = item.seq
        try:
            await self._send_say(item)
            if item.kind == "reply":
                self._note_reply_said()
                if not self.speaking:
                    self._set_speaking(True)
                    await self.state("speaking")
                self._client_drained = False
                if self._turn_acks is not None:
                    self._turn_acks.on_reply_audio()
                if not self._first_audio_logged and self._eot_at is not None:
                    self._first_audio_logged = True
                    await self.send({"type": "metrics",
                                     "eot_to_first_audio_ms": round((self.clock() - self._eot_at) * 1000),
                                     **self._ack_metrics, **_platform_facts()})
        finally:
            self._streaming_seq = None
        if item.kind == "reply" and item.rid:
            await self._maybe_say_end(item.rid)
        return True

    async def _tts_worker(self) -> None:
        while True:
            item = await self._say_q.get()
            self._tts_busy = True
            failed = False
            try:
                if item.gen != self.gen or item.dropped:
                    continue
                self.tr.at_sentence_boundary()          # a pending `link` rung/codec change
                if self._mode_next == "text":          # applied at the sentence boundary
                    await self._say_text_only(item)
                elif (item.pcm is not None and len(item.pcm)) or self._take_kept(item):
                    await self._send_cached(item)      # `replay_last`: the piece's kept audio
                else:
                    failed = not await self._stream_sentence(item)
            finally:
                self._tts_busy = False
            if failed and not self.speaking and self._say_q.empty():
                await self.state("listening" if self.listening else "paused")
            await self._maybe_finish_speaking()

    # ------------------------------------------------- acknowledgements #
    def _cancel_ack_task(self) -> None:
        if self._ack_task is not None:
            self._ack_task.cancel()
            self._ack_task = None

    def _start_turn_acks(self, text: str, t0: float) -> None:
        """The ack plan for the turn just sent (T5's pure gate, SPEC §6.2)."""
        self._cancel_ack_task()
        self._ack_earcon_done = False
        kit = self.acks
        intent = intent_of(text, after_barge_in=self._after_barge_in)
        self._after_barge_in = False
        turn = TurnAcks(kit.selector, bot_id=self.bot_id or kit.bot_id or "", voice=kit.voice,
                        intent=intent, t0=t0, eta_s=kit.estimator.eta_s,
                        last_turn_at=self._last_turn_at, last_acked=self._last_acked,
                        clip_s=kit.clip_s)
        turn.ACK_AFTER_S = float(self.settings.ack_after_s)
        self._turn_acks = turn
        self._last_turn_at = t0
        self._ack_metrics = {"ack_eta_ms": round(kit.estimator.eta_s * 1000), "ack_played": False,
                             "ack_tier": None, "tool_lines": 0}
        self._schedule_ack_poll()

    def _schedule_ack_poll(self) -> None:
        self._cancel_ack_task()
        turn = self._turn_acks
        if turn is None or self._closed:
            return
        now = self.clock()
        due = turn.next_due(now)
        if due is None:
            return
        # A line that is due but cannot play yet (the gate's `poll` says no
        # while `next_due` says now, e.g. a tool line while an ack plays) is
        # re-polled every 100 ms, never in a hot loop.
        self._ack_task = asyncio.create_task(self._ack_at(max(due, now + 0.1) if due <= now else due))

    def _poll_acks_soon(self) -> None:
        if self._turn_acks is None or self._closed:
            return
        self._cancel_ack_task()
        self._ack_task = asyncio.create_task(self._ack_at(self.clock()))

    async def _ack_at(self, when: float) -> None:
        delay = when - self.clock()
        if delay > 0:
            await asyncio.sleep(delay)
        await self._poll_acks()

    def _listener_talking(self) -> bool:
        """The listener is talking now — an utterance confirmed since the last `eot` and
        still open — or a merge hold waits for more of their turn: no acknowledgement,
        wait or tool line may play over them (review-T3s3 #3: a premature end at a comma
        acked 0.6 s into the listener's second clause)."""
        if self._hold:
            return True
        started = self._utt.started_at
        return bool(getattr(self.detector, "active", False) and started is not None
                    and (self._eot_at is None or started >= self._eot_at))

    async def _poll_acks(self) -> None:
        turn = self._turn_acks
        if turn is None:
            return
        now = self.clock()
        turn.on_speaking(self.speaking or now < self._line_until or self._listener_talking())
        got = turn.poll(now)
        if got is not None:
            kind, text = got
            played = await self._say_line(kind, text, intent=turn.intent if kind == "ack" else None)
            if kind == "ack":
                self._last_acked = played
                self._ack_metrics["ack_played"] = played
            elif kind == "tier":
                self._ack_metrics["ack_tier"] = "wait_15s" if turn.tier15_done else "wait_5s"
            else:
                self._ack_metrics["tool_lines"] = len(turn.tool_lines)
        elif (turn.ack_decided and not turn.acked and not self._ack_earcon_done
              and self.acks.mode != "off"
              and getattr(turn, "ack_outcome", "") in ("none_fit", "no_lines")):
            # The gate wanted a line (no reply queued, nothing playing, a real wait)
            # but none can play: this voice's lines are not rendered yet (the
            # render-in window of a new voice), are too long for the time left, or
            # there are no lines at all (the pools or the queue are unavailable).
            # The earcon stands in (SPEC §6.2, §10). A Reactions roll that SKIPPED
            # sends nothing (review-T3s2 #5: one rendered line elsewhere in the voice
            # no longer silences the gate, and a skip is never an earcon).
            self._ack_earcon_done = True
            self._last_acked = False
            await self.send({"type": "ack"})
        if turn.ack_decided and not turn.acked and not self._ack_metrics.get("ack_played"):
            self._last_acked = False
        self._schedule_ack_poll()

    async def _say_line(self, kind: str, text: str, *, intent: str | None = None) -> bool:
        """An acknowledgement (`ack`), wait/tool line (`tier`/`tool` → kind
        notice) or spoken notice, from the rendered lines, outside the hold.

        Audio mode: `say {kind}` + one audio frame. Text / phone-voice mode: an
        ack is the `ack` earcon (C.5 #12), everything else a `say {kind: notice}`
        alone. A tier/tool line never queues behind a reply piece that is
        already on its way (it would play after the answer started)."""
        if self._closed:
            return False
        if kind in ("tier", "tool") and (self._tts_busy or not self._say_q.empty() or self.speaking):
            return False
        if kind in ("ack", "tier", "tool") and self._listener_talking():
            return False
        wire_kind = "ack" if kind == "ack" else "notice"
        self.tr.at_sentence_boundary()
        mode = self._mode_next
        self.seq += 1
        item = _Say(self.gen, self.seq, text, None, wire_kind, intent=intent)
        if mode == "text":
            if kind == "ack":
                # The earcon: nothing was spoken, so nothing can echo — the line never
                # goes into `_recent_said` (review-T3s2 #6: a user who then says
                # "let me think" must not be dropped as our echo).
                await self.send({"type": "ack"})
            else:
                await self._send_say(item)        # records it in `_recent_said` once
            return True
        pcm = self.acks.pcm(text)
        if pcm is None and kind == "notice":
            pcm = await self._render_notice(text)
        if pcm is None or not len(pcm):
            if kind == "ack":
                await self.send({"type": "ack"})
                return False
            log.info("voice: %s line %r not rendered in this voice yet; skipped", kind, text)
            return False
        await self._send_say(item)
        self._client_drained = False
        self._line_until = self.clock() + len(pcm) / float(max(1, self.acks.rate))
        await self._send_audio(kind=wire_kind, gen=self.gen, seq=item.seq, pcm=np.asarray(pcm, "<i2"),
                               chunk=0, text=text, message_id=None, role="character", intent=intent,
                               last=True)
        return True

    async def _render_notice(self, text: str):
        try:
            if self._notice_fn is not None:
                pcm = await asyncio.to_thread(self._notice_fn, text)
            else:
                pcm, _ = await asyncio.to_thread(self.tts.synth, text, self.profile_id)
        except Exception:
            log.exception("voice: could not render the spoken notice")
            return None
        return None if pcm is None else np.asarray(pcm, "<i2")

    # ------------------------------------------------------------ audio #
    def _set_speaking(self, on: bool) -> None:
        if self.speaking and not on:
            self._audio_stopped_at = self.clock()
        self.speaking = on
        self._apply_threshold()
        if self._turn_acks is not None:
            self._turn_acks.on_speaking(on)
            if not on:
                self._poll_acks_soon()

    def _phone_voice_out(self) -> bool:
        """The phone's own voice is the output: text mode (the phone reads the
        `say` frames), or the uplink flags say it is speaking (C.2 bit0)."""
        return self._mode_next == "text" or self._local_speaking

    def _apply_threshold(self) -> None:
        """Layer 2 of the barge-in (SPEC §4.6, §13 C5), set while our own voice may
        reach the mic — a reply playing (server audio or the phone's voice) or the
        uplink's ``local_speaking`` flag: the sustained speech that counts as the
        driver talking over it is 0.25 s with server audio (the duck is reversible)
        and 0.6 s while the phone's voice speaks (no echo reference, no pause), and
        the Silero margin is 0.65. Otherwise the configured values."""
        cfg = self.detector.ep.cfg
        # An acknowledgement, notice or backchannel clip counts too (SPEC §6.2: playback
        # counts as speaking for the barge-in threshold; review-T3s2 #10).
        ours = self.speaking or self._local_speaking or self.clock() < self._line_until
        if ours:
            cfg.min_speech_s = (PHONE_VOICE_BARGE_IN_S if self._phone_voice_out()
                                else float(self.settings.barge_in_speech_s))
        else:
            cfg.min_speech_s = self._base[0]
        if hasattr(cfg, "threshold"):
            cfg.threshold = max(self._base_threshold, SPEAKING_VAD_THRESHOLD) if ours else self._base_threshold

    def _apply_patience(self, p: float) -> None:
        self.patience = p
        cfg = self.detector.ep.cfg
        cfg.turn_hard_silence_s = self._base[1] * p
        cfg.no_smart_turn_silence_s = self._base[2] * p

    async def barge_in(self, *, by_user_voice: bool, abort: bool | None = None,
                       keep_partial: bool = False) -> str | None:
        """Cut the reply: a new gen, the queue dropped, `barge_in {gen}`; with
        ``abort`` the host's turn is stopped too (returns its row id)."""
        self.gen += 1
        self._stop_tts.set()
        if self.rig is not None:
            # cancel{gen} to the rig first (L8 order: before abort_turn / Speculator.start).
            self.rig.cancel_gen(self.gen - 1)
        self._duck = False
        # The client flushes everything older than the new gen (C.4 `barge_in`): nothing it
        # holds is playing any more, so the send-ahead clock starts over at the next audio
        # (review-T3s4 #2: the reply after a tap waited 1.4-4.4 s for discarded audio).
        self._pace_t0 = None
        for r in self._replies.values():
            r.muted = True
            r.end_sent = True            # no say_end for a cut reply
        while not self._say_q.empty():
            self._say_q.get_nowait()
        self._set_speaking(False)
        # Audio of the cut gens that is still pending (never sent) is forgotten, and
        # whatever the engine still produces for them is dropped (Amendment T6-4).
        await self.tr.drop_older_gens(self.gen)
        await self.send({"type": "barge_in", "gen": self.gen})
        do_abort = (by_user_voice and self.settings.barge_abort_turn) if abort is None else abort
        if do_abort and self.host.abort:
            return await self._host_abort(keep_partial=keep_partial)
        return None

    async def _set_duck(self, on: bool) -> None:
        """Soft barge-in (H3). ``on``: someone may be talking over the reply —
        hold further audio and ask the client to duck/pause playback. Nothing
        is cancelled until the words prove it was not our own echo.
        With the phone voice (text mode) it is informational: the phone keeps
        speaking and cancels only on `barge_in` (feasibility H4)."""
        if self._duck == on:
            return
        self._duck = on
        if on:
            self._duck_at = self.clock()
            self._duck_verdict = False
            self._duck_informational = self._phone_voice_out()
        if self.rig is not None:
            # A false interruption pauses the rig's rendering too (tts.pause/resume).
            self.rig.pause() if on else self.rig.resume()
        await self.send({"type": "duck", "on": on, "gen": self.gen})

    async def _resume_reply(self, *, why: str = "") -> None:
        """A false interruption (echo, backchannel, noise, the timeout): end the
        duck and continue the PAUSED reply — ``resume {gen, rewind}`` rewinds
        the sentence when the pause lasted > 0.8 s (experience M3). Nothing to
        resume when no reply was playing (a plain ``duck off`` then). Over the
        phone's voice nothing was paused: ``rewind`` is never set there, so the
        phone never re-speaks a sentence because the mic heard it (H4)."""
        if not self._duck:
            return
        paused_s = self.clock() - self._duck_at
        informational = self._duck_informational
        self._duck_verdict = True
        await self._set_duck(False)
        self._hold = False
        if self.speaking or self._talking():
            rewind = paused_s > REWIND_AFTER_S and not informational
            await self.send({"type": "resume", "gen": self.gen, "rewind": rewind})
            await self.state("speaking")
            log.info("voice: reply resumed after %.2f s (%s%s)", paused_s, why or "false interruption",
                     ", rewind" if rewind else "")

    def _talking(self) -> bool:
        return self.speaking or not self._say_q.empty() or self._tts_busy

    def _decode_up(self, buf: bytes):
        """``Transport.decode_up``, with a malformed message returned as its ValueError."""
        try:
            return self.tr.decode_up(buf)
        except ValueError as e:
            return e

    async def _decode_batch(self, items: list[bytes]) -> list:
        """Decode a coalesced batch. A replay burst of Opus is ~1,500 packets (≈ 0.1 s of
        CPU, more under the cap): decoded in a worker thread, never on DisPatch's event
        loop (review-T3s4 #10). A live message or two (≈ 0.3 ms each) stays inline."""
        if len(items) > DECODE_INLINE_ITEMS or sum(len(b) for b in items) > DECODE_INLINE_BYTES:
            return await asyncio.to_thread(lambda: [self._decode_up(b) for b in items])
        return [self._decode_up(b) for b in items]

    async def _audio_worker(self) -> None:
        while True:
            chunk = await self._audio_q.get()
            # Coalesce whatever queued up while the last batch ran.
            items = [chunk]
            while not self._audio_q.empty() and len(items) < 50:
                items.append(self._audio_q.get_nowait())
            self._audio_q_bytes = max(0, self._audio_q_bytes - sum(len(b) for b in items))
            self._replay_q_bytes = max(0, self._replay_q_bytes - sum(len(b) for b in items if _is_replay(b)))
            batch: list[bytes] = []
            decoded = await self._decode_batch(items)
            for buf, got in zip(items, decoded, strict=True):
                if isinstance(got, ValueError):
                    await self._limited("bad_frame", "That audio wasn't understood.")
                    continue
                if got is None:                       # v2: raw PCM16, no offsets
                    pcm = buf[: len(buf) // 2 * 2]
                    batch.append(pcm)
                    self._pos = (self._pos + len(pcm) // 2) & wire3.U32
                    continue
                off, flags, pcm16 = got
                self._local_speaking = bool(flags & wire3.FLAG_LOCAL_SPEAKING)
                if flags & wire3.FLAG_REPLAY and len(pcm16):
                    end = (off + len(pcm16)) & wire3.U32
                    if self._replay_hi is None or wire3.u32_diff(end, self._replay_hi) > 0:
                        self._replay_hi = end
                try:
                    parts = self.tr.offsets.feed(off, pcm16)
                except GapTooLarge as e:
                    # A hole the zero-fill cannot (or must not) cover: what came before is
                    # a different stretch of time. Feed what is pending, then start over —
                    # an utterance that was open is lost with the detector's state.
                    await self._feed_detector(batch)
                    batch = []
                    log.info("voice: uplink %s of %d samples; turn detector reset", e.reason, e.gap)
                    self.detector.reset()
                    await self._drop_open_utterance()
                    parts = e.frames
                batch.extend(p.astype("<i2", copy=False).tobytes() for p in parts if len(p))
                self._pos = self.tr.offsets.expected
            self._apply_threshold()
            await self._feed_detector(batch)

    async def _feed_detector(self, batch: list[bytes]) -> None:
        """Run the turn detector over contiguous uplink audio and act on its events."""
        if not batch:
            return
        if not self.listening:
            return
        if not self.barge_in_enabled and self._talking():
            # "Interrupt by talking" is off: the mic is deaf while we talk.
            self._deaf = True
            return
        if self._deaf:
            self._deaf = False
            self.detector.reset()
            self._sstt_drop_open()
        data = b"".join(batch)
        # The batch is contiguous uplink audio ending at `_pos` (the transport's offsets on
        # v3, the received-sample count on v2). Each ≤ 1 s piece is fed with the offset
        # just past it, and `_pos` follows the piece being fed while its events are acted
        # on: `eot.upto`, the commit cursor and the open mark come from the detector's own
        # position, never from the end of a coalesced batch (review-T3s4 #3: a replay
        # burst of 30 s put them seconds late and the C.5 #8 belt then dropped a real
        # second utterance). The batch end is restored on the way out.
        end = self._pos
        base = (end - len(data) // 2) & wire3.U32
        try:
            for i in range(0, len(data), FEED_MAX_BYTES):
                if not self.listening or self._closed:
                    return
                piece = data[i:i + FEED_MAX_BYTES]
                at = (base + (i + len(piece)) // 2) & wire3.U32
                self._pos = at
                # ≤ 1 s of audio per call: the detector holds its lock for the whole call and
                # the decision calls (end_turn, reprobe) take it on the event loop — a 30 s
                # replay burst in one call stalled the loop (review-T3s3 #13).
                if self._feed_takes_end:
                    events = await asyncio.to_thread(self.detector.feed, piece, end=at)
                else:
                    events = await asyncio.to_thread(self.detector.feed, piece)
                await self._detect(events)
        finally:
            self._pos = end
        if (self.sstt is not None and getattr(self.detector, "active", False) and not self._closed
                and (self.rt_partials or self._duck or self._talking())
                and (self._tick_task is None or self._tick_task.done())):
            # Live captions (≤ 1/s) and the early verdict over a reply. With
            # captions off and nothing playing a tick would be a recogniser call
            # (0.4–1.2 s of CPU each) nobody reads: skipped.
            self._tick_task = asyncio.create_task(self._tick())
            self._turn_tasks.add(self._tick_task)
            self._tick_task.add_done_callback(self._turn_tasks.discard)

    async def _detect(self, events: list) -> None:
        """Act on the detector's events for one fed piece of audio."""
        for ev in events:
            if isinstance(ev, SpeechStart):
                self._cursor_open(getattr(ev, "start", None))
                await self._on_speech_start()
            elif isinstance(ev, Pause):
                pid = self._pause_id
                self._pause_task = self._spawn(self._turn_step("pause decision", self._on_pause(ev, pid)))
            elif isinstance(ev, Resumed):
                self._pause_id += 1           # a decision still in flight is stale
            elif isinstance(ev, TurnEnd):
                self._pause_id += 1
                # The utterance end's uplink offset, taken NOW (where the detector is), not
                # when the turn-end task runs (more audio may have been fed by then).
                upto = self._event_upto(ev, ev.silence_s)
                if self.sstt is not None:
                    # Off the audio worker: final() and the submit can take a second,
                    # and the mic must keep flowing meanwhile (the 64 KB queue holds
                    # 2 s). `_ending` is counted NOW, so speech that starts before the
                    # task runs is seen as speech during a turn end.
                    self._ending += 1
                    self._spawn(self._turn_step("turn end",
                                                self._end_turn(ev, silence_s=ev.silence_s, counted=True,
                                                               utt=self._utt, upto=upto)))
                else:
                    await self._on_turn_end(ev, upto=upto)
            elif isinstance(ev, Discarded):
                await self._drop_open_utterance()

    async def _undo_duck(self) -> None:
        """End a duck that never became anything: the paused reply resumes."""
        if self._duck:
            await self._resume_reply()

    async def _drop_open_utterance(self) -> None:
        """The open utterance is gone without a turn — noise that never reached the
        minimum speech (``Discarded``), or an uplink gap that reset the detector
        (``GapTooLarge``): a pause decision still in flight is stale, a merge hold
        waiting for it is released (and a turn end in progress must not re-arm it),
        its open mark leaves the cursor (a reconnect never replays it) and a duck it
        caused ends (the reply resumes)."""
        self._pause_id += 1
        self._hold = False
        self._resumed_during_end = False
        self._utt_while_playing = None
        self._sstt_drop_open()
        self._cursor_clear()
        await self._undo_duck()

    def _already_submitted(self, upto: int) -> bool:
        """C.5 #8, the server's belt to the `cursor {from}` answer: an utterance that ended
        inside REPLAYED uplink audio (flags bit1) at or below the stream's committed offset
        was submitted by an earlier socket (or before a restart) — a client that replayed
        from below the answer (one that gave up waiting for it on a slow link). It is
        dropped, never submitted twice. Live audio is never judged by this: a client whose
        offset clock restarted must not go deaf."""
        if self.cursor is None or self._replay_hi is None:
            return False
        if wire3.u32_diff(upto, self._replay_hi) > 0:
            return False                       # it ends in live audio
        try:
            committed = self.cursor.get(self.cursor_key)
        except Exception:          # a cursor that cannot be read never drops an utterance
            log.exception("voice: cursor read failed")
            return False
        return committed is not None and wire3.u32_diff(upto, committed) <= 0

    # ------------------------------------------------- the probe flow (v3) #
    def _on_frame(self, frame: np.ndarray | None, prob: float) -> None:
        """Detector frame sink (the audio thread): every utterance frame to the
        streaming STT with its speech probability; ``None`` = the detector ended the
        utterance (the boundary, see ``_sink_pending``)."""
        with self._sink_lock:
            if frame is None:
                self._sink_pending += 1
                if self._sink_pending > 1:
                    self._sink_buf.append(None)
                return
            if self._sink_pending:
                if len(self._sink_buf) < SINK_BUF_MAX:
                    self._sink_buf.append((frame, prob))
                return
            self.sstt.push(frame, vad_prob=prob)

    def _sink_busy(self) -> bool:
        """A turn end is still reading the stream: the open utterance's frames are
        waiting, so a pause or a tick cannot read this utterance's words yet."""
        with self._sink_lock:
            return self._sink_pending > 0

    def _sink_release(self) -> None:
        """The turn end that owns the oldest end mark is done with the stream: reset it,
        then hand it the frames that waited — the next utterance, up to ITS end mark."""
        with self._sink_lock:
            self.sstt.reset()
            if self._sink_pending <= 0:
                self._sink_pending = 0
                self._sink_buf.clear()
                return
            self._sink_pending -= 1
            while self._sink_buf:
                item = self._sink_buf.popleft()
                if item is None:
                    break                       # that utterance ended too: its end owns the rest
                self.sstt.push(item[0], vad_prob=item[1])

    def _sstt_drop_open(self) -> None:
        """The OPEN utterance is dropped (a discard, a detector reset): its frames go —
        from the stream when no turn end is reading it, else from the waiting frames
        after the last end mark (the ending utterance's own frames stay for its final())."""
        if self.sstt is None:
            return
        with self._sink_lock:
            if not self._sink_pending:
                self.sstt.reset()
                return
            items = list(self._sink_buf)
            last = max((i for i, x in enumerate(items) if x is None), default=-1)
            self._sink_buf = deque(items[:last + 1])

    def _mergeable(self, prev: _Turn | None, *, onset: float | None = None) -> bool:
        """Can ``prev`` still take an addendum: sent, answer not failed, nothing of
        its reply audible yet (SPEC §4.3: once a sentence plays it is an interruption)
        — judged at the speech ``onset`` when known: a reply that became audible only
        after the driver had started talking did not interrupt them."""
        if prev is None or self._last_submit is None or prev.abandoned or prev.failed:
            return False
        if not prev.said:
            return True
        return onset is not None and prev.said_at is not None and prev.said_at >= onset

    def _addendum_open(self, now: float, prev: _Turn | None, *, onset: float | None = None) -> bool:
        """The "oh, and…" window: within ``rt.addendum_s`` × patience of the previous
        transcript while nothing of its reply was heard (experience H1)."""
        if not self._mergeable(prev, onset=onset) or self.sstt is None:
            return False
        return now - prev.at < self.rt.addendum_s * max(0.1, self.patience)

    def _playing_now(self, now: float) -> bool:
        """Our voice is actually playing (or stopped a moment ago) — server audio,
        or the phone's voice per the uplink flag: the only time the echo verdict is
        trusted (Amendment T4-3)."""
        return (self.speaking or self._local_speaking
                or (now - self._audio_stopped_at) < ECHO_PLAYING_GRACE_S
                or (now - self._line_until) < ECHO_PLAYING_GRACE_S)   # an ack/notice/backchannel clip

    def _echo_candidates(self) -> list[str]:
        """The sentences our voice may be playing right now, for the echo rule. The
        client queues sentences ahead of the speaker (a fast engine, the phone's
        voice reading `say` frames), so "the last two the server said" can be
        sentences the speaker has not reached yet. With a fresh `heard` of this gen:
        the piece the listener is at and its two neighbours; without one: every piece
        of this gen said since the client last drained (at most six). Plus the last
        two said (Amendment T4-3's default)."""
        out: list[str] = []

        def add(t: str | None) -> None:
            if t and t not in out:
                out.append(t)

        h = self._last_heard
        if h is not None and h.gen == self.gen and self.clock() - h.at <= HEARD_STALE_S:
            for q in (h.seq - 1, h.seq, h.seq + 1):
                g_t = self._said_text.get(q)
                if g_t is not None and g_t[0] == self.gen:
                    add(g_t[1])
        else:
            lo = self._drained_seq or 0
            pending = [g_t[1] for q, g_t in sorted(self._said_text.items())
                       if q > lo and g_t[0] == self.gen]
            for t in pending[-6:]:
                add(t)
        for t in list(self._recent_said)[-2:]:
            add(t)
        return out

    async def _on_speech_start(self) -> None:
        now = self.clock()
        # A new utterance: its own state object. A turn end still running for the
        # previous one keeps that one's object (review-T3s3 #1(c)).
        self._utt_seq += 1
        self._utt = _Utt(self._utt_seq, started_at=now, while_playing=self._playing_now(now))
        self._interrupt_at = None             # a metric left over from an utterance that never became a turn
        self.bc_gate.new_turn()               # ≤ 3 backchannels per utterance (SPEC §4.7)
        prev = self._current
        if self._ending:
            # The utterance before this one is being turned into a turn right now
            # (final(), the submit): this speech is more of it — every reply waits,
            # and its end merges with the turn being sent (`_end_turn` keeps the hold).
            self._resume_gap = (now - self._eot_at) if self._eot_at is not None else None
            self._resumed_during_end = True
            self._hold = True
            if self.speaking:
                await self._set_duck(True)
            await self.state("hearing")
            return
        # The speech began min_speech_s (and a frame or two) before the detector said so.
        onset = now - float(getattr(self.detector.ep.cfg, "min_speech_s", 0.25) or 0.0)
        if self._eot_at is not None and self._mergeable(prev, onset=onset):
            self._resume_gap = now - self._eot_at
        else:
            self._resume_gap = None
        if (self._last_submit and self._mergeable(prev, onset=onset)
                and (now < self._merge_until or self._addendum_open(now, prev, onset=onset))):
            self._hold_onset = onset
            # Maybe the driver wasn't finished (the hold), or this is the
            # "oh, and…" addendum: hold the reply until we know whether this is
            # more of the same turn. A reply already playing is an interruption.
            self._hold = True
            if self.speaking:
                await self._set_duck(True)
        elif self._talking():
            await self._set_duck(True)
        await self.state("hearing")

    async def _on_turn_end(self, ev: TurnEnd, *, upto: int | None = None) -> None:
        self._eot_at = self.clock()
        merging = self._hold and self._last_submit is not None
        self._eot_upto = upto if upto is not None else self._upto(ev.silence_s)
        if self._already_submitted(self._eot_upto):
            await self._drop_submitted_utterance()
            return
        await self.send({"type": "eot", "reason": ev.reason, "upto": self._eot_upto})
        await self.state("transcribing")
        try:
            text = await asyncio.to_thread(self.stt.transcribe, ev.audio)
        except Exception:
            log.exception("voice: STT failed")
            self._hold = False
            await self._undo_duck()
            await self.error("stt_failed", "Couldn't make that out. Try again.")
            await self.state(self._idle_state())
            return
        stt_ms = round((self.clock() - self._eot_at) * 1000)
        if not text:
            self._hold = False
            self._utt_while_playing = None
            self._cursor_clear()
            await self.send({"type": "transcript", "text": "", "upto": self._eot_upto})
            await self._undo_duck()
            await self.state(self._idle_state())
            return
        # Words that ran over our voice: the same verdict as at a pause (echo,
        # backchannel, stop or real — classify_interruption, the echo rule only
        # for an utterance that began while our voice played; Amendment T4-3).
        cut = (self._duck or self._talking()) and not merging
        if ((cut or self._utt_while_playing) and not merging
                and await self._consumed_over_reply(text, self._speech_s_of(ev), self.clock())):
            return
        self._utt_while_playing = None
        if not self._check_live():
            return
        prev = self._current
        if not merging and not cut and await self._maybe_replay_command(text, prev):
            return
        if merging:
            prev_text, prev_id = self._last_submit
            prev_text = prev_text.rstrip().rstrip("?.!…").rstrip()
            text = f"{prev_text} {text}".strip()[:MAX_UTTERANCE_CHARS]
            # Abandon the early turn: stop it, mute anything it already said,
            # take its user row back, then send the whole utterance once.
            # A merge persists nothing (keep_partial=False) and writes no
            # heard note (SPEC §4.8).
            await self.barge_in(by_user_voice=False, abort=False)
            await self._abandon(prev, abort=True)
            if prev_id and self.host.retract:
                with contextlib.suppress(Exception):
                    await self.host.retract(self.thread_id, prev_id)
        elif cut:
            # Confirmed: real words over the reply. NOW cut it (and the run):
            # abort(keep_partial=True) → wait for the exact `heard` → note_heard.
            await self._interrupt(prev, by_user_voice=True, abort=self.settings.barge_abort_turn)
            self._after_barge_in = True
        elif prev is not None and not prev.closed:
            # Speech while still waiting on the last answer (review M4): the
            # new question replaces it — never two answers back to back.
            await self._abandon(prev, abort=True)
        self._hold = False
        try:
            msg_id = await self._submit(text)
        except Exception:
            log.exception("voice: submit failed")
            await self.send({"type": "transcript", "text": text, "merged": merging})
            await self._turn_failed(None, "agent_failed", "Could not send that message.")
            return
        # Track the turn BEFORE the next await: its echo and its run's
        # frames may already be on their way.
        turn = _Turn(msg_id=msg_id, text=text, at=self.clock(),
                     echoed=bool(msg_id) and msg_id in self._echoed_ids)
        self._current = turn
        if msg_id:
            self._turns.append(turn)
        if prev is not None and prev.abandoned and not turn.echoed:
            self._await_user_id = msg_id
        await self.send({"type": "transcript", "text": text, "merged": merging,
                         "message_id": msg_id, "upto": self._eot_upto})
        await self.send({"type": "metrics", "eot_to_transcript_ms": stt_ms,
                         "utterance_ms": round(len(ev.audio) / 16), "reason": ev.reason,
                         **self._interrupt_metrics()})
        self._last_submit = (text, msg_id)
        self._merge_until = self.clock() + self.settings.merge_window(self.patience)
        self._first_audio_logged = False
        self._awaiting_reply = True
        await self.state("thinking")
        self._start_turn_acks(text, self._eot_at)

    def _upto(self, silence_s: float) -> int:
        """The uplink sample offset of the utterance end (C.4 ``eot.upto``, C.5 #7):
        the stream offset of the last sample the detector was fed (the transport's
        offsets on v3, the received-sample count on v2) minus the trailing silence."""
        return (self._pos - round(max(0.0, silence_s) * 16000)) & wire3.U32

    def _event_upto(self, ev, silence_s: float | None = None) -> int:
        """``eot.upto`` for a detector event: the offset just past the frame it was decided
        at (``ev.pos``, the real detector's) minus ITS trailing silence; a detector that
        gives no position: the piece being fed (``_pos``) minus ``silence_s``."""
        pos = getattr(ev, "pos", None)
        if isinstance(pos, int) and not isinstance(pos, bool):
            return (pos - round(max(0.0, float(getattr(ev, "silence_s", 0.0) or 0.0)) * 16000)) & wire3.U32
        return self._upto(silence_s if silence_s is not None else float(getattr(ev, "silence_s", 0.0) or 0.0))

    def _utterance_start(self) -> int:
        """The uplink offset where the open utterance started (pre-roll included): what
        a reconnect must replay from if the socket drops mid-utterance (Amendment T6-1)."""
        frames_l = getattr(getattr(self.detector, "ep", None), "_frames", None)
        buf = getattr(self.detector, "_buf", None)
        frames = len(frames_l) if frames_l is not None else 0
        buffered = len(buf) if buf is not None else 0
        back = frames * VAD_FRAME + buffered if frames else 2 * 16000
        return (self._pos - back) & wire3.U32

    def _cursor_open(self, start: int | None = None) -> None:
        """The open mark (Amendment T6-1) at the utterance's first frame: ``start`` from the
        detector's SpeechStart when it carries one, else computed from its state now."""
        if self.cursor is not None:
            try:
                at = start if isinstance(start, int) and not isinstance(start, bool) else self._utterance_start()
                self.cursor.mark_open(self.cursor_key, at & wire3.U32)
            except Exception:
                log.exception("voice: cursor mark failed")

    def _cursor_clear(self) -> None:
        if self.cursor is not None:
            try:
                self.cursor.clear_open(self.cursor_key)
            except Exception:
                log.exception("voice: cursor clear failed")

    def _cursor_commit(self, text: str, message_id: str | None) -> None:
        """The utterance up to the last ``eot`` is submitted (C.5 #8): a reconnect never
        replays it into a second submit, and a client that missed the transcript gets it
        again (``replayed: true``)."""
        if self.cursor is None or self._eot_upto is None:
            return
        try:
            self.cursor.set(self.cursor_key, self._eot_upto,
                            meta={"text": text[:1000], "message_id": message_id})
        except Exception:
            log.exception("voice: cursor commit failed")

    async def _on_pause(self, ev: Pause, pid: int) -> None:
        """A probe mark inside a pause: Smart Turn (thread A) and the tail
        transcript (thread B) in parallel, then — if speech did not resume
        meanwhile — the interruption verdict when a reply is playing, and the
        end-of-turn decision (SPEC §4.1, §4.3, §4.6)."""
        loop = asyncio.get_running_loop()
        turn_fn = self.detector.ep.turn_fn
        rig_turn = getattr(self.sstt, "turn_fn_for", None)
        if callable(rig_turn):
            turn_fn = rig_turn(turn_fn)       # the rig's eou{p} while the utterance is on it
        st_fut = loop.run_in_executor(None, turn_fn, ev.audio) if turn_fn is not None else None
        partial = None
        if not self._sink_busy():
            # (While the previous turn end still reads the stream this utterance's frames
            # wait outside it: no words yet — the policy decides on Smart Turn alone.)
            try:
                partial = await asyncio.to_thread(self.sstt.at_pause)
            except Exception:
                log.exception("voice: streaming STT at_pause failed")
                partial = None
        p: float | None = None
        if st_fut is not None:
            try:
                got = await st_fut
                p = float(got) if got is not None else None
            except Exception:
                log.exception("voice: smart turn failed at the probe")
                p = None
        if pid != self._pause_id or self._closed or not self.listening:
            return                          # speech resumed, the turn ended, or we paused
        now = self.clock()
        self._last_p = p
        if partial is not None and partial.text:
            await self._send_partial(partial, force=True)
        # Speech over a reply that is playing (or still on its way): the words
        # decide what it was. Not while the turn is a merge candidate (the policy's
        # hold, the "oh, and…" addendum: nothing of the reply was said, so the
        # words join the turn — SPEC §4.3), and not twice for one utterance.
        over_reply = (self._duck or self._talking()) and not self._hold
        if over_reply and not self._interrupted_utt:
            verdict = self._classify(partial.text if partial else "", ev.speech_s, now)
            if verdict != "real":
                await self._false_or_stop(verdict, partial.text if partial else "")
                return
            # A real interruption: cut the reply and the run FIRST (abort before
            # Speculator.start, hardening L8), note what was heard; the words
            # become the next turn through the decision below.
            self._interrupted_utt = True
            self._duck_verdict = True
            await self._interrupt(self._current, by_user_voice=True, abort=self.settings.barge_abort_turn)
            self._after_barge_in = True
            if pid != self._pause_id or self._closed:
                return
        d = self.policy.decide(Signals(p_done=p, partial=partial, pause_s=ev.silence_s,
                                       patience=self.patience))
        if d.action == "reprobe":
            self.detector.request_reprobe(ev.silence_s + (d.hold_s or REPROBE_AFTER_S))
            return
        if d.action == "listen":
            if d.backchannel_ok:
                await self._maybe_backchannel(ev, now)
            return
        if d.action == "hold":
            await self.send({"type": "eot_soft", "p": round(p, 3) if p is not None else None,
                             "hold_ms": round(d.hold_s * 1000)})
            if pid != self._pause_id or self._closed:
                return                      # speech resumed while the frame went out
        turn_end = self.detector.end_turn(d.reason or d.action, p)
        if turn_end is None:
            return                          # ended meanwhile (hard silence beat us)
        self._pause_id += 1
        await self._end_turn(turn_end, partial=partial, speculate=d.speculate,
                             hold_s=d.hold_s if d.action == "hold" else 0.0, silence_s=ev.silence_s,
                             utt=self._utt, upto=self._event_upto(turn_end, ev.silence_s))

    def _classify(self, text: str, speech_s: float, now: float, *, record: bool = True,
                  utt: _Utt | None = None) -> str:
        """``classify_interruption`` over the last two sentences said, with the
        echo rule only while audio is playing (Amendment T4-3). No words at all
        (a cough, road noise) is a false interruption, never a backchannel for
        the two-in-3-s rule. ``utt``: the utterance the words are (its own
        "began while our voice played"; default the open one)."""
        if not _bc_tokens(text):
            return "noise"
        began = (utt if utt is not None else self._utt).while_playing
        playing = began if began is not None else self._playing_now(now)
        recent = self._echo_candidates() if (playing or self._playing_now(now)) else []
        v = classify_interruption(text, dur_s=speech_s, recent_said=recent,
                                  prev_verdict=self._prev_verdict, now=now)
        if record and v in ("backchannel", "stop"):
            # A tick's verdict is not recorded (record=False) unless acted on: an early,
            # unstable partial ("Mm-hmm." for the first 0.3 s of "Actually…") must not
            # arm the two-backchannels-in-3-s rule for the utterance's own pause.
            self._prev_verdict = (v, now)
        return v

    async def _false_or_stop(self, verdict: str, text: str, *, discard: bool = True,
                             utt: _Utt | None = None) -> None:
        """echo / backchannel / noise → the reply resumes; stop → the reply is cut
        like a tap (the answer finishes into the chat for reading, the listener's
        position is noted) and nothing becomes a turn. The utterance is dropped
        (``discard``: still open in the detector at a pause verdict; at the end of
        the utterance the detector has already moved on and must not be touched —
        a new utterance may have begun meanwhile)."""
        self._duck_verdict = True
        if discard:
            self._discard_utterance()
        else:
            # The utterance has ended (the end verdict): its state goes, and the session's
            # only when no newer utterance has begun meanwhile (it owns them now).
            utt = utt if utt is not None else self._utt
            utt.while_playing = None
            if self._own_utt(utt):
                self._hold = False
                self._cursor_clear()          # never replayed into a turn after a reconnect
        if verdict == "echo":
            await self.send({"type": "echo_ignored", "text": text})
        if verdict == "stop":
            self._hold = False
            cur = self._current
            await self._interrupt(cur if (cur is not None and not cur.closed) else None,
                                  by_user_voice=True, abort=False)
            await self.state("listening" if self.listening else "paused")
            log.info("voice: stop word over the reply (%r)", text[:40])
            return
        await self._resume_reply(why=verdict)
        if not (self.speaking or self._talking()):
            await self.state(self._idle_state())

    def _discard_utterance(self) -> None:
        self._pause_id += 1
        self._hold = False
        self._utt_while_playing = None
        try:
            self.detector.discard_utterance()
        except AttributeError:
            self.detector.reset()
        self._sstt_drop_open()
        # The open mark (Amendment T6-1) goes with it: a reconnect must never replay
        # our own echo or a backchannel into the next session as a turn.
        self._cursor_clear()

    async def _consumed_over_reply(self, text: str, speech_s: float, now: float, *,
                                   utt: _Utt | None = None) -> bool:
        """An utterance that ran over a reply and ENDED with no word verdict yet (the
        hard silence beat the pause probe, or the whole-utterance path): its words
        decide. True = consumed (echo, backchannel, noise or stop: nothing becomes a
        turn, the reply resumes or is cut); False = a real interruption."""
        verdict = self._classify(text, speech_s, now, utt=utt)
        if verdict == "real":
            return False
        if self.spec is not None:
            await self.spec.discard()
        await self._false_or_stop(verdict, text, discard=False, utt=utt)
        return True

    def _early_stop(self, text: str) -> bool:
        """A short tick partial (≤ 3 words) carrying a command word ("Actually, wait", "No,
        the other one") that our voice did not just say: cut now, like a stop word — the
        end of the utterance decides whether it was only that (nothing) or a turn. The
        command word completes long before a third word does ("Actually, wait, stop" →
        "wait" ends 0.6 s before "stop"). A command word the bot itself is saying may be
        its own echo: then the 3-word rule stands."""
        core = [w for w in _bc_tokens(text) if w not in FILLERS]
        if not core or len(core) > 3:
            return False
        cmds = {w for w in core if w in COMMAND_WORDS}
        if not cmds:
            return False
        ours = set(_bc_tokens(" ".join(self._echo_candidates())))
        return not (cmds & ours)

    def _stop_only(self, text: str) -> bool:
        """The words are nothing but a stop: a command word, a stop phrase."""
        return classify_interruption(text, dur_s=0.0, recent_said=(), prev_verdict=None,
                                     now=self.clock()) == "stop"

    async def _abort_after_stop_cut(self, utt: _Utt | None = None) -> None:
        """A tick's stop word cut the reply without aborting its run (a bare "stop"
        lets the answer finish into the chat); the utterance went on with real
        words, so the run is aborted now (keep_partial: the spoken prefix is the
        row) and the cut's heard note lands on the row the abort names."""
        utt = utt if utt is not None else self._utt
        if utt.stop_cut_aborted:
            return
        utt.stop_cut_aborted = True
        row = await self._host_abort(keep_partial=True)
        # The cut's note may be in either order with the abort: already deferred (the
        # row had no id) → noted here; still waiting for the exact `heard` → it picks
        # up `_cut_row` as its target.
        self._cut_row = row
        d = self._deferred_note
        if row and d is not None:
            self._deferred_note = None
            await self._call_note_heard(row, d[1])
        self._after_barge_in = True

    def _speech_s_of(self, ev: TurnEnd) -> float:
        """Speech in a finished utterance: its audio minus the pre-roll and the ≤ 0.2 s
        of trailing silence the endpointer keeps (the backchannel rule wants < 1.0 s)."""
        pre = float(getattr(getattr(getattr(self.detector, "ep", None), "cfg", None), "pre_roll_s", 0.3) or 0.0)
        return max(0.0, len(ev.audio) / 16000 - pre - min(0.2, max(0.0, ev.silence_s)))

    async def _tick(self) -> None:
        """Live captions while the listener talks (``tick()`` ≤ 1/s), and the
        early verdict over a reply: a tick partial that is an echo resumes the
        reply at once, one with ≥ 3 words of the listener's own cuts it without
        waiting for a pause (speech with no pause for seconds is not a cough)."""
        pid = self._pause_id
        over = (self._duck or self._talking()) and not self._hold and not self._interrupted_utt
        default_s = self._tick_default_s
        if hasattr(self.sstt, "tick_interval_s"):
            if default_s is None:
                default_s = self._tick_default_s = float(self.sstt.tick_interval_s)
            self.sstt.tick_interval_s = min(default_s, TICK_OVER_REPLY_S) if over else default_s
        if self._sink_busy():
            return                          # this utterance's frames wait for the previous end
        try:
            partial = await asyncio.to_thread(self.sstt.tick)
        except Exception:
            log.exception("voice: streaming STT tick failed")
            return
        if partial is None or pid != self._pause_id or self._closed or not self.listening:
            return
        if partial.text:
            await self._send_partial(partial)
        if (not (self._duck or self._talking()) or self._hold or self._interrupted_utt
                or not partial.text):
            return
        now = self.clock()
        verdict = self._classify(partial.text, max(0.0, now - (self._utt_started_at or now)), now,
                                 record=False)
        if verdict == "echo":
            # Our own bleed so far: the reply resumes at once — but the utterance stays
            # open. A listener who starts by quoting the bot ("Dentist at three? Can we
            # move it?") goes on, and the words at the pause or the end decide; a true
            # echo is dropped there (review-T3s3 #4: discarding here threw away the turn).
            self._duck_verdict = True
            await self._resume_reply(why="echo (tick)")
        elif verdict == "stop" or (verdict == "real" and self._early_stop(partial.text)):
            self._prev_verdict = ("stop", now)
            # "Wait" / "stop" / "hold on" heard while the listener is still talking:
            # the reply is cut NOW (like a tap: the run is not aborted yet), and the
            # utterance goes on — at its end, more words make it a turn (the run is
            # aborted then, before the submit) and a bare "stop" makes nothing.
            self._interrupted_utt = True
            self._stop_cut = True
            self._duck_verdict = True
            self._interrupt_at = now              # the cut, for interrupt_to_submit_ms
            cur = self._current
            await self._interrupt(cur if (cur is not None and not cur.closed) else None,
                                  by_user_voice=True, abort=False)
            log.info("voice: stop word over the reply (%r); listening on", partial.text[:40])
        elif verdict == "real" and len(_bc_tokens(partial.text)) >= TICK_MIN_WORDS:
            self._interrupted_utt = True
            self._duck_verdict = True
            await self._interrupt(self._current, by_user_voice=True, abort=self.settings.barge_abort_turn)
            self._after_barge_in = True

    async def _send_partial(self, partial, *, force: bool = False) -> None:
        """``partial {committed, tail, stable, t}`` (C.4): only with ``start.rt.partials``,
        ≤ 2/s, never a message id. A pause partial always goes out (``force``)."""
        if not self.rt_partials:
            return
        now = self.clock()
        if not force and now - self._last_partial_at < PARTIAL_MIN_GAP_S:
            return
        self._last_partial_at = now
        await self.send({"type": "partial", "committed": partial.committed, "tail": partial.tail,
                         "stable": int(partial.stable_words), "t": float(partial.t_audio)})

    async def _maybe_backchannel(self, ev: Pause, now: float) -> None:
        """A ``listen`` row: one pre-rendered line at −6 dB, ``kind: backchannel``,
        never bumping ``gen`` (SPEC §4.7) — only with Interrupt-by-talking on,
        Reactions Often (``start.rt.backchannels``) and the bot opted in, never
        over our own reply, never for the phone voice (its lines are earcons)."""
        if not (self.barge_in_enabled and self.rt_backchannels and self.bot_backchannel):
            return
        if self._talking() or self.speaking or self._mode_next == "text" or self._closed:
            return
        turn_s = now - (self._utt_started_at if self._utt_started_at is not None else now)
        if not self.bc_gate.ok(turn_s=turn_s, pause_s=ev.silence_s, now=now):
            return
        kit = self.acks
        try:
            line = kit.selector.pick("statement", bot_id=self.bot_id or kit.bot_id or "", voice=kit.voice,
                                     now=now, last_turn_at=None, last_acked=False,
                                     fits=lambda t: kit.pcm(t) is not None)
        except Exception:
            log.exception("voice: backchannel pick failed")
            return
        pcm = kit.pcm(line) if line else None
        if pcm is None or not len(pcm):
            return
        self.seq += 1
        item = _Say(self.gen, self.seq, line, None, "backchannel")
        self.tr.at_sentence_boundary()
        await self._emit_say(item)
        self._remember_seq(item.seq, None)
        self._said_text[item.seq] = (item.gen, line)
        self._recent_said.append(line)          # its echo is never a turn (C.5 #5)
        quiet = (np.asarray(pcm, np.float32) * BACKCHANNEL_GAIN).astype("<i2")
        self._line_until = self.clock() + len(quiet) / float(max(1, self.acks.rate))
        await self._send_audio(kind="backchannel", gen=self.gen, seq=item.seq, pcm=quiet, chunk=0,
                               text=line, message_id=None, role="character", last=True)
        self.bc_gate.sent(now)

    async def _end_turn(self, ev: TurnEnd, *, partial=None, speculate: bool = False,
                        hold_s: float = 0.0, silence_s: float | None = None,
                        counted: bool = False, utt: _Utt | None = None, upto: int | None = None) -> None:
        """The streaming end of turn: ``eot`` → Speculator.start (an end/hold row
        with text) ‖ ``final()`` → commit → merge/interrupt/replace rules →
        submit (``prefetched`` when adopted) → ``transcript`` → the reply hold
        (``hold_s``; 0 on a confident end) and the addendum window.

        Turn ends are serialised (a second utterance that ends while the first is
        still being submitted waits for it, so its merge/replace decision sees the
        turn it follows), and speech that starts while one is in progress holds
        every reply and joins that turn (the false end, SPEC §4.3). ``utt`` is the
        ending utterance's own state, taken when its end was decided (the next
        utterance may begin before this end runs). The stream is released exactly
        once per end, after its ``final()`` (or on the way out of a failure), so the
        frames that waited — the next utterance — go into a fresh stream."""
        if not counted:
            self._ending += 1
        utt = utt if utt is not None else self._utt
        entered = released = False

        def stream_done() -> None:
            nonlocal released
            if not released:
                released = True
                self._sink_release()

        try:
            async with self._end_lock:
                entered = True
                before = self._current
                await self._end_turn_locked(ev, partial=partial, speculate=speculate,
                                            hold_s=hold_s, silence_s=silence_s, utt=utt,
                                            stream_done=stream_done, upto=upto)
                submitted = self._current is not before
        finally:
            self._ending -= 1
            if entered and not released and self.sstt is not None:
                stream_done()
        if self._resumed_during_end and not self._ending:
            self._resumed_during_end = False
            if not getattr(self.detector, "active", True):
                # The speech that resumed during the turn end has ended already (its own
                # turn end, serialised after this one, merged it): nothing left to hold
                # for — re-arming here held the merged reply for ever (review-T3s3 #1).
                return
            now = self.clock()
            prev = self._current
            if submitted:
                self._hold = True          # the new speech is more of the turn just sent
            else:                          # nothing was sent: only a real merge window holds
                self._hold = bool(self._last_submit and self._mergeable(prev)
                                  and (now < self._merge_until or self._addendum_open(now, prev)))

    def _own_utt(self, utt: _Utt) -> bool:
        """The ending utterance is still the session's newest one (no SpeechStart since):
        its end may reset the session-level state the open utterance would own."""
        return self._utt is utt

    async def _spec_drop(self) -> None:
        """A turn end that submits nothing: its speculative run (if any) is discarded and
        the Speculator's per-turn counters start over (review-T3s3 #9)."""
        if self.spec is not None:
            with contextlib.suppress(Exception):
                await self.spec.discard()
            self.spec.new_turn()

    async def _spec_start_after(self, abort_task: asyncio.Task | None, text: str):
        """Speculator.start, but only once a stop-cut run is aborted (hardening L8:
        abort before the speculative start; the abort itself runs beside ``final()``)."""
        if abort_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(abort_task)
        return await self.spec.start(text)

    async def _end_turn_locked(self, ev: TurnEnd, *, partial=None, speculate: bool = False,
                               hold_s: float = 0.0, silence_s: float | None = None,
                               utt: _Utt | None = None, stream_done: Callable[[], None] | None = None,
                               upto: int | None = None) -> None:
        utt = utt if utt is not None else self._utt
        if stream_done is None:
            def stream_done() -> None:
                self._sink_release()
        self._eot_at = self.clock()
        now = self._eot_at
        prev = self._current
        # Speech that resumed while the previous turn end was still being sent is more of
        # that turn — also when it ENDED before that turn end finished (this end was
        # queued behind it): the earlier end cleared the hold before its submit, so the
        # hold alone would make it a second, replacing question (review-T3s3 #1).
        merging = ((self._hold or self._resumed_during_end)
                   and self._mergeable(prev, onset=utt.hold_onset))
        if merging:
            self._resumed_during_end = False
        silence = ev.silence_s if silence_s is None else silence_s
        # (taken when the end was decided — see `_event_upto`; a caller without it: now)
        self._eot_upto = upto if upto is not None else self._upto(silence)
        if self._already_submitted(self._eot_upto):
            stream_done()
            await self._drop_submitted_utterance(utt)
            return
        await self.send({"type": "eot", "reason": ev.reason, "upto": self._eot_upto})
        await self.state("transcribing")
        spec_task = None
        abort_task = None
        head_at_start = (self._head_moves, self._head_id)
        if (utt.stop_cut and partial is not None and partial.text
                and not self._stop_only(partial.text)):
            # More than "stop" was said after a stop cut: the run the cut left going is
            # aborted — beside final() (review-T3s3 #2: an inline abort of up to 1.5 s
            # was a window for the next words to land in the wrong turn), and always
            # BEFORE the speculation starts on the new words (L8).
            abort_task = asyncio.create_task(self._abort_after_stop_cut(utt))
        if (speculate and partial is not None and partial.text and self._speculation_allowed()
                and not merging and self._check_live()):     # fail closed first (review-T3s3 #10)
            spec_task = asyncio.create_task(self._spec_start_after(abort_task, partial.text))
        try:
            try:
                text = await asyncio.to_thread(self.sstt.final, timeout_s=self.rt.hold_max_s)
            except TypeError:
                text = await asyncio.to_thread(self.sstt.final)
            except Exception:
                log.exception("voice: streaming STT final failed")
                text = ""
        finally:
            # This end has read its stream: reset it, and the next utterance's frames
            # that waited go into the fresh one (review-T3s3 #2).
            stream_done()
        text = (text or "").strip()
        if abort_task is not None:
            with contextlib.suppress(Exception):
                await abort_task
        started = False
        if spec_task is not None:
            try:
                started = bool(await spec_task)
            except Exception:
                log.exception("voice: speculative start failed")
        stt_ms = round((self.clock() - self._eot_at) * 1000)
        own = self._own_utt(utt)
        if not text:
            if own:
                self._hold = False
                self._cursor_clear()
            utt.while_playing = None
            await self._spec_drop()
            await self.send({"type": "transcript", "text": "", "upto": self._eot_upto})
            await self._undo_duck()
            await self.state(self._idle_state())
            return
        if utt.stop_cut:
            utt.stop_cut = False
            if self._stop_only(text):
                # The whole utterance was "wait" / "stop": the cut was all of it.
                utt.while_playing = None
                if own:
                    self._hold = False
                    self._cursor_clear()
                await self._spec_drop()
                await self.state(self._idle_state())
                return
            await self._abort_after_stop_cut(utt)
        # Words over our voice that no pause verdict covered (the hard silence ended
        # the utterance first, or the reply ended while it was spoken): the same
        # verdict now — an echo, a backchannel or a "stop" never becomes a turn.
        cut = (self._duck or self._talking()) and not merging and not utt.interrupted
        if ((cut or (utt.while_playing and not utt.interrupted)) and not merging
                and await self._consumed_over_reply(text, self._speech_s_of(ev), self.clock(), utt=utt)):
            if self.spec is not None:
                self.spec.new_turn()
            return
        utt.while_playing = None
        if not self._check_live():
            await self._spec_drop()
            return
        if (not merging and not cut and not utt.interrupted
                and await self._maybe_replay_command(text, prev)):
            return
        run = None
        if self.spec is not None:
            if merging:
                await self.spec.discard()
            else:
                # The thread head moved since the speculative prompt was built (a typed
                # message, the previous reply landing): pass the head the frames show,
                # so the Speculator drops the run as stale (hardening M1); unchanged →
                # no claim either way, DisPatch's own check at adoption is the law.
                moved = self._head_moves != head_at_start[0]
                run = await self.spec.commit(text, head_id=self._head_id if moved else None)
            self.spec.new_turn()
        if merging:
            prev_text, prev_id = self._last_submit
            prev_text = prev_text.rstrip().rstrip("?.!…").rstrip()
            text = f"{prev_text} {text}".strip()[:MAX_UTTERANCE_CHARS]
            gap = self._resume_gap
            if gap is not None and gap <= self.rt.hold_max_s + 0.05:
                # The driver resumed within a second of the eot: the hold either
                # covered it (clean, feed the gap) or was too short (a false end).
                self._observe(false_end=gap > self._last_hold_s + 0.05 and gap < FALSE_END_S,
                              resume_gap_s=gap)
            await self.barge_in(by_user_voice=False, abort=False)
            await self._abandon(prev, abort=True)
            if prev_id and self.host.retract:
                with contextlib.suppress(Exception):
                    await self.host.retract(self.thread_id, prev_id)
        elif cut:
            # Real words over the reply that no pause classified (the hard silence
            # ended the utterance first): cut it now, as 2.1 did.
            await self._interrupt(prev, by_user_voice=True, abort=self.settings.barge_abort_turn)
            self._after_barge_in = True
        elif prev is not None and not prev.closed and not utt.interrupted:
            # Speech while still waiting on the last answer, past the addendum
            # window (review M4): the new question replaces it.
            await self._abandon(prev, abort=True)
        if own or not getattr(self.detector, "active", False):
            self._hold = False
        self._duck_verdict = True
        if self._duck:
            await self._set_duck(False)
        try:
            msg_id = await self._submit(text, prefetched=run)
        except Exception:
            log.exception("voice: submit failed")
            if run is not None:
                with contextlib.suppress(Exception):
                    await run.discard()
            await self.send({"type": "transcript", "text": text, "merged": merging})
            await self._turn_failed(None, "agent_failed", "Could not send that message.")
            return
        turn = _Turn(msg_id=msg_id, text=text, at=self.clock(),
                     echoed=bool(msg_id) and msg_id in self._echoed_ids,
                     speculated=started or run is not None)
        if run is not None:
            stale = bool(getattr(run, "stale", False)) or (hasattr(run, "usable") and not run.usable()
                                                             and not getattr(run, "adopted", False))
            turn.adopted = not stale
            if stale and self.spec is not None:
                self.spec.note_stale()
        self._current = turn
        if msg_id:
            self._turns.append(turn)
        if prev is not None and prev.abandoned and not turn.echoed:
            self._await_user_id = msg_id
        await self.send({"type": "transcript", "text": text, "merged": merging,
                         "message_id": msg_id, "upto": self._eot_upto})
        await self.send({"type": "metrics", "eot_to_transcript_ms": stt_ms,
                         "utterance_ms": round(len(ev.audio) / 16), "reason": ev.reason,
                         "p_done": self._last_p, "hold_ms": round(hold_s * 1000),
                         **self._speculation_metrics(), **self._interrupt_metrics()})
        self._last_submit = (text, msg_id)
        self._last_hold_s = float(hold_s)
        self._merge_until = now + float(hold_s)
        self._first_audio_logged = False
        self._awaiting_reply = True
        await self.state("thinking")
        self._start_turn_acks(text, self._eot_at)

    def _speculation_allowed(self) -> bool:
        """``rt.speculate``: ``advisor`` (default) only for the advisor backend,
        ``all`` for advisor and API bots, ``off`` never; a gateway bot never
        (tools and memory writes make a shadow run unsafe); the device may switch
        it off (``start.rt.speculate``)."""
        if self.spec is None or not self.rt_speculate or self.rt.speculate == "off":
            return False
        if self.rt.speculate == "advisor":
            return self.bot_backend == "advisor"
        return self.bot_backend in ("advisor", "api")

    async def _spec_start(self, text: str):
        """The Speculator's host hook: the same instructions the real submit
        would build (the pending interruption note included, not consumed)."""
        fn = self.host.speculate
        if fn is None:
            return None
        await self._settle_cut_note()
        return await fn(self.thread_id, text, self._turn_opts(consume=False))

    def _turn_opts(self, *, consume: bool) -> dict:
        note = self._pending_note
        if consume:
            self._pending_note = None            # one turn only (SPEC §4.8)
        return {"reply_style": self.reply_style, "follow_up": self.follow_up, "interrupted": note}

    async def _submit(self, text: str, prefetched=None, *, typed: bool = False) -> str | None:
        await self._settle_cut_note()        # a cut's note belongs to THIS turn's instructions
        opts = self._turn_opts(consume=True)
        if prefetched is not None:
            opts["prefetched"] = prefetched
        if typed:
            opts["typed"] = True
        # The commit cursor moves BEFORE the host call (C.5 #8): if the socket or the
        # process dies between the submit and the client's next `pong`, the replayed
        # audio can never become a second user row; it records the transcript so a
        # reconnecting client that missed it gets it again (`replayed: true`).
        # A typed message has no uplink audio behind it: the cursor is not its business.
        if not typed:
            self._cursor_commit(text, None)
        self._submit_at = self.clock()
        if self._submit_opts:
            call = self.host.submit(self.thread_id, text, opts)
        else:
            call = self.host.submit(self.thread_id, text)
        # Shielded: the cursor already says this utterance is submitted, so a session that
        # ends right now (a hang-up, a reconnect replacing it) must not cut the host's
        # submit in half (a row with no run, or no row for a committed utterance). The
        # submit finishes on its own; the reply then reaches whoever is listening.
        task = asyncio.ensure_future(call)
        try:
            msg_id = await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                task.add_done_callback(_log_task_failure)    # finishing alone: log a failure
            raise
        if msg_id and not typed:
            self._cursor_commit(text, msg_id)
        return msg_id

    async def _typed_turn(self, raw) -> None:
        """A message typed on the voice chat screen (C.3 `typed`, 1.2.0): the turn a
        transcript makes — submitted through the host with the voice persona, its
        reply spoken, Undo and the acks as usual — without audio, STT, a hold or a
        speculative start. For a headset or a quiet room: the microphone is usually
        muted, and its state is left alone either way. A message typed over a
        playing reply cuts it, as spoken words would (SPEC §4.6); one typed while
        the last answer is still pending replaces that question (review M4)."""
        text = raw.strip() if isinstance(raw, str) else ""
        if not text or len(text) > TYPED_MAX_CHARS:
            await self._limited("bad_frame", "That message wasn't understood.")
            return
        if not self._check_live():
            return
        prev = self._current
        if self._talking():
            await self._interrupt(prev if (prev is not None and not prev.closed) else None,
                                  by_user_voice=False, abort=self.settings.barge_abort_turn,
                                  background=True)
        elif prev is not None and not prev.closed:
            await self._abandon(prev, abort=True)
        self._hold = False
        self._merge_until = 0.0
        if self._duck:
            await self._set_duck(False)
        if self.spec is not None:
            await self._spec_drop()
        self._eot_at = self.clock()
        try:
            msg_id = await self._submit(text, typed=True)
        except Exception:
            log.exception("voice: typed submit failed")
            await self.send({"type": "transcript", "text": text, "merged": False, "typed": True})
            await self._turn_failed(None, "agent_failed", "Could not send that message.")
            return
        turn = _Turn(msg_id=msg_id, text=text, at=self.clock(),
                     echoed=bool(msg_id) and msg_id in self._echoed_ids)
        self._current = turn
        if msg_id:
            self._turns.append(turn)
        if prev is not None and prev.abandoned and not turn.echoed:
            self._await_user_id = msg_id
        await self.send({"type": "transcript", "text": text, "merged": False, "typed": True,
                         "message_id": msg_id})
        self._last_submit = (text, msg_id)
        self._first_audio_logged = False
        self._awaiting_reply = True
        await self.state("thinking")
        self._start_turn_acks(text, self._eot_at)

    # ----------------------------------------------------- heard tracking #
    def _note_heard_frame(self, msg: dict, *, gen_default: int | None = None) -> None:
        """`heard {gen, seq, played_ms, source}` (C.3), standalone or as the
        sub-object of stop_playback / cancel / undo (C.3a). The exact answer to a
        `barge_in` (the NEW gen) resolves the wait — or, when the fallback was
        already used, the refine that re-notes the row (review-T3s2 #4)."""
        if not isinstance(msg, dict):
            return
        gen = _int_field(msg, "gen")
        seq = _int_field(msg, "seq")
        played = _int_field(msg, "played_ms")
        if gen is None and gen_default is not None:
            gen = gen_default
        if gen is None or seq is None or played is None:
            return
        source = msg.get("source")
        source = source if source in ("server", "phone") else ("phone" if self._mode_next == "text" else "server")
        h = _Heard(gen, seq, played, source, self.clock())
        self._last_heard = h
        if self.rig is not None and source == "server":
            self.rig.heard(gen, seq, played)     # credit{played_ms} for the rig
        if gen == self._heard_gen:
            for fut in (self._heard_fut, self._refine_fut):
                if fut is not None and not fut.done():
                    fut.set_result(h)
                    if self._cut_rid is None or self._seq_rid.get(seq) == self._cut_rid:
                        self._cut_heard = h       # the listener's exact place at the cut (replay_last)

    def _arm_heard_wait(self) -> None:
        """Before the `barge_in` frame goes out: the client answers it with one
        exact `heard` for the NEW gen (C.3). Also notes WHICH reply is being cut
        (its gen and stream id), so a periodic `heard` of another reply is never
        taken for this one's position (review-T3s2 #2)."""
        loop = asyncio.get_running_loop()
        old = self._refine_fut
        if old is not None and not old.done():
            old.cancel()                       # a newer cut supersedes the refine of an older one
        self._refine_fut = None
        self._heard_fut = loop.create_future()
        self._heard_gen = self.gen + 1
        self._cut_gen = self.gen
        self._cut_row = None
        seq = self._cut_seq()
        self._cut_at_seq = seq
        self._cut_rid = self._seq_rid.get(seq) if seq else None
        # `replay_last` after this cut continues from where the listener was: the heard
        # the cut came with (a tap's `stop_playback {heard}`, a fresh periodic one), until
        # the exact post-`barge_in` one replaces it.
        self._cut_at = self.clock()
        self._cut_heard = self._last_heard if self._heard_of_cut(self._last_heard) else None
        self._continued = None

    def _cut_seq(self) -> int | None:
        """The reply piece the listener is in: the one being sent, else the one after
        the last drained, else the last reply piece said."""
        seq = self._streaming_seq
        if seq is None:
            seq = (self._drained_seq + 1) if self._drained_seq is not None else self._last_reply_seq
            if seq and seq not in self._seq_rid:
                seq = self._last_reply_seq or None
        return seq or None

    def _heard_of_cut(self, h: _Heard | None) -> bool:
        """A periodic `heard` names the reply being cut: the cut gen, and one of that
        reply's pieces (never the previous reply's, never a stale gen)."""
        if h is None or self.clock() - h.at > HEARD_STALE_S:
            return False
        if self._cut_gen is not None and h.gen != self._cut_gen:
            return False
        return self._cut_rid is None or self._seq_rid.get(h.seq) == self._cut_rid

    async def _heard_position(self) -> HeardPosition | None:
        """Where the listener was when the reply was cut: the post-barge_in
        `heard` (waited ≤ heard_wait_s — only from a client that sends `heard`
        frames at all; a 2.1 client never does and must not pay the wait),
        else the last periodic one when it names the reply being cut, else the
        piece being sent (everything before it heard — SPEC §10 "heard never
        arrives"). A hang-up during the wait cancels the caller: the
        CancelledError propagates (review-T3s2 #1 — swallowing it kept a dead
        session's worker alive and submitted the cut words after the socket
        was gone)."""
        fut = self._heard_fut
        h: _Heard | None = None
        if fut is not None and fut.done() and not fut.cancelled():
            h = fut.result()
        elif fut is not None and self._last_heard is not None:
            try:
                h = await asyncio.wait_for(asyncio.shield(fut), self.heard_wait_s)
            except TimeoutError:
                h = None
        if h is None and self._heard_of_cut(self._last_heard):
            h = self._last_heard
        if h is not None:
            pos = self.heard.locate(h.seq, h.played_ms, source=h.source)
            if pos is not None:
                return pos
        return self._fallback_position()

    def _fallback_position(self) -> HeardPosition | None:
        """No usable `heard`: the piece that was being sent when the reply was cut
        (taken before the `barge_in` cleared it) counts as the cut, everything
        before it as heard."""
        seq = self._cut_at_seq or self._cut_seq()
        return self.heard.locate(seq, 0) if seq else None

    def _persisted_id(self, noted_id: str | None) -> str | None:
        if not noted_id:
            return None
        final = self._final_ids.get(noted_id, noted_id)
        return None if final.startswith("run:") else final

    def _reply_audible(self, prev: _Turn | None) -> bool:
        """Something of the reply reached the listener: audio (or, with the phone's
        voice, a `say`) is playing, or a reply piece's `say` went out for this turn."""
        owner = self._owner if isinstance(self._owner, _Turn) else None
        return bool(self.speaking or (prev is not None and prev.said) or (owner is not None and owner.said))

    async def _interrupt(self, prev: _Turn | None, *, by_user_voice: bool, abort: bool,
                         background: bool = False) -> dict | None:
        """A real interruption of a playing reply (words over it, a tap, a
        cancel): `barge_in` → with ``abort``, abort(keep_partial=True) → the
        exact `heard` → `note_heard` (SPEC §4.6/§4.8, hardening H3/L7; without
        ``abort`` the reply finishes into the chat and the note waits for its
        row). Returns the note for the next turn.

        When NOTHING of the reply reached the listener yet (still synthesising,
        or held), there is nothing to annotate: the queued pieces are dropped and,
        with ``abort``, the turn is abandoned without keeping a partial — the new
        words replace the pending answer (review M4), and no heard note is written.

        ``background``: the cut (`barge_in`, the reply muted, the turn abandoned)
        happens now; the host abort and the wait for the exact `heard` run in a
        task. The control frames (a tap, `cancel`) use it — they arrive on the
        receive loop, which is also what reads the `heard` frame the wait is for:
        waiting inline would always time out, and would stall pings and audio."""
        self._last_acked = False
        if not self._reply_audible(prev):
            await self.barge_in(by_user_voice=by_user_voice, abort=False)
            if abort and prev is not None and not prev.abandoned:
                done = self._abandon(prev, abort=True, keep_partial=False)
                if background:
                    self._spawn(done)
                else:
                    await done
            return None
        self._arm_heard_wait()
        self._cut_turn = prev
        if by_user_voice and abort:
            self._interrupt_at = self.clock()
        await self.barge_in(by_user_voice=by_user_voice, abort=False)
        if prev is not None and not prev.abandoned:
            stop = self._abandon(prev, abort=abort, keep_partial=True)
        elif abort and self.host.abort:
            stop = self._host_abort(keep_partial=True)
        else:
            stop = None

        async def finish() -> dict | None:
            try:
                row_id = await stop if stop is not None else None
                return await self._note_interruption(row_id)
            finally:
                self._cut_settled_at = self.clock()

        # One task, remembered: the next turn's submit waits for it (its instructions
        # carry this note — a tick's cut runs beside the utterance's own turn end).
        task = self._note_task = self._spawn(finish())
        if background:
            return None
        return await task

    async def _settle_cut_note(self) -> None:
        """Before a turn is speculated or submitted: the note of a cut still being
        made (the abort, the wait for the exact `heard`) lands first, so the turn's
        instructions carry it (and it never leaks into the turn after)."""
        task = self._note_task
        if task is not None and not task.done():
            with contextlib.suppress(Exception):
                await asyncio.shield(task)

    def _spawn(self, coro) -> asyncio.Task:
        """A tracked background task of this session (cancelled and awaited at close);
        an exception it ends with is logged, never left unretrieved."""
        task = asyncio.create_task(coro)
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)
        task.add_done_callback(_log_task_failure)
        return task

    async def _note_interruption(self, row_id: str | None) -> dict | None:
        pos = await self._heard_position()
        fut = self._heard_fut
        self._heard_fut = None
        pending = fut is not None and not fut.done()
        if pending:
            self._refine_fut = fut            # the exact frame may still land: keep it reachable
        if pos is None or pos.complete or not (pos.heard or pos.unheard):
            if pending:
                self._exact_later(fut, row_id, None, None)
            return None
        target = row_id or self._persisted_id(pos.message_id) or self._cut_row
        noted = False
        if target:
            noted = await self._call_note_heard(target, pos)
        if not noted:
            self._deferred_note = (pos.message_id or "", pos, self.clock())
        note = {"heard": pos.heard, "unheard": pos.unheard}
        self._pending_note = note
        if pending:
            # The exact post-barge_in `heard` may still land (a client whose
            # first `heard` this is): re-note the row with it, within the window.
            self._exact_later(fut, target, pos.message_id, note)
        return note

    def _exact_later(self, fut: asyncio.Future, target: str | None, noted_id: str | None,
                     note: dict | None) -> None:
        async def refine() -> None:
            try:
                h = await asyncio.wait_for(asyncio.shield(fut), self.heard_wait_s)
            except TimeoutError:
                return
            finally:
                if self._refine_fut is fut:
                    self._refine_fut = None
            pos = self.heard.locate(h.seq, h.played_ms, source=h.source)
            if pos is None or pos.complete or not (pos.heard or pos.unheard):
                return
            if note is None or self._pending_note is note:
                # Still the note the next turn will read (never a later cut's).
                self._pending_note = {"heard": pos.heard, "unheard": pos.unheard}
            tid = target or self._persisted_id(pos.message_id)
            key = noted_id if noted_id is not None else (pos.message_id or "")
            if tid and await self._call_note_heard(tid, pos):
                if self._deferred_note is not None and self._deferred_note[0] == key:
                    self._deferred_note = None
            else:
                self._deferred_note = (key, pos, self.clock())

        task = asyncio.create_task(refine())
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _call_note_heard(self, message_id: str, pos: HeardPosition) -> bool:
        fn = self.host.note_heard
        if fn is None:
            return False
        try:
            return bool(await fn(self.thread_id, message_id, heard_upto=pos.upto,
                                 heard_text=pos.heard, unheard_text=pos.unheard))
        except Exception:
            log.exception("voice: note_heard failed")
            return False

    async def _resolve_deferred_note(self, f: dict) -> None:
        """A note waiting for its row id: the `stream_done` that maps the
        provisional id, or the row itself (A7: `note_heard` was False)."""
        d = self._deferred_note
        if d is None:
            return
        noted_id, pos, at = d
        if self.clock() - at > DEFERRED_NOTE_S:
            self._deferred_note = None
            return
        msg = f.get("message") or {}
        if msg.get("role") != "assistant":
            return
        mid = msg.get("id")
        if not mid or mid.startswith("run:"):
            return
        match = (f.get("provisional_id") == noted_id or mid == noted_id
                 or f.get("message_id") == noted_id
                 or (noted_id.startswith("run:") and bool((msg.get("metadata") or {}).get("voice_heard"))))
        if not match:
            return
        self._deferred_note = None
        await self._call_note_heard(mid, pos)

    # ----------------------------------------------- replay_last (SPEC §4.6) #
    async def _drop_submitted_utterance(self, utt: _Utt | None = None) -> None:
        """The utterance that just ended was submitted by an earlier socket (see
        :meth:`_already_submitted`): no `eot`, no transcript, no turn. (The streaming
        path has released its stream already.)"""
        log.info("voice: a replayed utterance the cursor already submitted (ends at %s); dropped",
                 self._eot_upto)
        utt = utt if utt is not None else self._utt
        utt.while_playing = None
        if self._own_utt(utt):
            self._hold = False
            self._cursor_clear()
        await self._undo_duck()
        await self.state(self._idle_state())

    @staticmethod
    def _replay_command(text: str) -> str | None:
        """The whole utterance, fillers aside, asks to hear the last reply again:
        ``"whole"`` ("say that again"), ``"last"`` ("repeat the last part": its last
        sentence), else None."""
        words = [w for w in _norm_words((text or "").replace("’", "'")) if w not in REPLAY_FILLERS]
        phrase = " ".join(words)
        if not phrase:
            return None
        if phrase in REPLAY_LAST_COMMANDS:
            return "last"
        return "whole" if phrase in REPLAY_COMMANDS else None

    @staticmethod
    def _is_replay_command(text: str) -> bool:
        return VoiceSession._replay_command(text) is not None

    async def _maybe_replay_command(self, text: str, prev: _Turn | None) -> bool:
        """"Say that again" / "repeat that" while nothing is being said or awaited (C.3
        "while listening"): the last reply again, from its kept audio — no LLM turn, no
        row (SPEC §4.6 item 4); "repeat the last part": its last sentence. True =
        consumed. Over a reply, or with a turn pending, the words are a turn as before
        (the agent hears them with the heard note)."""
        mode = self._replay_command(text)
        if (mode is None or self._talking() or self._awaiting_reply
                or (prev is not None and not prev.closed)
                or self._replay_target(mode) is None):
            return False
        if self.spec is not None:
            await self.spec.discard()
            self.spec.new_turn()
        self._hold = False
        self._cursor_clear()                  # a reconnect never replays the command into a turn
        self._queue_replay(mode)
        await self.state(self._idle_state())
        log.info("voice: %r → the last reply again (%s, no turn)", text[:40], mode)
        return True

    def _queue_replay(self, mode: str) -> None:
        """Run the replay on the frames worker, in order with the reply frames. ``mode``:
        ``continue`` (the `replay_last` frame), ``whole`` or ``last`` (the voice command)."""
        self._frame_seq += 1
        with contextlib.suppress(asyncio.QueueFull):
            self._frames_q.put_nowait((self._frame_seq, {"type": "_replay_last", "mode": mode}))

    def _replay_target(self, mode: str) -> tuple[str, bool] | None:
        """What a replay would say: ``(reply stream id, continued)``.

        ``continue``: a cut (a tap, a stop word, an interruption) younger than
        REPLAY_KEEP_S with no other reply said since CONTINUES its reply from the piece
        the listener was in ("Tap to continue"). Otherwise (and for ``whole``/``last``)
        the last reply said: "say that again". None when nothing is kept."""
        now = self.clock()
        rid = self._cut_rid
        if (mode == "continue" and rid is not None and self._cut_at is not None
                and now - self._cut_at <= REPLAY_KEEP_S
                and (self._last_said_rid in (None, rid)
                     or (self._last_said_at is not None and self._last_said_at <= self._cut_at))):
            rt = self._texts.get(rid)
            if rt is not None and rt.text and self._replies.get(rid) is not None:
                return rid, True
        rid = self._last_said_rid
        if rid is None or self._last_said_at is None or now - self._last_said_at > REPLAY_KEEP_S:
            return None
        rt = self._texts.get(rid)
        if rt is None or not rt.text:
            return None
        return rid, False

    def _skip_point(self, mode: str, continued: bool, r: _Reply | None) -> tuple[int | None, int]:
        """Where a replay of ``r`` starts: ``(source offset, count)`` — pieces whose span
        ends at or before the offset are skipped (None = none), else the first ``count``.

        A continue starts at the START of the piece the listener was in (the cut's best
        `heard`, else the piece being sent), or after it when they had heard it to its end
        (≥ 95 %). By source offset, never by a count of pieces (review-T3s4 #1). A cut with
        no known piece falls back to the pieces heard before a reconnect. ``last``: the
        reply's last piece only."""
        if r is None:
            return None, 0
        if continued:
            h = self._cut_heard
            seq = h.seq if h is not None else self._cut_at_seq
            p = next((x for x in r.log if seq is not None and x.seq == seq), None)
            if p is None:
                return None, r.skipped
            if h is not None:
                pos = self.heard.locate(h.seq, h.played_ms, source=h.source)
                if pos is not None and pos.fraction >= REPLAY_HEARD_FRACTION:
                    return p.end, 0
            return p.start, 0
        if mode == "last" and r.log:
            return r.log[-1].start, 0
        return None, 0

    def _forget_replayed(self, r: _Reply | None, skip_upto: int | None, count: int) -> None:
        """The pieces a replay says again leave the heard map (it locates a later cut over
        ONE copy of the reply)."""
        if r is None:
            return
        for i, p in enumerate(r.log):
            if p.seq is None:
                continue
            again = p.end > skip_upto if skip_upto is not None else i >= count
            if again:
                self.heard.forget(p.seq)
                self._seq_rid.pop(p.seq, None)

    async def _do_replay(self, mode: str = "continue") -> None:
        """`replay_last` / "say that again" (SPEC §4.6 item 4, C.3), on the frames worker:
        the reply is rebuilt from its text under the current gen with new seqs (C.5 #8:
        never one seq twice) — split exactly as it was spoken (`feed_canonical`). A
        continue skips what was heard before the cut, takes the cut's note back (the
        listener now hears the rest) and follows a run that is still going on into the
        chat (a tap-stop never aborts it), so its later sentences are spoken too; a replay
        says the whole reply ("repeat the last part": its last sentence). Pieces the engine
        said before go out from their kept audio (no engine call), the others are
        synthesised. Never while a reply is being said, a turn is ending or one is pending
        (C.3 "while listening")."""
        if (self._closed or self._ending or self.speaking or not self._say_q.empty()
                or self._awaiting_reply):
            log.info("voice: replay_last ignored (a reply or a turn is under way)")
            return
        target = self._replay_target(mode)
        if target is None:
            log.info("voice: replay_last ignored (nothing kept)")
            return
        rid, continued = target
        rt = self._texts[rid]
        old = self._replies.get(rid)
        mid = rt.mid or (old.mid if old is not None else None)
        skip_upto, count = self._skip_point(mode, continued, old)
        self._forget_replayed(old, skip_upto, count)
        self._pace_t0 = None                      # nothing of the cut reply is playing any more
        r = self._new_reply()
        r.mid, r.continued, r.skip_upto = mid, continued, skip_upto
        self._replies[rid] = r
        if mid and mid != rid:
            self._final_ids[rid] = mid
        if count:
            self._after_partial[rid] = count
        else:
            self._after_partial.pop(rid, None)
        if continued:
            self._pending_note = None             # the cut's note no longer holds
            self._continued = rid
            self._cut_at = None                   # one continue per cut; later: "say that again"
            t = self._cut_turn
            if t is not None and t.abandoned and t.abort_seq is None and not rt.done:
                t.abandoned = False               # the run goes on into the chat: follow it again
        else:
            self._continued = None
        for span, m, rrid in self._reply_text_update(rid, rt.text, None, rt.done, mid):
            await self._enqueue_reply(span, m, rrid)
        await self._apply_rewrites()
        if rt.done:
            await self._maybe_say_end(rid)
        log.info("voice: %s %s (%d pieces queued, %d skipped)",
                 "continuing" if continued else ("replaying the last part of" if mode == "last" else "replaying"),
                 rid, r.pieces, r.skipped)

    async def _renote_full(self, rid: str) -> None:
        """A continued reply played to its end: its row is noted as heard in full, so the
        next turn's history is not cut at the old interruption (`voice_heard.upto` ≥ the
        content's length means "all of it", C-§B.2)."""
        rt = self._texts.get(rid)
        row = (rt.mid if rt is not None else None) or self._persisted_id(rid)
        if rt is None or not rt.done or not row:
            return
        text = rt.text
        heard = text if len(text) <= 160 else "…" + text[-159:].lstrip()
        await self._call_note_heard(row, HeardPosition(row, 0, 0, 1.0, len(text), heard, "", True))

    # ---------------------------------------------------------- control #
    def _check_live(self) -> bool:
        try:
            return bool(self.host.session_live())
        except Exception:
            return False

    async def _resume_after(self, after: str | None, after_ids: list | None = None) -> None:
        """Speak what this device missed. ``after`` is the last reply it
        heard; ``after_ids`` (contract F1) are older ones, tried in order when
        ``after`` is unknown (a streamed reply's provisional id is never
        stored). ``replies_after`` returns None for an unknown id."""
        if not self.host.replies_after:
            return
        candidates = [a for a in [after, *(after_ids or [])] if isinstance(a, str) and a][:9]
        only_named = False
        if not candidates and self._after_partial:
            # `after_partial` names a reply that was cut by the drop, but the client sent
            # no `after` to anchor it (a stored reply is found only AFTER something): the
            # last utterance this stream submitted is the anchor — the commit cursor keeps
            # its user row — and only the reply `after_partial` names is spoken (T3s4).
            anchor = self._cursor_turn_id()
            if anchor:
                candidates, only_named = [anchor], True
        rows = None
        for cand in candidates:
            try:
                rows = await self.host.replies_after(self.thread_id, cand)
            except Exception:
                log.exception("voice: replies_after failed")
                return
            if rows is not None:
                break
        if not rows:
            return
        # `start.after_partial` may name the cut reply by the provisional id the OLD
        # socket saw (the drop came before its `say_end`); this session only knows the
        # stored row. The reply that answers `after` (the user row, Amendment T7b-3) is
        # the first one stored after it: the count applies to that row.
        orphans = [k for k in self._after_partial
                   if k.startswith("run:") and k not in self._replies and k not in self._final_ids]
        first = rows[0][0] if rows and isinstance(rows[0][0], str) else None
        if orphans and first and first not in self._after_partial:
            self._after_partial[first] = self._after_partial.pop(orphans[0])
        for mid, content in rows[-3:]:      # never read out a backlog of more than 3
            if mid in self._spoken_ids or (only_named and mid not in self._after_partial):
                continue
            if self._streaming_now(content):
                continue                    # the same reply is still streaming in: it speaks itself
            self._spoken_ids.append(mid)
            self._resumed_ids.append(mid)
            rt = self._text_row(mid)                 # replayable like a streamed reply
            rt.text, rt.done, rt.late, rt.at = (content or "")[:REPLY_TEXT_MAX], True, False, self.clock()
            rt.mid = mid
            for span, m, rid in self._reply_text_update(mid, content, None, True, mid):
                await self._enqueue_reply(span, m, rid)
            await self._apply_rewrites()
            await self._maybe_say_end(mid)

    def _cursor_turn_id(self) -> str | None:
        """The user row of the last utterance this stream submitted (the commit cursor's
        record survives the socket and a restart), else this session's own last turn."""
        if self.cursor is not None:
            try:
                e = self.cursor.entry(self.cursor_key)
            except Exception:      # an unreadable cursor anchors nothing
                log.exception("voice: cursor read failed")
                e = None
            mid = (e.meta or {}).get("message_id") if e is not None else None
            if isinstance(mid, str) and mid:
                return mid
        t = self._current
        return t.msg_id if t is not None and t.msg_id else None

    def _streaming_now(self, content: str) -> bool:
        """A reply this session is receiving as a stream (a snapshot or live chunks)
        whose text so far is this stored content's beginning: the same reply."""
        for r in self._replies.values():
            if r.done or r.muted or not r.head:
                continue
            if content.startswith(r.head[:HEAD_CHARS]):
                return True
        return False

    def _apply_start(self, msg: dict) -> None:
        if "patience" in msg:
            self._apply_patience(_patience_value(msg.get("patience"), self.settings.patience))
        if "barge_in" in msg:
            v = msg.get("barge_in")
            self.barge_in_enabled = v if isinstance(v, bool) else True
        if "reply_style" in msg:
            v = msg.get("reply_style")
            self.reply_style = v if v in REPLY_STYLES else "normal"
        if "follow_up" in msg:
            v = msg.get("follow_up")
            self.follow_up = v if isinstance(v, bool) else None     # null = the bot's default (T7a-7)
        if "acks" in msg:
            v = msg.get("acks")
            self.acks.set_mode(v if v in ACK_MODES else "sometimes")
        if "slow_link" in msg:
            v = msg.get("slow_link")
            self.tr.link.set_slow_link(v)
            if v in SLOW_LINK_MODES:
                self.slow_link = v
                if v == "always_phone":
                    self._mode_next = "text"     # the server never synthesises (SPEC §5.4)
                elif self.slow_link == "always_phone":
                    self._mode_next = "audio"
        rt = msg.get("rt")
        if isinstance(rt, dict):
            self._apply_start_rt(rt)

    async def _apply_resume_fields(self, msg: dict) -> None:
        """``start.cursor`` and ``start.after_partial`` (C.3; re-read only when their value
        changes, B4). ``cursor {stream, from}`` → ``cursor {from: resume_from}`` (C.5 #10,
        Amendment T6-1) plus the committed transcript again (``replayed: true``) when the
        client never saw it. ``after_partial {message_id, sentences}`` → that reply's first
        ``sentences`` pieces were heard on the old socket: never spoken again (C.5 #11)."""
        cur = msg.get("cursor")
        if isinstance(cur, dict) and cur != self._last_cursor:
            self._last_cursor = dict(cur)
            frm = cur.get("from")
            if (self.cursor is not None and cur.get("stream") == self.cursor_key[1]
                    and isinstance(frm, int) and not isinstance(frm, bool)):
                frm &= wire3.U32
                try:
                    answer = self.cursor.resume_from(self.cursor_key, frm)
                    entry = self.cursor.entry(self.cursor_key)
                except Exception:
                    log.exception("voice: cursor read failed")
                    answer, entry = frm, None
                await self.send({"type": "cursor", "from": answer})
                if (entry is not None and entry.upto is not None and entry.meta
                        and wire3.u32_diff(entry.upto, frm) > 0):
                    await self.send({"type": "transcript", "text": str(entry.meta.get("text") or ""),
                                     "merged": False, "message_id": entry.meta.get("message_id"),
                                     "upto": entry.upto, "replayed": True})
        ap = msg.get("after_partial")
        if isinstance(ap, dict) and ap != self._last_after_partial:
            self._last_after_partial = dict(ap)
            mid = ap.get("message_id")
            n = _int_field(ap, "sentences", 0)
            if isinstance(mid, str) and 0 < len(mid) <= ID_MAX_CHARS and n:
                # Bounded (review-T3s4 #9): a client re-sending `start` with ever-new values
                # must not grow per-session state; the newest few are all a resume needs.
                self._after_partial.pop(mid, None)
                self._after_partial[mid] = min(n, 500)
                while len(self._after_partial) > AFTER_PARTIAL_KEEP:
                    self._after_partial.pop(next(iter(self._after_partial)))

    def _apply_start_rt(self, rt: dict) -> None:
        """``start.rt {partials, speculate, backchannels, optimistic_duck, learned,
        learned_reset}`` (C.3, F.3, Amendment T7a-6): bools keep their value when
        absent; the device blob's learned values apply only when the server has
        none of its own (SPEC §4.3); ``learned_reset`` forgets them (one-shot)."""
        for key, attr in (("partials", "rt_partials"), ("speculate", "rt_speculate"),
                          ("backchannels", "rt_backchannels"), ("optimistic_duck", "rt_optimistic_duck")):
            if isinstance(rt.get(key), bool):
                setattr(self, attr, rt[key])
        if self.sstt is None:
            return
        if rt.get("learned_reset") is True:
            # The Voice lab's Reset (one-shot, Amendment T7a-6): back to the configured
            # defaults, here and in the per-device store; the device's blob is not
            # read again in this session.
            self.policy.reset()
            self._learned_from_server = True
            self.detector.ep.cfg.turn_probe_silence_s = self.policy.probe_silence_s
            if self._on_learned is not None:
                with contextlib.suppress(Exception):
                    self._on_learned(self.policy.learned)
            return
        blob = rt.get("learned")
        if isinstance(blob, dict) and not self._learned_from_server and not self._blob_applied:
            # The device blob (F.3) seeds a server that has nothing of its own for this
            # device (a restart since): once per session, and never over what this
            # session has learned since (a re-sent `start` must not reset the policy).
            learned = Learned.from_dict({"probe_ms": blob.get("probe_ms", blob.get("probeMs")),
                                         "hold_ms": blob.get("hold_ms", blob.get("holdMs"))})
            if learned is not None:
                self._blob_applied = True
                self.policy = TurnPolicy(self.policy.cfg, learned)
                self.detector.ep.cfg.turn_probe_silence_s = self.policy.probe_silence_s
                if self._on_learned is not None:
                    with contextlib.suppress(Exception):
                        self._on_learned(self.policy.learned)

    def _apply_link(self, msg: dict) -> None:
        """`link {mode, why}`: the next sentence takes the new path (C.5 #6)."""
        mode = msg.get("mode")
        if mode not in ("audio", "text"):
            return
        if self.slow_link == "always_phone" and mode == "audio":
            return                       # Settings said never to synthesise
        if mode != self._mode_next:
            why = msg.get("why") if isinstance(msg.get("why"), str) else ""
            log.info("voice: link mode %s → %s (%s)", self._mode_next, mode, why[:16] or "client")
        self._mode_next = mode

    @property
    def effective_follow_up(self) -> bool:
        return self.bot_follow_up if self.follow_up is None else self.follow_up

    async def _undo(self) -> None:
        t = self._current
        # Once the reply has started (a sentence queued) or the turn is
        # answered, Undo would delete the question and orphan the stored
        # answer below it (review M1): too late, like the 8 s window.
        if (t is None or not t.msg_id or t.abandoned or t.spoke or t.closed
                or self.clock() - t.at > UNDO_WINDOW_S):
            await self.error("undo_expired", "Too late to take that back.")
            return
        await self.barge_in(by_user_voice=False, abort=False)
        stop = self._abandon(t, abort=True)

        async def take_back() -> None:
            # Abort FIRST, then retract (Amendment T2-4: the retract also removes a
            # direct reply that answered the row, so it must see the turn stopped).
            # In the background: an abort may wait for a reply row that was already
            # being delivered (≤ 3 s), and the receive loop must keep reading.
            await stop
            if self.host.retract:
                with contextlib.suppress(Exception):
                    await self.host.retract(self.thread_id, t.msg_id)

        self._spawn(take_back())
        self._last_submit = None
        self._merge_until = 0.0
        self._hold = False
        self._pending_note = None
        await self.send({"type": "undone", "message_id": t.msg_id})
        await self.state("listening" if self.listening else "paused")

    async def _cancel(self) -> None:
        """Drop the pending/running turn. A reply that was playing counts as
        interrupted (its spoken prefix is kept and annotated); a turn still
        thinking is cancelled outright, nothing persisted. The host calls run in
        the background (the receive loop must keep reading)."""
        playing = self._talking()
        t = self._current
        if playing:
            await self._interrupt(t if (t is not None and not t.closed) else None,
                                  by_user_voice=False, abort=True, background=True)
        else:
            await self.barge_in(by_user_voice=False, abort=False)
            if t is not None and not t.closed:
                self._spawn(self._abandon(t, abort=True))
        self._merge_until = 0.0
        self._hold = False
        await self.state("listening" if self.listening else "paused")

    async def handle_control(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "start":
            self._apply_start(msg)
            if not self.listening:
                self.listening = True
                self.detector.reset()
                self._sstt_drop_open()
            self._prepare_voices()
            await self.state(self._idle_state())
            await self._apply_resume_fields(msg)
            if not self._started.is_set():
                # The first `start`: `after_partial` is known now. The reply frames that
                # waited for it (a still-streaming reply's subscribe snapshot) are spoken
                # with its skip, and BEFORE the stored replies below are looked up, so a
                # reply that is both streaming and stored is recognised as one (C.5 #11).
                self._started.set()
                await self._frames_sync()
            after = msg.get("after")
            after = after if isinstance(after, str) else None
            ids = msg.get("after_ids")
            ids = [i for i in ids if isinstance(i, str)][:8] if isinstance(ids, list) else []
            key = (after, tuple(ids))
            if key != self._last_after:        # a settings re-send is not a resume
                self._last_after = key
                await self._resume_after(after, ids)
        elif t == "pause":
            self.listening = False
            self._hold = False
            self.detector.reset()
            self._sstt_drop_open()
            await self._undo_duck()
            await self.state("paused")
        elif t == "stop_playback":
            # The tap while speaking: an interruption without a new turn. The
            # agent turn is not aborted (the answer finishes into the chat for
            # reading); the heard note lands on its row, deferred if need be.
            self._note_heard_frame(msg.get("heard"), gen_default=self.gen)
            if self._talking():
                cur = self._current
                await self._interrupt(cur if (cur is not None and not cur.closed) else None,
                                      by_user_voice=False, abort=False, background=True)
            else:
                await self.barge_in(by_user_voice=False)
            await self.state("listening" if self.listening else "paused")
        elif t == "playback_done":
            # Optional `seq`: a drain report for audio older than the last
            # reply sentence we sent (an ack, an earcon-length filler) never
            # ends the speaking state.
            seq = msg.get("seq")
            stale = isinstance(seq, int) and not isinstance(seq, bool) and seq < self._last_reply_seq
            if msg.get("gen") == self.gen and not stale:
                if isinstance(seq, int) and not isinstance(seq, bool):
                    self._drained_seq = seq
                    rid = self._continued
                    r = self._replies.get(rid) if rid else None
                    if r is not None and r.done and r.last_seq is not None and seq >= r.last_seq:
                        # A continued reply was heard to its end (replay_last).
                        self._continued = None
                        self._spawn(self._renote_full(rid))
                self._client_drained = True
                await self._maybe_finish_speaking()
        elif t == "cancel":
            self._note_heard_frame(msg.get("heard"), gen_default=self.gen)
            await self._cancel()
        elif t == "undo":
            self._note_heard_frame(msg.get("heard"), gen_default=self.gen)
            await self._undo()
        elif t == "heard":
            self._note_heard_frame(msg)
        elif t == "link":
            self._apply_link(msg)
            self.tr.link.request(msg)              # rung / decoder fallback at the next boundary
        elif t == "probe":
            n = _int_field(msg, "bytes", 0)
            if n:
                # Never inline: a stalled downlink holds the transport's send path.
                task = asyncio.create_task(self._answer_probe(n))
                self._turn_tasks.add(task)
                task.add_done_callback(self._turn_tasks.discard)
        elif t == "replay_last":
            # "Tap to continue" after a stop, or "say that again": no LLM turn (C.3,
            # SPEC §4.6 item 4). Ignored while a reply or a turn is under way.
            # (`_tts_busy` alone does not count: right after a tap-stop the engine is
            # still winding down the cut piece, and the replay simply queues behind it.)
            if self.listening and not (self._ending or self._awaiting_reply or self.speaking
                                       or not self._say_q.empty()):
                self._queue_replay("continue")
        elif t == "quiet_check":
            await self._quiet_check()
        elif t == "typed":
            await self._typed_turn(msg.get("text"))
        elif t == "ping":
            self._pinged = True
            # `pong {t, rx_off, srv_t}` (C.4): `t` echoed only when it is a finite number.
            await self.send(self.tr.pong(msg.get("t")))

    def _maybe_touch(self) -> None:
        if self.host.touch is None:
            return
        now = self.clock()
        if now - self._last_touch < TOUCH_EVERY_S:
            return
        self._last_touch = now
        try:
            self.host.touch()
        except Exception:
            log.exception("voice: session touch failed")

    def _prepare_voices(self) -> None:
        """A voice's one-off work (sliced prompt + LM prefix, Amendment T4-2) in
        the background at the first ``start`` — never on the ready path, never
        awaited (a stock voice not built yet takes seconds there)."""
        if self._prepared or self._prepare_fn is None:
            return
        self._prepared = True
        fn = self._prepare_fn

        async def go() -> None:
            try:
                await asyncio.to_thread(fn)
            except Exception as e:  # noqa: BLE001 - a preparation that fails costs the first sentence ~0.5 s
                log.info("voice: voice preparation skipped (%s)", e)

        task = asyncio.create_task(go())
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _quiet_check(self) -> None:
        """The client's "Hang up when quiet" timer fired (C.3): the bot's voice
        asks "Still there?" as a notice line (``say {kind: notice}`` + audio, a
        ``say`` alone in phone/text mode); ≤ 1 per 60 s. If nobody answers
        within 20 s the CLIENT hangs up (experience M9)."""
        now = self.clock()
        if now - self._last_quiet_check < QUIET_CHECK_EVERY_S:
            return
        self._last_quiet_check = now
        if self._talking() or self._awaiting_reply:
            return                           # not quiet on our side
        await self._say_notice(NOTICE_LINES["quiet_check"])

    async def _enqueue_audio(self, buf: bytes) -> None:
        """One uplink message into the audio queue (contracts §C.2, A11): over 64 KB it
        is dropped BEFORE queueing (``bad_frame``, rate-limited); the queue is bounded in
        bytes. Live audio that finds it full is dropped (the CPU is behind: dropping beats
        unbounded lag). A replay (the client's ring after a reconnect, flags bit1) is a
        burst by design and a dropped piece of it is a lost utterance, so while replayed
        audio is queued the bound is REPLAY_QUEUE_BYTES (512 KB, above the client's 320 KB
        ring) for it AND for the live audio behind it (dropping that would leave a hole
        mid-utterance). Nothing here ever waits: the receive loop must keep reading
        pings, `heard` and `stop_playback` (review-T3s4 #8: a full queue blocked it for up
        to 5 s per replayed message)."""
        if Transport.too_big(buf):
            self.tr.stats["bad_frames"] += 1
            await self._limited("bad_frame", "Audio message too large.")
            return
        n = len(buf)
        replay = _is_replay(buf)
        bound = REPLAY_QUEUE_BYTES if (replay or self._replay_q_bytes) else AUDIO_QUEUE_BYTES
        if self._audio_q_bytes and self._audio_q_bytes + n > bound:
            self._dropped_audio += 1
            if self._dropped_audio in (1, 100) or self._dropped_audio % 1000 == 0:
                log.info("voice: uplink queue full; %d audio messages dropped so far%s",
                         self._dropped_audio, " (a replay)" if replay else "")
            return
        self._audio_q_bytes += n
        if replay:
            self._replay_q_bytes += n
        self._audio_q.put_nowait(buf)

    def _take_token(self, *, ping: bool = False) -> bool:
        now = self.clock()
        if ping:
            self._ping_tokens = min(float(PING_BURST),
                                    self._ping_tokens + max(0.0, now - self._ping_tokens_at) * PING_PER_S)
            self._ping_tokens_at = now
            if self._ping_tokens >= 1.0:
                self._ping_tokens -= 1.0
                return True
            return False
        self._tokens = min(float(RATE_BURST),
                           self._tokens + max(0.0, now - self._tokens_at) * RATE_PER_S)
        self._tokens_at = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    async def _answer_probe(self, n: int) -> None:
        """Send the probe burst; when the limiter (and not v2, an empty size or a
        closed socket) refused it, say so with ``error {code: "rate_limited"}``, so
        the client's monitor fails the probe at once and tries again in 20 s instead
        of waiting out its 30 s timeout. A refused probe is not a protocol error: it
        does not count toward any flood limit."""
        before = self.tr.probes.refused
        sent = await self.tr.send_probe(n)
        if not sent and self.tr.probes.refused > before and not self._closed:
            await self._limited("rate_limited", "Probe refused: one per 18 s.")

    async def _limited(self, code: str, message: str) -> None:
        """At most one error frame per code every 2 s (a flood stays a flood
        on our side only)."""
        now = self.clock()
        if now - self._last_limited.get(code, float("-inf")) >= 2.0:
            self._last_limited[code] = now
            await self.error(code, message)

    async def _guarded(self, name: str, worker: Callable[[], Awaitable[None]]) -> None:
        """Run a worker; if it dies, tell the client and end the session.

        Without this a crashed worker left a zombie: the socket stayed open,
        the client kept showing "Listening", and nothing was ever heard again."""
        try:
            await worker()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice: %s worker failed; ending the session", name)
            with contextlib.suppress(Exception):
                await self.error("unavailable", "Voice chat hit an internal error and stopped.")
            self.end(1011, "internal error")

    async def _watchdog(self) -> None:
        """Timers that must fire even when the socket is silent: the session
        re-check (M7), the heartbeat (H2), a turn that never answers (H1) and
        a duck that never became a transcript (H3)."""
        while True:
            await asyncio.sleep(self.watch_tick_s)
            now = self.clock()
            if now - self._last_live_check >= LIVE_CHECK_S:
                self._last_live_check = now
                if not self._check_live():
                    await self.send({"type": "locked"})
                    self.end(1008, "locked")
                    return
            if self._pinged and now - self._last_rx > DEAD_AFTER_S:
                log.info("voice: no heartbeat for %.0f s; closing", now - self._last_rx)
                self.end(1001, "heartbeat lost")
                return
            # The server's own bound on an idle chat (FR-hardening L3): the
            # client's "Hang up when quiet" is a device setting ("Never" is a
            # choice), so with nothing said either way for rt.quiet_max_s the
            # session stops sliding the unlocked session and ends the chat.
            # A turn still waiting on its answer is not idle.
            quiet_max = float(getattr(self.rt, "quiet_max_s", 0) or 0)
            if (quiet_max > 0 and now - self._last_exchange > quiet_max
                    and not self._awaiting_reply and not self._talking()):
                log.info("voice: nothing said for %.0f s; voice chat ended", now - self._last_exchange)
                self.end(QUIET_CLOSE, QUIET_REASON)
                return
            # Idle-touch policy (SPEC §5.6, hardening M8): an attached chat with
            # the mic enabled and a client frame (pings count) < 10 s old is
            # activity — a gated driver who says nothing for ten minutes must
            # not be locked at the wheel. A half-open socket stops touching
            # within 10 s and dies at DEAD_AFTER_S.
            if self.listening and now - self._last_rx < IDLE_TOUCH_FRESH_S:
                self._maybe_touch()
            if (self._hold and not self._ending and not getattr(self.detector, "active", True)
                    and now - getattr(self, "_hold_since", now) > HOLD_STUCK_S):
                # Nothing is being said and no turn is ending: the speech this hold waited
                # for is gone (a reset dropped it). The held reply must not wait for ever.
                log.warning("voice: a reply hold outlived its speech by %.0f s; released",
                            now - self._hold_since)
                self._hold = False
            t = self._current
            if (t is not None and not t.closed and not t.spoke
                    and now - t.at > TURN_TIMEOUT_S):
                # Not abandoned: if the answer does land while the driver is
                # still here, it is read out like any other reply.
                await self._turn_failed(t, "timeout")
            if self._duck and not (self._hold and getattr(self.detector, "active", False)):
                # (A merge hold over open speech is the addendum still being spoken: its own
                # end decides — the timeout resumed over the listener and broke the merge,
                # review-T3s3 #5.)
                ducked_s = now - self._duck_at
                if not self._duck_verdict and ducked_s > self.rt.false_interrupt_timeout_s:
                    # No word verdict this long after the duck (SPEC §4.6): a false
                    # interruption — the reply resumes (rewound, the pause was long).
                    await self._resume_reply(why="timeout")
                elif ducked_s > DUCK_MAX_S:
                    await self._undo_duck()

    def _ready_frame(self) -> dict:
        power = "off"
        if self._power_state is not None:
            try:
                power = str(self._power_state())
            except Exception:
                power = "off"
        cfg = self.detector.ep.cfg
        hard = float(getattr(cfg, "turn_hard_silence_s", self.rt.hard_silence_s))
        # v3 (C.4): proto, codecs, rungs, rx_off from the transport; the gate below is
        # the session's own (the detector's live hard silence). Empty on v2.
        wire = {k: v for k, v in self.tr.ready_fields().items() if k != "gate"}
        return {**wire, "type": "ready", "sample_rate_in": 16000,
                "sample_rate_out": getattr(self.tts, "sample_rate", 24000),
                "smart_turn": self.detector.ep.turn_fn is not None,
                "patience": self.patience,
                "voice_acks": self.acks.ready_count(),
                "barge_in": self.barge_in_enabled,
                "reply_style": self.reply_style,
                "follow_up": self.effective_follow_up,
                "sample_format": getattr(self.tts, "sample_format", "pcm_s16le"),
                "streaming_tts": bool(getattr(self.tts, "streaming", False)),
                "sid": self.sid, "langs": ["en"], "say": True,
                "replay_last": True,          # C.3: answered (Amendment T7a-6 / T3s4-1)
                "acks": self.acks.mode, "power_hold": power,
                # C.4: the realtime facts and the client mic gate's timings (SPEC §5.6)
                "rt": {"stt": self._stt_name(), "speculate": self._speculation_allowed(),
                       # null when the server has nothing of its own for the device: the
                       # client keeps its blob (review-T3s3 #6: the defaults overwrote it)
                       "learned": self.policy.learned.as_dict() if self._learned_from_server else None,
                       "decoder_prompt_s": self.rt.decoder_prompt_s,
                       "backchannels": bool(self.bot_backchannel),
                       "streaming": self.sstt is not None},
                "gate": {"pre_roll_ms": round(float(getattr(cfg, "pre_roll_s", 0.3)) * 1000),
                         "hangover_s": round(max(hard * 1.5, self.rt.hold_max_s * 1.5) + 0.5, 2)}}

    def _stt_name(self) -> str:
        name = type(self.sstt).__name__.lower() if self.sstt is not None else ""
        return "moonshine" if "moonshine" in name else "parakeet"

    async def run(self) -> None:
        """Receive loop. Returns when the socket closes, the session locks,
        is replaced, or a worker fails; ``close_code`` says how to close."""
        self._last_live_check = self.clock()
        await self.send(self._ready_frame())
        tasks = [asyncio.create_task(self._guarded(n, c)) for n, c in
                 (("audio", self._audio_worker), ("tts", self._tts_worker),
                  ("frames", self._frames_worker), ("watchdog", self._watchdog))]
        if self.tr.v3 and self._relay is not None:
            tasks.append(asyncio.create_task(self._lookup_relay()))
        fatal = asyncio.create_task(self._fatal.wait())
        recv: asyncio.Task | None = None
        try:
            while True:
                recv = asyncio.create_task(self.ws.receive())
                await asyncio.wait({recv, fatal}, return_when=asyncio.FIRST_COMPLETED)
                if not recv.done():
                    break                     # ended from inside (worker, lapse, eviction)
                m = recv.result()
                recv = None
                if m.get("type") == "websocket.disconnect":
                    break
                now = self.clock()
                self._last_rx = now
                if now - self._last_live_check > 1.0:
                    self._last_live_check = now
                    if not self._check_live():
                        await self.send({"type": "locked"})
                        self.close_code = (1008, "locked")
                        break
                if m.get("bytes") is not None:
                    if self.listening:
                        self._maybe_touch()
                    await self._enqueue_audio(m["bytes"])
                elif m.get("text") is not None:
                    text = m["text"]
                    try:
                        msg = json.loads(text) if len(text) <= CONTROL_MAX_CHARS else None
                    except ValueError:
                        msg = None
                    ping = isinstance(msg, dict) and msg.get("type") == "ping"
                    if not self._take_token(ping=ping):
                        await self._limited("rate_limited", "Too many requests; slow down.")
                        continue
                    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
                        await self._limited("bad_frame", "That message wasn't understood.")
                        continue
                    if msg.get("type") != "ping":
                        self._maybe_touch()
                    try:
                        await self.handle_control(msg)
                    except asyncio.CancelledError:
                        raise
                    except Exception:   # one malformed frame must not end the chat
                        log.exception("voice: a %r frame failed", str(msg.get("type"))[:20])
                        await self._limited("bad_frame", "That message wasn't understood.")
        finally:
            for t in (recv, fatal):
                if t is not None and not t.done():
                    t.cancel()
            self._closed = True
            self._stop_tts.set()
            for t in [*tasks, *self._turn_tasks]:
                t.cancel()
            self._cancel_ack_task()
            for fut in (self._heard_fut, self._refine_fut):
                if fut is not None and not fut.done():
                    fut.cancel()
            await asyncio.gather(*tasks, *self._turn_tasks, return_exceptions=True)
            if self.spec is not None:
                # A speculative run nobody will adopt must not run on for its 60 s TTL (a
                # metered reply generated for no one; review-T3s3 #9).
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.spec.discard())
            # The speaking task is gone: free the codecs (idempotent; waits ≤ 2 s for an
            # encode still running in a worker thread — review-T6 H1).
            with contextlib.suppress(Exception):
                await self.tr.close()
