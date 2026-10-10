"""Voice-chat settings, read from the environment and ``<data>/voice.yaml``.

Voice is an OPTIONAL feature with an on/off switch (Settings › Voice). Who
decides whether it is on:

* ``DISPATCH_VOICE`` unset or ``auto`` (default) — the switch decides. It is
  ``enabled`` in ``<data>/voice.yaml`` and is OFF until someone turns it on,
  even when every package and model is present: a fresh install never loads
  ~1.5 GB of speech models nobody asked for. When it is on, voice works once
  the model files the chosen engines need are on disk; until then
  ``/api/voice/status`` lists what is missing.
* ``1`` / ``true`` / ``on`` / ``yes`` — forced on (the switch is shown on and
  locked); a missing piece is reported by ``/api/voice/status`` and the voice
  socket refuses with a reason.
* ``0`` / anything else — forced off (the switch is hidden, with a note).

The switch applies live: turning it off ends open voice chats with 1013 and
drops the loaded speech models; turning it on starts the background prewarm
(routes.apply_switch).

Model files (VAD, Smart Turn, Parakeet) live in ``DISPATCH_SPEECH_MODELS``
(default ``<data>/speech-models``) with the layout ``dispatch-voice
fetch-models`` produces; the Chatterbox models in ``DISPATCH_VOICE_MODELS``.

A few choices are PREFERENCES, not install decisions, and are changed from the
app (``/api/voice/settings``, unlocked only): the on/off switch, the built-in
voice (when a plugin provides built-in voices), the speaking speed, the Voice
lab's thinking level and two timing values (the turn-merge window and the
hard-silence end of turn), the number of voice chats the server allows at
once (``max_sessions``), the power-profile hold (``rt.power_hold``), the
role-play speech mode (``rp_speech``) and its narrator voice
(``narrator_voice``), whether a cloned voice renders expressive tags
(``expressive_tags``), the gateway trailer form (``gateway_trailer``) and the
per-bot conversation defaults (``bots.<id>.backchannel`` / ``follow_up``;
contracts §F.1 — add-on-owned, never new DisPatch bot fields). They
live in ``<data>/voice.yaml`` and override the matching environment
defaults; :func:`load_settings` reads the file (cached on its mtime), so a
change applies to the next turn without a restart.

The data dir and the model dirs come from the DisPatch extension context
when the add-on runs inside DisPatch (``host.context``), else from the
environment (``DISPATCH_DATA_DIR``), so the CLI and the tests see the same
files a DisPatch install does.

Voice PROFILES (cloned voices) are ``profiles_bridge``'s job: here a profile
is only an opaque ``profile_id`` handed to the TTS engine.

Setting names are engine-neutral. The names from DisPatch 2.1, when the
built-in voice was always Kokoro, still work as aliases:
``DISPATCH_VOICE_KOKORO_VOICE`` / ``DISPATCH_VOICE_KOKORO_SPEED`` (env) and
``kokoro_voice`` / ``kokoro_speed`` (voice.yaml). The new name wins when both
are set; saving from the app rewrites voice.yaml with the new names.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from ..rt.settings import RtSettings

log = logging.getLogger("dispatch.voice")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"DISPATCH_{name}", default)


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _default_models_dir() -> Path:
    from . import context
    return context.paths().models_dir


#: Patience presets the driver can pick (multiplies the merge window and the
#: hard-silence fallback). "patient" suits a driver who pauses mid-thought.
PATIENCE = {"quick": 0.7, "normal": 1.0, "patient": 1.5}

#: How long a spoken reply should be (``start.reply_style``; persona.py).
REPLY_STYLES = ("short", "normal", "longer")
#: Allowed thinking levels for voice turns (``voice_turn_thinking``).
THINKING_LEVELS = ("off", "low", "medium")
#: Speaking-speed bounds (engines that support a speed, e.g. a built-in voice
#: plugin; a cloned voice speaks at its own pace).
SPEED_MIN, SPEED_MAX = 0.8, 1.3
#: Voice lab timing bounds (seconds). Outside these the endpointer either cuts
#: people off mid-sentence or makes every reply wait noticeably; the env
#: variables are not limited, the app is.
MERGE_WINDOW_MIN, MERGE_WINDOW_MAX = 0.5, 4.0
HARD_SILENCE_MIN, HARD_SILENCE_MAX = 0.8, 3.0
#: Open voice chats the server allows at once (contracts §F.1 ``max_sessions``;
#: DisPatch's gate answers 1013 "busy" beyond it).
MAX_SESSIONS_MIN, MAX_SESSIONS_MAX, MAX_SESSIONS_DEFAULT = 1, 8, 3
#: ``rt.power_hold``: hold the performance profile while a chat is open, or not.
POWER_HOLD_MODES = ("performance", "off")
#: ``rp_speech`` (server default) and a bot's ``voice_rp`` (contracts §B.3 / §F.5):
#: how a role-play reply is spoken. The prototype names ``quotes`` /
#: ``quotes+narrator`` are mapped inside ``text/speech_text.py``.
RP_MODES = ("all", "skip_actions", "dialogue", "dialogue_narrator")
#: ``gateway_trailer``: the full app note at the end of a gateway agent's text,
#: or the bare ``[voice chat]`` tag once the standing rule is in its brief.
GATEWAY_TRAILERS = ("full", "tag")
#: A voice id as stored in ``narrator_voice`` / a bot's ``voice`` (opaque: a
#: profile id or ``builtin:<id>``); the routes check that it exists.
VOICE_ID_MAX = 80
#: ``opus``: negotiate Opus with clients that can (``auto``), or never (``off``: PCM16
#: both ways). ``DISPATCH_VOICE_OPUS=off`` also turns it off (``transport.opus.available``).
OPUS_MODES = ("auto", "off")
#: Downlink / uplink Opus bitrate rungs (kbps; contracts §F.1).
OPUS_DOWN_KBPS = (24, 16)
OPUS_UP_KBPS = (16, 12)
#: Per-bot keys of ``bots.<id>`` (contracts §F.1).
BOT_KEYS = ("backchannel", "follow_up")
#: Values of DISPATCH_VOICE that force voice on; ``auto`` (or unset) leaves it
#: to the switch; anything else forces it off.
FORCE_ON = ("1", "true", "on", "yes")


@dataclass(frozen=True)
class VoiceSettings:
    flag: str = "auto"
    #: The Settings › Voice switch (voice.yaml ``enabled``). Only consulted
    #: when ``flag`` is ``auto``. Off by default.
    switch: bool = False
    models_dir: Path = field(default_factory=_default_models_dir)
    stt_engine: str = "parakeet"          # parakeet
    #: auto (the cloned-voice engine ``chatterbox`` when its models are
    #: fetched, else the first ready registered engine) | chatterbox | minimax
    #: | any name a plugin registered (e.g. ``kokoro`` from dispatch-voice-kokoro)
    tts_engine: str = "auto"
    #: Built-in voice (a provider's id, without ``builtin:``) used when no
    #: profile applies. "" = the provider's own default.
    builtin_voice: str = ""
    speed: float = 1.05
    minimax_model: str = "speech-2.8-turbo"
    minimax_voice: str = "Wise_Woman"
    minimax_base_url: str = "https://api.minimax.io"
    #: Optional "HQ" engine (e.g. Qwen3-TTS on the GPU rig) behind an
    #: OpenAI-compatible POST /v1/audio/speech. Falls back to the local engine on ANY
    #: failure, timeout or 409/503 — a drive never waits on the rig.
    hq_url: str = ""
    hq_model: str = ""
    hq_timeout_s: float = 4.0
    # Endpointing (seconds unless noted)
    vad_threshold: float = 0.5
    min_speech_s: float = 0.25            # shorter blips are noise, not words
    turn_probe_silence_s: float = 0.25    # silence before Smart Turn is asked
    turn_hard_silence_s: float = 1.4      # silence that ends a turn regardless
    no_smart_turn_silence_s: float = 0.8  # fallback when Smart Turn is absent
    max_utterance_s: float = 30.0
    smart_turn_threshold: float = 0.5
    #: After a turn ends we keep listening: speech resuming within this window
    #: means the turn ended too early (Smart Turn misfired on mid-sentence
    #: pauses in 4/10 road-noise simulation runs). The in-flight reply is
    #: cancelled and the merged utterance re-submitted.
    merge_window_s: float = 1.8
    patience: float = 1.0                 # multiplier; see PATIENCE presets
    #: Sustained speech needed to count as the driver talking OVER the server's
    #: own audio (SPEC §4.6 layer 2, §13 C5: 0.25 s — the duck is reversible, and
    #: it only DUCKS the reply until the words are checked; the Silero margin is
    #: raised to 0.65 meanwhile). Over the phone's voice the session uses 0.6 s.
    barge_in_speech_s: float = 0.25
    #: When the acknowledgement line goes out after the end of turn (SPEC §6.2:
    #: 0.6 s). It must land INSIDE the client's turn-end tone hold
    #: (drive.js EOT_TONE_HOLD_MS, 800 ms) with room for the network, or every
    #: acknowledged turn makes two sounds: the tone, then the line (FR-works 5:
    #: 0.7 s here against a 650 ms hold played both, 50 ms apart).
    ack_after_s: float = 0.55
    barge_abort_turn: bool = True         # barge-in also aborts the agent turn
    voice_turn_thinking: str = "off"      # TurnOptions.thinking for voice turns ("" = agent default)
    threads: int = 6                      # onnxruntime intra-op threads (of 20)
    tts_threads: int = 8                  # onnxruntime threads for a local TTS engine
    #: Load the speech models in the background at app startup (H5), so the
    #: first voice chat after a restart does not pay for it.
    prewarm: bool = True
    #: Open voice chats at once (DisPatch's gate: 1013 "busy" beyond it).
    max_sessions: int = MAX_SESSIONS_DEFAULT
    #: ``rt.power_hold``: "performance" holds that profile while a chat is open.
    power_hold: str = "performance"
    #: Role-play speech: ``all`` (default) | ``skip_actions`` | ``dialogue`` |
    #: ``dialogue_narrator``; a bot's own ``voice_rp`` overrides it.
    rp_speech: str = "all"
    #: The narrator's voice under ``dialogue_narrator`` (a profile id or
    #: ``builtin:<id>``); "" or a voice that no longer exists = the character voice.
    narrator_voice: str = ""
    #: A cloned (Chatterbox) voice may render ``[laugh]``/``[chuckle]``/``[sigh]``;
    #: off = every bracket tag is removed from speech.
    expressive_tags: bool = True
    #: Gateway agents get the full app note (``full``) or the bare tag (``tag``).
    gateway_trailer: str = "full"
    #: The v3 wire's codec (contracts §F.1): ``auto`` = Opus with a client that can,
    #: ``off`` = PCM16 both ways; and the first Opus rung each way.
    opus: str = "auto"
    opus_down_kbps: int = 24
    opus_up_kbps: int = 16
    #: Per-bot conversation defaults: ``{bot id: {backchannel: bool, follow_up: bool}}``.
    bots: dict = field(default_factory=dict)
    #: The realtime pipeline's knobs (contracts §F.1 ``rt.*``; ``rt/settings.py``
    #: validates and clamps them). ``hard_silence_s`` is copied in from the
    #: top-level key; ``power_hold`` mirrors the field above.
    rt: RtSettings = field(default_factory=RtSettings)

    # ---- derived paths -------------------------------------------------- #
    @property
    def silero_path(self) -> Path:
        return self.models_dir / "silero_vad.onnx"

    @property
    def smart_turn_path(self) -> Path:
        return self.models_dir / "smart-turn-v3.2-cpu.onnx"

    @property
    def parakeet_dir(self) -> Path:
        return self.models_dir / "parakeet-tdt-0.6b-v2"

    # ---- 2.1 names (read-only aliases) ---------------------------------- #
    @property
    def kokoro_voice(self) -> str:
        return self.builtin_voice

    @property
    def kokoro_speed(self) -> float:
        return self.speed

    # ---- availability --------------------------------------------------- #
    def missing(self) -> list[str]:
        """Human-readable list of what stops voice from working (empty = ready)."""
        out: list[str] = []
        for mod in ("numpy", "onnxruntime"):
            if importlib.util.find_spec(mod) is None:
                out.append(f"python package {mod}")
        if not self.silero_path.exists():
            out.append(f"model {self.silero_path.name}")
        if self.stt_engine == "parakeet":
            if importlib.util.find_spec("onnx_asr") is None:
                out.append("python package onnx-asr")
            if not (self.parakeet_dir / "encoder-model.int8.onnx").exists():
                out.append("model parakeet-tdt-0.6b-v2")
        out += self._tts_missing()
        return out

    def _tts_missing(self) -> list[str]:
        """What the chosen TTS engine still needs (empty = it can speak)."""
        from . import registry
        name = self.tts_engine
        if name == "minimax":
            return [] if os.environ.get("MINIMAX_API_KEY") else ["MINIMAX_API_KEY (DISPATCH_VOICE_TTS=minimax)"]
        if name in ("chatterbox", "auto"):
            if _dispatch_voice_ready():
                return []
            if name == "chatterbox" or not any(s.ready(self) for s in registry._engines.values()):
                return ["dispatch-voice models (run `dispatch-voice fetch-models`)"]
            return []
        spec = registry.engine_spec(name)
        if spec is None:
            hint = " (it comes with the optional dispatch-voice-kokoro plugin)" if name == "kokoro" else ""
            return [f"TTS engine {name!r} is not installed{hint}"]
        return spec.missing_for(self)

    @property
    def forced(self) -> str | None:
        """"on" / "off" when DISPATCH_VOICE decides, None when the switch does."""
        f = (self.flag or "").strip().lower()
        if f == "auto":
            return None
        return "on" if f in FORCE_ON else "off"

    @property
    def switched_on(self) -> bool:
        """Has the owner asked for voice (env, else the switch)? Says nothing
        about whether the pieces are installed."""
        forced = self.forced
        return self.switch if forced is None else forced == "on"

    @property
    def enabled(self) -> bool:
        """Voice chat is offered: forced on, or switched on and ready."""
        forced = self.forced
        if forced is not None:
            return forced == "on"
        return self.switch and not self.missing()

    def merge_window(self, patience: float | None = None) -> float:
        return self.merge_window_s * (patience or self.patience)

    def rt_status(self) -> dict:
        """The ``rt`` block of ``/api/voice/status`` (contracts §D): the knobs a
        client shows, plus the hold."""
        rt = self.rt
        return {"stt": rt.stt_engine, "speculate": rt.speculate,
                "decoder_prompt_s": rt.decoder_prompt_s, "decoder_sessions": rt.decoder_sessions,
                "addendum_s": rt.addendum_s, "power_hold": self.power_hold}

    def rt_settings(self) -> dict:
        """The ``rt`` tree of ``GET/PATCH /api/voice/settings``: every §F.1 key."""
        d = self.rt.to_dict()
        d.pop("bots", None)
        d.pop("hard_silence_s", None)        # a top-level key, not an rt one
        d["power_hold"] = self.power_hold
        return d

    def bot(self, bot_id: str | None, *, backend: str | None = None) -> dict:
        """This bot's effective conversation defaults: ``backchannel`` is off
        unless opted in; ``follow_up`` is on unless the bot's backend is the
        advisor or the file says otherwise (SPEC §6.1 L6, §4.7 M2)."""
        conf = self.bots.get(bot_id or "", {}) if isinstance(self.bots, dict) else {}
        return {"backchannel": bool(conf.get("backchannel", False)),
                "follow_up": bool(conf.get("follow_up", backend != "advisor"))}

    def status(self) -> dict:
        """Safe to show an unlocked client: no paths, no keys."""
        miss = self.missing()
        forced = self.forced
        return {
            "enabled": forced == "on" if forced is not None else (self.switch and not miss),
            # The switch as stored (what Settings › Voice draws) and whether
            # DISPATCH_VOICE overrides it ("on" / "off" / None).
            "switch": self.switched_on,
            "forced": forced,
            "ready": not miss,
            "missing": miss,
            "stt": self.stt_engine,
            "tts": self.tts_engine,
            "hq": bool(self.hq_url),
            "smart_turn": self.smart_turn_path.exists(),
            "patience": self.patience,
            "patience_presets": PATIENCE,
            "speed": self.speed,
            "voice_turn_thinking": self.voice_turn_thinking or "off",
            "merge_window_s": self.merge_window_s,
            "hard_silence_s": self.turn_hard_silence_s,
            "max_sessions": self.max_sessions,
            "rt": self.rt_status(),
            "rp_speech": self.rp_speech,
            "narrator_voice": self.narrator_voice,
            "expressive_tags": self.expressive_tags,
            "gateway_trailer": self.gateway_trailer,
        }


#: What /api/voice/status says when nothing can speak. Deliberately not
#: "python package …": the Settings tab treats that prefix as "voice is off".
NO_TTS = ("a text-to-speech engine: run `dispatch-voice fetch-models`, "
          "or set DISPATCH_VOICE_TTS=minimax")


def _dispatch_voice_installed() -> bool:
    """The cloned-voice engine is this package: always installed."""
    return True


def _dispatch_voice_ready() -> bool:
    from .engines.chatterbox import available
    return available()


def _bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


# --------------------------------------------------------------------------- #
# <data>/voice.yaml — the in-app preferences (see the module docstring)
# --------------------------------------------------------------------------- #

PREF_KEYS = ("enabled", "builtin_voice", "speed", "voice_turn_thinking",
             "merge_window_s", "hard_silence_s", "max_sessions", "rt",
             "rp_speech", "narrator_voice", "expressive_tags", "gateway_trailer", "bots",
             "opus", "opus_down_kbps", "opus_up_kbps")
#: voice.yaml keys from 2.1 → today's names.
PREF_ALIASES = {"kokoro_voice": "builtin_voice", "kokoro_speed": "speed"}
#: voice.yaml key → VoiceSettings field, where the two differ.
PREF_FIELDS = {"enabled": "switch", "hard_silence_s": "turn_hard_silence_s"}
#: Keys of the ``rt:`` tree (contracts §F.1): ``power_hold`` plus every
#: ``RtSettings`` field except the per-bot block and the copied hard silence.
RT_KEYS = ("power_hold", *[f.name for f in fields(RtSettings)
                           if f.name not in ("bots", "hard_silence_s", "power_hold")])


def _num_in(v, lo: float, hi: float) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi:
        return round(float(v), 3)
    return None


def _int_in(v, lo: int, hi: int) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
        return int(v)
    if isinstance(v, float) and v.is_integer() and lo <= v <= hi:
        return int(v)
    return None


_VOICE_ID_RE = __import__("re").compile(rf"\A[A-Za-z0-9][A-Za-z0-9:._-]{{0,{VOICE_ID_MAX - 1}}}\Z")
_BOT_ID_RE = __import__("re").compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def clean_voice_id(v) -> str | None:
    """A stored voice id ("" = none), or None when the value is not one."""
    if v is None or v == "":
        return ""
    if isinstance(v, str) and _VOICE_ID_RE.match(v.strip()):
        return v.strip()
    return None


def clean_bots(raw) -> dict:
    """The ``bots:`` subtree: ``{bot id: {backchannel?: bool, follow_up?: bool}}``,
    validated (unknown keys and non-boolean values dropped, empty entries gone)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for bot_id, conf in raw.items():
        if not isinstance(bot_id, str) or not _BOT_ID_RE.match(bot_id) or not isinstance(conf, dict):
            continue
        entry = {k: conf[k] for k in BOT_KEYS if isinstance(conf.get(k), bool)}
        if entry:
            out[bot_id] = entry
    return out


def clean_rt(raw) -> dict:
    """The ``rt:`` subtree, validated (unknown keys dropped): ``power_hold`` by
    hand (a hand-edited ``off`` is YAML's boolean false), every other key through
    ``RtSettings.from_mapping`` — a value outside the §F.1 range is clamped, a
    wrong type is dropped (the default applies)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    ph = raw.get("power_hold")
    if isinstance(ph, bool):
        out["power_hold"] = "performance" if ph else "off"
    elif isinstance(ph, str) and ph.strip().lower() in POWER_HOLD_MODES:
        out["power_hold"] = ph.strip().lower()
    rest = {k: v for k, v in raw.items() if k in RT_KEYS and k != "power_hold"}
    if rest:
        clean = RtSettings.from_mapping(rest)
        defaults = RtSettings()
        for k, v in rest.items():
            got = getattr(clean, k)
            # from_mapping falls back to the default on a wrong type; keep the key
            # only when the raw value was of the right kind (so a typo'd string
            # does not silently pin the default into the file).
            if isinstance(getattr(defaults, k), bool):
                ok = isinstance(v, bool)
            elif isinstance(getattr(defaults, k), (int, float)):
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            else:
                ok = isinstance(v, str) and got == v
            if ok:
                out[k] = got
    return out


_PREFS_CACHE: tuple[float, dict] | None = None
_PREFS_LOCK = threading.Lock()
#: Serialises the read-modify-write of voice.yaml: two PATCHes at once used
#: to drop each other's key. (A save keeps only known keys; comments and
#: unknown keys in a hand-edited file are not preserved.)
_PREFS_WRITE_LOCK = threading.Lock()


def prefs_path() -> Path:
    from . import context
    return context.paths().voice_yaml


def clean_prefs(raw) -> dict:
    """Only known keys with valid values survive; anything else is dropped."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    raw = dict(raw)
    for old, new in PREF_ALIASES.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]
    v = raw.get("builtin_voice")
    if isinstance(v, str) and v.strip() and len(v) <= 64 and v.replace("_", "").replace("-", "").isalnum():
        out["builtin_voice"] = v.strip()
    sp = raw.get("speed")
    if isinstance(sp, (int, float)) and not isinstance(sp, bool) and SPEED_MIN <= float(sp) <= SPEED_MAX:
        out["speed"] = round(float(sp), 3)
    th = raw.get("voice_turn_thinking")
    if isinstance(th, str) and th.strip().lower() in THINKING_LEVELS:
        out["voice_turn_thinking"] = th.strip().lower()
    if isinstance(raw.get("enabled"), bool):
        out["enabled"] = raw["enabled"]
    mw = _num_in(raw.get("merge_window_s"), MERGE_WINDOW_MIN, MERGE_WINDOW_MAX)
    if mw is not None:
        out["merge_window_s"] = mw
    hs = _num_in(raw.get("hard_silence_s"), HARD_SILENCE_MIN, HARD_SILENCE_MAX)
    if hs is not None:
        out["hard_silence_s"] = hs
    ms = _int_in(raw.get("max_sessions"), MAX_SESSIONS_MIN, MAX_SESSIONS_MAX)
    if ms is not None:
        out["max_sessions"] = ms
    rt = clean_rt(raw.get("rt"))
    if rt:
        out["rt"] = rt
    rp = raw.get("rp_speech")
    if isinstance(rp, str) and rp.strip().lower() in RP_MODES:
        out["rp_speech"] = rp.strip().lower()
    nv = clean_voice_id(raw.get("narrator_voice")) if "narrator_voice" in raw else None
    if nv is not None:
        out["narrator_voice"] = nv
    if isinstance(raw.get("expressive_tags"), bool):
        out["expressive_tags"] = raw["expressive_tags"]
    gt = raw.get("gateway_trailer")
    if isinstance(gt, str) and gt.strip().lower() in GATEWAY_TRAILERS:
        out["gateway_trailer"] = gt.strip().lower()
    op = raw.get("opus")
    if isinstance(op, bool):                       # a hand-edited `off` is YAML's false
        out["opus"] = "auto" if op else "off"
    elif isinstance(op, str) and op.strip().lower() in OPUS_MODES:
        out["opus"] = op.strip().lower()
    dk = _int_in(raw.get("opus_down_kbps"), min(OPUS_DOWN_KBPS), max(OPUS_DOWN_KBPS))
    if dk in OPUS_DOWN_KBPS:
        out["opus_down_kbps"] = dk
    uk = _int_in(raw.get("opus_up_kbps"), min(OPUS_UP_KBPS), max(OPUS_UP_KBPS))
    if uk in OPUS_UP_KBPS:
        out["opus_up_kbps"] = uk
    if "bots" in raw:
        bots = clean_bots(raw.get("bots"))
        if bots or raw.get("bots") == {}:
            out["bots"] = bots
    return out


def forget_prefs(keys) -> dict:
    """Remove ``keys`` from voice.yaml (back to the env/default value)."""
    with _PREFS_WRITE_LOCK:
        return _write_prefs({k: v for k, v in load_prefs().items() if k not in set(keys)})


def load_prefs() -> dict:
    """voice.yaml, validated; {} when absent or unreadable. Cached on mtime."""
    global _PREFS_CACHE
    p = prefs_path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    with _PREFS_LOCK:
        if _PREFS_CACHE is not None and _PREFS_CACHE[0] == mtime:
            return dict(_PREFS_CACHE[1])
    try:
        import yaml
        prefs = clean_prefs(yaml.safe_load(p.read_text()) or {})
    except Exception:
        log.warning("voice: %s is unreadable; using the defaults", p)
        prefs = {}
    with _PREFS_LOCK:
        _PREFS_CACHE = (mtime, prefs)
    return dict(prefs)


def save_prefs(update: dict) -> dict:
    """Merge ``update`` (already validated) into voice.yaml atomically. The
    ``rt`` subtree is merged key by key, and ``bots`` per bot per key (a PATCH
    of one bot's ``backchannel`` leaves every other bot and its ``follow_up``
    alone), never replaced wholesale."""
    with _PREFS_WRITE_LOCK:
        current = load_prefs()
        clean = clean_prefs(update)
        if "rt" in clean or "rt" in current:
            clean["rt"] = {**current.get("rt", {}), **clean.get("rt", {})}
        if "bots" in clean or "bots" in current:
            merged = {k: dict(v) for k, v in current.get("bots", {}).items()}
            for bot_id, conf in clean.get("bots", {}).items():
                merged[bot_id] = {**merged.get(bot_id, {}), **conf}
            clean["bots"] = merged
        return _write_prefs({**current, **clean})


def _write_prefs(prefs: dict) -> dict:
    global _PREFS_CACHE
    import yaml
    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(prefs, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    with _PREFS_LOCK:
        _PREFS_CACHE = None
    return prefs


def _power_hold_env() -> str:
    v = _env("VOICE_POWER_HOLD", "performance").strip().lower()
    return v if v in POWER_HOLD_MODES else "performance"


def load_settings() -> VoiceSettings:
    models = _env("SPEECH_MODELS")
    base = VoiceSettings(
        flag=_env("VOICE", "auto"),
        models_dir=Path(models).expanduser() if models else _default_models_dir(),
        stt_engine=_env("VOICE_STT", "parakeet").strip().lower(),
        tts_engine=_env("VOICE_TTS", "auto").strip().lower(),
        builtin_voice=_env("VOICE_BUILTIN_VOICE", _env("VOICE_KOKORO_VOICE", "")).strip(),
        speed=_float("VOICE_SPEED", _float("VOICE_KOKORO_SPEED", 1.05)),
        minimax_model=_env("VOICE_MINIMAX_MODEL", "speech-2.8-turbo"),
        minimax_voice=_env("VOICE_MINIMAX_VOICE", "Wise_Woman"),
        minimax_base_url=_env("VOICE_MINIMAX_BASE_URL", "https://api.minimax.io"),
        hq_url=_env("VOICE_HQ_URL", ""),
        hq_model=_env("VOICE_HQ_MODEL", ""),
        hq_timeout_s=_float("VOICE_HQ_TIMEOUT", 4.0),
        turn_hard_silence_s=_float("VOICE_HARD_SILENCE", 1.4),
        no_smart_turn_silence_s=_float("VOICE_SILENCE", 0.8),
        merge_window_s=_float("VOICE_MERGE_WINDOW", 1.8),
        patience=_float("VOICE_PATIENCE", 1.0),
        barge_abort_turn=_env("VOICE_BARGE_ABORT", "1").strip().lower() in ("1", "true", "on", "yes"),
        voice_turn_thinking=_env("VOICE_THINKING", "off").strip(),
        threads=_int("VOICE_THREADS", 6),
        tts_threads=_int("VOICE_TTS_THREADS", 8),
        prewarm=_bool("VOICE_PREWARM", True),
        max_sessions=min(MAX_SESSIONS_MAX, max(MAX_SESSIONS_MIN, _int("VOICE_MAX_SESSIONS", MAX_SESSIONS_DEFAULT))),
        power_hold=_power_hold_env(),
    )
    prefs = {PREF_FIELDS.get(k, k): v for k, v in load_prefs().items()}
    rt = prefs.pop("rt", None) or {}
    if "power_hold" in rt:
        prefs["power_hold"] = rt["power_hold"]
    s = replace(base, **prefs) if prefs else base
    rt_settings = RtSettings.from_mapping({**rt, "power_hold": s.power_hold, "bots": s.bots},
                                          hard_silence_s=s.turn_hard_silence_s)
    return replace(s, rt=rt_settings)
