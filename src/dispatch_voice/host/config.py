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
contracts §F.1 — add-on-owned, never new DisPatch bot fields) and the
optional GPU rig (``rig.enabled`` / ``rig.url`` / its deadlines; the token
is never a setting, see ``host/rig.py``). They
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

from .. import fsperm
from ..rt.settings import RtSettings
from . import settings_schema as _sch
from .rig import RigSettings

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
    """Where the speech models are: the unified folder's ``speech`` subfolder, or the old flat
    location while models still sit there (host/models.py)."""
    from . import models
    return models.speech_dir()


#: The vocabularies and ranges below live in ``settings_schema`` (the one table); the names
#: are kept here because the rest of the add-on, and its plugins, import them from here.
PATIENCE = _sch.PATIENCE
REPLY_STYLES = _sch.REPLY_STYLES
THINKING_LEVELS = _sch.THINKING_LEVELS
SPEED_MIN, SPEED_MAX = _sch.SPEED_MIN, _sch.SPEED_MAX
MERGE_WINDOW_MIN, MERGE_WINDOW_MAX = _sch.MERGE_WINDOW_MIN, _sch.MERGE_WINDOW_MAX
HARD_SILENCE_MIN, HARD_SILENCE_MAX = _sch.HARD_SILENCE_MIN, _sch.HARD_SILENCE_MAX
MAX_SESSIONS_MIN, MAX_SESSIONS_MAX, MAX_SESSIONS_DEFAULT = (
    _sch.MAX_SESSIONS_MIN, _sch.MAX_SESSIONS_MAX, _sch.MAX_SESSIONS_DEFAULT)
POWER_HOLD_MODES = _sch.POWER_HOLD_MODES
RP_MODES = _sch.RP_MODES
GATEWAY_TRAILERS = _sch.GATEWAY_TRAILERS
VOICE_ID_MAX = _sch.VOICE_ID_MAX
OPUS_MODES = _sch.OPUS_MODES
OPUS_DOWN_KBPS = _sch.OPUS_DOWN_KBPS
OPUS_UP_KBPS = _sch.OPUS_UP_KBPS
BOT_KEYS = _sch.BOT_KEYS
BOT_LOCKED_VOICE = _sch.BOT_LOCKED_VOICE
FORCE_ON = _sch.FORCE_ON


@dataclass(frozen=True)
class VoiceSettings:
    flag: str = "auto"
    #: The Settings › Voice switch (voice.yaml ``enabled``). Only consulted
    #: when ``flag`` is ``auto``. Off by default.
    switch: bool = False
    models_dir: Path = field(default_factory=_default_models_dir)
    stt_engine: str = "parakeet"          # parakeet | moonshine (ONE setting: also RtSettings.stt_engine)
    #: Who recognises speech: cpu (this computer) | openai (an OpenAI-compatible server) | a plugin's
    #: provider id. The GPU rig is a separate switch (``rig.enabled``) with the CPU behind it.
    stt_provider: str = "cpu"
    stt_url: str = ""
    stt_model: str = ""
    stt_timeout_s: float = 20.0
    #: auto (the cloned-voice engine ``chatterbox`` when its models are
    #: fetched, else the first ready registered engine) | chatterbox | minimax
    #: | any name a plugin registered (e.g. ``kokoro`` from dispatch-voice-kokoro)
    tts_engine: str = "auto"
    #: auto (the HQ server when ``hq_url`` is set) | cpu (never the HQ server) | openai | a plugin's id.
    tts_provider: str = "auto"
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
    #: The optional GPU rig (``rig:``; host/rig.py). Off unless ``rig.enabled`` AND
    #: ``rig.url`` AND the SPEECHFORGE_TOKEN are all present.
    rig: RigSettings = field(default_factory=RigSettings)
    #: Hosts / ``*.suffix`` / CIDRs a PUBLIC rig or speech-server address may be (netpolicy).
    rig_allow: tuple = ()
    #: Enabled provider plugins: entry-point names or legacy ``module:setup`` targets.
    extensions: tuple = ()
    #: Where each setting's value came from: ``{id: {"source": "env"|"file"|"default", "env": NAME}}``.
    sources: dict = field(default_factory=dict, compare=False, repr=False)

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
        if self.stt_provider == "openai":
            if not self.stt_url:
                out.append("a transcription server address (stt_url)")
        elif self.stt_engine in ("parakeet", "moonshine"):         # Moonshine falls back to Parakeet
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

    def locked_voice(self, bot_id: str | None) -> str:
        """The voice a locked device hears for this bot, or "" (same as standard)."""
        conf = self.bots.get(bot_id or "", {}) if isinstance(self.bots, dict) else {}
        v = conf.get(BOT_LOCKED_VOICE)
        return v.strip() if isinstance(v, str) else ""

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
            "stt_provider": self.stt_provider,
            "tts": self.tts_engine,
            "tts_provider": self.tts_provider,
            "hq": bool(self.hq_url) and self.tts_provider != "cpu",
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
# <data>/voice.yaml: the in-app preferences (see the module docstring)
#
# What a key may hold, its range, its aliases and its environment names are in
# ``settings_schema``; this section only reads and writes the file.
# --------------------------------------------------------------------------- #

#: voice.yaml keys from 2.1 -> today's names.
PREF_ALIASES = dict(_sch.LEGACY_FILE_KEYS)
#: Keys of the ``rt:`` tree (contracts section F.1): ``power_hold`` plus every ``RtSettings``
#: field except the per-bot block and the copied hard silence.
RT_KEYS = _sch.rt_keys()
#: Top-level keys of voice.yaml.
PREF_KEYS = tuple(sorted({s.parts[0] for s in _sch.FILE_SETTINGS} | {"bots"}))

clean_voice_id = _sch.clean_voice_id
clean_bots = _sch.clean_bots


def clean_rt(raw) -> dict:
    """The ``rt:`` subtree, validated (unknown keys dropped, a number outside its range clamped,
    a wrong type dropped so the default applies)."""
    return _sch.clean_subtree("rt", raw)


def clean_prefs(raw) -> dict:
    """Only known keys with valid values survive; anything else is dropped. Aliases fold into
    the canonical key (``kokoro_speed`` -> ``speed``, a top-level ``power_hold`` -> ``rt.power_hold``)."""
    return _sch.clean_prefs(raw)


_PREFS_CACHE: tuple[tuple[int, int], dict] | None = None
_PREFS_LOCK = threading.Lock()
#: Serialises the read-modify-write of voice.yaml: two PATCHes at once used
#: to drop each other's key. (A save keeps only known keys; comments and
#: unknown keys in a hand-edited file are not preserved.)
_PREFS_WRITE_LOCK = threading.Lock()


def prefs_path() -> Path:
    from . import context
    return context.paths().voice_yaml


def forget_prefs(keys) -> dict:
    """Remove ``keys`` from voice.yaml (back to the env/default value)."""
    with _PREFS_WRITE_LOCK:
        return _write_prefs({k: v for k, v in load_prefs().items() if k not in set(keys)})


def load_prefs() -> dict:
    """voice.yaml, validated; {} when absent or unreadable. Cached on (mtime, size)."""
    global _PREFS_CACHE
    p = prefs_path()
    try:
        st = p.stat()
        # (mtime_ns, size): two writes inside one filesystem clock tick (15 ms on Windows)
        # share an mtime, and a hand edit that changes the text changes the size.
        mtime = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    with _PREFS_LOCK:
        if _PREFS_CACHE is not None and _PREFS_CACHE[0] == mtime:
            return dict(_PREFS_CACHE[1])
    try:
        import yaml
        prefs = clean_prefs(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    except Exception:
        log.warning("voice: %s is unreadable; using the defaults", p)
        prefs = {}
    with _PREFS_LOCK:
        _PREFS_CACHE = (mtime, prefs)
    return dict(prefs)


def save_prefs(update: dict) -> dict:
    """Merge ``update`` (already validated) into voice.yaml atomically. The ``rt`` and ``rig``
    subtrees are merged key by key, and ``bots`` per bot per key (a PATCH of one bot's
    ``backchannel`` leaves every other bot and its ``follow_up`` alone), never replaced wholesale."""
    with _PREFS_WRITE_LOCK:
        current = load_prefs()
        clean = clean_prefs(update)
        if "rt" in clean or "rt" in current:
            clean["rt"] = {**current.get("rt", {}), **clean.get("rt", {})}
        if "rig" in clean or "rig" in current:
            clean["rig"] = {**current.get("rig", {}), **clean.get("rig", {})}
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
    tmp.write_text(yaml.safe_dump(prefs, sort_keys=True), encoding="utf-8")
    fsperm.ensure_private(tmp)
    os.replace(tmp, p)
    with _PREFS_LOCK:
        _PREFS_CACHE = None
    return prefs


#: Schema ids that are not ``VoiceSettings`` fields (read where they are used, or kept elsewhere).
_NOT_FIELDS = frozenset({"models_root", "chatterbox_models", "model_revision", "data_dir", "ack_render",
                         "extensions", "rig.allow", "bots", "enabled"})
#: ``rt.*`` ids that ARE ``VoiceSettings`` fields (one definition, two readers).
_RT_FIELDS = frozenset({"rt.power_hold", "rt.stt_engine"})


def load_settings() -> VoiceSettings:
    """The settings in force: environment beats voice.yaml beats the default (1.6.0; 1.5.0 let
    voice.yaml beat every variable except DISPATCH_VOICE). ``settings.sources`` says which won."""
    prefs = load_prefs()
    values, sources = _sch.resolve(prefs, os.environ)
    names = {f.name for f in fields(VoiceSettings)}
    kw: dict = {}
    for s in _sch.SETTINGS:
        if s.scope != "server" or s.id not in values or s.secret or s.id in _NOT_FIELDS:
            continue
        head = s.id.split(".")[0]
        if head in ("rt", "rig") and s.id not in _RT_FIELDS:
            continue
        name = s.field or s.id.split(".")[-1]
        if name in names:
            kw[name] = values[s.id]
    kw["flag"] = _env("VOICE", "auto")
    kw["switch"] = bool(values.get("enabled", False))
    models = values.get("models_root") or ""
    kw["models_dir"] = Path(models).expanduser() if models else _default_models_dir()
    kw["bots"] = dict(values.get("bots") or {})
    kw["extensions"] = tuple(values.get("extensions") or ())
    kw["rig_allow"] = tuple(values.get("rig.allow") or ())

    def explicit(prefix: str) -> dict:
        """The keys of a subtree that something set (a default must not look like a choice)."""
        return {k[len(prefix):]: v for k, v in values.items()
                if k.startswith(prefix) and sources[k]["source"] != "default"}

    rt_raw = {k: v for k, v in explicit("rt.").items() if k != "power_hold"}
    rig_fields = {f.name for f in fields(RigSettings)}
    kw["rig"] = RigSettings.from_mapping({k: v for k, v in explicit("rig.").items() if k in rig_fields})
    kw["rt"] = RtSettings.from_mapping(
        {**rt_raw, "power_hold": kw["power_hold"], "stt_engine": kw["stt_engine"], "bots": kw["bots"]},
        hard_silence_s=kw["turn_hard_silence_s"])
    kw["sources"] = sources
    return VoiceSettings(**kw)
