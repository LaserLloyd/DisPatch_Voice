"""The ONE declarative table of every DisPatch Voice setting.

Before 1.6.0 a setting lived in up to five places: a constant for its range, a
``VoiceSettings`` field, a hand-written ``clean_prefs`` branch (for voice.yaml), a
hand-written ``PATCH /api/voice/settings`` branch, and the Settings pane's
JavaScript. This table is the single source: each :class:`Setting` names its
type, default, range, scope, level, secrecy and environment aliases, and the
rest is GENERATED from it:

* :func:`clean_prefs` / :func:`clean_subtree` - what voice.yaml may hold
  (unknown keys dropped, a bad value dropped or clamped);
* :func:`validate_patch` - the strict ``PATCH /api/voice/settings`` check
  (``SettingError`` becomes a 422 with the reason);
* :func:`resolve` - the value of every key with its source: **environment beats
  the file beats the default** (this inverts 1.5.0, where voice.yaml beat env);
* :func:`schema_json` - ``GET /api/voice/settings/schema`` (the Settings pane
  renders from it);
* :func:`env_table` - the README's environment table (a drift test keeps the
  README honest).

Ids are dotted paths in the shape of the file and the API: ``speed``,
``rt.power_hold``, ``rig.url``. A concept has ONE entry; the older spellings are
``aliases`` (accepted on input, folded into the canonical id): the 2.1
``kokoro_speed``, the top-level ``power_hold`` / ``stt_engine``, and
``rt.hard_silence_s``. This module is stdlib-only and imports nothing from the
add-on, so ``rt/settings.py`` and ``host/rig.py`` can take their ranges from it.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

SCOPES = ("server", "device", "bot")
LEVELS = ("basic", "advanced")
#: Where a value is kept. ``file`` = voice.yaml (and, from the host-store release, the
#: DisPatch settings store); ``env`` = environment only; ``device`` = the browser's
#: ``dispatch-drive`` blob; ``bot`` = the DisPatch bot record or ``bots:`` in the file;
#: ``credential`` = the rig token file (never a setting value).
STORES = ("file", "env", "device", "bot", "credential")

#: The environment prefix of every setting without an explicit env name.
ENV_PREFIX = "DISPATCH_VOICE_"

MISSING: Any = object()


class SettingError(ValueError):
    """A PATCH value the schema refuses; ``str(e)`` is the 422 text."""


@dataclass(frozen=True)
class Setting:
    id: str
    type: str                       # bool | int | float | enum | str | url | list | map | secret | voice_id
    default: Any = None
    scope: str = "server"
    level: str = "advanced"
    store: str = "file"
    lo: float | None = None
    hi: float | None = None
    #: A number outside [lo, hi]: ``drop`` the value (the default applies) or ``clamp`` it
    #: when READ from voice.yaml. A PATCH is always refused, never clamped.
    out_of_range: str = "drop"
    choices: tuple = ()
    secret: bool = False
    #: Environment names, highest priority first. ``DISPATCH_VOICE_*`` unless full.
    env: tuple[str, ...] = ()
    #: Also accepted in files and PATCH bodies (dotted), folded into ``id``.
    aliases: tuple[str, ...] = ()
    #: Applying a new value needs a restart of DisPatch.
    restart: bool = False
    resettable: bool = False
    patchable: bool = True
    group: str = ""                 # stt | tts | input | voices | connection | models | system
    label: str = ""
    help: str = ""
    #: ``VoiceSettings`` attribute when it differs from the id's last segment ("" = not a field).
    field: str | None = None
    pattern: str | None = None
    max_len: int | None = None
    #: Accepted URL schemes for a ``url`` setting.
    schemes: tuple[str, ...] = ("http", "https")
    #: A ``list`` setting's item pattern (default: a plugin name or ``module:setup`` target).
    item_pattern: str | None = None
    #: An env value outside the range is clamped (max_sessions) instead of passed through.
    env_clamp: bool = False
    #: The environment may hold a value the file would refuse (an experimental
    #: thinking level): it is passed through after type checks.
    env_loose: bool = False
    #: Custom environment parser ``str -> value`` (``MISSING`` = treat as unset).
    env_parse: Callable[[str], Any] | None = dc_field(default=None, compare=False)
    #: A short note about who else reads it (documentation only).
    note: str = ""

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.id.split("."))

    @property
    def persisted(self) -> bool:
        return self.store in ("file", "bot")

    @property
    def env_names(self) -> tuple[str, ...]:
        return tuple(self.env)


# --------------------------------------------------------------------------- #
# Shared vocabularies (the old constants live here now)
# --------------------------------------------------------------------------- #

PATIENCE = {"quick": 0.7, "normal": 1.0, "patient": 1.5}
REPLY_STYLES = ("short", "normal", "longer")
THINKING_LEVELS = ("off", "low", "medium")
SPEED_MIN, SPEED_MAX = 0.8, 1.3
MERGE_WINDOW_MIN, MERGE_WINDOW_MAX = 0.5, 4.0
HARD_SILENCE_MIN, HARD_SILENCE_MAX = 0.8, 3.0
MAX_SESSIONS_MIN, MAX_SESSIONS_MAX, MAX_SESSIONS_DEFAULT = 1, 8, 3
POWER_HOLD_MODES = ("performance", "off")
RP_MODES = ("all", "skip_actions", "dialogue", "dialogue_narrator")
GATEWAY_TRAILERS = ("full", "tag")
OPUS_MODES = ("auto", "off")
OPUS_DOWN_KBPS = (24, 16)
OPUS_UP_KBPS = (16, 12)
STT_ENGINES = ("parakeet", "moonshine")
SPECULATE_MODES = ("advisor", "all", "off")
BOT_KEYS = ("backchannel", "follow_up")
BOT_LOCKED_VOICE = "locked_voice"
VOICE_ID_MAX = 80
URL_MAX = 256
FORCE_ON = ("1", "true", "on", "yes")
INTERRUPT_SENSITIVITIES = ("low", "normal", "high")
SLOW_LINK = ("phone", "wait", "text", "always_phone")
REACTIONS = ("off", "sometimes", "often")
HANGUP_QUIET_MIN = (5, 15, 30, 0)

VOICE_ID_RE = re.compile(rf"\A[A-Za-z0-9][A-Za-z0-9:._-]{{0,{VOICE_ID_MAX - 1}}}\Z")
BOT_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SLUG_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
ENGINE_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,31}\Z")
#: A plugin target (``pkg.module:setup``) or an entry-point name.
PLUGIN_RE = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.:-]{0,95}\Z")
HOSTPAT_RE = re.compile(r"\A[A-Za-z0-9_*][A-Za-z0-9_.*:/\[\]-]{0,127}\Z")
LIST_MAX = 16


def _force_flag(raw: str) -> Any:
    """``DISPATCH_VOICE``: ``auto``/unset leaves the switch to the file; a truthy word
    forces voice on; anything else forces it off."""
    v = raw.strip().lower()
    if v in ("", "auto"):
        return MISSING
    return v in FORCE_ON


def _env_list(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _s(id_: str, type_: str, default: Any = None, **kw: Any) -> Setting:
    return Setting(id_, type_, default, **kw)


def _rt_float(name: str, default: float, lo: float, hi: float, **kw: Any) -> Setting:
    return _s(f"rt.{name}", "float", default, lo=lo, hi=hi, out_of_range="clamp", group="advanced_timing", **kw)


def _rt_int(name: str, default: int, lo: int, hi: int, **kw: Any) -> Setting:
    return _s(f"rt.{name}", "int", default, lo=lo, hi=hi, out_of_range="clamp", group="advanced_timing", **kw)


def _rt_bool(name: str, default: bool, **kw: Any) -> Setting:
    return _s(f"rt.{name}", "bool", default, group="advanced_timing", **kw)


# --------------------------------------------------------------------------- #
# THE TABLE
# --------------------------------------------------------------------------- #

SETTINGS: tuple[Setting, ...] = (
    # ---- the switch -------------------------------------------------------- #
    _s("enabled", "bool", False, level="basic", env=("DISPATCH_VOICE",), env_parse=_force_flag,
       field="switch", group="system", label="Voice chat",
       help="Turns voice chat on. DISPATCH_VOICE=1/0 forces it (the switch is then locked)."),

    # ---- 1. speech to text ------------------------------------------------- #
    _s("stt_provider", "str", "cpu", level="basic", env=("DISPATCH_VOICE_STT_PROVIDER",), pattern=ENGINE_RE.pattern,
       max_len=32, group="stt", label="Speech-to-text provider",
       help="cpu (this computer), openai (an OpenAI-compatible server), or a plugin's id. The GPU rig is "
            "switched with rig.enabled."),
    _s("stt_url", "url", "", env=("DISPATCH_VOICE_STT_URL",), group="stt", label="Transcription server",
       help="A server with POST /v1/audio/transcriptions (whisper.cpp, faster-whisper-server, ...)."),
    _s("stt_model", "str", "", env=("DISPATCH_VOICE_STT_MODEL",), max_len=96, group="stt",
       label="Transcription model"),
    _s("stt_timeout_s", "float", 20.0, lo=1.0, hi=120.0, env=("DISPATCH_VOICE_STT_TIMEOUT",), group="stt",
       label="Transcription timeout (s)"),
    _s("stt_api_key", "secret", "", store="env", secret=True, env=("DISPATCH_VOICE_STT_API_KEY",), group="stt",
       label="Transcription server API key", help="Only ever read from the environment."),
    _s("rt.stt_engine", "enum", "parakeet", level="basic", choices=STT_ENGINES, aliases=("stt_engine",),
       env=("DISPATCH_VOICE_STT",), field="stt_engine", group="stt", label="Speech-to-text engine",
       help="Which recogniser hears you. Parakeet is the default; Moonshine is an optional extra "
            "(it falls back to Parakeet when missing)."),
    _s("rt.speculate", "enum", "advisor", choices=SPECULATE_MODES, group="stt", label="Speculative replies",
       help="Start a reply before the turn is certain to be over (advisor bots only by default)."),
    _s("rt.stt_spinning", "bool", False, group="stt", label="STT thread spinning",
       help="ORT intra-op spinning for the recogniser (measured: off is faster under load)."),
    _s("rt.stt_threads_while_speaking", "int", 4, lo=1, hi=8, out_of_range="clamp", group="stt",
       label="STT threads while a reply plays"),
    _s("threads", "int", 6, lo=1, hi=64, store="env", env=("DISPATCH_VOICE_THREADS",), group="stt",
       label="ONNX threads (speech recognition)", help="onnxruntime intra-op threads."),

    # ---- 2. text to speech ------------------------------------------------- #
    _s("tts_provider", "str", "auto", level="basic", env=("DISPATCH_VOICE_TTS_PROVIDER",), pattern=ENGINE_RE.pattern,
       max_len=32, group="tts", label="Text-to-speech provider",
       help="auto (an OpenAI-compatible server when hq_url is set, else this computer), cpu (never the remote "
            "server), openai, or a plugin's id. The GPU rig is switched with rig.enabled."),
    _s("tts_engine", "str", "auto", level="basic", env=("DISPATCH_VOICE_TTS",), pattern=ENGINE_RE.pattern,
       max_len=32, group="tts", label="Text-to-speech engine",
       help="auto, chatterbox, minimax, or the name a plugin registered (kokoro)."),
    _s("builtin_voice", "str", "", level="basic", env=("DISPATCH_VOICE_BUILTIN_VOICE", "DISPATCH_VOICE_KOKORO_VOICE"),
       aliases=("kokoro_voice",), pattern=SLUG_RE.pattern, max_len=64, patchable=False, group="voices",
       label="Default built-in voice", help="Changed with default_voice (it also accepts a cloned-voice profile)."),
    _s("speed", "float", 1.05, level="basic", lo=SPEED_MIN, hi=SPEED_MAX,
       env=("DISPATCH_VOICE_SPEED", "DISPATCH_VOICE_KOKORO_SPEED"), aliases=("kokoro_speed",),
       group="voices", label="Speaking speed", help="Engines that support a speed (a cloned voice keeps its own pace)."),
    _s("rp_speech", "enum", "all", level="basic", choices=RP_MODES, group="voices", label="Role-play speech"),
    _s("narrator_voice", "voice_id", "", group="voices", label="Narrator voice",
       help="A voice id (profile or builtin:<id>); empty = the character's voice."),
    _s("expressive_tags", "bool", True, group="voices", label="Expressive tags",
       help="A cloned voice may render [laugh] / [sigh]; off removes every bracket tag."),
    _s("gateway_trailer", "enum", "full", choices=GATEWAY_TRAILERS, group="voices", label="Gateway trailer"),
    _s("tts_threads", "int", 8, lo=1, hi=64, store="env", env=("DISPATCH_VOICE_TTS_THREADS",), group="tts",
       label="ONNX threads (speech synthesis)", note="read once, by the settings loader"),
    _s("hq_url", "url", "", level="basic", env=("DISPATCH_VOICE_HQ_URL",), group="tts",
       label="OpenAI-compatible speech server",
       help="A server with POST /v1/audio/speech; the reply text is sent to it, and any failure falls back to "
            "this computer's voice."),
    _s("hq_model", "str", "", env=("DISPATCH_VOICE_HQ_MODEL",), max_len=96, group="tts", label="Speech server model"),
    _s("hq_timeout_s", "float", 4.0, lo=0.5, hi=60.0, env=("DISPATCH_VOICE_HQ_TIMEOUT",),
       group="tts", label="Speech server timeout (s)"),
    _s("hq_api_key", "secret", "", store="env", secret=True, env=("DISPATCH_VOICE_HQ_API_KEY",), group="tts",
       label="Speech server API key", help="Only ever read from the environment."),
    _s("minimax_model", "str", "speech-2.8-turbo", store="env", env=("DISPATCH_VOICE_MINIMAX_MODEL",), group="tts",
       label="MiniMax model"),
    _s("minimax_voice", "str", "Wise_Woman", store="env", env=("DISPATCH_VOICE_MINIMAX_VOICE",), group="tts",
       label="MiniMax voice"),
    _s("minimax_base_url", "url", "https://api.minimax.io", store="env", schemes=("https", "http"),
       env=("DISPATCH_VOICE_MINIMAX_BASE_URL",), group="tts", label="MiniMax address"),
    _s("minimax_api_key", "secret", "", store="env", secret=True, env=("MINIMAX_API_KEY",), group="tts",
       label="MiniMax API key", help="Only ever read from the environment."),

    # ---- 3. input mode, turn taking, devices ------------------------------- #
    _s("merge_window_s", "float", 1.8, level="basic", lo=MERGE_WINDOW_MIN, hi=MERGE_WINDOW_MAX,
       env=("DISPATCH_VOICE_MERGE_WINDOW",), resettable=True, group="input", label="Turn merge window (s)"),
    _s("hard_silence_s", "float", 1.4, lo=HARD_SILENCE_MIN, hi=HARD_SILENCE_MAX, aliases=("rt.hard_silence_s",),
       env=("DISPATCH_VOICE_HARD_SILENCE",), resettable=True, field="turn_hard_silence_s", group="input",
       label="Hard silence (s)", help="Silence that ends a turn regardless of the turn-taking model."),
    _s("voice_turn_thinking", "enum", "off", choices=THINKING_LEVELS, env=("DISPATCH_VOICE_THINKING",),
       env_loose=True, resettable=True, group="input", label="Thinking during voice turns"),
    _s("no_smart_turn_silence_s", "float", 0.8, lo=0.1, hi=5.0, store="env", env=("DISPATCH_VOICE_SILENCE",),
       field="no_smart_turn_silence_s", group="input", label="End-of-turn silence without Smart Turn (s)"),
    _s("patience", "float", 1.0, lo=0.5, hi=3.0, store="env", env=("DISPATCH_VOICE_PATIENCE",), group="input",
       label="Server patience multiplier", note="per-device `patience` overrides it"),
    _s("barge_abort_turn", "bool", True, store="env", env=("DISPATCH_VOICE_BARGE_ABORT",), group="input",
       label="Barge-in also aborts the agent turn", note="per-device `bargeIn` switches barge-in itself"),
    _s("rt.probe_silence_s", "float", 0.25, lo=0.2, hi=0.5, out_of_range="clamp", group="advanced_timing",
       label="Pause that triggers the turn model (s)"),
    _s("rt.hold_min_s", "float", 0.3, lo=0.2, hi=1.0, out_of_range="clamp", group="advanced_timing",
       label="Learned-hold minimum (s)"),
    _s("rt.hold_max_s", "float", 1.2, lo=0.5, hi=2.0, out_of_range="clamp", group="advanced_timing",
       label="Learned-hold maximum (s)"),
    _rt_float("addendum_s", 3.0, 1.0, 5.0, label="'Oh, and...' merge window (s)"),
    _rt_float("echo_guard_s", 1.5, 0.5, 5.0, label="No optimistic duck in a reply's first seconds"),
    _rt_float("false_interrupt_timeout_s", 2.0, 1.0, 5.0, label="Resume after a false interruption (s)"),
    _rt_float("quiet_max_s", 2400.0, 600.0, 14400.0, label="Server bound on an idle chat (s)"),
    _rt_int("speculate_max_per_turn", 2, 1, 3, label="Speculations per turn"),
    _rt_float("speculate_min_gap_s", 3.0, 1.0, 10.0, label="Gap between speculations (s)"),
    _rt_bool("preemptive_tts", False, label="Start speaking before the reply is complete"),
    _rt_float("decoder_prompt_s", 4.0, 2.0, 10.0, label="Voice prompt length (s)"),
    _rt_int("first_chunk_tokens", 25, 10, 40, label="First speech chunk (tokens)"),
    _rt_int("decoder_sessions", 1, 1, 2, label="Decoder sessions"),
    _s("rt.power_hold", "enum", "performance", choices=POWER_HOLD_MODES, aliases=("power_hold",),
       env=("DISPATCH_VOICE_POWER_HOLD",), field="power_hold", group="system", label="Power-profile hold",
       help="Hold the performance power profile while a chat is open (Linux only)."),
    _s("opus", "enum", "auto", choices=OPUS_MODES, env=("DISPATCH_VOICE_OPUS",), group="system", label="Opus codec",
       note="needs libopus on the server"),
    _s("opus_down_kbps", "enum", 24, choices=OPUS_DOWN_KBPS, group="system", label="Opus downlink (kbps)"),
    _s("opus_up_kbps", "enum", 16, choices=OPUS_UP_KBPS, group="system", label="Opus uplink (kbps)"),
    _s("max_sessions", "int", MAX_SESSIONS_DEFAULT, lo=MAX_SESSIONS_MIN, hi=MAX_SESSIONS_MAX,
       env=("DISPATCH_VOICE_MAX_SESSIONS",), env_clamp=True, restart=True, group="system",
       label="Voice chats at once"),
    _s("prewarm", "bool", True, store="env", env=("DISPATCH_VOICE_PREWARM",), group="system",
       label="Load the models at start-up"),

    # ---- SpeechForge rig --------------------------------------------------- #
    _s("rig.enabled", "bool", False, level="basic", env=("DISPATCH_VOICE_RIG_ENABLED",), group="connection",
       label="Use the GPU rig"),
    _s("rig.url", "url", "", level="basic", env=("DISPATCH_VOICE_RIG_URL",), max_len=URL_MAX, group="connection",
       schemes=("ws", "wss"),
       label="Rig address", help="ws:// or wss:// address of the speech server. Checked against a host allow-list."),
    _s("rig.stt_final_deadline_ms", "int", 350, lo=100, hi=1000, env=("DISPATCH_VOICE_RIG_STT_FINAL_DEADLINE_MS",),
       out_of_range="clamp", group="connection", label="Rig: final-transcript deadline (ms)"),
    _s("rig.tts_start_deadline_ms", "int", 800, lo=200, hi=2000, env=("DISPATCH_VOICE_RIG_TTS_START_DEADLINE_MS",),
       out_of_range="clamp", group="connection", label="Rig: first-audio deadline (ms)"),
    _s("rig.ahead_ms", "int", 3000, lo=1000, hi=10000, env=("DISPATCH_VOICE_RIG_AHEAD_MS",), out_of_range="clamp",
       group="connection", label="Rig: audio rendered ahead (ms)"),
    _s("rig.first_chunk_tokens", "int", 15, lo=4, hi=100, env=("DISPATCH_VOICE_RIG_FIRST_CHUNK_TOKENS",),
       out_of_range="clamp", group="connection", label="Rig: first chunk (tokens)"),
    _s("rig.allow", "list", [], env=("DISPATCH_VOICE_RIG_ALLOW",), group="connection", item_pattern=HOSTPAT_RE.pattern,
       label="Rig address allow-list",
       help="Hosts, *.suffixes or CIDRs a PUBLIC rig address may be. Private, tailnet and loopback addresses "
            "need no entry; link-local and metadata addresses are never allowed."),
    _s("rig.token", "secret", "", level="basic", store="credential", secret=True, env=("SPEECHFORGE_TOKEN",),
       patchable=False, group="connection", label="Rig token",
       help="Write-only: saved to a private file, never shown, never logged."),

    # ---- plugins, models, paths ------------------------------------------- #
    _s("extensions", "list", [], level="basic", env=("DISPATCH_VOICE_EXTENSIONS",),
       restart=True, group="models", label="Enabled provider plugins",
       help="Entry-point names from the dispatch_voice.providers group (or legacy module:setup targets)."),
    _s("models_root", "str", "", store="env", env=("DISPATCH_SPEECH_MODELS",), patchable=False, group="models",
       label="Speech-model folder", note="legacy name; the unified folder is DisPatch's models_dir"),
    _s("chatterbox_models", "str", "", store="env", env=("DISPATCH_VOICE_MODELS",), patchable=False, group="models",
       label="Chatterbox model folder", note="legacy name; now a subfolder of the unified models folder"),
    _s("model_revision", "str", "", store="env", env=("DISPATCH_VOICE_MODEL_REVISION",), patchable=False,
       group="models", label="Chatterbox model revision"),
    _s("data_dir", "str", "", store="env", env=("DISPATCH_DATA_DIR", "XDG_DATA_HOME"), patchable=False,
       group="models", label="Data folder", note="XDG_DATA_HOME/local-chat when DISPATCH_DATA_DIR is unset"),
    _s("ack_render", "bool", True, store="env", env=("DISPATCH_VOICE_ACK_RENDER",), group="voices",
       label="Pre-render acknowledgement lines"),

    # ---- per bot ----------------------------------------------------------- #
    _s("bots", "map", {}, scope="bot", level="basic", group="voices", label="Per-character voice behaviour",
       help="{bot id: {backchannel, follow_up, locked_voice}}"),
    _s("bots.*.backchannel", "bool", False, scope="bot", store="bot", group="voices", label="Back-channel ('mm-hm')"),
    _s("bots.*.follow_up", "bool", True, scope="bot", store="bot", group="voices", label="Follow-up question",
       note="on unless the bot's backend is the advisor"),
    _s("bots.*.locked_voice", "voice_id", "", scope="bot", store="bot", group="voices",
       label="Voice on a locked device", help="'' = the same as the standard voice."),
    _s("bot.voice", "voice_id", "", scope="bot", store="bot", patchable=False, group="voices", label="Bot voice",
       note="a DisPatch bot field, validated by this add-on"),
    _s("bot.voice_rp", "enum", "", scope="bot", store="bot", choices=("",) + RP_MODES, patchable=False,
       group="voices", label="Bot role-play speech", note="a DisPatch bot field"),

    # ---- per device (the browser's dispatch-drive blob) -------------------- #
    _s("device.patience", "enum", "normal", scope="device", store="device", level="basic", choices=tuple(PATIENCE),
       patchable=False, group="input", label="Patience", help="How long the server waits after you stop talking."),
    _s("device.bargeIn", "bool", True, scope="device", store="device", level="basic", patchable=False,
       group="input", label="Talking over a reply stops it"),
    _s("device.interruptSensitivity", "enum", "normal", scope="device", store="device", choices=INTERRUPT_SENSITIVITIES,
       patchable=False, group="input", label="Interrupt sensitivity"),
    _s("device.autoListen", "bool", True, scope="device", store="device", level="basic", patchable=False,
       group="input", label="Hands-free (off = push to talk)"),
    _s("device.typedInput", "bool", False, scope="device", store="device", level="basic", patchable=False,
       group="input", label="Type and speak (replies spoken, you type)"),
    _s("device.micId", "str", "", scope="device", store="device", level="basic", patchable=False, max_len=256,
       group="input", label="Microphone"),
    _s("device.sinkId", "str", "", scope="device", store="device", level="basic", patchable=False, max_len=256,
       group="input", label="Speaker"),
    _s("device.replyStyle", "enum", "normal", scope="device", store="device", choices=REPLY_STYLES, patchable=False,
       group="voices", label="Reply length"),
    _s("device.followUp", "bool", None, scope="device", store="device", patchable=False, group="voices",
       label="Follow-up question (this device)", note="null = the character's default"),
    _s("device.sounds", "bool", True, scope="device", store="device", patchable=False, group="input", label="Tones"),
    _s("device.screenOn", "bool", True, scope="device", store="device", patchable=False, group="input",
       label="Keep the screen on"),
    _s("device.captions", "bool", True, scope="device", store="device", patchable=False, group="input",
       label="Captions"),
    _s("device.showTimings", "bool", False, scope="device", store="device", patchable=False, group="input",
       label="Show latency numbers"),
    _s("device.reactions", "enum", "sometimes", scope="device", store="device", choices=REACTIONS, patchable=False,
       group="voices", label="Reactions"),
    _s("device.slowLink", "enum", "phone", scope="device", store="device", choices=SLOW_LINK, patchable=False,
       group="input", label="When the connection is slow"),
    _s("device.phoneVoice", "str", "", scope="device", store="device", patchable=False, max_len=256,
       group="input", label="Phone voice"),
    _s("device.optimisticDuck", "bool", True, scope="device", store="device", patchable=False, group="input",
       label="Lower the reply as soon as you speak"),
    _s("device.hangupQuietMin", "enum", 15, scope="device", store="device", choices=HANGUP_QUIET_MIN,
       patchable=False, group="input", label="Hang up when quiet (min, 0 = never)"),
    _s("device.learned", "map", None, scope="device", store="device", patchable=False, group="input",
       label="Learned pause/hold", note="written by the server's learner"),
)

BY_ID: dict[str, Setting] = {s.id: s for s in SETTINGS}
assert len(BY_ID) == len(SETTINGS), "duplicate setting id"

#: Settings that live in voice.yaml / the API tree (scope server, persisted).
FILE_SETTINGS: tuple[Setting, ...] = tuple(
    s for s in SETTINGS if s.store == "file" and s.scope == "server" and not s.id.startswith(("bots.", "bot.", "device.")))
#: Ids that are subtrees of the file (``rt``, ``rig``) and the maps.
TREES = ("rt", "rig")

#: Every dotted id or alias a file/PATCH may use -> its canonical setting.
BY_NAME: dict[str, Setting] = {}
for _s_ in FILE_SETTINGS:
    BY_NAME[_s_.id] = _s_
for _s_ in FILE_SETTINGS:
    for _a in _s_.aliases:
        BY_NAME.setdefault(_a, _s_)
del _s_, _a

#: voice.yaml keys written by 2.1.
LEGACY_FILE_KEYS = {"kokoro_voice": "builtin_voice", "kokoro_speed": "speed"}


def setting(id_: str) -> Setting:
    return BY_ID[id_]


def rt_ranges() -> dict[str, tuple[float, float]]:
    """``(lo, hi)`` of every numeric ``rt.*`` key plus ``hard_silence_s`` (what
    ``rt/settings.py`` clamps with)."""
    out = {s.id[3:]: (s.lo, s.hi) for s in SETTINGS
           if s.id.startswith("rt.") and s.type in ("int", "float") and s.lo is not None}
    h = BY_ID["hard_silence_s"]
    out["hard_silence_s"] = (h.lo, h.hi)
    return out


def rig_ranges() -> dict[str, tuple[int, int]]:
    return {s.id[4:]: (int(s.lo), int(s.hi)) for s in SETTINGS
            if s.id.startswith("rig.") and s.type == "int"}


def rt_keys() -> tuple[str, ...]:
    """The keys of the ``rt:`` tree (contracts section F.1)."""
    return tuple(s.id[3:] for s in FILE_SETTINGS if s.id.startswith("rt."))


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _label(s: Setting) -> str:
    return s.id


def check_url(v: str, *, schemes: tuple[str, ...] = ("ws", "wss", "http", "https")) -> bool:
    """Shape only (scheme, host, no credentials/query/fragment, <= 256 chars); WHICH hosts
    are acceptable is :mod:`host.netpolicy`'s call."""
    from urllib.parse import urlsplit
    if not isinstance(v, str) or not v or len(v) > URL_MAX or any(c.isspace() for c in v):
        return False
    try:
        u = urlsplit(v)
        _ = u.port
    except ValueError:
        return False
    return (u.scheme in schemes and bool(u.hostname) and not u.username and not u.password
            and not u.query and not u.fragment)


def coerce(s: Setting, v: Any, *, strict: bool = False, env: bool = False) -> Any:
    """``v`` as this setting's value, or :data:`MISSING` when it is unusable (lenient).
    ``strict`` raises :class:`SettingError` with the reason instead (a PATCH).

    Lenient reading of a number outside its range follows ``out_of_range``: dropped, or
    clamped (the ``rt.*`` and ``rig.*`` deadlines). ``env=True`` reads an environment
    string: it is parsed to the type first, and a range is NOT enforced (the app limits
    what a person can set, not what an operator can) unless ``env_clamp``."""
    name = _label(s)

    def bad(msg: str) -> Any:
        if strict:
            raise SettingError(msg)
        return MISSING

    t = s.type
    if env:
        if not isinstance(v, str):
            return MISSING
        raw = v.strip()
        if s.env_parse is not None:
            return s.env_parse(raw)
        if t == "bool":
            return raw.lower() in FORCE_ON
        if t in ("int", "float"):
            try:
                n = float(raw) if t == "float" else int(raw)
            except ValueError:
                return MISSING
            if s.env_clamp and s.lo is not None:
                n = min(s.hi, max(s.lo, n))
            return n
        if t == "list":
            return _env_list(raw)[:LIST_MAX]
        if t == "enum":
            low = raw.lower()
            if s.choices and all(isinstance(c, int) for c in s.choices):
                try:
                    return int(low) if int(low) in s.choices else MISSING
                except ValueError:
                    return MISSING
            if s.choices and low not in s.choices and not s.env_loose:
                return MISSING
            return low
        t = "str" if t in ("secret", "url", "voice_id") else t
        if t == "str":
            return raw.lower() if s.id in ("tts_engine", "stt_provider", "tts_provider") else raw
        return MISSING

    if t == "bool":
        if isinstance(v, bool):
            return v
        return bad(f"{name} must be true or false")
    if t in ("int", "float"):
        if not _is_num(v):
            return bad(f"{name} must be a number" if t == "float" else f"{name} must be a whole number")
        if t == "int":
            if isinstance(v, float) and not v.is_integer():
                if strict or s.out_of_range == "drop":
                    return bad(f"{name} must be a whole number between {int(s.lo)} and {int(s.hi)}")
            n: float = int(v)
        else:
            n = float(v)
        if s.lo is not None:
            if not s.lo <= n <= s.hi:
                if strict or s.out_of_range == "drop":
                    if t == "int":
                        return bad(f"{name} must be a whole number between {int(s.lo)} and {int(s.hi)}")
                    return bad(f"{name} must be between {s.lo} and {s.hi}")
                n = min(s.hi, max(s.lo, n))
        return round(float(n), 3) if t == "float" else int(n)
    if t == "enum":
        want = v
        if isinstance(v, str):
            want = v.strip().lower()
            if s.choices and all(isinstance(c, int) for c in s.choices):
                want = v
        if isinstance(want, bool):
            # YAML reads a hand-edited `off` as false and `on` as true.
            if s.id in ("rt.power_hold", "opus") and not strict:
                return s.choices[0] if want else "off"
            return bad(f"{name} must be one of {', '.join(map(str, s.choices))}")
        if want in s.choices:
            return want
        if isinstance(want, float) and want.is_integer() and int(want) in s.choices:
            return int(want)
        return bad(f"{name} must be one of {', '.join(map(str, s.choices))}")
    if t in ("str", "secret", "voice_id"):
        if v is None and t == "voice_id":
            return ""
        if not isinstance(v, str):
            return bad(f"{name} must be text")
        v = v.strip()
        if t == "voice_id":
            if v == "":
                return ""
            if VOICE_ID_RE.match(v):
                return v
            return bad(f"Unknown voice: {v[:80]}")
        if s.max_len is not None and len(v) > s.max_len:
            return bad(f"{name} must be at most {s.max_len} characters")
        if s.pattern and v:
            if s.id in ("tts_engine", "stt_provider", "tts_provider"):
                v = v.lower()
            if not re.match(s.pattern, v):
                return bad(f"{name} has characters that are not allowed")
        if s.id in ("tts_engine", "stt_provider", "tts_provider") and not v:
            return bad(f"{name} must not be empty")
        return v
    if t == "url":
        if v is None or v == "":
            return ""
        if not isinstance(v, str) or not check_url(v.strip(), schemes=s.schemes):
            if s.id == "rig.url":
                return bad("rig.url must be a ws:// or wss:// address with no credentials or query "
                           "(the token stays out of URLs)")
            return bad(f"{name} must be an address with no credentials or query")
        return v.strip()
    if t == "list":
        if not isinstance(v, (list, tuple)):
            return bad(f"{name} takes a list of names")
        out: list[str] = []
        for item in v:
            pat = re.compile(s.item_pattern) if s.item_pattern else PLUGIN_RE
            if not isinstance(item, str) or not pat.match(item.strip()):
                if strict:
                    what = "host names, *.suffixes or CIDRs" if s.item_pattern else "plugin names or module:setup targets"
                    raise SettingError(f"{name} takes a list of {what}")
                continue
            if item.strip() not in out:
                out.append(item.strip())
        if len(out) > LIST_MAX:
            return bad(f"{name} takes at most {LIST_MAX} entries")
        return out
    return bad(f"{name} cannot be set here")


def _dig(raw: Mapping, parts: tuple[str, ...]) -> Any:
    cur: Any = raw
    for p in parts:
        if not isinstance(cur, Mapping) or p not in cur:
            return MISSING
        cur = cur[p]
    return cur


def _put(out: dict, parts: tuple[str, ...], value: Any) -> None:
    cur = out
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _raw_value(raw: Mapping, s: Setting) -> Any:
    """The raw value for ``s`` in ``raw``: its canonical path first, then each alias
    (the canonical spelling wins when both are present)."""
    v = _dig(raw, s.parts)
    if v is not MISSING:
        return v
    for a in s.aliases:
        v = _dig(raw, tuple(a.split(".")))
        if v is not MISSING:
            return v
    for old, new in LEGACY_FILE_KEYS.items():
        if new == s.id and old in raw:
            return raw[old]
    return MISSING


# --------------------------------------------------------------------------- #
# bots: the one map-shaped setting
# --------------------------------------------------------------------------- #

def clean_voice_id(v: Any) -> str | None:
    """A stored voice id ("" = none), or None when the value is not one."""
    if v is None or v == "":
        return ""
    if isinstance(v, str) and VOICE_ID_RE.match(v.strip()):
        return v.strip()
    return None


def clean_bots(raw: Any) -> dict:
    """``bots:`` validated: ``{bot id: {backchannel?, follow_up?, locked_voice?}}``; unknown
    keys and non-boolean flags dropped, an empty entry gone. ``locked_voice`` "" is kept
    (it is how a PATCH clears the override)."""
    out: dict = {}
    if not isinstance(raw, Mapping):
        return out
    for bot_id, conf in raw.items():
        if not isinstance(bot_id, str) or not BOT_ID_RE.match(bot_id) or not isinstance(conf, Mapping):
            continue
        entry = {k: conf[k] for k in BOT_KEYS if isinstance(conf.get(k), bool)}
        lv = clean_voice_id(conf.get(BOT_LOCKED_VOICE)) if BOT_LOCKED_VOICE in conf else None
        if lv is not None:
            entry[BOT_LOCKED_VOICE] = lv
        if entry:
            out[bot_id] = entry
    return out


def validate_bots(raw: Any, *, known: Mapping[str, Any] | None = None,
                  voice_ok: Callable[[str], bool] | None = None) -> dict:
    """The strict ``bots`` PATCH value: a known bot (when ``known`` lists any), boolean
    flags, and ``locked_voice`` (a voice id ``voice_ok`` accepts, or "")."""
    if not isinstance(raw, Mapping):
        raise SettingError("bots takes an object of {bot id: {backchannel, follow_up}}")
    raw = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in raw.items()}
    for bot_id, conf in raw.items():
        if not isinstance(bot_id, str) or not bot_id:
            raise SettingError("bots takes an object of {bot id: {backchannel, follow_up}}")
        if known and bot_id not in known:
            raise SettingError(f"Unknown bot: {bot_id[:40]}")
        if not isinstance(conf, dict):
            raise SettingError(f"bots.{bot_id[:40]} takes an object")
        extra = set(conf) - set(BOT_KEYS) - {BOT_LOCKED_VOICE}
        if extra:
            raise SettingError(f"Unknown setting: bots.{bot_id[:40]}.{min(extra)}")
        if BOT_LOCKED_VOICE in conf:
            lv = conf[BOT_LOCKED_VOICE]
            if lv is None or lv == "":
                conf[BOT_LOCKED_VOICE] = ""
            elif (not isinstance(lv, str) or clean_voice_id(lv) is None
                  or (voice_ok is not None and not voice_ok(lv))):
                raise SettingError(f"Unknown voice: {str(lv)[:80]}")
        for k, v in conf.items():
            if k != BOT_LOCKED_VOICE and not isinstance(v, bool):
                raise SettingError(f"bots.{bot_id[:40]}.{k} must be true or false")
    return clean_bots(raw)


# --------------------------------------------------------------------------- #
# Generated validators
# --------------------------------------------------------------------------- #

def clean_prefs(raw: Any) -> dict:
    """voice.yaml validated against the schema: only known keys with valid values survive;
    aliases fold into the canonical key (the new spelling wins); the result has the file's
    shape (``rt`` / ``rig`` subtrees, flat otherwise)."""
    out: dict = {}
    if not isinstance(raw, Mapping):
        return out
    for s in FILE_SETTINGS:
        if s.id == "bots":
            continue
        v = _raw_value(raw, s)
        if v is MISSING:
            continue
        if s.id == "rig.url" and (v == "" or v is None):
            _put(out, s.parts, "")
            continue
        if s.id in ("tts_engine", "stt_provider", "tts_provider") and not isinstance(v, str):
            continue
        c = coerce(s, v)
        if c is MISSING:
            continue
        _put(out, s.parts, c)
    if "bots" in raw:
        bots = clean_bots(raw.get("bots"))
        if bots or raw.get("bots") == {}:
            out["bots"] = bots
    return out


def clean_subtree(tree: str, raw: Any) -> dict:
    """``rt`` or ``rig`` alone, validated (what ``clean_rt`` / ``clean_rig`` were)."""
    got = clean_prefs({tree: raw}) if isinstance(raw, Mapping) else {}
    return got.get(tree, {})


def _unknown(raw: Mapping, tree: str | None) -> None:
    prefix = f"{tree}." if tree else ""
    names = {s.id[len(prefix):].split(".")[0] for s in FILE_SETTINGS if s.id.startswith(prefix)}
    if tree:
        names |= {a[len(prefix):] for s in FILE_SETTINGS for a in s.aliases if a.startswith(prefix)}
    for k in sorted(raw, key=str):
        if k not in names:
            raise SettingError(f"Unknown setting: {prefix}{k}")


def validate_subtree(tree: str, raw: Any) -> dict:
    """Strict ``rt`` / ``rig`` PATCH subtree: unknown key, wrong type or out-of-range value
    -> :class:`SettingError`. Returns the cleaned update."""
    keys = [s.id[len(tree) + 1:] for s in FILE_SETTINGS if s.id.startswith(tree + ".")]
    if not isinstance(raw, Mapping):
        extra = ", ".join(keys)
        raise SettingError(f"{tree} takes an object with: {extra}")
    _unknown(raw, tree)
    out: dict = {}
    for k, v in raw.items():
        s = BY_NAME.get(f"{tree}.{k}")
        if s is None:
            raise SettingError(f"Unknown setting: {tree}.{k}")
        if tree == "rt":
            c = _strict_rt(s, v)
        else:
            c = coerce(s, v.strip() if isinstance(v, str) and s.type == "url" else v, strict=True)
        _put(out, s.parts, c)
    return out


def _strict_rt(s: Setting, v: Any) -> Any:
    """The ``rt.*`` wording (kept from 1.5.0): 'must be a number between lo and hi'."""
    try:
        return coerce(s, v, strict=True)
    except SettingError:
        if s.type == "float" and s.lo is not None:
            raise SettingError(f"{s.id} must be a number between {s.lo} and {s.hi}") from None
        if s.type == "int" and s.lo is not None:
            if _is_num(v) and float(v) != int(float(v)):
                raise SettingError(f"{s.id} must be a whole number between {int(s.lo)} and {int(s.hi)}") from None
            raise SettingError(f"{s.id} must be a number between {int(s.lo)} and {int(s.hi)}") from None
        raise


def validate_patch(body: Mapping, *, extra_keys: frozenset[str] = frozenset()) -> dict:
    """The strict flat/nested update of a ``PATCH /api/voice/settings`` body for every
    schema-managed key. ``extra_keys`` are keys the route handles itself (``default_voice``,
    ``reset``, ``bots``, ``enabled``...) and are skipped here. Raises :class:`SettingError`."""
    out: dict = {}
    for k, v in body.items():
        if k in extra_keys:
            continue
        if k in TREES:
            _merge(out, validate_subtree(k, v))
            continue
        s = BY_NAME.get(k)
        if s is None or not s.patchable or "." in k:
            raise SettingError(f"Unknown setting: {k}")
        c = coerce(s, v, strict=True)
        _put(out, s.parts, c)
    return out


def _merge(dst: dict, src: Mapping) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = dict(v) if isinstance(v, Mapping) else v


def patchable_keys() -> frozenset[str]:
    """Every top-level key a PATCH body may carry (the route adds its own specials)."""
    keys = {s.id for s in FILE_SETTINGS if s.patchable and s.id.split(".")[0] not in TREES}
    keys |= {a for s in FILE_SETTINGS if s.patchable for a in s.aliases if "." not in a}    # power_hold, stt_engine
    keys |= set(TREES)
    return frozenset(keys)


# --------------------------------------------------------------------------- #
# Resolution: env > file > default, with the source of each key
# --------------------------------------------------------------------------- #

def env_value(s: Setting, environ: Mapping[str, str]) -> tuple[Any, str | None]:
    """``(value, env var name)`` of the first environment alias that is set and parses;
    ``(MISSING, None)`` when none does."""
    for name in s.env:
        raw = environ.get(name)
        if raw is None:
            continue
        if s.id == "data_dir" and name == "XDG_DATA_HOME":
            continue                    # a base dir for the default, not an override of the data dir
        if raw.strip() == "" and s.type not in ("str",) and s.env_parse is None:
            continue
        got = coerce(s, raw, env=True)
        if got is not MISSING:
            return got, name
    return MISSING, None


def resolve(prefs: Mapping | None = None, environ: Mapping[str, str] | None = None) -> tuple[dict, dict]:
    """``(values, sources)`` for every server-scope setting, keyed by id. ``prefs`` is the
    CLEANED voice.yaml tree. Order: environment, file, default. ``sources[id]`` is
    ``{"source": "env"|"file"|"default", "env": <name or omitted>}``."""
    environ = os.environ if environ is None else environ
    prefs = prefs or {}
    values: dict = {}
    sources: dict = {}
    for s in SETTINGS:
        if s.scope != "server" or s.store in ("device",):
            continue
        ev, name = env_value(s, environ)
        fv = _dig(prefs, s.parts) if s.store == "file" else MISSING
        if s.secret:
            values[s.id] = ev is not MISSING          # whether it is set; never the value
            sources[s.id] = {"source": "env" if ev is not MISSING else "default",
                             **({"env": name} if name else {})}
        elif ev is not MISSING and s.id != "enabled":
            values[s.id] = ev
            sources[s.id] = {"source": "env", "env": name}
        elif ev is not MISSING and s.id == "enabled":
            # DISPATCH_VOICE forces the switch; the stored switch stays what the file says.
            values[s.id] = fv if fv is not MISSING else s.default
            sources[s.id] = {"source": "env", "env": name, "forced": bool(ev)}
        elif fv is not MISSING:
            values[s.id] = fv
            sources[s.id] = {"source": "file"}
        else:
            values[s.id] = s.default
            sources[s.id] = {"source": "default"}
    if "bots" in prefs:
        values["bots"] = prefs["bots"]
        sources["bots"] = {"source": "file"}
    return values, sources


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #

def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    return v


def schema_json() -> dict:
    """``GET /api/voice/settings/schema``: every key with its metadata. Secrets carry
    no value, ever (there is no value here at all: this is the schema)."""
    items = []
    for s in SETTINGS:
        d = {"id": s.id, "type": s.type, "default": _jsonable(s.default), "scope": s.scope, "level": s.level,
             "store": s.store, "secret": s.secret, "group": s.group, "label": s.label, "help": s.help,
             "env": list(s.env), "aliases": list(s.aliases), "patchable": s.patchable,
             "restart_required": s.restart, "resettable": s.resettable}
        if s.lo is not None:
            d["min"], d["max"] = s.lo, s.hi
        if s.choices:
            d["choices"] = list(s.choices)
        if s.note:
            d["note"] = s.note
        items.append(d)
    return {"version": 1, "scopes": list(SCOPES), "levels": list(LEVELS),
            "groups": ["stt", "tts", "input", "voices", "connection", "models", "system", "advanced_timing"],
            "settings": items}


def env_table() -> str:
    """The README's environment table, generated (a test fails when the README differs)."""
    rows = ["| Environment variable | Setting | Default | Notes |", "|---|---|---|---|"]
    for s in SETTINGS:
        if not s.env or s.scope == "bot":
            continue
        for i, name in enumerate(s.env):
            default = "(unset)" if s.secret else ("-" if s.default in (None, "", []) else str(s.default))
            note = s.help or s.label
            if i:
                note = f"older name of `{s.env[0]}`" if name.startswith(ENV_PREFIX) else note
            rows.append(f"| `{name}` | `{s.id}` | {default} | {note.replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def _readme_block() -> str:
    return "<!-- env-table:start -->\n" + env_table() + "<!-- env-table:end -->"


def main(argv: list[str] | None = None) -> int:
    """``python -m dispatch_voice.host.settings_schema [--write-readme | --print]``: print the
    environment table, or rewrite the README between its markers."""
    import argparse
    import re
    from pathlib import Path
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--write-readme", action="store_true")
    a = ap.parse_args(argv)
    if not a.write_readme:
        print(env_table(), end="")
        return 0
    readme = Path(__file__).resolve().parents[3] / "README.md"
    text = readme.read_text(encoding="utf-8")
    new, n = re.subn(r"<!-- env-table:start -->\n.*?<!-- env-table:end -->", lambda m: _readme_block(), text,
                     flags=re.S)
    if not n:
        print("README.md has no <!-- env-table:start --> / <!-- env-table:end --> markers")
        return 1
    readme.write_text(new, encoding="utf-8", newline="")
    print("README.md environment table updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
