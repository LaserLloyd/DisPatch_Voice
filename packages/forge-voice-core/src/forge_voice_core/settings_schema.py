"""Declarative settings: the primitives, the engine-level keys, and the generic :class:`Schema`.

A *setting* is one :class:`Setting` row: its type, default, range, scope, secrecy and environment
aliases. Everything a host needs is generated from a table of rows:

* :meth:`Schema.clean_prefs` / :meth:`Schema.clean_subtree` - what a settings file may hold (unknown
  keys dropped, a bad value dropped or clamped);
* :meth:`Schema.validate_patch` - the strict check of an update (``SettingError`` carries the reason);
* :meth:`Schema.resolve` - the value of every key with its source: **environment beats the file
  beats the default**;
* :meth:`Schema.schema_json` - the rows as JSON, for a Settings page to render;
* :meth:`Schema.env_table` - the environment table for a README.

This module owns the PRIMITIVES (the row type, coercion, URL shape, environment parsing, the
vocabularies the engines share) and the ENGINE-LEVEL rows (:data:`ENGINE_SETTINGS`: speech-to-text,
text-to-speech, the realtime ``rt.*`` tuning and the optional speech rig's ``rig.*``), because the
engines in this package read and clamp them (``rt/settings.py``, ``rig.py``). A host adds its own
rows (an on/off switch, its bots, its devices) and builds a :class:`Schema` over both; DisPatch Voice
does exactly that in ``dispatch_voice/host/settings_schema.py``.

Stdlib only, and nothing here imports the rest of the package.

The environment names are the ones DisPatch Voice shipped (``DISPATCH_VOICE_*``); a host that wants
its own prefix passes ``Schema(..., env_prefix="CHATFORGE_VOICE_")``, which renames every
``DISPATCH_VOICE_``-prefixed name of the rows it is given (the unprefixed ones, such as
``MINIMAX_API_KEY`` and ``SPEECHFORGE_TOKEN``, are left alone).
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any

SCOPES = ("server", "device", "bot")
LEVELS = ("basic", "advanced")
#: Where a value is kept. ``file`` = the host's settings file; ``env`` = environment only;
#: ``device`` = the browser's blob; ``bot`` = the host's bot record; ``credential`` = a credential
#: store (never a setting value).
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
    #: when READ from a settings file. A PATCH is always refused, never clamped.
    out_of_range: str = "drop"
    choices: tuple = ()
    secret: bool = False
    #: Environment names, highest priority first. ``DISPATCH_VOICE_*`` unless full.
    env: tuple[str, ...] = ()
    #: Also accepted in files and PATCH bodies (dotted), folded into ``id``.
    aliases: tuple[str, ...] = ()
    #: Applying a new value needs a restart of the host.
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
# Shared vocabularies
# --------------------------------------------------------------------------- #

SPEED_MIN, SPEED_MAX = 0.8, 1.3
HARD_SILENCE_MIN, HARD_SILENCE_MAX = 0.8, 3.0
POWER_HOLD_MODES = ("performance", "off")
STT_ENGINES = ("parakeet", "moonshine")
SPECULATE_MODES = ("advisor", "all", "off")
VOICE_ID_MAX = 80
URL_MAX = 256
FORCE_ON = ("1", "true", "on", "yes")

VOICE_ID_RE = re.compile(rf"\A[A-Za-z0-9][A-Za-z0-9:._-]{{0,{VOICE_ID_MAX - 1}}}\Z")
SLUG_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
ENGINE_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,31}\Z")
#: A plugin target (``pkg.module:setup``) or an entry-point name.
PLUGIN_RE = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.:-]{0,95}\Z")
HOSTPAT_RE = re.compile(r"\A[A-Za-z0-9_*][A-Za-z0-9_.*:/\[\]-]{0,127}\Z")
LIST_MAX = 16

#: The settings-file keys written by DisPatch Voice 2.1 (the built-in voice was always Kokoro).
LEGACY_FILE_KEYS = {"kokoro_voice": "builtin_voice", "kokoro_speed": "speed"}


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
# THE ENGINE-LEVEL ROWS (what the engines in this package read)
# --------------------------------------------------------------------------- #

ENGINE_SETTINGS: tuple[Setting, ...] = (
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
    _s("hard_silence_s", "float", 1.4, lo=HARD_SILENCE_MIN, hi=HARD_SILENCE_MAX, aliases=("rt.hard_silence_s",),
       env=("DISPATCH_VOICE_HARD_SILENCE",), resettable=True, field="turn_hard_silence_s", group="input",
       label="Hard silence (s)", help="Silence that ends a turn regardless of the turn-taking model."),
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
)

ENGINE_BY_ID: dict[str, Setting] = {s.id: s for s in ENGINE_SETTINGS}
assert len(ENGINE_BY_ID) == len(ENGINE_SETTINGS), "duplicate engine setting id"


def engine(id_: str) -> Setting:
    """The engine-owned row ``id_`` (a host builds its table around these)."""
    return ENGINE_BY_ID[id_]


def rt_ranges(table: Sequence[Setting] = ENGINE_SETTINGS) -> dict[str, tuple[float, float]]:
    """``(lo, hi)`` of every numeric ``rt.*`` key plus ``hard_silence_s`` (what
    ``rt/settings.py`` clamps with)."""
    by_id = {s.id: s for s in table}
    out = {s.id[3:]: (s.lo, s.hi) for s in table
           if s.id.startswith("rt.") and s.type in ("int", "float") and s.lo is not None}
    h = by_id["hard_silence_s"]
    out["hard_silence_s"] = (h.lo, h.hi)
    return out


def rig_ranges(table: Sequence[Setting] = ENGINE_SETTINGS) -> dict[str, tuple[int, int]]:
    return {s.id[4:]: (int(s.lo), int(s.hi)) for s in table
            if s.id.startswith("rig.") and s.type == "int"}


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _label(s: Setting) -> str:
    return s.id


def check_url(v: str, *, schemes: tuple[str, ...] = ("ws", "wss", "http", "https")) -> bool:
    """Shape only (scheme, host, no credentials/query/fragment, <= 256 chars); WHICH hosts
    are acceptable is :mod:`netpolicy`'s call."""
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


def clean_voice_id(v: Any) -> str | None:
    """A stored voice id ("" = none), or None when the value is not one."""
    if v is None or v == "":
        return ""
    if isinstance(v, str) and VOICE_ID_RE.match(v.strip()):
        return v.strip()
    return None


def _merge(dst: dict, src: Mapping) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = dict(v) if isinstance(v, Mapping) else v


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


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    return v


def setting_row(s: Setting) -> dict:
    """One row of the published schema (no value of any kind: secrets stay secret)."""
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
    return d


#: The groups a Settings page shows, in order.
GROUPS = ["stt", "tts", "input", "voices", "connection", "models", "system", "advanced_timing"]


def schema_document(settings: Sequence[Setting]) -> dict:
    """The published schema of ``settings``: every key with its metadata."""
    return {"version": 1, "scopes": list(SCOPES), "levels": list(LEVELS), "groups": list(GROUPS),
            "settings": [setting_row(s) for s in settings]}


# --------------------------------------------------------------------------- #
# The generic schema
# --------------------------------------------------------------------------- #

class Schema:
    """Validators, resolution and publication generated from one ordered table of rows.

    ``map_cleaners`` names the map-shaped settings (DisPatch: ``bots``) that are validated by their
    own function instead of by :func:`coerce`: ``{id: clean(raw) -> dict}``. ``skip_prefixes`` are the
    row ids that never live in the settings file (per bot / per device). ``trees`` are the dotted
    prefixes that are sub-objects of the file (``rt``, ``rig``)."""

    def __init__(self, settings: Sequence[Setting], *, map_cleaners: Mapping[str, Callable[[Any], dict]] | None = None,
                 skip_prefixes: tuple[str, ...] = ("bots.", "bot.", "device."), trees: tuple[str, ...] = ("rt", "rig"),
                 env_prefix: str | None = None) -> None:
        if env_prefix is not None and env_prefix != ENV_PREFIX:
            settings = [replace(s, env=tuple(env_prefix + n[len(ENV_PREFIX):] if n.startswith(ENV_PREFIX) else n
                                             for n in s.env)) for s in settings]
        self.SETTINGS: tuple[Setting, ...] = tuple(settings)
        self.map_cleaners = dict(map_cleaners or {})
        self.TREES = tuple(trees)
        self.BY_ID: dict[str, Setting] = {s.id: s for s in self.SETTINGS}
        assert len(self.BY_ID) == len(self.SETTINGS), "duplicate setting id"
        #: Settings that live in the settings file / the API tree (scope server, persisted).
        self.FILE_SETTINGS: tuple[Setting, ...] = tuple(
            s for s in self.SETTINGS
            if s.store == "file" and s.scope == "server" and not s.id.startswith(skip_prefixes))
        #: Every dotted id or alias a file/PATCH may use -> its canonical setting.
        self.BY_NAME: dict[str, Setting] = {}
        for s in self.FILE_SETTINGS:
            self.BY_NAME[s.id] = s
        for s in self.FILE_SETTINGS:
            for a in s.aliases:
                self.BY_NAME.setdefault(a, s)

    # ---- lookups -------------------------------------------------------- #
    def setting(self, id_: str) -> Setting:
        return self.BY_ID[id_]

    def rt_ranges(self) -> dict[str, tuple[float, float]]:
        return rt_ranges(self.SETTINGS)

    def rig_ranges(self) -> dict[str, tuple[int, int]]:
        return rig_ranges(self.SETTINGS)

    def rt_keys(self) -> tuple[str, ...]:
        """The keys of the ``rt:`` tree (contracts section F.1)."""
        return tuple(s.id[3:] for s in self.FILE_SETTINGS if s.id.startswith("rt."))

    # ---- generated validators ------------------------------------------- #
    def clean_prefs(self, raw: Any) -> dict:
        """A settings file validated against the schema: only known keys with valid values survive;
        aliases fold into the canonical key (the new spelling wins); the result has the file's
        shape (``rt`` / ``rig`` subtrees, flat otherwise)."""
        out: dict = {}
        if not isinstance(raw, Mapping):
            return out
        for s in self.FILE_SETTINGS:
            if s.id in self.map_cleaners:
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
        for key, cleaner in self.map_cleaners.items():
            if key in raw:
                cleaned = cleaner(raw.get(key))
                if cleaned or raw.get(key) == {}:
                    out[key] = cleaned
        return out

    def clean_subtree(self, tree: str, raw: Any) -> dict:
        """``rt`` or ``rig`` alone, validated (what ``clean_rt`` / ``clean_rig`` were)."""
        got = self.clean_prefs({tree: raw}) if isinstance(raw, Mapping) else {}
        return got.get(tree, {})

    def _unknown(self, raw: Mapping, tree: str | None) -> None:
        prefix = f"{tree}." if tree else ""
        names = {s.id[len(prefix):].split(".")[0] for s in self.FILE_SETTINGS if s.id.startswith(prefix)}
        if tree:
            names |= {a[len(prefix):] for s in self.FILE_SETTINGS for a in s.aliases if a.startswith(prefix)}
        for k in sorted(raw, key=str):
            if k not in names:
                raise SettingError(f"Unknown setting: {prefix}{k}")

    def validate_subtree(self, tree: str, raw: Any) -> dict:
        """Strict ``rt`` / ``rig`` PATCH subtree: unknown key, wrong type or out-of-range value
        -> :class:`SettingError`. Returns the cleaned update."""
        keys = [s.id[len(tree) + 1:] for s in self.FILE_SETTINGS if s.id.startswith(tree + ".")]
        if not isinstance(raw, Mapping):
            extra = ", ".join(keys)
            raise SettingError(f"{tree} takes an object with: {extra}")
        self._unknown(raw, tree)
        out: dict = {}
        for k, v in raw.items():
            s = self.BY_NAME.get(f"{tree}.{k}")
            if s is None:
                raise SettingError(f"Unknown setting: {tree}.{k}")
            if tree == "rt":
                c = _strict_rt(s, v)
            else:
                c = coerce(s, v.strip() if isinstance(v, str) and s.type == "url" else v, strict=True)
            _put(out, s.parts, c)
        return out

    def validate_patch(self, body: Mapping, *, extra_keys: frozenset[str] = frozenset()) -> dict:
        """The strict flat/nested update of a ``PATCH`` body for every schema-managed key.
        ``extra_keys`` are keys the route handles itself and are skipped here. Raises
        :class:`SettingError`."""
        out: dict = {}
        for k, v in body.items():
            if k in extra_keys:
                continue
            if k in self.TREES:
                _merge(out, self.validate_subtree(k, v))
                continue
            s = self.BY_NAME.get(k)
            if s is None or not s.patchable or "." in k:
                raise SettingError(f"Unknown setting: {k}")
            c = coerce(s, v, strict=True)
            _put(out, s.parts, c)
        return out

    def patchable_keys(self) -> frozenset[str]:
        """Every top-level key a PATCH body may carry (the route adds its own specials)."""
        keys = {s.id for s in self.FILE_SETTINGS if s.patchable and s.id.split(".")[0] not in self.TREES}
        keys |= {a for s in self.FILE_SETTINGS if s.patchable for a in s.aliases if "." not in a}
        keys |= set(self.TREES)
        return frozenset(keys)

    # ---- resolution: env > file > default ------------------------------- #
    def resolve(self, prefs: Mapping | None = None, environ: Mapping[str, str] | None = None) -> tuple[dict, dict]:
        """``(values, sources)`` for every server-scope setting, keyed by id. ``prefs`` is the
        CLEANED settings tree. Order: environment, file, default. ``sources[id]`` is
        ``{"source": "env"|"file"|"default", "env": <name or omitted>}``."""
        environ = os.environ if environ is None else environ
        prefs = prefs or {}
        values: dict = {}
        sources: dict = {}
        for s in self.SETTINGS:
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
                # The host's force flag decides the switch; the stored switch stays what the file says.
                values[s.id] = fv if fv is not MISSING else s.default
                sources[s.id] = {"source": "env", "env": name, "forced": bool(ev)}
            elif fv is not MISSING:
                values[s.id] = fv
                sources[s.id] = {"source": "file"}
            else:
                values[s.id] = s.default
                sources[s.id] = {"source": "default"}
        for key in self.map_cleaners:
            if key in prefs:
                values[key] = prefs[key]
                sources[key] = {"source": "file"}
        return values, sources

    # ---- publication ---------------------------------------------------- #
    def schema_json(self) -> dict:
        """Every key with its metadata. Secrets carry no value, ever (there is no value here at
        all: this is the schema)."""
        return schema_document(self.SETTINGS)

    def env_table(self) -> str:
        """The README's environment table, generated (a test fails when the README differs)."""
        rows = ["| Environment variable | Setting | Default | Notes |", "|---|---|---|---|"]
        for s in self.SETTINGS:
            if not s.env or s.scope == "bot":
                continue
            for i, name in enumerate(s.env):
                default = "(unset)" if s.secret else ("-" if s.default in (None, "", []) else str(s.default))
                note = s.help or s.label
                if i:
                    note = f"older name of `{s.env[0]}`" if name.startswith(ENV_PREFIX) else note
                rows.append(f"| `{name}` | `{s.id}` | {default} | {note.replace('|', '/')} |")
        return "\n".join(rows) + "\n"


#: The schema of the engine-level rows alone: what a host with no table of its own gets, and what
#: ``rig.clean_rig`` validates the ``rig`` tree with.
ENGINE_SCHEMA = Schema(ENGINE_SETTINGS)
