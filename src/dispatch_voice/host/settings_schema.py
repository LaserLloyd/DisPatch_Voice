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
``rt.hard_silence_s``.

Since the core extraction the machinery (the row type, coercion, :class:`Schema`) and the engine-level
rows (speech-to-text, text-to-speech, ``rt.*``, ``rig.*``) live in ``forge_voice_core.settings_schema``,
where ``rt/settings.py`` and ``rig.py`` take their ranges from them; this module is THE TABLE of
DisPatch's keys, written in DisPatch's order with ``E("id")`` standing for an engine-level row. It keeps
every name it always exported, so ``host.config``, ``host.routes`` and the tests are unchanged.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from forge_voice_core import settings_schema as _core
from forge_voice_core.settings_schema import (  # noqa: F401  (re-exported: the module's public surface)
    ENGINE_RE,
    ENV_PREFIX,
    FORCE_ON,
    HARD_SILENCE_MAX,
    HARD_SILENCE_MIN,
    HOSTPAT_RE,
    LEGACY_FILE_KEYS,
    LEVELS,
    LIST_MAX,
    MISSING,
    PLUGIN_RE,
    POWER_HOLD_MODES,
    SCOPES,
    SLUG_RE,
    SPECULATE_MODES,
    SPEED_MAX,
    SPEED_MIN,
    STORES,
    STT_ENGINES,
    URL_MAX,
    VOICE_ID_MAX,
    VOICE_ID_RE,
    Setting,
    SettingError,
    _dig,
    _env_list,
    _is_num,
    _jsonable,
    _label,
    _merge,
    _put,
    _raw_value,
    _rt_bool,
    _rt_float,
    _rt_int,
    _s,
    _strict_rt,
    check_url,
    clean_voice_id,
    coerce,
    env_value,
)

#: The engine-owned rows keep one definition, in the core; this table names them in DisPatch's order.
E = _core.engine

PATIENCE = {"quick": 0.7, "normal": 1.0, "patient": 1.5}
REPLY_STYLES = ("short", "normal", "longer")
THINKING_LEVELS = ("off", "low", "medium")
MERGE_WINDOW_MIN, MERGE_WINDOW_MAX = 0.5, 4.0
MAX_SESSIONS_MIN, MAX_SESSIONS_MAX, MAX_SESSIONS_DEFAULT = 1, 8, 3
RP_MODES = ("all", "skip_actions", "dialogue", "dialogue_narrator")
GATEWAY_TRAILERS = ("full", "tag")
OPUS_MODES = ("auto", "off")
OPUS_DOWN_KBPS = (24, 16)
OPUS_UP_KBPS = (16, 12)
BOT_KEYS = ("backchannel", "follow_up")
BOT_LOCKED_VOICE = "locked_voice"
INTERRUPT_SENSITIVITIES = ("low", "normal", "high")
SLOW_LINK = ("phone", "wait", "text", "always_phone")
REACTIONS = ("off", "sometimes", "often")
HANGUP_QUIET_MIN = (5, 15, 30, 0)

BOT_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _force_flag(raw: str) -> Any:
    """``DISPATCH_VOICE``: ``auto``/unset leaves the switch to the file; a truthy word
    forces voice on; anything else forces it off."""
    v = raw.strip().lower()
    if v in ("", "auto"):
        return MISSING
    return v in FORCE_ON


# --------------------------------------------------------------------------- #
# bots: the one map-shaped setting
# --------------------------------------------------------------------------- #

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
# THE TABLE
# --------------------------------------------------------------------------- #

SETTINGS: tuple[Setting, ...] = (
    # ---- the switch -------------------------------------------------------- #
    _s("enabled", "bool", False, level="basic", env=("DISPATCH_VOICE",), env_parse=_force_flag,
       field="switch", group="system", label="Voice chat",
       help="Turns voice chat on. DISPATCH_VOICE=1/0 forces it (the switch is then locked)."),

    # ---- 1. speech to text ------------------------------------------------- #
    E("stt_provider"),
    E("stt_url"),
    E("stt_model"),
    E("stt_timeout_s"),
    E("stt_api_key"),
    E("rt.stt_engine"),
    E("rt.speculate"),
    E("rt.stt_spinning"),
    E("rt.stt_threads_while_speaking"),
    E("threads"),

    # ---- 2. text to speech ------------------------------------------------- #
    E("tts_provider"),
    E("tts_engine"),
    E("builtin_voice"),
    E("speed"),
    _s("rp_speech", "enum", "all", level="basic", choices=RP_MODES, group="voices", label="Role-play speech"),
    _s("narrator_voice", "voice_id", "", group="voices", label="Narrator voice",
       help="A voice id (profile or builtin:<id>); empty = the character's voice."),
    _s("expressive_tags", "bool", True, group="voices", label="Expressive tags",
       help="A cloned voice may render [laugh] / [sigh]; off removes every bracket tag."),
    _s("gateway_trailer", "enum", "full", choices=GATEWAY_TRAILERS, group="voices", label="Gateway trailer"),
    E("tts_threads"),
    E("hq_url"),
    E("hq_model"),
    E("hq_timeout_s"),
    E("hq_api_key"),
    E("minimax_model"),
    E("minimax_voice"),
    E("minimax_base_url"),
    E("minimax_api_key"),

    # ---- 3. input mode, turn taking, devices ------------------------------- #
    _s("merge_window_s", "float", 1.8, level="basic", lo=MERGE_WINDOW_MIN, hi=MERGE_WINDOW_MAX,
       env=("DISPATCH_VOICE_MERGE_WINDOW",), resettable=True, group="input", label="Turn merge window (s)"),
    E("hard_silence_s"),
    _s("voice_turn_thinking", "enum", "off", choices=THINKING_LEVELS, env=("DISPATCH_VOICE_THINKING",),
       env_loose=True, resettable=True, group="input", label="Thinking during voice turns"),
    _s("no_smart_turn_silence_s", "float", 0.8, lo=0.1, hi=5.0, store="env", env=("DISPATCH_VOICE_SILENCE",),
       field="no_smart_turn_silence_s", group="input", label="End-of-turn silence without Smart Turn (s)"),
    _s("patience", "float", 1.0, lo=0.5, hi=3.0, store="env", env=("DISPATCH_VOICE_PATIENCE",), group="input",
       label="Server patience multiplier", note="per-device `patience` overrides it"),
    _s("barge_abort_turn", "bool", True, store="env", env=("DISPATCH_VOICE_BARGE_ABORT",), group="input",
       label="Barge-in also aborts the agent turn", note="per-device `bargeIn` switches barge-in itself"),
    E("rt.probe_silence_s"),
    E("rt.hold_min_s"),
    E("rt.hold_max_s"),
    E("rt.addendum_s"),
    E("rt.echo_guard_s"),
    E("rt.false_interrupt_timeout_s"),
    E("rt.quiet_max_s"),
    E("rt.speculate_max_per_turn"),
    E("rt.speculate_min_gap_s"),
    E("rt.preemptive_tts"),
    E("rt.decoder_prompt_s"),
    E("rt.first_chunk_tokens"),
    E("rt.decoder_sessions"),
    E("rt.power_hold"),
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
    E("rig.enabled"),
    E("rig.url"),
    E("rig.stt_final_deadline_ms"),
    E("rig.tts_start_deadline_ms"),
    E("rig.ahead_ms"),
    E("rig.first_chunk_tokens"),
    E("rig.allow"),
    E("rig.token"),

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

#: Validators, resolution and publication over this table (the machinery is the core's).
_SCHEMA = _core.Schema(SETTINGS, map_cleaners={"bots": clean_bots})

BY_ID: dict[str, Setting] = _SCHEMA.BY_ID
#: Settings that live in voice.yaml / the API tree (scope server, persisted).
FILE_SETTINGS: tuple[Setting, ...] = _SCHEMA.FILE_SETTINGS
#: Ids that are subtrees of the file (``rt``, ``rig``) and the maps.
TREES = _SCHEMA.TREES
#: Every dotted id or alias a file/PATCH may use -> its canonical setting.
BY_NAME: dict[str, Setting] = _SCHEMA.BY_NAME

setting = _SCHEMA.setting
rt_ranges = _SCHEMA.rt_ranges
rig_ranges = _SCHEMA.rig_ranges
rt_keys = _SCHEMA.rt_keys
clean_prefs = _SCHEMA.clean_prefs
clean_subtree = _SCHEMA.clean_subtree
validate_subtree = _SCHEMA.validate_subtree
validate_patch = _SCHEMA.validate_patch
patchable_keys = _SCHEMA.patchable_keys
resolve = _SCHEMA.resolve
schema_json = _SCHEMA.schema_json
env_table = _SCHEMA.env_table
_unknown = _SCHEMA._unknown


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

