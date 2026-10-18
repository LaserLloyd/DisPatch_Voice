"""The settings schema (host/settings_schema.py): the one table every validator, the
precedence rule, the schema endpoint and the README's environment table are generated from."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dispatch_voice.host import settings_schema as S

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "dispatch_voice"


# --------------------------------------------------------------------------- #
# the table itself
# --------------------------------------------------------------------------- #

def test_ids_are_unique_and_every_entry_is_well_formed():
    ids = [s.id for s in S.SETTINGS]
    assert len(ids) == len(set(ids))
    for s in S.SETTINGS:
        assert s.scope in S.SCOPES and s.level in S.LEVELS and s.store in S.STORES, s.id
        assert s.type in {"bool", "int", "float", "enum", "str", "url", "list", "map", "secret", "voice_id"}, s.id
        if s.type in ("int", "float") and s.lo is not None:
            assert s.lo <= s.hi and s.lo <= s.default <= s.hi, s.id
        if s.type == "enum":
            assert s.default in s.choices or s.default is None, s.id
        if s.secret:
            assert s.store in ("env", "credential") and s.default in ("", None), s.id


def test_secrets_never_carry_a_value_in_the_schema_or_in_resolve():
    sch = S.schema_json()
    secret_rows = [r for r in sch["settings"] if r["secret"]]
    assert {r["id"] for r in secret_rows} >= {"rig.token", "minimax_api_key", "stt_api_key", "hq_api_key"}
    assert all(r["default"] in ("", None) for r in secret_rows)
    values, sources = S.resolve({}, {"SPEECHFORGE_TOKEN": "tok-value", "MINIMAX_API_KEY": "k-value"})
    assert values["rig.token"] is True and values["minimax_api_key"] is True       # set / not set, never the value
    assert "tok-value" not in repr(values) and "k-value" not in repr(sources)
    assert sources["rig.token"] == {"source": "env", "env": "SPEECHFORGE_TOKEN"}


def test_every_environment_variable_the_code_reads_is_in_the_schema():
    """A new os.environ read without a schema entry is a drift: the README table, the UI badge and
    the precedence rule would all miss it."""
    known = {n for s in S.SETTINGS for n in s.env}
    # not settings: process plumbing, other products' variables, test switches, deprecated seams
    plumbing = {"DISPATCH_BACKEND", "DISPATCH_EXTENSIONS", "DISPATCH_VOICE_PKG", "LOCAL_CHAT_DATA_DIR",
                "DISPATCH_VOICE_TEST_HARD_EXIT", "DISPATCH_VOICE_REQUIRE_FFMPEG", "DISPATCH_MIRROR",
                "DISPATCH_VOICE_PREWARM_DONE", "DISPATCH_SPEECH_MODELS_ROOT",
                "DISPATCH_VOICE_KOKORO_DIR", "DISPATCH_VOICE_KOKORO_MODEL"}   # read from a unit file for the doctor
    pat = re.compile(r"\b(DISPATCH_[A-Z0-9_]+|SPEECHFORGE_[A-Z0-9_]+|MINIMAX_[A-Z0-9_]+)\b")
    seen: dict[str, str] = {}
    for f in SRC.rglob("*.py"):
        if f.name in ("settings_schema.py",):
            continue
        for m in pat.finditer(f.read_text(encoding="utf-8")):
            seen.setdefault(m.group(1), f.name)
    missing = {n: f for n, f in seen.items()
               if n not in known and n not in plumbing and not n.endswith("_") and n not in {"SPEECHFORGE_CHECKOUT"}}
    assert missing == {}, f"environment names without a schema entry: {missing}"


# --------------------------------------------------------------------------- #
# generated file validation (what clean_prefs was)
# --------------------------------------------------------------------------- #

def test_clean_prefs_keeps_the_file_shape_and_drops_the_bad():
    assert S.clean_prefs({"max_sessions": 8.0, "rt": {"power_hold": "OFF"}}) == {
        "max_sessions": 8, "rt": {"power_hold": "off"}}
    assert S.clean_prefs({"max_sessions": 0, "rt": {"power_hold": "turbo"}}) == {}
    assert S.clean_prefs({"max_sessions": True}) == {}
    assert S.clean_prefs({"speed": 9, "merge_window_s": 99, "voice_turn_thinking": "high"}) == {}
    assert S.clean_prefs({"rt": {"decoder_prompt_s": 99, "bogus": 1}}) == {"rt": {"decoder_prompt_s": 10.0}}
    assert S.clean_prefs({"rig": {"ahead_ms": 99999, "nope": 1, "enabled": "yes"}}) == {"rig": {"ahead_ms": 10000}}
    assert S.clean_prefs({"rig": {"url": ""}}) == {"rig": {"url": ""}}
    assert S.clean_prefs("not a mapping") == {}


def test_yaml_booleans_for_off_and_on():
    # a hand edit writes `off`, which YAML reads as false
    assert S.clean_prefs({"rt": {"power_hold": False}}) == {"rt": {"power_hold": "off"}}
    assert S.clean_prefs({"rt": {"power_hold": True}}) == {"rt": {"power_hold": "performance"}}
    assert S.clean_prefs({"opus": False}) == {"opus": "off"}


def test_legacy_and_duplicate_keys_collapse_to_one_definition():
    assert S.clean_prefs({"kokoro_voice": "af_bella", "kokoro_speed": 1.1}) == {"builtin_voice": "af_bella", "speed": 1.1}
    assert S.clean_prefs({"kokoro_voice": "af_bella", "builtin_voice": "am_adam"}) == {"builtin_voice": "am_adam"}
    # the top-level spellings of the rt keys are aliases of the one definition
    assert S.clean_prefs({"power_hold": "off", "stt_engine": "moonshine"}) == {
        "rt": {"power_hold": "off", "stt_engine": "moonshine"}}
    assert S.clean_prefs({"rt": {"hard_silence_s": 2.0}}) == {"hard_silence_s": 2.0}
    # the canonical spelling wins when both are present
    assert S.clean_prefs({"power_hold": "off", "rt": {"power_hold": "performance"}}) == {
        "rt": {"power_hold": "performance"}}


def test_bots_and_voice_ids():
    assert S.clean_prefs({"bots": {"nova": {"backchannel": True, "junk": 1}, "BAD ID": {"follow_up": True},
                                   "sage": {"locked_voice": ""}}}) == {
        "bots": {"nova": {"backchannel": True}, "sage": {"locked_voice": ""}}}
    assert S.clean_prefs({"narrator_voice": "builtin:bf_emma"}) == {"narrator_voice": "builtin:bf_emma"}
    assert S.clean_prefs({"narrator_voice": "../etc"}) == {}


def test_new_persisted_keys():
    got = S.clean_prefs({"tts_engine": "Kokoro", "stt_provider": "openai", "stt_url": "https://stt.example/v1",
                         "hq_url": "http://10.0.0.5:8000", "extensions": ["kokoro", "pkg.mod:setup", "no spaces"],  # scrub-ok: address-range fixture
                         "rig": {"allow": ["10.0.0.0/8", "*.example.com", "bad host"]}})  # scrub-ok: address-range fixture
    assert got == {"tts_engine": "kokoro", "stt_provider": "openai", "stt_url": "https://stt.example/v1",
                   "hq_url": "http://10.0.0.5:8000", "extensions": ["kokoro", "pkg.mod:setup"],  # scrub-ok: address-range fixture
                   "rig": {"allow": ["10.0.0.0/8", "*.example.com"]}}  # scrub-ok: address-range fixture
    assert S.clean_prefs({"hq_url": "ftp://x"}) == {} and S.clean_prefs({"rig": {"url": "http://x"}}) == {}


# --------------------------------------------------------------------------- #
# generated PATCH validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body, text", [
    ({"speed": 9}, "speed must be between 0.8 and 1.3"),
    ({"speed": "fast"}, "speed must be a number"),
    ({"voice_turn_thinking": "high"}, "voice_turn_thinking must be one of off, low, medium"),
    ({"max_sessions": 99}, "max_sessions must be a whole number between 1 and 8"),
    ({"max_sessions": 2.5}, "max_sessions must be a whole number between 1 and 8"),
    ({"expressive_tags": "yes"}, "expressive_tags must be true or false"),
    ({"opus_down_kbps": 20}, "opus_down_kbps must be one of 24, 16"),
    ({"rt": {"power_hold": "turbo"}}, "rt.power_hold must be one of performance, off"),
    ({"rt": {"decoder_prompt_s": 99}}, "rt.decoder_prompt_s must be a number between 2.0 and 10.0"),
    ({"rt": {"decoder_sessions": 1.5}}, "rt.decoder_sessions must be a whole number between 1 and 2"),
    ({"rt": {"nope": 1}}, "Unknown setting: rt.nope"),
    ({"rt": "x"}, "rt takes an object with"),
    ({"rig": {"url": "ws://u:p@h/x"}}, "rig.url must be a ws:// or wss:// address"),
    ({"rig": {"ahead_ms": 5}}, "rig.ahead_ms must be a whole number between 1000 and 10000"),
    ({"rig": {"nope": 1}}, "Unknown setting: rig.nope"),
    ({"rig": "on"}, "rig takes an object"),
    ({"tts_engine": ""}, "tts_engine must not be empty"),
    ({"tts_engine": "bad engine"}, "tts_engine has characters that are not allowed"),
    ({"extensions": ["bad name"]}, "extensions takes a list of plugin names"),
    ({"nope": 1}, "Unknown setting: nope"),
    ({"rig.url": "ws://h"}, "Unknown setting: rig.url"),
    ({"builtin_voice": "x"}, "Unknown setting: builtin_voice"),       # changed with default_voice
])
def test_validate_patch_refuses_with_the_reason(body, text):
    with pytest.raises(S.SettingError) as e:
        S.validate_patch(body)
    assert text in str(e.value)


def test_validate_patch_accepts_and_folds_aliases():
    assert S.validate_patch({"speed": 1.2, "power_hold": "off", "rt": {"stt_engine": "moonshine"},
                             "rig": {"url": " ws://10.0.0.2:1236/v1/realtime ", "enabled": True}}) == {  # scrub-ok: address-range fixture
        "speed": 1.2, "rt": {"power_hold": "off", "stt_engine": "moonshine"},
        "rig": {"url": "ws://10.0.0.2:1236/v1/realtime", "enabled": True}}  # scrub-ok: address-range fixture
    assert S.validate_patch({"rt": {"hard_silence_s": 1.5}}) == {"hard_silence_s": 1.5}
    assert S.validate_patch({"rig": {"url": ""}}) == {"rig": {"url": ""}}
    assert S.validate_patch({"enabled": True, "bots": {}}, extra_keys=frozenset({"enabled", "bots"})) == {}


def test_validate_bots_strict():
    ok = S.validate_bots({"nova": {"backchannel": True, "locked_voice": None}}, known={"nova": 1},
                         voice_ok=lambda v: True)
    assert ok == {"nova": {"backchannel": True, "locked_voice": ""}}
    for bad, text in [({"zed": {}}, "Unknown bot"), ({"nova": {"backchannel": 1}}, "must be true or false"),
                      ({"nova": {"x": True}}, "Unknown setting: bots.nova.x"), ("x", "bots takes an object"),
                      ({"nova": {"locked_voice": "nope"}}, "Unknown voice")]:
        with pytest.raises(S.SettingError, match=text):
            S.validate_bots(bad, known={"nova": 1}, voice_ok=lambda v: False)


# --------------------------------------------------------------------------- #
# precedence: environment beats the file beats the default
# --------------------------------------------------------------------------- #

def test_environment_beats_the_file_beats_the_default_and_the_source_is_named():
    prefs = {"speed": 1.2, "merge_window_s": 2.0}
    values, sources = S.resolve(prefs, {"DISPATCH_VOICE_SPEED": "0.9"})
    assert values["speed"] == 0.9 and sources["speed"] == {"source": "env", "env": "DISPATCH_VOICE_SPEED"}
    assert values["merge_window_s"] == 2.0 and sources["merge_window_s"] == {"source": "file"}
    assert values["hard_silence_s"] == 1.4 and sources["hard_silence_s"] == {"source": "default"}


def test_environment_aliases_new_name_first_and_bad_values_are_ignored():
    v, s = S.resolve({}, {"DISPATCH_VOICE_KOKORO_SPEED": "1.15"})
    assert v["speed"] == 1.15 and s["speed"]["env"] == "DISPATCH_VOICE_KOKORO_SPEED"
    v, s = S.resolve({}, {"DISPATCH_VOICE_KOKORO_SPEED": "1.15", "DISPATCH_VOICE_SPEED": "0.95"})
    assert v["speed"] == 0.95
    v, s = S.resolve({"speed": 1.1}, {"DISPATCH_VOICE_SPEED": "fast"})
    assert v["speed"] == 1.1 and s["speed"]["source"] == "file"          # an unparsable env value does not win


def test_environment_is_not_range_limited_except_where_it_always_was():
    v, _ = S.resolve({}, {"DISPATCH_VOICE_MERGE_WINDOW": "0", "DISPATCH_VOICE_MAX_SESSIONS": "99"})
    assert v["merge_window_s"] == 0.0           # the app limits people, not the operator
    assert v["max_sessions"] == 8               # max_sessions was always clamped


def test_dispatch_voice_force_flag():
    v, s = S.resolve({"enabled": True}, {"DISPATCH_VOICE": "0"})
    assert v["enabled"] is True and s["enabled"] == {"source": "env", "env": "DISPATCH_VOICE", "forced": False}
    _, s = S.resolve({}, {"DISPATCH_VOICE": "auto"})
    assert s["enabled"]["source"] == "default"
    _, s = S.resolve({}, {"DISPATCH_VOICE": "yes"})
    assert s["enabled"]["forced"] is True


def test_power_hold_and_enum_env():
    assert S.resolve({}, {"DISPATCH_VOICE_POWER_HOLD": "off"})[0]["rt.power_hold"] == "off"
    assert S.resolve({}, {"DISPATCH_VOICE_POWER_HOLD": "nonsense"})[0]["rt.power_hold"] == "performance"
    assert S.resolve({}, {"DISPATCH_VOICE_THINKING": "high"})[0]["voice_turn_thinking"] == "high"   # loose env
    assert S.resolve({}, {"DISPATCH_VOICE_EXTENSIONS": "a.b:setup, kokoro"})[0]["extensions"] == ["a.b:setup", "kokoro"]


def test_rig_environment_overrides():
    v, s = S.resolve({"rig": {"url": "ws://10.0.0.1:1/x"}}, {"DISPATCH_VOICE_RIG_URL": "wss://10.0.0.9:2/y",  # scrub-ok: address-range fixture
                                                                 "DISPATCH_VOICE_RIG_AHEAD_MS": "4000"})
    assert v["rig.url"] == "wss://10.0.0.9:2/y" and s["rig.url"]["source"] == "env"  # scrub-ok: address-range fixture
    assert v["rig.ahead_ms"] == 4000


# --------------------------------------------------------------------------- #
# publication
# --------------------------------------------------------------------------- #

def test_schema_json_describes_every_scope():
    sch = S.schema_json()
    by = {r["id"]: r for r in sch["settings"]}
    assert by["speed"]["min"] == 0.8 and by["speed"]["max"] == 1.3 and by["speed"]["env"][0] == "DISPATCH_VOICE_SPEED"
    assert by["rt.power_hold"]["aliases"] == ["power_hold"] and by["max_sessions"]["restart_required"] is True
    assert {r["scope"] for r in sch["settings"]} == {"server", "device", "bot"}
    assert by["device.interruptSensitivity"]["choices"] == ["low", "normal", "high"]
    assert by["bots.*.locked_voice"]["scope"] == "bot"
    assert by["rig.url"]["level"] == "basic" and by["rig.token"]["secret"] is True
    assert all("value" not in r for r in sch["settings"])


def test_ranges_for_the_runtime_modules_come_from_the_table():
    from dispatch_voice.host import rig
    from dispatch_voice.rt import settings as rt_settings
    assert rt_settings.RANGES == S.rt_ranges() and rig.RIG_RANGES == S.rig_ranges()
    assert rt_settings.RtSettings().decoder_prompt_s == S.BY_ID["rt.decoder_prompt_s"].default


def test_runtime_defaults_match_the_table():
    """The dataclasses still carry their defaults; this pins them to the table."""
    from dispatch_voice.host.config import VoiceSettings
    from dispatch_voice.host.rig import RigSettings
    from dispatch_voice.rt.settings import RtSettings
    vs, rt, rg = VoiceSettings(), RtSettings(), RigSettings()
    for s in S.SETTINGS:
        if s.scope != "server" or s.secret or s.store == "credential" or s.id in ("enabled", "extensions", "rig.allow") \
                or s.id in ("models_root", "chatterbox_models", "model_revision", "data_dir", "bots", "ack_render"):
            continue
        parts = s.parts
        if parts[0] == "rt":
            got = getattr(rt, parts[1], None)
            if parts[1] == "stt_engine":
                got = vs.stt_engine
        elif parts[0] == "rig":
            got = getattr(rg, parts[1], None)
        else:
            got = getattr(vs, s.field or parts[0], None)
        assert got is not None or s.default in ("", None), s.id
        if got is not None:
            assert got == s.default, (s.id, got, s.default)


# --------------------------------------------------------------------------- #
# README table (generated; this fails when someone edits it by hand)
# --------------------------------------------------------------------------- #

def test_readme_environment_table_is_the_generated_one():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- env-table:start -->\n(.*?)<!-- env-table:end -->", readme, re.S)
    assert m, "README.md needs the <!-- env-table:start/end --> markers"
    assert m.group(1) == S.env_table(), "README environment table is stale: run `python -m dispatch_voice.host.settings_schema --write-readme`"
