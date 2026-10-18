"""The generic Schema over the engine-level rows, and its single-source guarantees."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from forge_voice_core import settings_schema as S
from forge_voice_core.rig import RIG_RANGES, RigSettings, clean_rig
from forge_voice_core.rt.settings import RANGES, RtSettings

ROOT = Path(__file__).resolve().parents[1]


def test_the_realtime_ranges_come_from_the_engine_rows():
    assert RANGES == S.rt_ranges() == S.ENGINE_SCHEMA.rt_ranges()
    assert RIG_RANGES == S.rig_ranges()
    assert RANGES["addendum_s"] == (1.0, 5.0) and RANGES["hard_silence_s"] == (S.HARD_SILENCE_MIN, S.HARD_SILENCE_MAX)


def test_rt_settings_clamp_with_those_ranges():
    rt = RtSettings.from_mapping({"addendum_s": 99, "quiet_max_s": 1, "decoder_sessions": 2})
    assert rt.addendum_s == 5.0 and rt.quiet_max_s == 600.0 and rt.first_chunk_tokens == 15


def test_clean_rig_drops_unknown_clamps_and_checks_the_url():
    got = clean_rig({"url": "ws://192.0.2.1:1/x", "ahead_ms": 99999, "bogus": 1, "stt_final_deadline_ms": "no"})
    assert got == {"url": "ws://192.0.2.1:1/x", "ahead_ms": 10000}
    assert "url" not in clean_rig({"url": "http://x"})
    assert RigSettings.from_mapping({"enabled": True, "ahead_ms": 1}).ahead_ms == 1000


def test_clean_prefs_validate_patch_and_resolve_over_the_engine_rows():
    sch = S.ENGINE_SCHEMA
    assert sch.clean_prefs({"speed": 9, "kokoro_voice": "af", "rt": {"power_hold": False}, "nonsense": 1}) == {
        "builtin_voice": "af", "rt": {"power_hold": "off"}}
    assert sch.validate_patch({"speed": 1.1, "rt": {"quiet_max_s": 700}}) == {"speed": 1.1, "rt": {"quiet_max_s": 700.0}}
    with pytest.raises(S.SettingError, match="between"):
        sch.validate_patch({"speed": 3})
    with pytest.raises(S.SettingError, match="Unknown setting"):
        sch.validate_patch({"bogus": 1})
    values, sources = sch.resolve({"speed": 1.2}, {"DISPATCH_VOICE_SPEED": "1.1", "DISPATCH_VOICE_THREADS": "2"})
    assert values["speed"] == 1.1 and sources["speed"]["source"] == "env"        # environment beats the file
    assert values["threads"] == 2
    values, sources = sch.resolve({"speed": 1.2}, {})
    assert values["speed"] == 1.2 and sources["speed"] == {"source": "file"}     # the file beats the default
    assert sch.resolve({}, {})[0]["tts_engine"] == "auto"


def test_secrets_resolve_to_a_flag_never_a_value():
    values, _ = S.ENGINE_SCHEMA.resolve({}, {"DISPATCH_VOICE_HQ_API_KEY": "topsecret"})
    assert values["hq_api_key"] is True
    assert "topsecret" not in str(S.ENGINE_SCHEMA.schema_json())


def test_a_host_can_rename_the_environment_prefix_and_add_its_own_rows():
    mine = S.Setting("greeting", "str", "hi", env=("CHATFORGE_GREETING",), group="voices")
    sch = S.Schema([*S.ENGINE_SETTINGS, mine], env_prefix="CHATFORGE_VOICE_")
    assert sch.setting("speed").env[0] == "CHATFORGE_VOICE_SPEED"
    assert sch.setting("hq_api_key").env == ("CHATFORGE_VOICE_HQ_API_KEY",)
    assert sch.setting("minimax_api_key").env == ("MINIMAX_API_KEY",)             # unprefixed names stay
    assert sch.resolve({}, {"CHATFORGE_VOICE_SPEED": "1.2", "DISPATCH_VOICE_SPEED": "1.0"})[0]["speed"] == 1.2
    assert sch.resolve({}, {"CHATFORGE_GREETING": "yo"})[0]["greeting"] == "yo"
    assert S.ENGINE_BY_ID["speed"].env[0] == "DISPATCH_VOICE_SPEED"              # the shared rows were not mutated


def test_the_schema_json_shape():
    doc = S.ENGINE_SCHEMA.schema_json()
    assert doc["version"] == 1 and {r["id"] for r in doc["settings"]} == {s.id for s in S.ENGINE_SETTINGS}
    assert all("value" not in r for r in doc["settings"])


def test_check_url_is_shape_only():
    assert S.check_url("ws://host:1/x", schemes=("ws",)) and not S.check_url("ws://u:p@host/x")
    assert not S.check_url("http://host/x?token=1") and not S.check_url("http://" + "a" * 300)


def test_the_system_requirements_table_matches_what_sysreq_checks():
    from forge_voice_core import sysreq
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = meta["tool"]["forge-voice-core"]["system-requirements"]
    assert set(declared) == {r.id for r in sysreq.REQUIREMENTS} == {"ffmpeg", "libopus"}
    assert all(isinstance(v, str) and v for v in declared.values())
    assert any(d.startswith("websockets") for d in meta["project"]["optional-dependencies"]["speechforge"])
    assert not any(d.lower().startswith(("fastapi", "starlette", "websockets")) for d in meta["project"]["dependencies"])


def test_engine_rows_are_a_closed_set_with_unique_ids():
    ids = [s.id for s in S.ENGINE_SETTINGS]
    assert len(ids) == len(set(ids)) == 46
    assert {i.split(".")[0] for i in ids if "." in i} == {"rt", "rig"}
