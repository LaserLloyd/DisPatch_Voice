"""The provider catalog as the host builds it."""
from __future__ import annotations

import json

from dispatch_voice.host import providers as P


def test_the_builtin_catalog_lists_cpu_rig_and_openai(host_env):
    from dispatch_voice.host.config import load_settings
    cat = P.REGISTRY.catalog(load_settings())
    ids = {(d["kind"], d["id"]) for d in cat}
    assert {("stt", "cpu"), ("stt", "rig"), ("stt", "openai"), ("tts", "cpu"), ("tts", "rig"),
            ("tts", "openai"), ("vad", "cpu")} <= ids
    assert all(d["health"]["state"] in P.STATES and "capabilities" in d for d in cat)
    json.dumps(cat)


def test_the_openai_factories_read_the_servers_allow_list(host_env):
    """`rig_allow` lives on VoiceSettings (not on RigSettings): a public server the administrator
    allow-listed must not be reported `url_not_allowed` by the catalog."""
    from dispatch_voice.host.config import VoiceSettings
    s = VoiceSettings(stt_provider="openai", stt_url="https://stt.example.com/", hq_url="https://tts.example.com/",
                      rig_allow=("stt.example.com", "tts.example.com"))
    assert P.REGISTRY.build("stt", "openai", s).health().state == "ok"
    assert P.REGISTRY.build("tts", "openai", s).health().state == "ok"
    bare = VoiceSettings(stt_provider="openai", stt_url="https://stt.example.com/", hq_url="https://tts.example.com/")
    assert P.REGISTRY.build("stt", "openai", bare).health().code == "url_not_allowed"
    assert P.REGISTRY.build("tts", "openai", bare).health().code == "url_not_allowed"


def test_the_minimax_engine_builds(host_env, monkeypatch):
    """1.6.0 put `import os` inside build_engine's HQ branch, which made `os` a local for the
    whole function: `tts_engine: minimax` raised UnboundLocalError before reaching MiniMax."""
    from dispatch_voice.host import tts
    from dispatch_voice.host.config import VoiceSettings
    monkeypatch.setenv("MINIMAX_API_KEY", "k-not-real")  # scrub-ok: test fixture
    eng = tts.build_engine(VoiceSettings(tts_engine="minimax"))
    assert isinstance(eng, tts.MiniMaxTTS)


def test_the_network_policy_applies_when_a_remote_address_is_used_not_only_on_patch(host_env, monkeypatch, caplog):
    """An address from the environment never went through PATCH: the HQ server is not wrapped in and
    the transcription server counts as missing when the policy refuses the address."""
    import logging

    from dispatch_voice.host import tts
    from dispatch_voice.host.config import VoiceSettings
    monkeypatch.setenv("MINIMAX_API_KEY", "k-not-real")  # scrub-ok: test fixture
    caplog.set_level(logging.WARNING, logger="dispatch.voice")
    blocked = VoiceSettings(tts_engine="minimax", hq_url="http://169.254.169.254/")
    assert isinstance(tts.build_engine(blocked), tts.MiniMaxTTS)                # no FallbackTTS(HQTTS) in front
    assert "url_blocked_metadata" in caplog.text
    fine = VoiceSettings(tts_engine="minimax", hq_url="http://127.0.0.1:8000/")
    assert isinstance(tts.build_engine(fine), tts.FallbackTTS)
    stt = VoiceSettings(stt_provider="openai", stt_url="http://2852039166/")       # 169.254.169.254 in decimal
    assert any("network policy" in m and "url_blocked_metadata" in m for m in stt.missing())
    assert not any("network policy" in m for m in VoiceSettings(stt_provider="openai", stt_url="http://10.0.0.5/").missing())  # scrub-ok: address-range fixture
