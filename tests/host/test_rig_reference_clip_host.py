"""``rig.send_reference_clip`` as a setting (voice.yaml, the settings PATCH, the status
the Settings row reads) and the deletion sync of every rig voice a profile has."""
from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import rig, routes
from dispatch_voice.profiles import ProfileStore

SECRET = "rig-token-value-not-real"  # scrub-ok: test fixture
URL = "ws://192.0.2.7:1236/v1/realtime"


@pytest.fixture(autouse=True)
def _isolated_token(monkeypatch, tmp_path):
    rig._MISCONFIGURED.clear()
    rig._LAST.clear()
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    path = tmp_path / "local-chat" / "speechforge.env"
    monkeypatch.setattr(rig, "token_path", lambda: path)
    yield path
    rig._MISCONFIGURED.clear()


def test_the_setting_is_off_by_default_and_cleaned():
    assert rig.RigSettings().send_reference_clip is False
    assert rig.clean_rig({"send_reference_clip": True}) == {"send_reference_clip": True}
    assert rig.clean_rig({"send_reference_clip": "yes"}) == {}
    assert rig.RigSettings.from_mapping({"send_reference_clip": 1}).send_reference_clip is False


def test_voice_yaml_keeps_it_beside_the_rig_switch(host_env):
    vconfig.save_prefs({"rig": {"enabled": True, "url": URL}})
    assert vconfig.load_settings().rig.send_reference_clip is False
    vconfig.save_prefs({"rig": {"send_reference_clip": True}})              # merged key by key
    st = vconfig.load_settings().rig
    assert (st.enabled, st.url, st.send_reference_clip) == (True, URL, True)


def test_the_settings_patch_takes_a_bool_only():
    assert routes._validate_rig({"send_reference_clip": True}) == {"send_reference_clip": True}
    assert routes._validate_rig({"send_reference_clip": False}) == {"send_reference_clip": False}
    for bad in (1, "true", None):
        with pytest.raises(HTTPException) as e:
            routes._validate_rig({"send_reference_clip": bad})
        assert e.value.status_code == 422 and "send_reference_clip" in e.value.detail


def test_the_status_carries_it_for_the_settings_row(host_env, monkeypatch):
    monkeypatch.setenv(rig.TOKEN_ENV, SECRET)
    vconfig.save_prefs({"rig": {"enabled": True, "url": URL}})
    assert rig.status(vconfig.load_settings())["send_reference_clip"] is False
    vconfig.save_prefs({"rig": {"send_reference_clip": True}})
    st = rig.status(vconfig.load_settings())
    assert st["send_reference_clip"] is True and SECRET not in json.dumps(st)


def test_deleting_a_profile_deletes_its_rig_voices_of_every_format(host_env, monkeypatch, tmp_path):
    from dispatch_voice.host import profiles_bridge as pb
    root = tmp_path / "voices"
    store = ProfileStore(root, None)
    d = root / "gone-voice"
    d.mkdir()
    (d / "voice.npz").write_bytes(b"tensors")
    (d / "profile.json").write_text(json.dumps({"name": "Gone", "created": 1}))
    a, b = "a" * 64, "b" * 64
    store.set_rig_voice("gone-voice", "dotstts-ref", {"id": a, "transcript": "hi", "reference_sha256": "c" * 64})
    store.set_rig_voice("gone-voice", "qwen3tts-ref", {"id": b, "transcript": "hi", "reference_sha256": "c" * 64})
    store.set_rig_voice("gone-voice", "broken", {"id": "../../etc"})     # never a path, never sent
    monkeypatch.setattr(pb, "store", lambda: store)
    assert rig.ids_for_delete(store, "gone-voice") == []                 # no rig set up: nothing read
    vconfig.save_prefs({"rig": {"url": URL}})                           # switched off, but an address
    monkeypatch.setenv(rig.TOKEN_ENV, SECRET)
    seen = []
    import httpx
    monkeypatch.setattr(httpx, "delete", lambda url, **kw: seen.append(url) or SimpleNamespace(status_code=200))
    pb._RigAwareStore().delete("gone-voice")
    for t in threading.enumerate():
        if t.name == "voice-rig-delete":
            t.join(2)
    sha = hashlib.sha256(b"tensors").hexdigest()
    base = "http://192.0.2.7:1236/v1/voices/"
    assert sorted(seen) == sorted([base + sha, base + a, base + b])
    assert not d.exists()
