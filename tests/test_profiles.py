import json

import pytest

from conftest import speechlike, wav_bytes
from fakes import assert_private

from dispatch_voice.profiles import FILLERS, ProfileError


def make(store, name="Morning host", consent="synthetic"):
    return store.create(name, wav_bytes(speechlike(14.0)), "host.wav", consent, "test")


def test_create_writes_profile(store):
    p = make(store)
    d = store.root / p["id"]
    for f in ("original.wav", "reference.wav", "voice.npz", "profile.json", "audition.wav"):
        assert (d / f).exists(), f
    meta = json.loads((d / "profile.json").read_text(encoding="utf-8"))
    assert meta["consent"]["kind"] == "synthetic" and meta["fillers"] == "pending"
    assert meta["quality"]["match"] == 0.9
    assert_private(store.root, directory=True)


def test_consent_and_name_required(store):
    with pytest.raises(ProfileError, match="consent"):
        make(store, consent="")
    with pytest.raises(ProfileError, match="name"):
        make(store, name="  ")


def test_bad_clip_refused_and_nothing_left_behind(store):
    with pytest.raises(ProfileError):
        store.create("x", wav_bytes(speechlike(2.0)), "x.wav", "self")
    assert list(store.root.iterdir()) == []


def test_fillers_render(store):
    p = make(store)
    store.render_fillers(p["id"])
    assert store.get(p["id"])["fillers"] == "ready"
    assert all(store.file(p["id"], f"fillers/{i}.wav").exists() for i in range(len(FILLERS)))


def test_default_is_exclusive(store):
    a, b = make(store, "A"), make(store, "B")
    store.update(a["id"], default=True)
    store.update(b["id"], default=True)
    assert store.default_id() == b["id"]
    assert not store.get(a["id"])["default"]


def test_delete_removes_everything(store):
    p = make(store)
    store.delete(p["id"])
    assert not (store.root / p["id"]).exists()
    with pytest.raises(KeyError):
        store.get(p["id"])


@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", "", "UPPER", ".hidden", "x" * 60])
def test_ids_cannot_escape(store, bad):
    with pytest.raises((ProfileError, KeyError)):
        store.get(bad)


def test_file_allowlist(store):
    p = make(store)
    with pytest.raises(ProfileError):
        store.file(p["id"], "voice.npz")
    with pytest.raises(ProfileError):
        store.file(p["id"], "../profile.json")
