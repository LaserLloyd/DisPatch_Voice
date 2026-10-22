import json

import pytest

from conftest import speechlike, wav_bytes
from helpers import assert_private

from forge_voice_core.profiles import FILLERS, ProfileError


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


# ---------------------------------------------------------------- owners (1.6.0)

def test_household_voice_writes_no_owner_key(store):
    p = make(store)
    meta = json.loads((store.root / p["id"] / "profile.json").read_text(encoding="utf-8"))
    assert "owner" not in meta                      # old readers see an unchanged file
    assert p["owner"] == "household" and store.owner_of(p["id"]) == "household"
    explicit = store.create("Other", wav_bytes(speechlike(14.0)), "h.wav", "synthetic", owner="household")
    assert "owner" not in json.loads((store.root / explicit["id"] / "profile.json").read_text(encoding="utf-8"))


def test_personal_voice_records_its_owner(store):
    p = store.create("Mine", wav_bytes(speechlike(14.0)), "m.wav", "self", owner="prof-a")
    assert p["owner"] == "prof-a" and store.owner_of(p["id"]) == "prof-a"
    assert store.get(p["id"])["owner"] == "prof-a"
    assert [m["owner"] for m in store.list()] == ["prof-a"]


def test_profile_json_without_owner_is_household(store):
    p = make(store)
    path = store.root / p["id"] / "profile.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("owner", None)
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert store.owner_of(p["id"]) == "household"
    path.write_text(json.dumps({**meta, "owner": ""}), encoding="utf-8")
    assert store.owner_of(p["id"]) == "household"


def test_set_owner_round_trip_is_atomic_and_clears_default(store):
    p = make(store)
    store.update(p["id"], default=True)
    out = store.set_owner(p["id"], "prof-a")
    assert out["owner"] == "prof-a" and out["default"] is False
    assert not list((store.root / p["id"]).glob("*.tmp"))
    out = store.set_owner(p["id"], None)
    assert out["owner"] == "household"
    assert "owner" not in json.loads((store.root / p["id"] / "profile.json").read_text(encoding="utf-8"))
    with pytest.raises(ProfileError):
        store.set_owner(p["id"], "  ")
    with pytest.raises(KeyError):
        store.set_owner("nope-123456", "x")


def test_personal_voice_cannot_be_default(store):
    p = store.create("Mine", wav_bytes(speechlike(14.0)), "m.wav", "self", owner="prof-a")
    with pytest.raises(ProfileError):
        store.update(p["id"], default=True)
    assert store.default_id() is None
