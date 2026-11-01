"""Per-profile cloned voices (add-on 1.6.0, BUILD-PLAN-v2 1.12), against the fake 4.0 host.

The rule table: who lists / creates / renames / deletes / tries / downloads / defaults /
publishes which clone, who may speak with it, what the purge removes, and what a host
without the hooks (DisPatch < 4.0) still gets. No model is loaded; the Chatterbox engine
is a fake and a "clone" is a profile directory written by ``make_clone``.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import (
    CALLER_A,
    CALLER_ADMIN,
    CALLER_B,
    CALLER_BASIC,
    BotView,
    FakeCaller,
    FakeContext,
    FakeTenantContext,
)
from fastapi.testclient import TestClient

from dispatch_voice.host import acks, registry, routes, setup, voice_access
from dispatch_voice.host import profiles_bridge as pb
from dispatch_voice.host import rig as host_rig
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.engines import chatterbox
from dispatch_voice.profiles import ProfileStore

WAV = ("a.wav", b"RIFF....WAVE", "audio/wav")


class FakeEngine:
    sample_rate = 24000
    dtype = "fake"
    watermarking = False


def make_clone(st, pid: str, owner: str | None = None, default: bool = False, name: str | None = None) -> None:
    """A profile directory as the store writes it (voice.npz, audition, one filler, reference)."""
    import soundfile as sf

    from dispatch_voice.engine import VoiceTensors
    d = st.root / pid
    (d / "fillers").mkdir(parents=True)
    VoiceTensors(np.zeros((1, 2, 4), np.float32), np.zeros((1, 3), np.int64),
                 np.zeros((1, 192), np.float32), np.zeros((1, 5, 80), np.float32)).save(d / "voice.npz")
    meta = {"name": name or pid, "created": 1, "default": default}
    if owner:
        meta["owner"] = owner
    (d / "profile.json").write_text(json.dumps(meta), encoding="utf-8")
    for f in ("reference.wav", "audition.wav", "fillers/0.wav"):
        sf.write(d / f, np.full(480, 0.1, np.float32), 24000)


@pytest.fixture
def env(tenant_env, monkeypatch):
    """The 4.0 host, set up, with the cloned-voice engine mounted and a recorder for rig deletes."""
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: FakeEngine())
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(tenant_env.paths.models_dir / "dv-models"))
    rig_deleted: list[str] = []
    monkeypatch.setattr(host_rig, "sha_for_delete", lambda store, pid: f"sha-{pid}")
    monkeypatch.setattr(host_rig, "delete_voice_async", lambda sha: rig_deleted.append(sha))
    created: list[dict] = []

    def fake_create(self, name, data, filename, consent, consent_note="", allow_warnings=True, owner=None):
        pid = f"{name.lower().replace(' ', '-')}-{len(created):03d}"
        make_clone(self, pid, owner=owner, name=name)
        created.append({"id": pid, "owner": owner})
        return self.get(pid)

    monkeypatch.setattr(ProfileStore, "create", fake_create)
    monkeypatch.setattr(ProfileStore, "speak_line", lambda self, pid, text: np.zeros(2400, np.float32))
    setup(tenant_env)
    tenant_env.seal()
    client = TestClient(tenant_env.app)
    return SimpleNamespace(ctx=tenant_env, client=client, st=pb.store(), rig_deleted=rig_deleted, created=created)


def as_(env, caller):
    env.ctx.current_caller = caller


def ids(resp) -> set[str]:
    return {v["id"] for v in resp.json()["voices"]}


@pytest.fixture
def world(env):
    """household: hh (default); A: a-mine; B: b-mine; admin: adm-mine."""
    make_clone(env.st, "hh", default=True)
    make_clone(env.st, "a-mine", owner="prof-a")
    make_clone(env.st, "b-mine", owner="prof-b")
    make_clone(env.st, "adm-mine", owner="admin-1")
    return env


# ---------------------------------------------------------------- registration

def test_setup_registers_the_hooks_on_a_4_0_host(env):
    assert env.ctx.voice_access is voice_access.voice_access
    assert env.ctx.profile_count is voice_access.profile_count
    assert env.ctx.profile_purge is voice_access.profile_purge


def test_setup_is_quiet_on_a_host_without_the_hooks(host_env, monkeypatch):
    """DisPatch 3.x: no caller / bots_visible / register_*: nothing is registered, nothing is required."""
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    assert not hasattr(host_env, "register_voice_access") and not hasattr(host_env, "caller")
    setup(host_env)
    assert routes._LOADED == ["dispatch-voice"]


def test_registration_after_setup_is_the_hosts_to_refuse(env):
    with pytest.raises(RuntimeError):
        env.ctx.register_voice_access(lambda v, o: True)


# ---------------------------------------------------------------- list / get

def test_list_is_household_plus_the_callers_own(world):
    as_(world, CALLER_A)
    r = world.client.get("/api/voice/profiles")
    assert ids(r) == {"hh", "a-mine"}
    as_(world, CALLER_B)
    assert ids(world.client.get("/api/voice/profiles")) == {"hh", "b-mine"}


def test_an_admin_never_sees_another_profiles_clone(world):
    as_(world, CALLER_ADMIN)
    assert ids(world.client.get("/api/voice/profiles")) == {"hh", "adm-mine"}


def test_signed_out_reads_household_only(world):
    as_(world, None)
    assert ids(world.client.get("/api/voice/profiles")) == {"hh"}


def test_rows_say_what_the_caller_may_do_and_hide_the_owner(world):
    as_(world, CALLER_A)
    rows = {v["id"]: v for v in world.client.get("/api/voice/profiles").json()["voices"]}
    assert all("owner" not in v for v in rows.values())            # no other profile id leaks
    assert rows["a-mine"]["mine"] and rows["a-mine"]["can_manage"] and rows["a-mine"]["can_download"]
    assert rows["hh"]["household"] and not rows["hh"]["can_manage"] and not rows["hh"]["can_download"]
    as_(world, CALLER_ADMIN)
    hh = {v["id"]: v for v in world.client.get("/api/voice/profiles").json()["voices"]}["hh"]
    assert hh["can_manage"] and hh["can_download"] and hh["household"]


def test_get_one_other_profiles_clone_is_404(world):
    as_(world, CALLER_B)
    assert world.client.get("/api/voice/profiles/a-mine").status_code == 404
    assert world.client.get("/api/voice/profiles/hh").status_code == 200
    as_(world, CALLER_ADMIN)
    assert world.client.get("/api/voice/profiles/a-mine").status_code == 404      # not even an Admin
    assert world.client.get("/api/voice/profiles/nope-1").status_code == 404
    as_(world, CALLER_A)
    assert world.client.get("/api/voice/profiles/a-mine").json()["mine"] is True


# ---------------------------------------------------------------- create / analyse

def form(**over):
    return {"data": {"name": "My voice", "consent": "self", **over}, "files": {"file": WAV}}


def test_create_makes_a_personal_clone_for_the_caller(world):
    as_(world, CALLER_A)
    r = world.client.post("/api/voice/profiles", **form())
    assert r.status_code == 200 and r.json()["mine"] is True
    assert world.created[-1]["owner"] == "prof-a"
    as_(world, CALLER_B)
    assert r.json()["id"] not in ids(world.client.get("/api/voice/profiles"))


def test_household_flag_is_for_admins_only(world):
    as_(world, CALLER_A)
    assert world.client.post("/api/voice/profiles", **form(household="true")).status_code == 422
    assert world.created == []
    as_(world, CALLER_ADMIN)
    r = world.client.post("/api/voice/profiles", **form(household="true"))
    assert r.status_code == 200 and r.json()["household"] is True and world.created[-1]["owner"] is None
    r = world.client.post("/api/voice/profiles", **form())
    assert r.json()["mine"] is True and world.created[-1]["owner"] == "admin-1"


def test_create_needs_voice_lab_and_a_signed_in_caller(world):
    as_(world, CALLER_BASIC)                     # use_voice only
    assert world.client.post("/api/voice/profiles", **form()).status_code == 403
    assert world.client.post("/api/voice/profiles/analyse", files={"file": WAV}).status_code == 403
    as_(world, None)
    assert world.client.post("/api/voice/profiles", **form()).status_code == 403
    assert world.created == []


def test_an_open_install_without_a_profile_makes_household_voices(world):
    as_(world, FakeCaller(None, is_admin=True, role="admin"))
    r = world.client.post("/api/voice/profiles", **form())
    assert r.status_code == 200 and world.created[-1]["owner"] is None and r.json()["household"] is True


# ---------------------------------------------------------------- patch / delete / say

def test_rename_owner_yes_other_profile_404(world):
    as_(world, CALLER_A)
    r = world.client.patch("/api/voice/profiles/a-mine", json={"name": "Renamed"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed"
    as_(world, CALLER_B)
    assert world.client.patch("/api/voice/profiles/a-mine", json={"name": "Mine now"}).status_code == 404
    as_(world, CALLER_ADMIN)
    assert world.client.patch("/api/voice/profiles/a-mine", json={"name": "Mine now"}).status_code == 404
    assert world.st.get("a-mine")["name"] == "Renamed"


def test_household_clone_is_the_admins_to_change(world):
    as_(world, CALLER_A)
    assert world.client.patch("/api/voice/profiles/hh", json={"name": "x"}).status_code == 404
    assert world.client.delete("/api/voice/profiles/hh").status_code == 404
    assert world.client.post("/api/voice/profiles/hh/say", json={"text": "hi"}).status_code == 404
    as_(world, CALLER_ADMIN)
    assert world.client.patch("/api/voice/profiles/hh", json={"name": "House"}).json()["name"] == "House"


def test_publish_and_unpublish_are_admin_only(world):
    as_(world, CALLER_A)
    r = world.client.patch("/api/voice/profiles/a-mine", json={"household": True})
    assert r.status_code == 403 and world.st.owner_of("a-mine") == "prof-a"
    as_(world, CALLER_ADMIN)
    assert world.client.patch("/api/voice/profiles/a-mine", json={"household": True}).status_code == 404
    r = world.client.patch("/api/voice/profiles/adm-mine", json={"household": True})
    assert r.status_code == 200 and r.json()["household"] is True
    as_(world, CALLER_B)
    assert "adm-mine" in ids(world.client.get("/api/voice/profiles"))
    as_(world, CALLER_ADMIN)
    r = world.client.patch("/api/voice/profiles/adm-mine", json={"household": False})
    assert r.json()["mine"] is True and world.st.owner_of("adm-mine") == "admin-1"
    as_(world, CALLER_B)
    assert "adm-mine" not in ids(world.client.get("/api/voice/profiles"))
    as_(world, CALLER_ADMIN)
    assert world.client.patch("/api/voice/profiles/hh", json={"household": "yes"}).status_code == 422


def test_default_voice_is_household_and_admin_only(world):
    make_clone(world.st, "hh2")
    as_(world, CALLER_ADMIN)
    assert world.client.patch("/api/voice/profiles/hh2", json={"default": True}).status_code == 200
    assert world.st.default_id() == "hh2"
    assert world.client.patch("/api/voice/profiles/adm-mine", json={"default": True}).status_code == 422
    assert world.client.patch("/api/voice/profiles/adm-mine", json={"household": False, "default": True}).status_code == 422
    as_(world, CALLER_A)
    assert world.client.patch("/api/voice/profiles/a-mine", json={"default": True}).status_code == 403
    assert world.st.default_id() == "hh2"


def test_making_the_default_clone_private_clears_the_default(world):
    as_(world, CALLER_ADMIN)
    assert world.st.default_id() == "hh"
    r = world.client.patch("/api/voice/profiles/hh", json={"household": False})
    assert r.status_code == 200 and world.st.default_id() is None


def test_delete_owner_yes_others_404_and_the_rig_copy_goes(world):
    as_(world, CALLER_B)
    assert world.client.delete("/api/voice/profiles/a-mine").status_code == 404
    assert world.rig_deleted == []
    as_(world, CALLER_A)
    assert world.client.delete("/api/voice/profiles/a-mine").json() == {"deleted": "a-mine"}
    assert world.rig_deleted == ["sha-a-mine"] and not (world.st.root / "a-mine").exists()
    as_(world, CALLER_ADMIN)
    assert world.client.delete("/api/voice/profiles/b-mine").status_code == 404
    assert (world.st.root / "b-mine").exists()
    assert world.client.delete("/api/voice/profiles/hh").status_code == 200


def test_try_a_line_is_the_owners(world):
    as_(world, CALLER_A)
    r = world.client.post("/api/voice/profiles/a-mine/say", json={"text": "hello there"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    as_(world, CALLER_B)
    assert world.client.post("/api/voice/profiles/a-mine/say", json={"text": "hello"}).status_code == 404
    as_(world, None)
    assert world.client.post("/api/voice/profiles/hh/say", json={"text": "hello"}).status_code == 403


# ---------------------------------------------------------------- recordings

@pytest.mark.parametrize("name", ["reference.wav", "audition.wav", "fillers/0.wav"])
def test_recordings_owner_only_for_a_personal_clone(world, name):
    url = f"/api/voice/profiles/a-mine/files/{name}"
    as_(world, CALLER_A)
    assert world.client.get(url).status_code == 200
    as_(world, CALLER_B)
    assert world.client.get(url).status_code == 404
    as_(world, CALLER_ADMIN)
    assert world.client.get(url).status_code == 404              # an Admin cannot download a personal clone
    as_(world, None)
    assert world.client.get(url).status_code == 404


def test_recordings_of_a_household_clone_are_the_admins(world):
    url = "/api/voice/profiles/hh/files/reference.wav"
    as_(world, CALLER_ADMIN)
    assert world.client.get(url).status_code == 200
    as_(world, CALLER_A)
    assert world.client.get(url).status_code == 404


# ---------------------------------------------------------------- preview / status

@pytest.fixture
def on(world, monkeypatch, no_models):
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setattr(routes, "_preview_wav", lambda settings, voice, text: b"RIFFxxxx")
    return world


def test_preview_of_another_profiles_clone_is_404(on):
    as_(on, CALLER_A)
    assert on.client.get("/api/voice/preview", params={"voice": "a-mine"}).status_code == 200
    assert on.client.get("/api/voice/preview", params={"voice": "hh"}).status_code == 200
    assert on.client.get("/api/voice/preview", params={"voice": "b-mine"}).status_code == 404
    as_(on, CALLER_ADMIN)
    assert on.client.get("/api/voice/preview", params={"voice": "a-mine"}).status_code == 404


def test_status_lists_only_what_the_caller_may_use(on):
    on.ctx.visible_bot_ids = {"sage", "swift"}
    on.ctx.bots._bots["swift"] = BotView("swift", "Swift", backend="api", scope="personal", owner_profile="prof-a")
    as_(on, CALLER_A)
    st = on.client.get("/api/voice/status").json()
    assert set(st["bots"]) == {"sage", "swift"}
    assert {p["id"] for p in st["profiles"]} == {"hh", "a-mine"}
    as_(on, CALLER_B)
    on.ctx.visible_bot_ids = {"sage"}
    st = on.client.get("/api/voice/status").json()
    assert set(st["bots"]) == {"sage"} and {p["id"] for p in st["profiles"]} == {"hh", "b-mine"}
    as_(on, None)
    on.ctx.visible_bot_ids = set()
    st = on.client.get("/api/voice/status").json()
    assert st["bots"] == {} and {p["id"] for p in st["profiles"]} == {"hh"}


def test_locked_voice_is_inert_in_status_on_a_4_0_host(on):
    on.client.patch("/api/voice/settings", json={"bots": {"sage": {"locked_voice": "hh"}}})   # still accepted
    as_(on, CALLER_A)
    assert "locked_voice" not in on.client.get("/api/voice/status").json()["bots"]["sage"]


def test_status_acks_never_name_another_profiles_clone(on, monkeypatch):
    """``/api/voice/status.acks`` is keyed by voice id with the voice's name: a personal clone's
    row (it appears once its owner has talked with it) is the owner's alone, like the picker."""
    from dispatch_voice.host import acks
    rows = {k: {"ready": 1, "total": 2, "failed": 0, "name": k.upper()}
            for k in ("default", "hh", "a-mine", "b-mine", "adm-mine", "builtin:af_heart", "gone-1")}
    monkeypatch.setattr(acks, "status", lambda: {k: dict(v) for k, v in rows.items()})
    as_(on, CALLER_A)
    assert set(on.client.get("/api/voice/status").json()["acks"]) == {"default", "hh", "a-mine", "builtin:af_heart"}
    as_(on, CALLER_ADMIN)
    assert set(on.client.get("/api/voice/status").json()["acks"]) == {"default", "hh", "adm-mine", "builtin:af_heart"}
    as_(on, None)
    assert set(on.client.get("/api/voice/status").json()["acks"]) == {"default", "hh", "builtin:af_heart"}


def test_deleting_a_clone_drops_its_acknowledgement_lines(on, monkeypatch):
    """A deleted clone's ack row (named) must not outlive it: with no profile.json left it would
    read as a household voice and show its name to every profile."""
    from dispatch_voice.host import acks
    forgot = []
    monkeypatch.setattr(acks, "forget", lambda v: forgot.append(v))
    as_(on, CALLER_A)
    assert on.client.delete("/api/voice/profiles/a-mine").status_code == 200
    assert forgot == ["a-mine"]


def test_acks_rebuild_of_another_profiles_clone_is_404(on, monkeypatch):
    """The Voice lab's Rebuild (an Admin) renders with the voice and answers the acks map: never
    for another profile's clone, and the map it answers is filtered like the status one."""
    from dispatch_voice.host import acks
    rebuilt = []
    monkeypatch.setattr(acks, "rebuild", lambda v: rebuilt.append(v) or True)
    monkeypatch.setattr(acks, "status", lambda: {"hh": {"ready": 1, "total": 1, "failed": 0, "name": "HH"},
                                                  "a-mine": {"ready": 1, "total": 1, "failed": 0, "name": "A"}})
    as_(on, CALLER_ADMIN)
    assert on.client.post("/api/voice/acks/rebuild", params={"voice": "a-mine"}).status_code == 404
    r = on.client.post("/api/voice/acks/rebuild", params={"voice": "hh"})
    assert r.status_code == 200 and set(r.json()["acks"]) == {"hh"} and rebuilt == ["hh"]


SETUP_DETAIL = ("config", "settings_env", "providers", "models", "system")


def test_status_shows_addresses_and_setup_detail_to_admins_only(on, monkeypatch):
    """``/api/voice/status`` is ``use_voice`` (every role, the unsigned default profile too): the rig
    and server addresses, the provider catalog, install commands and the settings' sources are the
    Admin's. Everyone keeps what the call screen and the device settings need."""
    import json as _json
    from dispatch_voice.host import config as vconfig
    vconfig.save_prefs({"rig": {"enabled": True, "url": "ws://192.0.2.7:1236/v1/realtime"},
                        "stt_url": "http://192.0.2.9:8000/v1"})
    monkeypatch.setattr(vconfig.VoiceSettings, "missing", lambda self: [
        "python package onnx-asr", "dispatch-voice models (run `dispatch-voice fetch-models`)",
        "MINIMAX_API_KEY (DISPATCH_VOICE_TTS=minimax)",
        "a transcription server address the network policy allows (stt_url: blocked)"])
    leaks = ("fetch-models", "MINIMAX_API_KEY", "DISPATCH_VOICE", "stt_url", "onnx-asr", "`")
    as_(on, CALLER_ADMIN)
    st = on.client.get("/api/voice/status").json()
    assert st["rig"]["host"] == "192.0.2.7:1236" and "config" in st and st["plugins"]["builtin"].get("suggested")
    assert any("fetch-models" in m for m in st["missing"]) and any("MINIMAX_API_KEY" in m for m in st["missing"])
    for who in (CALLER_A, CALLER_BASIC, None):
        as_(on, who)
        st = on.client.get("/api/voice/status").json()
        text = _json.dumps(st)
        assert "192.0.2.7" not in text and "192.0.2.9" not in text
        for k in SETUP_DETAIL:
            assert k not in st, k
        assert set(st["rig"]) <= {"enabled", "active", "configured", "state"}
        assert "suggested" not in st["plugins"]["builtin"]
        assert st["missing"] and not any(x in m for m in st["missing"] for x in leaks)
        for part in st["plugins"].values():
            assert not any(x in str(m) for m in part.get("missing", []) for x in leaks)
        assert all(set(h) <= {"id", "state", "code"} for h in st["health"])
        for k in ("enabled", "ready", "profiles", "bots", "proto", "opus", "default_voice", "health_state"):
            assert k in st, k


def test_status_says_whether_the_host_knows_profiles(on):
    """The Settings pane words its hints for the host it runs on (4.0: Settings → Characters and
    My bots, no locked-voice picker); ``host_profiles`` is how it knows."""
    as_(on, CALLER_A)
    assert on.client.get("/api/voice/status").json()["host_profiles"] is True


def test_settings_default_and_narrator_refuse_a_personal_clone(on):
    for key in ("default_voice", "narrator_voice"):
        r = on.client.patch("/api/voice/settings", json={key: "a-mine"})
        assert r.status_code == 422, key
    assert on.client.patch("/api/voice/settings", json={"default_voice": "hh"}).status_code == 200


def test_lab_pickers_list_household_voices_only(on):
    ch = routes._voice_choices(VoiceSettings(), voice_access.HOUSEHOLD_ONLY)
    assert {p["id"] for p in ch["profiles"]} == {"hh"}
    assert routes._admin_view() is voice_access.HOUSEHOLD_ONLY             # what the settings routes pass
    with pytest.raises(ValueError):
        pb.set_default_profile("a-mine")


# ---------------------------------------------------------------- voice_access / resolve

def test_voice_access_rule(world):
    f = voice_access.voice_access
    assert f("hh", None) and f("hh", "prof-a")                 # household voices: any bot
    assert f("a-mine", "prof-a")                               # a personal clone: its owner's bot
    assert not f("a-mine", "prof-b") and not f("a-mine", None)  # not another profile's, not a household bot
    assert f("builtin:af_heart", None) and f("stock-host", None) and f("missing-1", None)


def test_resolve_profile_falls_back_to_the_default_voice(world):
    world.ctx.bots._bots["swift"] = BotView("swift", "Swift", backend="api", voice="a-mine",
                                            scope="personal", owner_profile="prof-a")
    assert pb.resolve_profile("swift") == "a-mine"                          # no owner given: no check
    assert pb.resolve_profile("swift", owner="prof-a") == "a-mine"
    assert pb.resolve_profile("swift", owner="prof-b") == "hh"              # refused -> the (household) default
    assert pb.resolve_profile("swift", owner=None) == "hh"                  # a household bot
    assert pb.resolve_profile("swift", voice="hh", owner=None) == "hh"
    assert routes.voice_for_bot(VoiceSettings(), None, "swift", bot_owner="prof-a") == "a-mine"
    assert routes.voice_for_bot(VoiceSettings(), None, "swift", bot_owner="prof-b") == "hh"
    assert routes.voice_for_bot(VoiceSettings(), None, "swift", bot_owner=None) == "hh"


def test_bot_owner_of():
    own = SimpleNamespace(scope="personal", owner_profile="prof-a")
    assert voice_access.bot_owner_of(own) == "prof-a"
    assert voice_access.bot_owner_of(SimpleNamespace(scope="personal", owner_profile=""),
                                     SimpleNamespace(profile_id="prof-z")) == "prof-z"
    assert voice_access.bot_owner_of(SimpleNamespace(scope="household", owner_profile="")) is None
    assert voice_access.bot_owner_of(SimpleNamespace()) is None            # an older BotView


# ---------------------------------------------------------------- the session

def _serve_pieces(monkeypatch, fake_engines):
    from dispatch_voice.host import power, vad_turn
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0")

    async def fake_warm(settings):
        return fake_engines.stt, fake_engines.tts

    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: fake_engines.OneTurnDetector())
    monkeypatch.setattr(power, "acquire", lambda settings: "held")
    monkeypatch.setattr(power, "release", lambda: "off")


async def test_session_resolves_the_voice_with_the_bots_owner(world, monkeypatch, fake_engines, no_models, caplog):
    from settling import settle
    _serve_pieces(monkeypatch, fake_engines)
    ctx = world.ctx
    ctx.bots._bots["swift"] = BotView("swift", "Swift", backend="api", voice="a-mine",
                                      scope="personal", owner_profile="prof-a")
    ctx.add_thread("t-a", "swift", "A's chat")
    caplog.set_level(logging.INFO, logger="dispatch.voice")

    async def run(profile_id):
        ctx.current_caller = FakeCaller(profile_id)
        ws = fake_engines.FakeWS()
        import asyncio
        task = asyncio.create_task(ctx.connect("voice", "t-a", ws))
        await settle(12)
        assert ws.json_of("ready")
        ws.push_json({"type": "stop"})
        await settle(5)
        task.cancel()

    caplog.clear()
    await run("prof-a")
    assert any("speaks with a-mine" in m for m in caplog.messages)
    # The same bot opened by a socket of another profile (its owner is the bot's, not the socket's):
    ctx.bots._bots["swift"] = BotView("swift", "Swift", backend="api", voice="a-mine", scope="household")
    caplog.clear()
    await run("prof-b")
    assert not any("a-mine" in m for m in caplog.messages if "speaks with" in m)
    assert any("speaks with hh" in m for m in caplog.messages)


def test_a_clone_is_never_synced_to_the_rig_for_another_profile(world, monkeypatch):
    """The rig sync only ever receives the resolved voice id; a refused clone resolves to the default."""
    world.ctx.bots._bots["swift"] = BotView("swift", "Swift", voice="a-mine", scope="personal", owner_profile="prof-a")
    assert routes.voice_for_bot(VoiceSettings(), None, "swift", bot_owner="prof-b") != "a-mine"


def test_the_narrator_must_be_permitted_too(world, monkeypatch):
    from dispatch_voice.host import ws as ws_mod
    s = VoiceSettings(rp_speech="dialogue_narrator", narrator_voice="a-mine")
    bot = SimpleNamespace(id="swift", voice_rp="")
    monkeypatch.setattr(acks, "kit_for", lambda *a, **k: acks.AckKit.silent())
    assert ws_mod.session_options(s, None, bot, None, bot_owner="prof-a")["narrator_id"] == "a-mine"
    assert ws_mod.session_options(s, None, bot, None, bot_owner="prof-b")["narrator_id"] is None
    assert ws_mod.session_options(s, None, bot, None, bot_owner=None)["narrator_id"] is None


# ---------------------------------------------------------------- the profile-data hooks

def test_count_gives_numbers_only(world):
    assert voice_access.profile_count("prof-a") == {"voices": 1}
    assert voice_access.profile_count("prof-zzz") == {"voices": 0}
    assert voice_access.profile_count("household") == {"voices": 0}        # household is nobody's


def test_purge_removes_only_that_profiles_clones(world, monkeypatch):
    make_clone(world.st, "a-second", owner="prof-a")
    forgotten: list[str] = []
    monkeypatch.setattr(acks, "forget", lambda pid: forgotten.append(pid))
    assert voice_access.profile_purge("prof-a") == 2
    assert sorted(forgotten) == ["a-mine", "a-second"]
    assert sorted(world.rig_deleted) == ["sha-a-mine", "sha-a-second"]
    left = {m["id"] for m in world.st.list()}
    assert left == {"hh", "b-mine", "adm-mine"}
    assert voice_access.profile_purge("prof-a") == 0
    assert voice_access.profile_purge("household") == 0 and (world.st.root / "hh").exists()


def test_purge_drops_the_rendered_acknowledgement_lines(world):
    cache = world.ctx.paths.cache_dir
    for tag in ("cb-1234abcd-a-mine-1", "cb-1234abcd-hh-1"):
        (cache / tag).mkdir(parents=True)
        (cache / tag / "x.npy").write_bytes(b"0")
    voice_access.profile_purge("prof-a")
    assert not (cache / "cb-1234abcd-a-mine-1").exists() and (cache / "cb-1234abcd-hh-1").exists()


def test_purge_keeps_going_when_one_delete_fails(world, monkeypatch):
    make_clone(world.st, "a-second", owner="prof-a")
    real = pb._RigAwareStore.delete

    def flaky(self, pid):
        if pid == "a-mine":
            raise OSError("disk")
        return real(self, pid)

    monkeypatch.setattr(pb._RigAwareStore, "delete", flaky)
    assert voice_access.profile_purge("prof-a") == 1
    assert (world.st.root / "a-mine").exists() and not (world.st.root / "a-second").exists()


# ---------------------------------------------------------------- the older host (1.5.x behaviour)

@pytest.fixture
def legacy(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox.LazyEngine, "_get", lambda self: FakeEngine())
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(host_env.paths.models_dir / "dv-models"))
    created = []

    def fake_create(self, name, data, filename, consent, consent_note="", allow_warnings=True, owner=None):
        created.append(owner)
        make_clone(self, "made-1", owner=owner)
        return self.get("made-1")

    monkeypatch.setattr(ProfileStore, "create", fake_create)
    monkeypatch.setattr(host_rig, "sha_for_delete", lambda store, pid: None)
    assert type(host_env) is FakeContext and not hasattr(host_env, "caller")
    setup(host_env)
    return SimpleNamespace(ctx=host_env, client=TestClient(host_env.app), st=pb.store(), created=created)


def test_legacy_host_is_single_tenant_as_in_1_5(legacy):
    make_clone(legacy.st, "hh", default=True)
    make_clone(legacy.st, "x-owned", owner="someone")           # even an owned folder: every profile is listed
    r = legacy.client.get("/api/voice/profiles")
    assert ids(r) == {"hh", "x-owned"}
    assert legacy.client.get("/api/voice/profiles/x-owned").status_code == 200
    assert legacy.client.get("/api/voice/profiles/x-owned/files/reference.wav").status_code == 200
    assert legacy.client.patch("/api/voice/profiles/hh", json={"name": "N", "default": True}).json()["name"] == "N"
    assert legacy.client.post("/api/voice/profiles", **form(household="true")).status_code == 200
    assert legacy.created == [None]                              # household stays the only kind
    assert legacy.client.delete("/api/voice/profiles/x-owned").status_code == 200
    legacy.ctx.operator = False
    assert legacy.client.get("/api/voice/profiles").status_code == 403       # the host's operator gate, as before


def test_legacy_host_status_lists_every_bot(legacy, no_models, monkeypatch):
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    st = legacy.client.get("/api/voice/status").json()
    assert set(st["bots"]) == {"sage", "swift", "nova", "alpha"}
    assert st["host_profiles"] is False


def test_a_host_whose_caller_returns_none_cannot_write(world):
    as_(world, None)
    assert world.client.get("/api/voice/profiles").status_code == 200
    for call in (lambda: world.client.post("/api/voice/profiles", **form()),
                 lambda: world.client.patch("/api/voice/profiles/hh", json={"name": "x"}),
                 lambda: world.client.delete("/api/voice/profiles/hh")):
        assert call().status_code == 403
    assert world.created == [] and (world.st.root / "hh").exists()


def test_a_caller_that_raises_is_signed_out(world):
    def boom(request):
        raise RuntimeError("host bug")

    world.ctx.caller = boom
    assert ids(world.client.get("/api/voice/profiles")) == {"hh"}
    assert world.client.delete("/api/voice/profiles/hh").status_code == 403
