"""Where the rig token lives (review F7c): the host credential store first, then a private file in
the add-on's state folder (<data>/ext/voice/, so it survives an image update and a read-only root),
with the old home-folder file still read. The value is never logged."""
from __future__ import annotations

import logging

import pytest
from fakes import make_private

from dispatch_voice.host import rig

SECRET = "rig-token-value-not-real"  # scrub-ok: test fixture
OTHER = "other-token-value-not-real"  # scrub-ok: test fixture


class Store:
    """The host's ``ctx.secrets``: get/set/delete by name."""

    def __init__(self, fail: bool = False):
        self.d: dict[str, str] = {}
        self.fail = fail

    def get(self, name):
        if self.fail:
            raise RuntimeError("locked")
        return self.d.get(name)

    def set(self, name, value):
        if self.fail:
            raise RuntimeError("locked")
        self.d[name] = value

    def delete(self, name):
        self.d.pop(name, None)


@pytest.fixture
def env(host_env, monkeypatch, tmp_path):
    monkeypatch.delenv(rig.TOKEN_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    rig._MISCONFIGURED.clear()
    return host_env


def put(path, token, private=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{rig.TOKEN_ENV}={token}\n", encoding="utf-8")
    make_private(path, private=private)


def test_the_token_file_is_in_the_state_folder(env):
    assert rig.token_path() == env.paths.state_dir / "speechforge.env"
    assert rig.legacy_token_path().name == "speechforge.env" and "home" in rig.legacy_token_path().parts


def test_outside_dispatch_the_old_home_file_stays_the_default(monkeypatch, tmp_path):
    from dispatch_voice.host import context
    context.clear_context()
    monkeypatch.delenv("DISPATCH_DATA_DIR", raising=False)
    assert rig.token_path() == rig.legacy_token_path()
    monkeypatch.setenv("DISPATCH_DATA_DIR", str(tmp_path / "d"))
    assert rig.token_path() == tmp_path / "d" / "ext" / "voice" / "speechforge.env"


def test_write_without_a_store_makes_a_private_file_in_the_state_folder(env):
    p = rig.write_token(SECRET)
    assert p == env.paths.state_dir / "speechforge.env" and p.exists()
    assert not rig.legacy_token_path().exists()                       # nothing under the home folder
    t = rig.read_token()
    assert t.value == SECRET and t.source == "file"
    assert rig.clear_token() is True and rig.read_token() == rig.Token()


def test_the_old_home_file_is_still_read_and_cleared(env):
    put(rig.legacy_token_path(), SECRET)
    assert rig.read_token().value == SECRET
    put(env.paths.state_dir / "speechforge.env", OTHER)               # the new place wins
    assert rig.read_token().value == OTHER
    assert rig.clear_token() is True
    assert not rig.legacy_token_path().exists() and rig.read_token() == rig.Token()


def test_the_credential_store_is_the_first_home(env):
    env.secrets = Store()
    assert rig.write_token(SECRET) is None                             # no file written
    assert not (env.paths.state_dir / "speechforge.env").exists()
    t = rig.read_token()
    assert t.value == SECRET and t.source == "secret" and SECRET not in repr(t)
    assert env.secrets.d == {rig.TOKEN_SECRET: SECRET}
    put(env.paths.state_dir / "speechforge.env", OTHER)                # the store outranks a file
    assert rig.read_token().value == SECRET
    assert rig.clear_token() is True
    assert env.secrets.d == {} and rig.read_token() == rig.Token()


def test_the_environment_beats_the_store(env, monkeypatch):
    env.secrets = Store()
    rig.write_token(SECRET)
    monkeypatch.setenv(rig.TOKEN_ENV, OTHER)
    assert rig.read_token().source == "env" and rig.read_token().value == OTHER


def test_a_store_that_fails_falls_back_to_the_private_file(env, caplog):
    env.secrets = Store(fail=True)
    with caplog.at_level(logging.DEBUG):
        p = rig.write_token(SECRET)
        t = rig.read_token()
    assert p is not None and t.value == SECRET and t.source == "file"
    assert SECRET not in caplog.text


def test_saving_to_the_store_removes_an_old_state_file(env):
    put(env.paths.state_dir / "speechforge.env", OTHER)
    env.secrets = Store()
    rig.write_token(SECRET)
    assert not (env.paths.state_dir / "speechforge.env").exists()
    assert rig.read_token().value == SECRET


def test_a_state_file_others_can_read_is_refused(env):
    put(env.paths.state_dir / "speechforge.env", SECRET, private=False)
    t = rig.read_token()
    assert t.value == "" and t.problem == "permissions"


def test_a_home_folder_that_cannot_be_written_does_not_matter(env, monkeypatch, tmp_path):
    """The container case: HOME is read-only (or nowhere); the state folder is on the data volume."""
    monkeypatch.setenv("HOME", str(tmp_path / "no" / "such" / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no" / "such" / "home"))
    assert rig.write_token(SECRET) is not None and rig.read_token().value == SECRET
    assert not (tmp_path / "no").exists()
