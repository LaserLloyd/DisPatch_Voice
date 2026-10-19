"""migrate_voice_yaml(store_writer, dry_run=True): the one-shot move of voice.yaml into a settings
store. Implemented and tested here; the host wires it when the DisPatch store exists."""
from __future__ import annotations

import json

import pytest

from dispatch_voice.host import migration
from dispatch_voice.host.migration import flatten, migrate_voice_yaml

OLD = """\
kokoro_voice: af_bella
kokoro_speed: 1.1
enabled: true
max_sessions: 5
power_hold: off
rt:
  stt_engine: moonshine
  decoder_prompt_s: 99
  bogus: 1
rig:
  enabled: true
  url: ws://127.0.0.1:1236/v1/realtime
  ahead_ms: 4000
bots:
  nova: {backchannel: true, locked_voice: ""}
speed_typo: 3
opus_down_kbps: 20
"""


class Store:
    def __init__(self, fail_after: int | None = None) -> None:
        self.data: dict = {}
        self.fail_after = fail_after

    def __call__(self, key, value):
        if self.fail_after is not None and len(self.data) >= self.fail_after:
            raise RuntimeError("store is down")
        self.data[key] = value


@pytest.fixture(autouse=True)
def _store_is_read(request, monkeypatch):
    """The move itself (writes, marker, backup) is what most tests here pin; it runs for real only in
    a release that reads the store back. ``store_not_read`` tests see this release's behaviour."""
    if request.node.get_closest_marker("store_not_read") is None:
        monkeypatch.setattr(migration, "STORE_IS_READ", True)


def write(tmp_path, text=OLD):
    p = tmp_path / "voice.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    p = write(tmp_path)
    store = Store()
    rep = migrate_voice_yaml(store, path=p)
    assert rep["dry_run"] is True and rep["status"] == "dry_run" and store.data == {} and rep["written"] == 0
    keys = {i["key"] for i in rep["items"]}
    assert {"voice.builtin_voice", "voice.speed", "voice.enabled", "voice.max_sessions", "voice.rt.power_hold",
            "voice.rt.stt_engine", "voice.rt.decoder_prompt_s", "voice.rig.enabled", "voice.rig.url",
            "voice.rig.ahead_ms", "voice.bots"} <= keys
    assert not (tmp_path / "voice.yaml.bak").exists() and not (tmp_path / "voice.yaml.migrated").exists()


def test_legacy_names_arrive_under_their_one_canonical_key(tmp_path):
    store = Store()
    migrate_voice_yaml(store, dry_run=False, path=write(tmp_path))
    assert store.data["voice.builtin_voice"] == "af_bella" and store.data["voice.speed"] == 1.1
    assert "voice.kokoro_speed" not in store.data and "voice.power_hold" not in store.data
    assert store.data["voice.rt.power_hold"] == "off"                       # YAML's `off` is false; it means off
    assert store.data["voice.rt.decoder_prompt_s"] == 10.0                  # out of range: clamped, like loading
    assert store.data["voice.bots"] == {"nova": {"backchannel": True, "locked_voice": ""}}
    assert store.data["voice.rig.url"] == "ws://127.0.0.1:1236/v1/realtime"


def test_what_the_schema_refuses_is_listed_by_name_and_never_written(tmp_path):
    store = Store()
    rep = migrate_voice_yaml(store, path=write(tmp_path))
    assert set(rep["dropped"]) == {"speed_typo", "opus_down_kbps", "rt.bogus"}
    assert not any("typo" in k or "bogus" in k or "opus" in k for k in store.data)
    assert "3" not in str(rep["dropped"])                                    # names, never values


def test_a_real_run_writes_backs_up_and_marks_the_file(tmp_path):
    p = write(tmp_path)
    store = Store()
    rep = migrate_voice_yaml(store, dry_run=False, path=p)
    assert rep["status"] == "migrated" and rep["written"] == len(rep["items"]) == len(store.data)
    assert rep["backup"] == "voice.yaml.bak" and (tmp_path / "voice.yaml.bak").read_text(encoding="utf-8") == OLD
    assert p.read_text(encoding="utf-8") == OLD                              # kept: it is read for one more release
    assert json.loads((tmp_path / "voice.yaml.migrated").read_text(encoding="utf-8"))["keys"] == len(store.data)
    assert rep["marker"] == "voice.yaml.migrated"


def test_a_second_run_is_a_no_op_unless_forced(tmp_path):
    p = write(tmp_path)
    first = Store()
    migrate_voice_yaml(first, dry_run=False, path=p)
    second = Store()
    rep = migrate_voice_yaml(second, dry_run=False, path=p)
    assert rep["status"] == "already_migrated" and second.data == {} and rep["written"] == 0
    assert migrate_voice_yaml(second, dry_run=True, path=p)["status"] == "already_migrated"
    forced = Store()
    assert migrate_voice_yaml(forced, dry_run=False, path=p, force=True)["status"] == "migrated"
    assert forced.data == first.data


def test_a_failing_store_leaves_no_marker_and_a_retry_completes(tmp_path):
    p = write(tmp_path)
    bad = Store(fail_after=3)
    rep = migrate_voice_yaml(bad, dry_run=False, path=p)
    assert rep["status"] == "error" and rep["written"] == 3 and "RuntimeError" in rep["error"]
    assert not (tmp_path / "voice.yaml.migrated").exists() and not (tmp_path / "voice.yaml.bak").exists()
    good = Store()
    assert migrate_voice_yaml(good, dry_run=False, path=p)["status"] == "migrated"
    assert set(good.data) >= set(bad.data)


def test_missing_empty_and_broken_files(tmp_path):
    store = Store()
    assert migrate_voice_yaml(store, dry_run=False, path=tmp_path / "none.yaml")["status"] == "nothing_to_migrate"
    assert migrate_voice_yaml(store, dry_run=False, path=write(tmp_path, ""))["status"] == "nothing_to_migrate"
    rep = migrate_voice_yaml(store, dry_run=False, path=write(tmp_path, ": : not yaml : ["))
    assert rep["status"] == "error" and "could not be read" in rep["error"] and store.data == {}
    assert not (tmp_path / "voice.yaml.migrated").exists()


def test_default_path_is_the_hosts_voice_yaml(host_env):
    host_env.paths.voice_yaml.write_text("speed: 1.2\n", encoding="utf-8")
    rep = migrate_voice_yaml(Store())
    assert rep["items"] == [{"key": "voice.speed", "value": 1.2}]


def test_flatten_keeps_maps_and_lists_whole():
    got = dict(flatten({"rt": {"a": 1}, "rig": {"allow": ["x"], "url": "u"}, "bots": {"n": {"follow_up": False}},
                        "extensions": ["k"], "speed": 1.0}))
    assert got == {"voice.rt.a": 1, "voice.rig.allow": ["x"], "voice.rig.url": "u",
                   "voice.bots": {"n": {"follow_up": False}}, "voice.extensions": ["k"], "voice.speed": 1.0}


@pytest.mark.store_not_read
def test_a_real_run_is_deferred_while_the_addon_does_not_read_the_store(tmp_path):
    """Ops check 2026-10-19: DisPatch 3.0 calls the move for real at start, but 1.6 reads and writes
    voice.yaml only. The store copy was a frozen snapshot from that first start (a later Settings
    change went to voice.yaml alone, so GET /api/settings and the backups' settings.json showed
    stale voice values), and the marker would stop the release that does read the store from ever
    taking the current file. So a host's real run writes nothing yet: no key, no marker, no backup."""
    assert migration.STORE_IS_READ is False
    p = write(tmp_path)
    store = Store()
    rep = migrate_voice_yaml(store, dry_run=False, path=p)
    assert rep["status"] == "deferred" and rep["written"] == 0
    assert store.data == {}
    assert rep["items"]                                          # still says what it would move
    assert not (tmp_path / "voice.yaml.migrated").exists()
    assert not (tmp_path / "voice.yaml.bak").exists()
    assert p.read_text(encoding="utf-8") == OLD
    # an explicit operator run still works
    assert migrate_voice_yaml(Store(), dry_run=False, path=p, force=True)["status"] == "migrated"


@pytest.mark.store_not_read
def test_the_dry_run_report_is_unchanged_while_deferred(tmp_path):
    rep = migrate_voice_yaml(Store(), path=write(tmp_path))
    assert rep["status"] == "dry_run" and rep["items"]
