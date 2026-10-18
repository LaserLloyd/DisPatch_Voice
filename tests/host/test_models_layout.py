"""One models folder with subfolders (host/models.py): the resolvers keep an old install
working untouched, a fresh install gets the new layout, and the move is idempotent,
dry-run by default and never overwrites."""
from __future__ import annotations

import pytest

from dispatch_voice.host import context, models


@pytest.fixture
def layout(host_env, monkeypatch, tmp_path):
    monkeypatch.delenv(models.SPEECH_ENV, raising=False)
    monkeypatch.delenv(models.CHATTERBOX_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    return context.paths().models_dir


def put(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_a_fresh_install_gets_the_new_subfolders(layout):
    assert models.speech_dir() == layout / "speech"
    assert models.chatterbox_dir() == layout / "chatterbox"


def test_an_existing_flat_install_keeps_working_untouched(layout, tmp_path):
    put(layout / "silero_vad.onnx")
    put(layout / "parakeet-tdt-0.6b-v2" / "vocab.txt")
    old_cb = tmp_path / "xdg" / "dispatch-voice" / "models"
    put(old_cb / "tokenizer.json")
    assert models.speech_dir() == layout and models.chatterbox_dir() == old_cb
    assert models.legacy_present()


def test_dry_run_reports_and_moves_nothing(layout, tmp_path):
    put(layout / "silero_vad.onnx")
    put(tmp_path / "xdg" / "dispatch-voice" / "models" / "tokenizer.json")
    rep = models.migrate()                                   # dry_run is the default
    assert rep["dry_run"] and rep["moved"] == 0
    assert {m["family"] for m in rep["moves"]} == {"speech", "chatterbox"}
    assert all(m["state"] == "move" for m in rep["moves"])
    assert (layout / "silero_vad.onnx").exists() and not (layout / "speech").exists()


def test_the_move_is_complete_idempotent_and_the_resolvers_follow(layout, tmp_path):
    put(layout / "silero_vad.onnx", b"s")
    put(layout / "smart-turn-v3.2-cpu.onnx", b"t")
    put(layout / "parakeet-tdt-0.6b-v2" / "encoder-model.int8.onnx", b"p")
    old_cb = tmp_path / "xdg" / "dispatch-voice" / "models"
    put(old_cb / "tokenizer.json", b"tok")
    put(old_cb / "onnx" / "speech_encoder_q4.onnx", b"e")
    rep = models.migrate(dry_run=False)
    assert rep["moved"] == 4 and rep["errors"] == []
    sp, cb = layout / "speech", layout / "chatterbox"
    assert (sp / "silero_vad.onnx").read_bytes() == b"s"
    assert (sp / "parakeet-tdt-0.6b-v2" / "encoder-model.int8.onnx").exists()
    assert (cb / "tokenizer.json").read_bytes() == b"tok" and (cb / "onnx" / "speech_encoder_q4.onnx").exists()
    assert not (layout / "silero_vad.onnx").exists() and not (old_cb / "tokenizer.json").exists()
    assert models.speech_dir() == sp and models.chatterbox_dir() == cb and not models.legacy_present()
    again = models.migrate(dry_run=False)                    # idempotent
    assert again["moved"] == 0 and again["moves"] == []


def test_it_never_overwrites_an_existing_destination(layout):
    put(layout / "silero_vad.onnx", b"old")
    put(layout / "speech" / "silero_vad.onnx", b"new")
    rep = models.migrate(dry_run=False)
    assert rep["moved"] == 0 and rep["moves"][0]["state"] == "skip_exists"
    assert (layout / "silero_vad.onnx").read_bytes() == b"old"
    assert (layout / "speech" / "silero_vad.onnx").read_bytes() == b"new"


def test_an_explicit_env_folder_is_honoured_and_never_moved(layout, monkeypatch, tmp_path):
    mine = tmp_path / "mine"
    put(layout / "silero_vad.onnx")
    monkeypatch.setenv(models.SPEECH_ENV, str(mine))
    monkeypatch.setenv(models.CHATTERBOX_ENV, str(tmp_path / "cb"))
    assert models.speech_dir() == mine and models.chatterbox_dir() == tmp_path / "cb"
    rep = models.migrate(dry_run=False)
    assert rep["moved"] == 0 and {m["state"] for m in rep["moves"]} == {"explicit"}
    assert (layout / "silero_vad.onnx").exists()
