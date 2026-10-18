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


# ---- review F8: DisPatch 3.0 renamed its folder to <data>/models ------------------------------ #

def test_the_default_folder_is_named_models_and_the_old_host_name_is_still_read(layout):
    assert layout.name == "models"
    old = layout.parent / "speech-models"
    put(old / "speech" / "silero_vad.onnx", b"s")                     # a 2.x install with the 1.6.0 layout
    put(old / "chatterbox" / "tokenizer.json", b"t")
    assert models.speech_dir() == old / "speech" and models.chatterbox_dir() == old / "chatterbox"
    assert models.legacy_host_root() == old


def test_a_flat_2x_install_in_the_old_host_folder_still_works(layout):
    old = layout.parent / "speech-models"
    put(old / "silero_vad.onnx")
    assert models.speech_dir() == old and models.legacy_present()


def test_the_add_on_can_move_what_the_host_left_behind(layout, tmp_path):
    old = layout.parent / "speech-models"
    put(old / "silero_vad.onnx", b"flat")
    put(old / "chatterbox" / "tokenizer.json", b"tok")
    put(old / "chatterbox" / "onnx" / "speech_encoder_q4.onnx", b"e")
    rep = models.migrate()                                           # dry run first
    assert rep["moved"] == 0 and {m["state"] for m in rep["moves"]} == {"move"}
    assert (old / "silero_vad.onnx").exists()
    rep = models.migrate(dry_run=False)
    assert rep["errors"] == [] and rep["moved"] == 2
    assert (layout / "speech" / "silero_vad.onnx").read_bytes() == b"flat"
    assert (layout / "chatterbox" / "onnx" / "speech_encoder_q4.onnx").read_bytes() == b"e"
    assert not old.exists()                                          # emptied, so removed
    assert models.speech_dir() == layout / "speech" and models.chatterbox_dir() == layout / "chatterbox"
    assert models.migrate(dry_run=False)["moves"] == []              # idempotent


def test_two_sources_for_one_file_move_only_the_first(layout):
    old = layout.parent / "speech-models"
    put(layout / "silero_vad.onnx", b"in-new-root")
    put(old / "silero_vad.onnx", b"in-old")
    rep = models.migrate(dry_run=False)
    assert [m["state"] for m in rep["moves"]] == ["moved", "skip_exists"]
    assert (layout / "speech" / "silero_vad.onnx").read_bytes() == b"in-new-root"
    assert (old / "silero_vad.onnx").read_bytes() == b"in-old"       # never overwritten, never deleted


def _whole_chatterbox_set(d):
    from dispatch_voice.engine import COMPONENTS, _filename
    put(d / "tokenizer.json", b"tok")
    put(d / "config.json", b"{}")
    for c in COMPONENTS:
        put(d / "onnx" / _filename(c, "q4"), b"g")
        put(d / "onnx" / (_filename(c, "q4") + "_data"), b"w")


def test_a_chatterbox_move_that_stopped_half_way_resumes_and_keeps_loading(layout, tmp_path, monkeypatch):
    """Windows refuses to move a file in use (the engine holds its graphs): the move stopped
    after config.json. The old code then took the half-filled new folder as THE models
    folder (engine unavailable) and reported the rest "skip_exists" forever. The half-way
    state is laid down by hand: which child a directory listing returns first differs
    between file systems, so the test must not depend on it."""
    import shutil
    from dispatch_voice.host.engines import chatterbox
    old_cb = tmp_path / "xdg" / "dispatch-voice" / "models"
    _whole_chatterbox_set(old_cb)
    put(layout / "chatterbox" / "config.json", b"{}")                # what an earlier run moved
    (old_cb / "config.json").unlink()
    real_move = shutil.move

    def busy(src, dst, *a, **k):
        if src.endswith("onnx"):
            raise PermissionError("in use")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "move", busy)
    rep = models.migrate(dry_run=False)
    assert rep["errors"] and (layout / "chatterbox" / "config.json").exists()   # the earlier run's file
    assert not (layout / "chatterbox" / "tokenizer.json").exists()   # folders go first: nothing else moved
    assert models.chatterbox_dir() == old_cb and chatterbox.complete(models.chatterbox_dir())
    assert models.legacy_present()                                   # still something to move
    monkeypatch.setattr(shutil, "move", real_move)
    rep = models.migrate(dry_run=False)                              # the retry finishes the job
    assert rep["errors"] == [] and rep["moved"] == 1
    assert models.chatterbox_dir() == layout / "chatterbox" and chatterbox.complete(layout / "chatterbox")
    assert not models.legacy_present()


def test_no_legacy_root_when_the_models_folder_is_the_old_name(host_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DISPATCH_SPEECH_MODELS", str(tmp_path / "speech-models"))
    from dispatch_voice.host import context
    context.clear_context()
    assert models.legacy_host_root() is None
