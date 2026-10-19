"""Regression (ops check 2026-10-19): a dev-box run of the suite deleted the developer's real rig
token (~/.config/local-chat/speechforge.env) through rig.clear_token(). The conftest guard
(`_private_home`) gives every test its own home folder."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import REAL_HOME

from dispatch_voice.host import rig

REAL_TOKEN = Path(REAL_HOME) / ".config" / "local-chat" / "speechforge.env"


def _inside(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def test_the_legacy_token_file_is_the_tests_own(_private_home):
    assert Path(rig.legacy_token_path()) != REAL_TOKEN
    assert _inside(rig.legacy_token_path(), _private_home)


def test_clearing_the_token_only_touches_the_tests_home(_private_home):
    mine = rig.legacy_token_path()
    mine.parent.mkdir(parents=True, exist_ok=True)
    mine.write_text(f"{rig.TOKEN_ENV}=not-a-real-token-value\n", encoding="utf-8")
    rig.clear_token()                            # what DELETE /api/voice/rig/token runs
    assert not mine.exists() and _inside(mine, _private_home)


def test_the_default_dispatch_data_folder_is_the_tests_own(_private_home, monkeypatch):
    monkeypatch.delenv("DISPATCH_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCAL_CHAT_DATA_DIR", raising=False)
    default = Path(os.path.expanduser("~/.local/share")) / "local-chat"
    assert _inside(default, _private_home)


@pytest.mark.models
def test_a_models_test_keeps_the_real_home_but_not_the_real_token_file(_private_home):
    assert Path(os.path.expanduser("~")) == Path(REAL_HOME)
    assert _inside(rig.legacy_token_path(), _private_home)
