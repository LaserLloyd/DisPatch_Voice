"""Regression (ops check 2026-10-19): no test of the core suite may reach the real home folder's
token file (rig.clear_token() removes every token file it knows)."""
from __future__ import annotations

from pathlib import Path

from conftest import REAL_HOME

from forge_voice_core import rig


def test_the_legacy_token_file_is_the_tests_own(_private_home):
    p = Path(rig.legacy_token_path()).resolve()
    assert not str(p).startswith(str(Path(REAL_HOME).resolve() / ".config") + "/")
    assert Path(_private_home).resolve() in p.parents
