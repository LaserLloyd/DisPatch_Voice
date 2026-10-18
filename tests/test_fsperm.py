"""fsperm on Windows: "private" means nobody but the owner, SYSTEM and Administrators is
granted access, not merely "no well-known group" (a file shared with one other account is not
private). POSIX keeps the mode-bit rule, covered by the token tests."""
from __future__ import annotations

import subprocess

import pytest

from dispatch_voice import fsperm

pytestmark = pytest.mark.skipif(not fsperm.IS_WINDOWS, reason="Windows ACLs")


def _grant(path, spec: str) -> None:
    subprocess.run(["icacls", str(path), "/grant", spec], capture_output=True, check=True)


def test_a_grant_to_another_named_account_is_not_private(tmp_path):
    p = tmp_path / "speechforge.env"
    p.write_text("x", encoding="utf-8")
    assert fsperm.restrict(p) and fsperm.others_can_read(p) is False
    # Remote Desktop Users: not one of the well-known "everybody" groups the old deny-list named,
    # standing in for a second local account.
    _grant(p, "*S-1-5-32-555:R")
    assert fsperm.others_can_read(p) is True
    assert fsperm.ensure_private(p) and fsperm.others_can_read(p) is False


def test_owner_system_and_administrators_are_private(tmp_path):
    p = tmp_path / "voice.yaml"
    p.write_text("x", encoding="utf-8")
    assert fsperm.restrict(p)
    _grant(p, "*S-1-5-32-544:F")                  # Administrators
    assert fsperm.others_can_read(p) is False


def test_an_inherit_only_entry_does_not_grant_the_folder_itself(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    assert fsperm.restrict(d, directory=True)
    _grant(d, "*S-1-5-32-555:(OI)(CI)(IO)R")       # applies to children created later, not to d
    assert fsperm.others_can_read(d) is False
