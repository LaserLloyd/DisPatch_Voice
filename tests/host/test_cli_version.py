"""``dispatch-voice --version``: prints ``dispatch-voice <version>`` and exits 0, next to the
subcommands (which are still required)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from dispatch_voice import __version__, cli


def test_version_flag_prints_the_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"dispatch-voice {__version__}"


def test_version_flag_works_as_a_process():
    r = subprocess.run([sys.executable, "-m", "dispatch_voice.cli", "--version"], capture_output=True,
                       text=True, timeout=60, check=False)
    assert r.returncode == 0 and r.stdout.strip() == f"dispatch-voice {__version__}"


def test_subcommands_are_still_required(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2 and "cmd" in capsys.readouterr().err
