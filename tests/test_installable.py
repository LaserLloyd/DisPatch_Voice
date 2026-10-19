"""The add-on installs with a plain pip / uv, with no path dependency.

1.6.0 declared its engines as `forge-voice-core[speechforge]>=0.1.0` and found them only through
`[tool.uv.sources]` (a path to `packages/forge-voice-core`). pip never reads that table, and uv reads it
only for a directory or a git checkout: an archive URL, an sdist or a wheel failed with "No solution found",
and an uv install from a directory left an editable `.pth` pointing at the checkout. These tests keep the
published metadata self-sufficient. The end-to-end proof (build, install into a clean venv, import) is
`scripts/check_installable.py`, which CI runs in the "Installable" job.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CORE = "forge-voice-core"
CORE_URL = "git+https://github.com/LaserLloyd/DisPatch_Voice@main#subdirectory=packages/forge-voice-core"


def _core_requirement() -> str:
    found = [d for d in PYPROJECT["project"]["dependencies"] if re.match(rf"{CORE}\b", d)]
    assert len(found) == 1, found
    return found[0]


def test_core_is_declared_as_a_url_not_a_path():
    req = _core_requirement()
    assert req.replace(" ", "") == f"{CORE}[speechforge]@{CORE_URL}"


def test_no_dependency_is_a_local_path():
    for dep in PYPROJECT["project"]["dependencies"]:
        assert "file:" not in dep and ".." not in dep and "\\" not in dep, dep
    for extra, deps in PYPROJECT["project"].get("optional-dependencies", {}).items():
        for dep in deps:
            assert "file:" not in dep and ".." not in dep, (extra, dep)


def test_the_build_backend_allows_the_direct_reference():
    # hatchling refuses a `name @ url` dependency unless this is on; without it nothing builds.
    assert PYPROJECT["tool"]["hatch"]["metadata"]["allow-direct-references"] is True


def test_the_path_source_is_for_a_checkout_only():
    # uv (checkout, `uv sync`) swaps the URL for the in-tree package; the URL stays the published truth.
    src = PYPROJECT["tool"]["uv"]["sources"][CORE]
    assert src["path"] == "packages/forge-voice-core"
    assert (ROOT / src["path"] / "pyproject.toml").is_file()


def test_core_package_metadata_matches_what_the_addon_asks_for():
    core = tomllib.loads((ROOT / "packages" / "forge-voice-core" / "pyproject.toml").read_text(encoding="utf-8"))
    assert core["project"]["name"] == CORE
    assert "speechforge" in core["project"]["optional-dependencies"]


def test_sdist_is_scoped_to_the_sources():
    # An unscoped sdist swallows untracked virtualenvs and node_modules (and fails on a venv symlink).
    inc = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/src" in inc and not any("venv" in i or "node_modules" in i for i in inc)
