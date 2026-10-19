#!/usr/bin/env python3
"""Prove the add-on installs with a plain pip, from its built artifacts, with no path dependency.

    python scripts/check_installable.py              # wheel and sdist
    python scripts/check_installable.py --keep       # keep the scratch folder and print it

For each of the wheel and the sdist built from this checkout it creates a clean virtualenv, runs a plain
`pip install <artifact>` (needs no git) from a folder that is NOT the checkout (so nothing on sys.path or in a .pth
leaks in), and imports `dispatch_voice` and `forge_voice_core` with `python -I`. It also asserts the
metadata names no local path and that the engines landed in site-packages (not editable, not the
checkout). Needs network access to PyPI and to github.com: the add-on's `forge-voice-core` requirement
is an archive URL of `main` (see pyproject.toml). Needs `uv` (builds the artifacts; creates the venvs).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "import sys, dispatch_voice, forge_voice_core, importlib.metadata as m;"
    "from pathlib import Path;"
    f"root = Path({str(ROOT)!r}).resolve();"
    "files = [Path(dispatch_voice.__file__).resolve(), Path(forge_voice_core.__file__).resolve()];"
    "assert not any(root in f.parents for f in files), files;"
    "assert all('site-packages' in f.parts for f in files), files;"
    "print('dispatch-voice', m.version('dispatch-voice'), 'forge-voice-core', m.version('forge-voice-core'))"
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder")
    ap.add_argument("--python", default="3.12", help="Python for the clean venvs (default 3.12)")
    args = ap.parse_args()
    uv = shutil.which("uv")
    if not uv:
        print("uv is required (https://docs.astral.sh/uv/)", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="dv-installable-"))
    try:
        dist = tmp / "dist"
        run([uv, "build", "--sdist", "--wheel", "--out-dir", str(dist), str(ROOT)])
        wheel = next(dist.glob("*.whl"))
        sdist = next(dist.glob("*.tar.gz"))
        with zipfile.ZipFile(wheel) as z:
            meta = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode("utf-8")
        reqs = re.findall(r"^Requires-Dist: (.+)$", meta, re.M)
        bad = [r for r in reqs if "file:" in r or re.search(r"@\s*(\.|/|[A-Za-z]:)", r)]
        assert not bad, f"a local path in the published metadata: {bad}"
        assert any(r.startswith("forge-voice-core") for r in reqs), reqs
        for label, artifact in (("wheel", wheel), ("sdist", sdist)):
            venv = tmp / f"venv-{label}"
            run([uv, "venv", "--seed", "--python", args.python, str(venv)])
            py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            run([str(py), "-m", "pip", "install", "--disable-pip-version-check", str(artifact)], cwd=tmp)
            run([str(py), "-I", "-c", PROBE], cwd=tmp)
            print(f"OK {label}")
        # The same wheel against THIS checkout's engines (what a change to packages/forge-voice-core needs
        # proven): uv's override replaces the published reference with the in-tree package, non-editable.
        venv = tmp / "venv-local-core"
        run([uv, "venv", "--seed", "--python", args.python, str(venv)])
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        override = tmp / "override.txt"
        override.write_text(
            f"forge-voice-core[speechforge] @ {(ROOT / 'packages' / 'forge-voice-core').as_uri()}\n", encoding="utf-8"
        )
        run([uv, "pip", "install", "--python", str(py), "--no-sources", "--override", str(override), str(wheel)], cwd=tmp)
        run([str(py), "-I", "-c", PROBE], cwd=tmp)
        print("OK wheel with this checkout's forge-voice-core")
        return 0
    finally:
        if args.keep:
            print("scratch kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
