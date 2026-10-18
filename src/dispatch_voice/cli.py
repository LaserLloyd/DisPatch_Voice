"""dispatch-voice: fetch the models, check the install, run the Voices panel standalone, say a line.

    dispatch-voice --version
    dispatch-voice fetch-models [--only speech|chatterbox] [--force] [--verify] [--no-stock]
    dispatch-voice migrate-models [--apply]
    dispatch-voice doctor [--unit local-chat] [--backend DIR] [--url http://127.0.0.1:8765] [--check-update] [--json]
    dispatch-voice serve --data DIR [--port 8769]
    dispatch-voice say <profile> "text" [-o out.wav]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, audio

REPO_URL = "https://github.com/LaserLloyd/DisPatch_Voice"
#: The moving tag the absent page installs from; its VERSION file says what the latest 1.x is.
VERSION_URL = "https://raw.githubusercontent.com/LaserLloyd/DisPatch_Voice/v1/VERSION"
DEFAULT_SERVER = "http://127.0.0.1:8765"


def _data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from .host import context
    return context.paths().voices_dir


def _serve_refusal(arg: str | None) -> str | None:
    """Why ``serve`` will not start, or None. The standalone page has no login,
    so it never serves DisPatch's real voice store — a cloned voice is a
    person's recording (FR-hardening L6)."""
    if not arg:
        return ("dispatch-voice serve: pass --data DIR, a voice profile folder of its own. "
                "This test page has no login, so it never opens DisPatch's voices.")
    from .host import context
    try:
        live = context.paths().voices_dir.resolve()
        mine = Path(arg).expanduser().resolve()
    except OSError:
        return None
    if mine == live:
        return (f"dispatch-voice serve: {arg} is DisPatch's own voice store; this test page has "
                "no login. Copy the profiles you need to another folder and pass that instead.")
    return None


def standalone_app(store, port: int):
    """The ``dispatch-voice serve`` app: the Voices panel on loopback, no login.
    It answers only a request whose Host is the loopback address it listens on:
    a DNS-rebinding page (a hostile name resolving to 127.0.0.1) carries its own
    name in Host and gets 421, GETs of recorded voices included (FR-hardening L6)."""
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse, RedirectResponse

    from .router import build_router
    app = FastAPI(title="dispatch-voice (standalone)")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.middleware("http")
    async def loopback_host_only(request, call_next):
        if request.headers.get("host", "") not in allowed_hosts:
            return PlainTextResponse("Wrong host for this loopback-only page.", status_code=421)
        return await call_next(request)

    app.include_router(build_router(store, allow_open=True), prefix="/voices")
    app.get("/")(lambda: RedirectResponse("/voices/ui"))
    return app


def _speech_models_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    from .host import models
    return models.speech_dir()


# --------------------------------------------------------------------------- #
# fetch-models: ALL the core models (SPEC §3.2)
# --------------------------------------------------------------------------- #

def cmd_fetch_models(a) -> int:
    """The same code the in-app Download button runs (host.download.fetch_all)."""
    from .host import download
    return download.fetch_all(only=a.only, force=a.force, verify=a.verify, stock=not a.no_stock, dtype=a.dtype,
                              speech_dir=_speech_models_dir(a.speech_models),
                              models_dir=Path(a.models) if a.models else None)


def cmd_migrate_models(a) -> int:
    """Move the models from the old two folders into the unified one (dry run unless --apply)."""
    from .host import models
    rep = models.migrate(dry_run=not a.apply)
    if not rep["moves"]:
        print("nothing to move: the models are already in the unified folder (or not downloaded yet)")
        return 0
    for m in rep["moves"]:
        print(f"  {m['state']:11} {m['family']:10} {m['src']} -> {m['dst']}")
    if rep["dry_run"]:
        print("dry run: nothing was moved. Re-run with --apply to move.")
    else:
        print(f"moved {rep['moved']}" + (f"; errors: {rep['errors']}" if rep["errors"] else ""))
    return 1 if rep["errors"] else 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

def _run(argv: list[str], cwd: Path | None, timeout: float) -> tuple[int | None, str]:
    """A fixed-argv subprocess; (returncode or None on timeout, combined output tail)."""
    try:
        r = subprocess.run(argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except FileNotFoundError:
        return None, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f}s"
    out = (r.stdout + r.stderr).strip()
    return r.returncode, out[-400:]


def _uv() -> str:
    from shutil import which
    return which("uv") or os.path.expanduser("~/.local/bin/uv")


def probe_dispatch_import(backend: Path | None = None) -> dict:
    """Does the handshake pass? ``EXT_API`` is read from DisPatch's extension
    host when THIS environment can import it (``source: "import"``); else,
    with ``--backend DIR``, from ``DIR/app/extensions.py`` itself
    (``source: "file"``; nothing is imported, so it works from any
    environment and touches no data dir)."""
    import importlib.util
    import re

    from . import EXTENSION
    lo, hi = EXTENSION["api"]
    out: dict = {"importable": False, "ext_api": None, "compatible": None, "source": None}
    try:
        # find_spec on a dotted name imports the parent package: check "app" first,
        # so an environment without DisPatch is simply "not importable", not an error.
        if importlib.util.find_spec("app") is not None and importlib.util.find_spec("app.extensions") is not None:
            mod = __import__("app.extensions", fromlist=["EXT_API"])
            api = int(mod.EXT_API)
            out.update(importable=True, ext_api=api, compatible=lo <= api <= hi, source="import")
            return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    host = (backend / "app" / "extensions.py") if backend is not None else None
    if host is None or not host.exists():
        return out
    try:
        m = re.search(r"^EXT_API\s*=\s*(\d+)\s*(#.*)?$", host.read_text(encoding="utf-8"), re.MULTILINE)
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out
    if m is None:
        out["error"] = f"no EXT_API line in {host}"
        return out
    api = int(m.group(1))
    out.update(ext_api=api, compatible=lo <= api <= hi, source="file")
    return out


def probe_backend(backend: Path, checkout: Path | None) -> dict:
    """Two probes in DisPatch's backend dir, both offline and without syncing
    its venv: is the add-on installed IN the venv (a `uv sync` removes a
    pip-installed add-on), and is the editable overlay resolvable offline
    (the live unit's `--offline --with-editable` line, hardening H2)."""
    uv = _uv()
    code = "import dispatch_voice as d; print(d.__version__)"
    out: dict = {"dir": str(backend)}
    rc, text = _run([uv, "run", "--offline", "--no-sync", "python", "-c", code], backend, 60)
    out["installed_in_venv"] = {"ok": rc == 0, "version": text.strip() if rc == 0 else None,
                                "detail": None if rc == 0 else text}
    if checkout is not None:
        rc, text = _run([uv, "run", "--offline", "--no-sync", "--with-editable", str(checkout),
                         "python", "-c", code], backend, 120)
        out["overlay_resolvable_offline"] = {"ok": rc == 0, "version": text.strip() if rc == 0 else None,
                                             "detail": None if rc == 0 else text}
    return out


def probe_running(url: str) -> dict:
    """``GET /api/extensions`` on the running DisPatch (loopback is machine-inbound
    for that route): which add-on version is RUNNING vs this checkout."""
    import httpx
    out: dict = {"url": url, "reachable": False}
    try:
        r = httpx.get(url.rstrip("/") + "/api/extensions", timeout=3.0)
    except Exception as e:
        out["detail"] = f"{type(e).__name__}"
        return out
    out["reachable"] = True
    out["status_code"] = r.status_code
    if r.status_code != 200:
        return out
    try:
        rows = r.json().get("extensions") or []
    except ValueError:
        return out
    voice = next((e for e in rows if e.get("id") == "voice"), None)
    out["voice"] = voice
    return out


def probe_update() -> dict:
    """Compare this checkout with the repo's VERSION at the moving ``v1`` tag."""
    import urllib.request

    from . import __version__
    out = {"checkout": __version__, "latest": None}
    try:
        req = urllib.request.Request(VERSION_URL, headers={"User-Agent": "dispatch-voice-doctor"})
        with urllib.request.urlopen(req, timeout=5) as r:   # a fixed https URL
            out["latest"] = r.read(64).decode("ascii", "replace").strip()
    except Exception as e:
        out["error"] = f"{type(e).__name__}"
    return out


def probe_rig(settings) -> dict:
    """The GPU rig line: set up? switched on? and, when an address and a token are
    set, ``GET /health`` + ``GET /ready`` on it (3 s each). The token is never printed."""
    from .host import rig
    rs = settings.rig
    tok = rig.read_token()
    out: dict = {"enabled": rs.enabled, "url_set": bool(rs.url), "url_ok": bool(rs.url) and rig.valid_url(rs.url),
                 "host": rig.host_of(rs.url), "token": tok.source, "token_problem": tok.problem,
                 "token_file": str(rig.token_path())}
    if not (out["url_ok"] and tok.value):
        return out
    import httpx
    base = rig.http_base(rs.url)
    try:
        r = httpx.get(base + "/health", timeout=3.0)
        out["health"] = r.status_code
        out["version"] = (r.json() or {}).get("version") if r.status_code == 200 else None
    except Exception as e:
        out["health"] = None
        out["detail"] = type(e).__name__
        return out
    try:
        r = httpx.get(base + "/ready", timeout=3.0, headers={"Authorization": f"Bearer {tok.value}"})
        out["ready"] = r.status_code
        if r.status_code == 200:
            body = r.json() or {}
            out["gpu"] = (body.get("gpu") or {}).get("name")
            out["engine_shas"] = body.get("engine_shas")
    except Exception as e:
        out["ready"] = None
        out["detail"] = type(e).__name__
    return out


def rig_line(r: dict) -> tuple[str, int]:
    """(the doctor's GPU rig line, problems it counts)."""
    where = f" ({r['host']})" if r.get("host") else ""
    if not r["url_set"] and not r["enabled"]:
        return "GPU rig: not set up (optional; see docs/voice-chat/rig.md)", 0
    if r["url_set"] and not r["url_ok"]:
        return "GPU rig: rig.url in voice.yaml is not a ws:// or wss:// address", 1
    if r.get("token_problem") == "permissions":
        return f"GPU rig: {r['token_file']} is readable by other users: chmod 600 it", 1
    if not r.get("token"):
        line = f"GPU rig{where}: no SPEECHFORGE_TOKEN (set it in {r['token_file']}, mode 600)"
        return line, 1 if r["enabled"] else 0
    if not r["url_set"]:
        return "GPU rig: a token is set but rig.url is not (voice.yaml)", 1 if r["enabled"] else 0
    state = "on" if r["enabled"] else "off (rig.enabled is false)"
    if r.get("health") is None:
        return f"GPU rig{where}: {state}; not reachable ({r.get('detail', 'no answer')}): this computer speaks", 0
    if r.get("ready") == 401:
        return f"GPU rig{where}: {state}; the rig refused the token (401)", 1
    if r.get("ready") == 200:
        gpu = f" on {r['gpu']}" if r.get("gpu") else ""
        return f"GPU rig{where}: {state}; SpeechForge {r.get('version') or '?'} ready{gpu}", 0
    return (f"GPU rig{where}: {state}; SpeechForge {r.get('version') or '?'} reachable, not ready "
            f"(HTTP {r.get('ready')}): this computer speaks until it is"), 0


class DoctorContext:
    """What the doctor hands the plugins in place of DisPatch's context: a
    plugin that mounts routes (``api.include_router``) is a plugin that loads,
    so the call is recorded instead of refused (inside DisPatch the same call
    mounts the router under ``/api/voice`` behind the gate)."""

    def __init__(self) -> None:
        self.prefixes: list[str] = []

    def include_router(self, router, *, prefix: str = "", **_kw) -> None:
        self.prefixes.append(prefix)          # the full prefix (ExtensionAPI adds /api/voice)


def doctor_report(*, backend: Path | None, url: str, check_update: bool) -> dict:
    from . import EXTENSION, __version__
    from .host import context, registry
    from .host.config import load_settings
    from .host.engines import chatterbox
    checkout = Path(__file__).resolve().parents[2]
    editable = (checkout / "pyproject.toml").exists()
    rep: dict = {
        "version": __version__,
        "checkout": str(checkout) if editable else None,
        "extension": {"id": EXTENSION["id"], "api": list(EXTENSION["api"]), "setup": EXTENSION["setup"]},
        "dispatch": probe_dispatch_import(backend),
    }
    settings = load_settings()
    missing = settings.missing()
    rep["speech_models"] = {"dir": str(settings.models_dir), "missing": [m for m in missing if m.startswith("model ")]}
    rep["chatterbox_models"] = {"dir": str(chatterbox.model_dir()), "ok": chatterbox.available()}
    from .stock import cached_tensors_path
    rep["stock_voice"] = {"tensors": cached_tensors_path().exists()}
    rep["python_packages_missing"] = [m for m in missing if m.startswith("python package")]
    # What is switched on: the `extensions` setting (Settings > Voice, voice.yaml, or the deprecated
    # DISPATCH_VOICE_EXTENSIONS variable, which wins); entry-point plugins that are installed but off.
    from .host import providers
    declared = list(settings.extensions)
    spec = ",".join(declared)
    routers = DoctorContext()
    loaded = registry.mount_extensions(routers, spec) if spec else []
    if chatterbox.available():
        chatterbox.register()
    enabled = {t for t in declared if ":" not in t}
    rep["plugins"] = {"declared": declared, "loaded": loaded, "routers": routers.prefixes}
    rep["provider_plugins"] = providers.discover_entry_points(enabled)
    rep["engines"] = registry.engine_names()
    rep["builtin_voices"] = len(registry.builtin_voices(settings))
    rep["switch"] = {"enabled": settings.switched_on, "forced": settings.forced,
                     "voice_yaml": str(context.paths().voice_yaml)}
    rep["rig"] = probe_rig(settings)
    from .host import sysreq
    rep["system"] = sysreq.check()
    rep["running"] = probe_running(url)
    if backend is not None:
        rep["backend"] = probe_backend(backend, checkout if editable else None)
    if check_update:
        rep["update"] = probe_update()
    return rep


def _yes(ok: bool) -> str:
    return "yes" if ok else "no"


def print_doctor(rep: dict) -> int:
    problems = 0
    print(f"dispatch-voice {rep['version']}" + (f" (checkout {rep['checkout']})" if rep["checkout"] else ""))
    e = rep["extension"]
    print(f"extension: id {e['id']}, api {e['api']}, setup {e['setup']}")
    d = rep["dispatch"]
    if d.get("ext_api") is not None:
        where = "" if d["importable"] else f" (read from {rep.get('backend', {}).get('dir', '--backend')}/app/extensions.py)"
        print(f"DisPatch extension host: EXT_API {d['ext_api']} -> compatible: {_yes(bool(d['compatible']))}{where}")
        problems += 0 if d["compatible"] else 1
    elif rep.get("backend") and not d.get("error"):
        print(f"DisPatch extension host: {rep['backend']['dir']} has no app/extensions.py "
              "(a DisPatch before 2.2.0: update DisPatch)")
        problems += 1
    else:
        print("DisPatch extension host: not importable from this environment "
              "(run `dispatch-voice doctor` inside DisPatch's backend env, or pass --backend DIR)"
              + (f"; {d['error']}" if d.get("error") else ""))
    sm = rep["speech_models"]
    print(f"speech models ({sm['dir']}): " + ("ok" if not sm["missing"] else "MISSING " + ", ".join(sm["missing"])))
    problems += 1 if sm["missing"] else 0
    cb = rep["chatterbox_models"]
    print(f"Chatterbox models ({cb['dir']}): " + ("ok" if cb["ok"] else "MISSING (run `dispatch-voice fetch-models`)"))
    problems += 0 if cb["ok"] else 1
    print(f"stock voice tensors: {'ok' if rep['stock_voice']['tensors'] else 'not built (fetch-models builds them)'}")
    if rep["python_packages_missing"]:
        print("python packages MISSING: " + ", ".join(rep["python_packages_missing"]))
        problems += 1
    p = rep["plugins"]
    off = [a["name"] for a in rep.get("provider_plugins", []) if not a["enabled"]]
    if off:
        print(f"provider plugins installed but off: {', '.join(off)} (switch them on in Settings > Voice)")
    if p["declared"]:
        failed = [t for t in p["declared"] if t not in p["loaded"]]
        print(f"plugins (extensions): loaded {p['loaded'] or 'none'}"
              + (f"; routes under {', '.join(p['routers'])} (mounted inside DisPatch)"
                 if p.get("routers") else "")
              + (f"; FAILED {failed}" if failed else ""))
        problems += 1 if failed else 0
    else:
        print("plugins: none switched on")
    print(f"engines: {', '.join(rep['engines']) or 'none'}; built-in voices: {rep['builtin_voices']}")
    sw = rep["switch"]
    print(f"switch: {'on' if sw['enabled'] else 'off'}" + (f" (forced by DISPATCH_VOICE={sw['forced']})" if sw["forced"] else "")
          + f"  [{sw['voice_yaml']}]")
    if rep.get("rig") is not None:
        line, bad = rig_line(rep["rig"])
        print(line)
        problems += bad
    warnings = 0
    for s in rep.get("system", []):
        if s["ok"]:
            print(f"system program {s['id']}: ok")
        else:
            # Not a "problem": voice chat runs without them. They are what a feature needs.
            kind = "optional" if s["optional"] else "needed"
            print(f"system program {s['id']}: MISSING ({kind} for {s['required_for']})")
            print(f"  install ({s['package_manager']}): {s['fix']}")
            warnings += 1
    r = rep["running"]
    if not r["reachable"]:
        print(f"running server ({r['url']}): not reachable")
    elif r.get("voice") is None:
        print(f"running server ({r['url']}): HTTP {r.get('status_code')}, no voice extension listed"
              + ("" if r.get("status_code") == 200 else " (not an operator/loopback call?)"))
    else:
        v = r["voice"]
        running = v.get("version")
        line = f"running server ({r['url']}): voice {running} {v.get('state')} (api {v.get('api')})"
        if running and running != rep["version"]:
            line += f"  <- running {running}, checkout {rep['version']}: restart local-chat"
        print(line)
    b = rep.get("backend")
    if b:
        iv = b["installed_in_venv"]
        ov = b.get("overlay_resolvable_offline")
        if iv["ok"]:
            hint = f" ({iv['version']})"
        elif ov is not None and ov["ok"]:
            hint = "  (not installed; the --with-editable overlay resolves, which is the other way to run it)"
        else:
            hint = "  (a `uv sync` removes a pip-installed add-on; re-run the install line)"
        print(f"backend ({b['dir']}): add-on installed in its venv: {_yes(iv['ok'])}{hint}")
        if ov is not None:
            print(f"backend ({b['dir']}): overlay resolvable offline: {_yes(ov['ok'])}"
                  + ("" if ov["ok"] else "  (warm it once online: see docs/voice-chat/host.md)"))
            problems += 0 if ov["ok"] else 1
    u = rep.get("update")
    if u:
        if u.get("latest"):
            same = u["latest"] == u["checkout"]
            print(f"update check: latest 1.x is {u['latest']}; this checkout is {u['checkout']}"
                  + (" (up to date)" if same else " (a newer release exists: `git pull` and restart)"))
        else:
            print(f"update check: could not read {VERSION_URL} ({u.get('error')})")
    if warnings:
        print(f"warnings: {warnings} (system programs above)")
    print(f"problems: {problems}")
    return 1 if problems else 0


#: The voice settings a systemd unit may carry that change what the doctor finds.
#: Only these are read from the unit: never PATH, never a key or token.
UNIT_ENV_KEYS = ("DISPATCH_SPEECH_MODELS", "DISPATCH_VOICE_MODELS", "DISPATCH_VOICE_EXTENSIONS",
                 "DISPATCH_VOICE_KOKORO_DIR", "DISPATCH_VOICE_KOKORO_MODEL", "DISPATCH_DATA_DIR",
                 "LOCAL_CHAT_DATA_DIR", "DISPATCH_VOICE", "DISPATCH_VOICE_TTS", "XDG_DATA_HOME")


def unit_environment(unit: str) -> tuple[dict[str, str], str | None]:
    """The unit's own voice settings and working directory (``systemctl --user
    show``, fixed argv): run without them the doctor looked in the default data
    dir and reported the speech models "missing" and no plugins (V1 #5)."""
    import re
    import shlex
    if not re.fullmatch(r"[A-Za-z0-9@_.:-]{1,128}", unit or ""):
        raise ValueError(f"not a unit name: {unit!r}")
    try:
        r = subprocess.run(["systemctl", "--user", "show", unit, "-p", "Environment",
                            "-p", "WorkingDirectory"], capture_output=True, text=True, timeout=10,
                           check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"systemctl --user show {unit}: {type(e).__name__}") from e
    if r.returncode != 0:
        raise RuntimeError(f"systemctl --user show {unit} failed: {(r.stderr or '').strip()[-200:]}")
    out = r.stdout
    env: dict[str, str] = {}
    wd = None
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key == "WorkingDirectory":
            wd = value.strip() or None
        elif key == "Environment":
            try:
                pairs = shlex.split(value)
            except ValueError:
                pairs = value.split()
            for pair in pairs:
                k, sep, v = pair.partition("=")
                if sep and k in UNIT_ENV_KEYS:
                    env[k] = v
    return env, wd


def cmd_doctor(a) -> int:
    unit_wd = None
    if a.unit:
        try:
            env, unit_wd = unit_environment(a.unit)
        except (RuntimeError, ValueError) as e:
            print(f"doctor: {e}", file=sys.stderr)
            return 2
        os.environ.update(env)
        print(f"(settings read from the {a.unit} unit: {', '.join(sorted(env)) or 'none'})")
    backend = Path(a.backend).expanduser() if a.backend else (
        Path(os.environ["DISPATCH_BACKEND"]).expanduser() if os.environ.get("DISPATCH_BACKEND")
        else (Path(unit_wd) if unit_wd else None))
    rep = doctor_report(backend=backend, url=a.url, check_update=a.check_update)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    return print_doctor(rep)


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dispatch-voice")
    p.add_argument("--version", action="version", version=f"dispatch-voice {__version__}")
    p.add_argument("--models", help="Chatterbox model directory (default DISPATCH_VOICE_MODELS, else the "
                                    "chatterbox folder of the unified models folder)")
    p.add_argument("--dtype", default="q4", help="ONNX precision: q4 (default, fastest on CPU), q8, fp16, fp32")
    sub = p.add_subparsers(dest="cmd", required=True)
    fm = sub.add_parser("fetch-models", help="download every core model: Silero VAD, Smart Turn, Parakeet "
                                             "(~680 MB) and Chatterbox-Turbo (~720 MB at q4), then build the stock voice")
    fm.add_argument("--speech-models", help="where the VAD/Smart Turn/Parakeet files go (default "
                                            "DISPATCH_SPEECH_MODELS, else the speech folder of the unified "
                                            "models folder)")
    fm.add_argument("--only", choices=("speech", "chatterbox"), default=None, help="fetch one set only")
    fm.add_argument("--force", action="store_true", help="re-download the speech models even if present")
    fm.add_argument("--verify", action="store_true", help="only check what is on disk (sha256 for the speech models)")
    fm.add_argument("--no-stock", action="store_true",
                    help="skip building the stock voice tensors (built on first use instead)")
    fm.set_defaults(fn=cmd_fetch_models)
    mm = sub.add_parser("migrate-models", help="move the models from the old two folders into the one models "
                                               "folder (dry run unless --apply; safe to repeat)")
    mm.add_argument("--apply", action="store_true", help="actually move the files")
    mm.set_defaults(fn=cmd_migrate_models)
    dr = sub.add_parser("doctor", help="print the DisPatch handshake, the models, the engines and the "
                                       "running-vs-checkout version")
    dr.add_argument("--backend", help="DisPatch's backend directory (default $DISPATCH_BACKEND): also checks "
                                      "that the add-on is installed there and resolvable offline")
    dr.add_argument("--url", default=DEFAULT_SERVER, help=f"the running DisPatch (default {DEFAULT_SERVER})")
    dr.add_argument("--check-update", action="store_true",
                    help="compare this checkout with the repository's latest 1.x release (needs the network)")
    dr.add_argument("--unit", help="read the voice settings (DISPATCH_SPEECH_MODELS, "
                                   "DISPATCH_VOICE_EXTENSIONS, …) and the backend directory from this "
                                   "systemd --user unit, e.g. local-chat, so the doctor checks what the "
                                   "service really uses")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("serve", help="run the Voices panel on loopback for testing")
    s.add_argument("--data", help="a voice profile dir of its own (required; never DisPatch's live "
                                  "<data>/voices: this page has no login)")
    s.add_argument("--port", type=int, default=8769)
    say = sub.add_parser("say", help="speak a line in a profile's voice to a wav file")
    say.add_argument("profile")
    say.add_argument("text")
    say.add_argument("-o", "--out", default="say.wav")
    say.add_argument("--data")
    a = p.parse_args(argv)

    if a.cmd in ("fetch-models", "doctor", "migrate-models"):
        return a.fn(a)

    if a.cmd == "serve":
        why = _serve_refusal(a.data)
        if why:
            print(why, file=sys.stderr)
            return 2

    from .engine import ChatterboxOnnx
    from .profiles import ProfileStore
    models = Path(a.models) if a.models else None
    engine = ChatterboxOnnx(models, a.dtype)
    store = ProfileStore(_data_dir(a.data), engine)
    if a.cmd == "say":
        t = time.perf_counter()
        wav = store.speak_line(a.profile, a.text)
        Path(a.out).write_bytes(audio.to_wav_bytes(wav))
        print(f"{a.out}: {len(wav) / audio.SAMPLE_RATE:.2f}s audio in {time.perf_counter() - t:.2f}s {engine.last_stats}")
        return 0

    import uvicorn
    app = standalone_app(store, a.port)
    print(f"Voices panel: http://127.0.0.1:{a.port}/voices/ui  (loopback only, no auth)", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
