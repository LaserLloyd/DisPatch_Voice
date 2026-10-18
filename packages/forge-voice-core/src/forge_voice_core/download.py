"""Fetch the speech-recognition and turn-taking models into a local directory.

    dispatch-voice fetch-models                 # all core models (this list + Chatterbox + the stock voice)
    python -m dispatch_voice.host.download [--dest DIR] [--force] [--verify]

DIR defaults to the same place the server looks (``DISPATCH_SPEECH_MODELS``,
else ``<data dir>/models/speech``). Standard library only. Every file comes
from a public, ungated release; nothing here needs an account or a token.

Total on disk: ~680 MB. That is Parakeet int8 (~670 MB), Smart Turn (~8 MB)
and Silero (~2 MB). The voice that speaks comes separately: Chatterbox-Turbo
(``dispatch_voice.engine.fetch_models``, ~720 MB at q4, fetched by the same
``dispatch-voice fetch-models`` command) and, optionally, the Kokoro built-in
voices from the separately installed GPL-3.0 plugin ``dispatch-voice-kokoro``
(``dispatch-voice-kokoro fetch-models``). This script never fetches Kokoro.

Licences: Silero VAD (MIT), Smart Turn v3 (BSD-2), Parakeet TDT 0.6B v2
(CC-BY-4.0, NVIDIA; ONNX int8 conversion by istupakov). See
THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

_HF = "https://huggingface.co"
_GH = "https://github.com"

# Pinned revisions: a moving branch (`main`, `master`) could hand a fresh
# install different bytes from the ones this release was tested with. Every
# file is ALSO checked against its sha256 below, so a re-tagged release asset
# or a tampered mirror fails loudly instead of being loaded.
_SILERO_REV = "1e261b036686cd0017d500ee96acd1c4ba572a9d"
_SMART_TURN_REV = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
_PARAKEET_REV = "0bbb45a3365852604aef28b538a8f066f4ccaa85"

#: (relative path under DEST, URL, sha256). Relative paths are what config.py expects.
FILES: list[tuple[str, str, str]] = [
    ("silero_vad.onnx",
     f"{_GH}/snakers4/silero-vad/raw/{_SILERO_REV}/src/silero_vad/data/silero_vad.onnx",
     "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"),
    ("smart-turn-v3.2-cpu.onnx",
     f"{_HF}/pipecat-ai/smart-turn-v3/resolve/{_SMART_TURN_REV}/smart-turn-v3.2-cpu.onnx",
     "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"),
] + [
    (f"parakeet-tdt-0.6b-v2/{name}",
     f"{_HF}/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/{_PARAKEET_REV}/{name}", sha)
    for name, sha in (
        ("config.json", "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
        ("vocab.txt", "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
        ("nemo128.onnx", "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f"),
        ("encoder-model.int8.onnx",
         "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155"),
        ("decoder_joint-model.int8.onnx",
         "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3"),
    )
]


def files_for() -> list[tuple[str, str, str]]:
    """The download list (speech recognition + turn taking only)."""
    return list(FILES)


KOKORO_MOVED = ("Kokoro is no longer fetched here: it is the optional, separately installed "
                "dispatch-voice-kokoro plugin (GPL-3.0). Use `dispatch-voice-kokoro fetch-models "
                "[--int8]` (docs/voice-chat/host.md).")


class ChecksumError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    """The operator cancelled the download (the partial file is removed)."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str, *, progress: Callable[[int, int], None] | None = None,
           cancel=None) -> None:
    """Download to a .part file, verify, then move into place. A file that
    fails its checksum never reaches `dest`. ``progress(bytes_done, bytes_total)`` is called
    after every chunk (total 0 when the server does not say); ``cancel`` (a ``threading.Event``)
    stops the download between chunks, removing the partial file (:class:`Cancelled`)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dispatch-voice-download"})
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0) if getattr(r, "headers", None) else 0
            while chunk := r.read(1 << 20):
                if cancel is not None and cancel.is_set():
                    raise Cancelled("cancelled")
                h.update(chunk)
                f.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ChecksumError(f"sha256 mismatch (got {h.hexdigest()}, want {sha256})")
    tmp.replace(dest)


#: The real downloader, kept so a test double (or an operator's fetcher) is told apart from it.
_REAL_FETCH = _fetch


def _approx_mb(rel: str) -> float:
    from .fetch_job import APPROX_MB
    return float(APPROX_MB["speech"].get(Path(rel).name, 1))


def fetch_speech_models(dest: Path, *, force: bool = False, verify: bool = False,
                        echo: Callable[[str], None] = print, fetch: Callable | None = None,
                        progress: Callable[[dict], None] | None = None, cancel=None) -> int:
    """Fetch (or with ``verify`` only check) every file in :data:`FILES` under
    ``dest``. Returns 0 when everything is present and verified, 1 otherwise.
    ``progress(event)`` gets ``{phase, file, index, total, bytes, bytes_total, weight_mb, done}``;
    ``cancel`` is a ``threading.Event`` (raises :class:`Cancelled`)."""
    fetch = fetch or _fetch
    dest = Path(dest)
    echo(f"speech models -> {dest}")
    bad = 0
    files = files_for()
    for idx, (rel, url, sha) in enumerate(files, 1):
        if cancel is not None and cancel.is_set():
            raise Cancelled("cancelled")
        target = dest / rel
        ev = {"phase": "speech", "file": rel, "index": idx, "total": len(files), "weight_mb": _approx_mb(rel)}
        have = target.exists() and target.stat().st_size > 0
        if verify or (have and not force):
            if not have:
                echo(f"  MISSING {rel}")
                bad += 1
            elif sha256_of(target) != sha:
                echo(f"  BAD    {rel}: sha256 mismatch (re-run with --force)")
                bad += 1
            else:
                echo(f"  have  {rel}")
            if progress is not None:
                progress({**ev, "done": True})
            continue
        echo(f"  fetch {rel}")
        try:
            if fetch is _REAL_FETCH:
                fetch(url, target, sha, cancel=cancel,
                      progress=(lambda d, t, ev=ev: progress({**ev, "bytes": d, "bytes_total": t}))
                      if progress is not None else None)
            else:
                fetch(url, target, sha)                     # a test double or a custom fetcher
        except Cancelled:
            raise
        except Exception as e:
            echo(f"  FAILED {rel}: {e}")
            return 1
        if progress is not None:
            progress({**ev, "done": True})
    return 1 if bad else 0


def fetch_all(*, only: str | None = None, force: bool = False, verify: bool = False, stock: bool = True,
              dtype: str = "q4", speech_dir: Path | None = None, models_dir: Path | None = None,
              echo: Callable[[str], None] = print, progress: Callable[[dict], None] | None = None,
              cancel=None, fetch: Callable | None = None, fetch_chatterbox: Callable | None = None,
              build_stock: Callable | None = None) -> int:
    """Everything ``dispatch-voice fetch-models`` does, callable from the CLI and from the
    in-app download job (:mod:`host.fetch_job`): the speech models, the Chatterbox graphs, the
    stock voice. ``only`` is ``"speech"``, ``"chatterbox"`` or None (both). Returns 0 / 1."""
    import time as _time

    from . import models as _models
    rc = 0
    if only in (None, "speech"):
        rc = fetch_speech_models(Path(speech_dir) if speech_dir else _models.speech_dir(), force=force,
                                 verify=verify, echo=echo, fetch=fetch, progress=progress, cancel=cancel)
    if only in (None, "chatterbox"):
        if cancel is not None and cancel.is_set():
            raise Cancelled("cancelled")
        from .engine import ChatterboxOnnx
        from .engine import fetch_models as _fetch_chatterbox
        target = Path(models_dir) if models_dir else None
        if verify:
            from .engines import chatterbox
            ok = chatterbox.available()
            echo(f"Chatterbox models ({chatterbox.model_dir()}): {'have' if ok else 'MISSING'}")
            rc = rc or (0 if ok else 1)
        else:
            fc = fetch_chatterbox or _fetch_chatterbox

            def on_file(name: str, idx: int, total: int, done: bool) -> None:
                if progress is not None:
                    progress({"phase": "chatterbox", "file": name, "index": idx, "total": total,
                              "weight_mb": 720.0 / max(1, total), "done": done})

            d = fc(target or _models.chatterbox_dir(), dtype, on_file=on_file, cancel=cancel)
            echo(f"Chatterbox models -> {d}")
            if stock:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("cancelled")
                if build_stock is None:
                    from .stock import build_tensors as build_stock_voice
                else:
                    build_stock_voice = build_stock
                t = _time.perf_counter()
                path = build_stock_voice(ChatterboxOnnx(d, dtype, watermark=False), d)
                echo(f"stock voice: {path} ({_time.perf_counter() - t:.1f}s)")
                if progress is not None:
                    progress({"phase": "stock", "file": "stock voice", "index": 1, "total": 1, "weight_mb": 20.0,
                              "done": True})
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--verify", action="store_true",
                    help="only check the files already on disk against their sha256")
    ap.add_argument("--kokoro-int8", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.kokoro_int8:
        print(f"note: {KOKORO_MOVED}", file=sys.stderr)
    if args.dest is None:
        from . import hostlink
        args.dest = hostlink.get().models_root()
    return fetch_speech_models(args.dest, force=args.force, verify=args.verify)


if __name__ == "__main__":
    sys.exit(main())
