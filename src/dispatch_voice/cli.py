"""dispatch-voice: fetch models, run the Voices panel standalone, or say a line."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import audio


def _data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "local-chat" / "voices"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dispatch-voice")
    p.add_argument("--models", help="model directory (default ~/.local/share/dispatch-voice/models)")
    p.add_argument("--dtype", default="q4", help="ONNX precision: q4 (default, fastest on CPU), q8, fp16, fp32")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-models", help="download the Chatterbox-Turbo ONNX files (~720 MB at q4)")
    s = sub.add_parser("serve", help="run the Voices panel on loopback for testing")
    s.add_argument("--data", help="voice profile dir (default ~/.local/share/local-chat/voices)")
    s.add_argument("--port", type=int, default=8769)
    say = sub.add_parser("say", help="speak a line in a profile's voice to a wav file")
    say.add_argument("profile")
    say.add_argument("text")
    say.add_argument("-o", "--out", default="say.wav")
    say.add_argument("--data")
    a = p.parse_args(argv)

    from .engine import ChatterboxOnnx, fetch_models
    models = Path(a.models) if a.models else None
    if a.cmd == "fetch-models":
        print(fetch_models(models, a.dtype))
        return 0

    from .profiles import ProfileStore
    engine = ChatterboxOnnx(models, a.dtype)
    store = ProfileStore(_data_dir(a.data), engine)
    if a.cmd == "say":
        t = time.perf_counter()
        wav = store.speak_line(a.profile, a.text)
        Path(a.out).write_bytes(audio.to_wav_bytes(wav))
        print(f"{a.out}: {len(wav) / audio.SAMPLE_RATE:.2f}s audio in {time.perf_counter() - t:.2f}s {engine.last_stats}")
        return 0

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from .router import build_router
    app = FastAPI(title="dispatch-voice (standalone)")
    app.include_router(build_router(store, allow_open=True), prefix="/voices")
    app.get("/")(lambda: RedirectResponse("/voices/ui"))
    print(f"Voices panel: http://127.0.0.1:{a.port}/voices/ui  (loopback only, no auth)", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
