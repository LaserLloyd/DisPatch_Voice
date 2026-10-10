"""A stand-in DisPatch for the wire round trip (tests/e2e_voice_roundtrip.py --host fake).

The FakeContext of tests/fakes.py plus the add-on's own ``setup(ctx)``, served
by uvicorn, so the whole speech pipeline runs end to end on the new seam
with no DisPatch checkout: the real engines, the real ``ws.serve``, the real
``/api/voice`` routes, and DisPatch's part played by the contract double.

    DISPATCH_DATA_DIR=<dir> DISPATCH_VOICE=1 DISPATCH_SPEECH_MODELS=<speech models> \\
      python -m uvicorn --app-dir tests/host fake_host:app --host 127.0.0.1 --port <port>

Routes beyond the add-on's: ``POST /api/threads {bot_id}``,
``GET /api/threads/{id}/messages``, ``/ws/voice/{id}`` (the FakeContext gate),
``GET /_fake/calls`` (every ``submit_turn`` the add-on made: the instruction
placements, the thinking level, the metadata). The bot ``voicetest`` answers
every turn with :data:`REPLY` as an advisor-style stream.
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # tests/: fakes.py

from fakes import BotView, FakeContext
from fastapi import WebSocket

REPLY = ("You have two things this afternoon. A dentist appointment at three, and a call "
         "with the school at four thirty. Want me to remind you before each one?")
_ids = itertools.count(1)


def build_app(data_dir: Path):
    from dispatch_voice.host import setup
    voice = os.environ.get("FAKE_HOST_VOICE", "")
    ctx = FakeContext(data_dir, bots=[
        BotView("voicetest", "Voice Test", backend="api", voice=voice),
        BotView("alpha", "Alpha", safe=True, backend="gateway"),
    ])
    ctx.reply_for = lambda text: REPLY
    # A direct bot: abort_turn(keep_partial=True) persists the spoken prefix as the
    # assistant row (hardening H3), which the session then annotates with `voice_heard`.
    ctx.partial_on_abort = REPLY.split(". ")[0] + "."
    setup(ctx)
    app = ctx.app

    @app.post("/api/threads")
    async def create_thread(body: dict):
        bot_id = str(body.get("bot_id") or "voicetest")
        if ctx.bots.get(bot_id) is None:
            return {"error": "no such bot"}
        tid = f"t-{next(_ids)}"
        ctx.add_thread(tid, bot_id, "Voice e2e")
        return {"id": tid, "bot_id": bot_id}

    @app.get("/api/threads/{tid}/messages")
    async def messages(tid: str):
        return {"messages": [m.model_dump() for m in ctx.db().messages(tid)]}

    @app.websocket("/ws/voice/{thread_id}")
    async def voice_socket(websocket: WebSocket, thread_id: str):
        await ctx.connect("voice", thread_id, websocket)

    @app.get("/_fake/calls")
    async def calls():
        out = []
        for name, args, kw in ctx.calls:
            if name != "submit_turn":
                continue
            instr = kw.get("instructions")
            out.append({"thread_id": args[0], "text": args[1],
                        "instructions": dataclasses.asdict(instr) if dataclasses.is_dataclass(instr) else instr,
                        "thinking": kw.get("thinking"), "metadata": kw.get("metadata")})
        return {"submit_turn": out, "power_hold": __import__("dispatch_voice.host.power", fromlist=["state"]).state()}

    @app.on_event("startup")
    async def run_startup_hooks():
        for fn, threaded in ctx.startup:          # the add-on's prewarm, as DisPatch runs it
            if threaded:
                threading.Thread(target=fn, name="fake-host-startup", daemon=True).start()
            else:
                fn()

    return app


app = build_app(Path(os.environ.get("DISPATCH_DATA_DIR") or tempfile.mkdtemp(prefix="dispatch-voice-fake-host-")))
