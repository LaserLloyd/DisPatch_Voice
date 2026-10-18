"""Day-0 test doubles shared by every track (owned by T0; see tests/README.md).

* Contract mirrors of the DisPatch extension types (``VoiceInstructions``,
  ``BotView``, ``ExtPaths``, ``ToolSpec``, ``ThreadSocket``). DisPatch defines the
  real ones in ``app/extensions.py``; the add-on never imports DisPatch, so its
  tests build against these shapes, which follow the v3 contracts §B.1 field for
  field.
* ``FakeContext``: an ``ExtensionContext`` stand-in with an in-memory database.
  Every call is recorded; ``subscribe_frames`` replays a recorded B.1a frame
  sequence (snapshot first) through a bounded queue, as DisPatch does.
* ``FakeSTT`` / ``FakeTTS`` / ``FakeWS`` / ``OneTurnDetector`` /
  ``ScriptedDetector``: the speech-engine doubles of the pre-split test suite.

Nothing here touches a model, the network or a real DisPatch install.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from fastapi import Request  # module level: FastAPI resolves the gate's annotation here

from dispatch_voice import fsperm


def make_private(path, *, private: bool = True) -> None:
    """Give ``path`` the protection the token-file rules look at: owner-only (``private``)
    or readable by everyone. POSIX: chmod 600 / 644. Windows: the ACL (restrict, or grant
    Everyone read)."""
    import os
    import subprocess
    if fsperm.IS_WINDOWS:
        if private:
            assert fsperm.restrict(path)
        else:
            subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"], capture_output=True, check=True)
        return
    os.chmod(path, 0o600 if private else 0o644)


def assert_private(path, *, directory: bool = False) -> None:
    """POSIX: exactly 0600 (0700 for a directory). Windows: the mode bits mean nothing there
    (always 0o666/0o777), so the ACL is checked instead: nobody but the owner and the system
    may be granted access. A filesystem whose ACL cannot be read is skipped, not failed."""
    import os
    import stat
    if fsperm.IS_WINDOWS:
        got = fsperm.others_can_read(path)
        if got is None:
            import pytest
            pytest.skip("cannot read the ACL on this filesystem")
        assert got is False, f"{path} grants access to other users"
        return
    want = 0o700 if directory else 0o600
    assert stat.S_IMODE(os.stat(path).st_mode) == want, oct(stat.S_IMODE(os.stat(path).st_mode))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

Backend = Literal["gateway", "api", "advisor"]

# --------------------------------------------------------------------------- #
# Contract mirrors (C-§B.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VoiceInstructions:
    system: str = ""
    trailer: str = ""


@dataclass(frozen=True)
class BotView:
    id: str
    name: str
    safe: bool = False
    voice: str = ""
    voice_rp: str = ""
    backend: Backend | None = "gateway"


@dataclass(frozen=True)
class ExtPaths:
    voice_yaml: Path
    acks_yaml: Path
    voices_dir: Path
    cache_dir: Path
    models_dir: Path
    state_dir: Path

    @classmethod
    def under(cls, data: Path, models_dir: Path | None = None) -> ExtPaths:
        state = data / "ext" / "voice"
        state.mkdir(parents=True, exist_ok=True)
        fsperm.ensure_private(state, directory=True)
        return cls(voice_yaml=data / "voice.yaml", acks_yaml=data / "voice-acks.yaml",
                   voices_dir=data / "voices", cache_dir=data / "voice-cache",
                   models_dir=models_dir or data / "speech-models", state_dir=state)


@dataclass
class ToolSpec:
    id: str
    feature: str
    title: str
    icon: str
    opt_in: bool
    probe: Callable[[], bool]


@dataclass
class ThreadSocket:
    thread: Any
    bot: BotView
    token: str | None
    session_live: Callable[[], bool]
    touch: Callable[[], None]
    close_unavailable: Callable[[str], Awaitable[None]]
    #: Amendment T1-1: callbacks the host runs before it closes this socket
    #: 4001 "replaced" (a newer socket of the same session took the slot).
    on_replaced: Callable[[Callable[[], Any]], None] = field(default_factory=lambda: (lambda fn: None))
    #: DisPatch 2.3.4: the socket came from a locked (Safe-Mode) device.
    locked: bool = False


# --------------------------------------------------------------------------- #
# In-memory database
# --------------------------------------------------------------------------- #

_ids = itertools.count(1)


def new_id(prefix: str = "m") -> str:
    return f"{prefix}-{next(_ids):06d}"


@dataclass
class FakeMessage:
    id: str
    thread_id: str
    role: str
    content: str
    created_at: str
    media_url: str | None = None
    metadata: dict[str, Any] | None = None

    def model_dump(self) -> dict:
        """The MessageOut dict DisPatch puts in `message` frames."""
        return {"id": self.id, "thread_id": self.thread_id, "role": self.role,
                "content": self.content, "created_at": self.created_at,
                "media_url": self.media_url,
                "metadata": json.loads(json.dumps(self.metadata)) if self.metadata else None}


class FakeDatabase:
    """The few `Database` calls the voice host makes, kept in memory."""

    def __init__(self) -> None:
        self.threads: dict[str, SimpleNamespace] = {}
        self.rows: dict[str, FakeMessage] = {}
        self.order: dict[str, list[str]] = {}

    def add_thread(self, thread_id: str, bot_id: str, title: str = "") -> SimpleNamespace:
        t = SimpleNamespace(id=thread_id, bot_id=bot_id, title=title, status="idle")
        self.threads[thread_id] = t
        self.order.setdefault(thread_id, [])
        return t

    def get_thread(self, thread_id: str) -> SimpleNamespace | None:
        return self.threads.get(thread_id)

    def add_message(self, thread_id: str, role: str, content: str, *,
                    metadata: dict | None = None, msg_id: str | None = None) -> FakeMessage:
        m = FakeMessage(id=msg_id or new_id(), thread_id=thread_id, role=role, content=content,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        metadata=dict(metadata) if metadata else None)
        self.rows[m.id] = m
        self.order.setdefault(thread_id, []).append(m.id)
        return m

    def get_message(self, msg_id: str) -> FakeMessage | None:
        return self.rows.get(msg_id)

    def messages(self, thread_id: str) -> list[FakeMessage]:
        return [self.rows[i] for i in self.order.get(thread_id, []) if i in self.rows]

    def last_message_id(self, thread_id: str) -> str | None:
        ids = [i for i in self.order.get(thread_id, []) if i in self.rows]
        return ids[-1] if ids else None

    def merge_message_metadata(self, msg_id: str, patch: dict) -> FakeMessage | None:
        m = self.rows.get(msg_id)
        if m is None:
            return None
        meta = dict(m.metadata or {})
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(meta.get(k), dict):
                meta[k] = {**meta[k], **v}      # merged, never rewritten (C-§F.6)
            else:
                meta[k] = v
        m.metadata = meta
        return m

    def delete_message(self, msg_id: str) -> bool:
        return self.rows.pop(msg_id, None) is not None


# --------------------------------------------------------------------------- #
# Recorded B.1a frames
# --------------------------------------------------------------------------- #


def recorded_frames() -> dict:
    """tests/fixtures/frames_b1a.json: named B.1a frame sequences (see its `$about`)."""
    return json.loads((FIXTURES / "frames_b1a.json").read_text(encoding="utf-8"))


def frames_for(name: str, thread_id: str | None = None, bot_id: str | None = None) -> list[dict]:
    """One recorded sequence, optionally re-targeted at another thread/bot."""
    seq = json.loads(json.dumps(recorded_frames()["sequences"][name]))
    for f in seq:
        if thread_id is not None:
            f["thread_id"] = thread_id
            msg = f.get("message")
            if isinstance(msg, dict):
                msg["thread_id"] = thread_id
        if bot_id is not None and "bot_id" in f:
            f["bot_id"] = bot_id
    return seq


# --------------------------------------------------------------------------- #
# Speculation double (C-§B.1 SpeculativeRun)
# --------------------------------------------------------------------------- #


class FakeSpeculativeRun:
    def __init__(self, text_in: str, *, head_id: str | None, reply: str = "") -> None:
        self.text_in = text_in
        self.text = reply
        self.done = False
        self.head_id = head_id
        self.prompt_chars = 1200 + len(text_in)
        self.discarded = False

    def feed(self, more: str, *, done: bool = False) -> None:
        self.text += more
        self.done = done

    async def discard(self) -> None:      # idempotent; never persists
        self.discarded = True
        self.done = True


# --------------------------------------------------------------------------- #
# FakeContext (C-§B.1 ExtensionContext)
# --------------------------------------------------------------------------- #

ALLOWED_STATIC = {"js", "mjs", "css", "json", "map", "wasm", "onnx", "wav", "mp3", "ogg",
                  "png", "jpg", "webp", "woff2"}

DEFAULT_BOTS = (
    BotView("sage", "Sage", backend="advisor", voice="stock-host"),
    BotView("swift", "Swift", backend="api"),
    BotView("nova", "Nova", backend="gateway", voice="builtin:af_heart"),
    BotView("alpha", "Alpha", safe=True, backend="gateway"),
)


class FakeBots:
    def __init__(self, bots: Iterable[BotView]) -> None:
        self._bots = {b.id: b for b in bots}

    def get(self, bot_id: str) -> BotView | None:
        return self._bots.get(bot_id)

    def all(self) -> list[BotView]:
        return list(self._bots.values())


class FakeContext:
    """Records every call; behaves like DisPatch 2.2.0's host where tests need it to.

    Knobs (plain attributes, set them in a test):
      operator             is_operator()/require_operator() outcome (True)
      reply_for            None, or fn(user_text) -> reply text: after submit_turn the
                           ctx emits an advisor-style stream for that reply and
                           persists the assistant row (frames as main.py emits them)
      speculation_reply    text a new FakeSpeculativeRun starts with ("")
      partial_on_abort     visible text persisted by abort_turn(keep_partial=True)
      turn_running         set of thread ids with a turn task in flight
    """

    api_version = 1
    app_version = "2.2.0"
    ext_id = "voice"

    def __init__(self, data_dir: Path, *, bots: Iterable[BotView] = DEFAULT_BOTS,
                 env: dict[str, str] | None = None, models_dir: Path | None = None,
                 queue_max: int = 256) -> None:
        from fastapi import FastAPI

        self.log = logging.getLogger("local-chat.ext.voice")
        self.bots = FakeBots(bots)
        self.paths = ExtPaths.under(Path(data_dir), models_dir)
        self.env = dict(env or {})
        self.app = FastAPI()
        self._db = FakeDatabase()
        self.calls: list[tuple[str, tuple, dict]] = []
        self.operator = True
        self.reply_for: Callable[[str], str] | None = None
        self.speculation_reply = ""
        self.partial_on_abort: str | None = None
        self.turn_running: set[str] = set()
        self.speculations: dict[str, FakeSpeculativeRun] = {}
        self.validators: dict[str, Callable[[str], None]] = {}
        self.tools: list[ToolSpec] = []
        self.startup: list[tuple[Callable, bool]] = []
        self.shutdown: list[Callable] = []
        self.websockets: dict[str, dict] = {}
        self.open_sockets: dict[str, int] = {}
        self.static: dict | None = None
        self.routers: list[tuple[Any, str, dict]] = []
        self.body_limits: dict[str, int] = {}
        self.touched: list[str | None] = []
        self._queue_max = queue_max
        self._subs: dict[str, list[_Subscription]] = {}
        self._tasks: dict[asyncio.Task, str] = {}
        self.dropped_frames = 0

    # -- helpers for tests ------------------------------------------------- #

    def _rec(self, name: str, *a, **kw) -> None:
        self.calls.append((name, a, kw))

    def calls_to(self, name: str) -> list[tuple[tuple, dict]]:
        return [(a, kw) for n, a, kw in self.calls if n == name]

    def add_thread(self, thread_id: str, bot_id: str = "sage", title: str = "") -> SimpleNamespace:
        return self._db.add_thread(thread_id, bot_id, title)

    async def drain(self) -> None:
        """Wait until every scheduled frame delivery and scripted reply has run."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await asyncio.sleep(0)
        for subs in self._subs.values():
            for s in subs:
                await s.flush()

    def emit(self, frame: dict) -> None:
        """Push one frame to the subscribers of its thread (the broadcast tap)."""
        for s in list(self._subs.get(frame.get("thread_id") or "", [])):
            s.put(frame)

    # -- ExtensionContext surface ------------------------------------------ #

    def db(self) -> FakeDatabase:
        return self._db

    def is_operator(self, request) -> bool:
        return bool(self.operator)

    def require_operator(self, request) -> None:
        from fastapi import HTTPException
        if not self.operator:
            raise HTTPException(403, "Forbidden")       # the decoy 403

    def include_router(self, router, *, prefix: str, body_limits: dict[str, int] | None = None) -> None:
        from fastapi import Depends
        from starlette.routing import WebSocketRoute

        self._rec("include_router", router, prefix=prefix, body_limits=body_limits)
        if not (prefix == "/api/voice" or prefix.startswith("/api/voice/")):
            raise ValueError(f"extension routes must live under /api/voice, not {prefix!r}")
        for r in getattr(router, "routes", []):
            if isinstance(r, WebSocketRoute) or type(r).__name__ == "APIWebSocketRoute":
                raise ValueError("thread_websocket is the only websocket path for an extension")

        def _gate(request: Request) -> None:
            self.require_operator(request)

        self.app.include_router(router, prefix=prefix, dependencies=[Depends(_gate)])
        self.routers.append((router, prefix, dict(body_limits or {})))
        self.body_limits.update(body_limits or {})

    def serve_static(self, directory: Path, *, entry: str, css: Sequence[str] = (),
                     locales: bool = False) -> str:
        directory = Path(directory)
        files = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        h = hashlib.sha256()
        for p in files:
            h.update(p.relative_to(directory).as_posix().encode() + b"\0" + p.read_bytes())
        version = h.hexdigest()[:16]
        not_served = [p.relative_to(directory).as_posix() for p in files
                      if p.suffix.lstrip(".").lower() not in ALLOWED_STATIC]
        self.static = {"directory": directory, "entry": entry, "css": list(css), "locales": locales,
                       "assets_version": version, "not_served": not_served}
        self._rec("serve_static", directory, entry=entry, css=tuple(css), locales=locales)
        return version

    def thread_websocket(self, name: str, handler, *, max_per_session: int = 2, max_total: int = 3) -> None:
        self._rec("thread_websocket", name, handler, max_per_session=max_per_session, max_total=max_total)
        self.websockets[name] = {"handler": handler, "max_per_session": max_per_session,
                                 "max_total": max_total}

    def socket_count(self, name: str) -> tuple[int, int]:
        return self.open_sockets.get(name, 0), self.websockets.get(name, {}).get("max_total", 3)

    def thread_socket(self, thread_id: str, *, ws=None, token: str | None = "tok-unlocked",
                      live: Callable[[], bool] | None = None, locked: bool = False) -> ThreadSocket:
        """The ThreadSocket the gate would hand the handler (a test builds one directly)."""
        thread = self._db.get_thread(thread_id) or self.add_thread(thread_id)
        bot = self.bots.get(thread.bot_id) or BotView(thread.bot_id, thread.bot_id)
        closed: list[str] = []

        async def close_unavailable(reason: str) -> None:
            closed.append(reason[:120])
            if ws is not None:
                await ws.accept()
                await ws.close(1013, reason[:120])

        replaced: list[Callable[[], Any]] = []
        sock = ThreadSocket(thread=thread, bot=BotView(bot.id, bot.name, False, bot.voice, bot.voice_rp, bot.backend),
                            token=token, session_live=live or (lambda: True),
                            touch=lambda: self.touched.append(token),
                            close_unavailable=close_unavailable,
                            on_replaced=replaced.append, locked=locked)
        sock.unavailable_reasons = closed      # type: ignore[attr-defined]
        sock.replaced_callbacks = replaced     # type: ignore[attr-defined]

        async def evict() -> None:
            """What DisPatch's `_Slot.evict` does: the callbacks, then close 4001."""
            for fn in list(replaced):
                res = fn()
                if hasattr(res, "__await__"):
                    await res
            if ws is not None:
                await ws.close(4001, "replaced")

        sock.evict = evict                     # type: ignore[attr-defined]
        return sock

    async def connect(self, name: str, thread_id: str, ws, *, token: str | None = "tok-unlocked") -> None:
        """Run the gate's last steps and the registered handler (Origin/PIN are DisPatch's tests)."""
        thread = self._db.get_thread(thread_id)
        bot = self.bots.get(thread.bot_id) if thread else None
        if thread is None or bot is None:
            await ws.close(1008)                        # pre-accept refusal
            return
        reg = self.websockets[name]
        if self.open_sockets.get(name, 0) >= reg["max_total"]:
            await ws.accept()
            await ws.close(1013, f"Voice is busy on this server ({reg['max_total']} chats open)")
            return
        self.open_sockets[name] = self.open_sockets.get(name, 0) + 1
        try:
            await reg["handler"](ws, self.thread_socket(thread_id, ws=ws, token=token))
        finally:
            self.open_sockets[name] -= 1

    async def submit_turn(self, thread_id: str, text: str, *, instructions=None, thinking=None,
                          metadata: dict | None = None, prefetched=None) -> str | None:
        self._rec("submit_turn", thread_id, text, instructions=instructions, thinking=thinking,
                  metadata=metadata, prefetched=prefetched)
        if not (text or "").strip():
            return None
        if prefetched is not None and prefetched.head_id != self._db.last_message_id(thread_id):
            await prefetched.discard()                  # stale head: submit normally (rule 4)
            self.calls.append(("speculation_stale", (thread_id,), {}))
            prefetched = None
        thread = self._db.get_thread(thread_id) or self.add_thread(thread_id)
        row = self._db.add_message(thread_id, "user", text.strip(), metadata=metadata)
        self.emit({"type": "message", "thread_id": thread_id, "bot_id": thread.bot_id,
                   "message": row.model_dump()})       # broadcast BEFORE the turn task
        if self.reply_for is not None:
            reply = self.reply_for(text)
            if prefetched is not None and prefetched.text and not reply.startswith(prefetched.text):
                reply = prefetched.text + reply
            self._spawn(thread_id, self._scripted_reply(thread_id, thread.bot_id, reply))
        return row.id

    async def _scripted_reply(self, thread_id: str, bot_id: str, reply: str) -> None:
        self.turn_running.add(thread_id)
        prov = f"run:advisor-{new_id('p')}"
        try:
            self.emit({"type": "thinking", "thread_id": thread_id, "bot_id": bot_id, "status": "started"})
            self.emit({"type": "stream_start", "thread_id": thread_id, "bot_id": bot_id, "message_id": prov})
            words = reply.split(" ")
            for i in range(0, len(words), 4):
                await asyncio.sleep(0)
                piece = (" " if i else "") + " ".join(words[i:i + 4])
                self.emit({"type": "stream_chunk", "thread_id": thread_id, "bot_id": bot_id,
                           "message_id": prov, "text": piece})
            row = self._db.add_message(thread_id, "assistant", reply)
            self.emit({"type": "stream_done", "thread_id": thread_id, "bot_id": bot_id,
                       "message_id": row.id, "message": row.model_dump(), "provisional_id": prov})
            self.emit({"type": "thinking", "thread_id": thread_id, "bot_id": bot_id, "status": "stopped"})
        finally:
            self.turn_running.discard(thread_id)

    async def speculate_turn(self, thread_id: str, text: str, *, instructions=None,
                             thinking=None) -> FakeSpeculativeRun | None:
        self._rec("speculate_turn", thread_id, text, instructions=instructions, thinking=thinking)
        thread = self._db.get_thread(thread_id)
        if thread is None or self.bot_backend(thread.bot_id) not in ("advisor", "api"):
            return None
        if thread_id in self.turn_running:
            return None
        old = self.speculations.pop(thread_id, None)
        if old is not None:
            await old.discard()
        run = FakeSpeculativeRun(text, head_id=self._db.last_message_id(thread_id),
                                 reply=self.speculation_reply)
        self.speculations[thread_id] = run
        return run

    async def abort_turn(self, thread_id: str, *, keep_partial: bool = True) -> str | None:
        self._rec("abort_turn", thread_id, keep_partial=keep_partial)
        run = self.speculations.pop(thread_id, None)
        if run is not None:
            await run.discard()
        for t, tid in list(self._tasks.items()):
            if tid == thread_id:
                t.cancel()
        self.turn_running.discard(thread_id)
        if not keep_partial or not self.partial_on_abort:
            return None
        thread = self._db.get_thread(thread_id)
        row = self._db.add_message(thread_id, "assistant", self.partial_on_abort,
                                   metadata={"voice_heard": {"interrupted": True}})
        self.emit({"type": "message", "thread_id": thread_id,
                   "bot_id": thread.bot_id if thread else None, "message": row.model_dump()})
        return row.id

    async def retract_message(self, thread_id: str, message_id: str, *,
                              only_metadata_key: str = "voice") -> None:
        self._rec("retract_message", thread_id, message_id, only_metadata_key=only_metadata_key)
        m = self._db.get_message(message_id)
        if m is not None and m.thread_id == thread_id and (m.metadata or {}).get(only_metadata_key):
            self._db.delete_message(message_id)

    async def replies_after(self, thread_id: str, after_id: str) -> list[tuple[str, str]] | None:
        self._rec("replies_after", thread_id, after_id)
        ids = [m.id for m in self._db.messages(thread_id)]
        if after_id not in ids:
            return None
        tail = self._db.messages(thread_id)[ids.index(after_id) + 1:]
        return [(m.id, m.content) for m in tail if m.role == "assistant"]

    async def note_heard(self, thread_id: str, message_id: str, *, heard_upto: int,
                         heard_text: str, unheard_text: str) -> bool:
        self._rec("note_heard", thread_id, message_id, heard_upto=heard_upto,
                  heard_text=heard_text, unheard_text=unheard_text)
        m = self._db.get_message(message_id)
        if message_id.startswith("run:") or m is None or m.thread_id != thread_id:
            return False
        m = self._db.merge_message_metadata(message_id, {"voice_heard": {
            "upto": int(heard_upto), "interrupted": True,
            "heard": heard_text[:160], "unheard": unheard_text[:240]}})
        thread = self._db.get_thread(thread_id)
        self.emit({"type": "message_update", "thread_id": thread_id,
                   "bot_id": thread.bot_id if thread else None,
                   "message_id": m.id, "message": m.model_dump()})
        return True

    async def subscribe_frames(self, thread_id: str, on_frame: Callable[[dict], None], *,
                               token: str | None, replay: Sequence[dict] | None = None
                               ) -> Callable[[], Awaitable[None]]:
        """Subscribe to one thread. `replay` (fake-only) is delivered first, e.g.
        ``frames_for("snapshot_mid_stream", thread_id)`` — a recorded snapshot."""
        self._rec("subscribe_frames", thread_id, token=token)
        sub = _Subscription(on_frame, self._queue_max, self)
        self._subs.setdefault(thread_id, []).append(sub)
        for f in replay or ():
            sub.put(f)

        async def unsubscribe() -> None:
            if sub in self._subs.get(thread_id, []):
                self._subs[thread_id].remove(sub)
            await sub.close()

        return unsubscribe

    def bot_backend(self, bot_id: str) -> Backend | None:
        b = self.bots.get(bot_id)
        return b.backend if b else None

    def set_bot_field_validator(self, field_name: str, fn: Callable[[str], None]) -> None:
        self._rec("set_bot_field_validator", field_name, fn)
        self.validators[field_name] = fn

    def validate_bot_field(self, field_name: str, value: str) -> None:
        """What PUT /api/bots/order does with a field (ValueError -> 422)."""
        fn = self.validators.get(field_name)
        if fn is None:
            raise ValueError(f"no validator for {field_name}")
        fn(value)

    def register_tools(self, rows: list[ToolSpec]) -> None:
        self._rec("register_tools", rows)
        self.tools.extend(rows)

    def on_startup(self, fn: Callable[[], Any], *, thread: bool = False) -> None:
        self._rec("on_startup", fn, thread=thread)
        self.startup.append((fn, thread))

    def on_shutdown(self, fn: Callable[[], Any]) -> None:
        self._rec("on_shutdown", fn)
        self.shutdown.append(fn)

    def touch_session(self, token: str | None) -> None:
        self._rec("touch_session", token)
        self.touched.append(token)

    # -- internals ---------------------------------------------------------- #

    def _spawn(self, thread_id: str, coro) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks[t] = thread_id
        t.add_done_callback(lambda done: self._tasks.pop(done, None))


class _Subscription:
    """A bounded per-subscriber queue drained by a task (never inside emit())."""

    def __init__(self, on_frame, maxlen: int, ctx: FakeContext) -> None:
        self.on_frame = on_frame
        self.q: deque[dict] = deque()
        self.maxlen = maxlen
        self.ctx = ctx
        self.task: asyncio.Task | None = None
        self.logged = False

    def put(self, frame: dict) -> None:
        if len(self.q) >= self.maxlen:
            self.q.popleft()                     # overflow drops the oldest, logs once
            self.ctx.dropped_frames += 1
            if not self.logged:
                self.ctx.log.warning("frame queue full; dropping the oldest")
                self.logged = True
        self.q.append(json.loads(json.dumps(frame)))
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self._pump())

    async def _pump(self) -> None:
        await asyncio.sleep(0)
        while self.q:
            self.on_frame(self.q.popleft())
            await asyncio.sleep(0)

    async def flush(self) -> None:
        while self.task is not None and not self.task.done():
            await self.task

    async def close(self) -> None:
        self.q.clear()
        await self.flush()


# --------------------------------------------------------------------------- #
# Speech-engine doubles (from the pre-split suite)
# --------------------------------------------------------------------------- #


class FakeWS:
    """A Starlette-ish websocket: push frames in, read what the server sent."""

    def __init__(self, headers=None, cookies=None, query: dict | None = None):
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.query_params = query or {}
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list = []          # dicts (JSON) and bytes, in order
        self.accepted = False
        self.closed_with = None
        self.close_reason = None

    async def accept(self, *a, **kw):
        self.accepted = True

    async def close(self, code=1000, reason=None):
        self.closed_with = code
        self.close_reason = reason

    async def send_text(self, s):
        self.sent.append(json.loads(s))

    async def send_json(self, obj):
        self.sent.append(json.loads(json.dumps(obj)))

    async def send_bytes(self, b):
        self.sent.append(bytes(b))

    async def receive(self):
        return await self.inbox.get()

    def push_json(self, obj):
        self.inbox.put_nowait({"type": "websocket.receive", "text": json.dumps(obj)})

    def push_audio(self, b=b"\x00\x00" * 320):
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": b})

    def hang_up(self):
        self.inbox.put_nowait({"type": "websocket.disconnect"})

    def json_of(self, t):
        return [m for m in self.sent if isinstance(m, dict) and m.get("type") == t]


class FakeSTT:
    def __init__(self, text: str = "what's the weather"):
        self.text = text
        self.heard: list[int] = []

    def transcribe(self, audio):
        self.heard.append(int(getattr(audio, "size", 0)))
        return self.text


class FakeTTS:
    sample_rate = 24000
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def __init__(self, chunks: int = 3, chunk_samples: int = 160):
        self.said: list[str] = []
        self.voices: list = []
        self.chunks = chunks
        self.chunk_samples = chunk_samples

    def synth(self, text, voice=None):
        self.said.append(text)
        return np.zeros(480, "<i2"), self.sample_rate

    def stream(self, text, voice=None, stop=None):
        self.said.append(text)
        self.voices.append(voice)
        for _ in range(self.chunks):          # a sentence arrives as several chunks
            if stop is not None and stop.is_set():
                return
            yield np.zeros(self.chunk_samples, "<i2")


def _vad():
    from dispatch_voice.host import vad_turn
    return vad_turn


class OneTurnDetector:
    """Every audio chunk ends a turn (the real endpointing is unit-tested)."""

    def __init__(self, reason: str = "silence", seconds: float = 1.0):
        self.ep = SimpleNamespace(cfg=_vad().EndpointerConfig(), turn_fn=None)
        self.reason, self.seconds = reason, seconds
        self.resets = 0

    def feed(self, pcm):
        return [_vad().TurnEnd(audio=np.zeros(int(16000 * self.seconds), np.float32), reason=self.reason)]

    def reset(self):
        self.resets += 1


class ScriptedDetector:
    """feed() returns the next scripted event list, one per audio chunk."""

    def __init__(self, script):
        self.script = list(script)
        self.ep = SimpleNamespace(cfg=_vad().EndpointerConfig(), turn_fn=None)
        self.resets = 0

    def feed(self, pcm):
        return self.script.pop(0) if self.script else []

    def reset(self):
        self.resets += 1


def utterance(seconds: float = 1.0, reason: str = "smart_turn", prob: float = 0.9):
    return _vad().TurnEnd(audio=np.zeros(int(16000 * seconds), np.float32), reason=reason, turn_prob=prob)


@dataclass
class FakeEngines:
    stt: FakeSTT = field(default_factory=FakeSTT)
    tts: FakeTTS = field(default_factory=FakeTTS)
    detector: Any = None
    FakeWS: type = FakeWS
    ScriptedDetector: type = ScriptedDetector
    OneTurnDetector: type = OneTurnDetector
    utterance: Callable = utterance

    def __post_init__(self):
        if self.detector is None:
            self.detector = OneTurnDetector()


# --------------------------------------------------------------------------- #
# Fixture audio
# --------------------------------------------------------------------------- #


def load_clip(name: str, *, band: Literal["16k", "8k"] = "16k") -> np.ndarray:
    """A fixture clip as float32 mono 16 kHz in [-1, 1] (`clips/` or `clips-8k/`)."""
    import wave

    d = FIXTURES / ("clips" if band == "16k" else "clips-8k")
    stem = name.removesuffix(".wav")
    p = d / f"{stem}.wav"
    if band == "16k" and stem == "addendum":
        p = FIXTURES / "addendum.wav"         # C-§G puts the 16 kHz addendum at the fixtures root
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def clip_meta() -> dict:
    return json.loads((FIXTURES / "clips" / "meta.json").read_text(encoding="utf-8"))
