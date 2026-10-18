#!/usr/bin/env python3
"""A stub OpenAI-compatible LLM + embeddings server for the voice harnesses.

    python tests/stub_llm.py --port 19001                 # 1.2 s to first token
    python tests/stub_llm.py --port 19001 --embed-delay 1.7 --reply long

    from stub_llm import StubLLM
    with StubLLM(delay_s=1.2) as llm:          # port 0 = any free loopback port
        bot_api = {"provider": "openai", "base_url": llm.base_url, "model": "stub"}
        ...
        llm.configure(embed_delay_s=2.5, reply="long")
        llm.requests                           # what DisPatch sent, newest last

Routes (all under /v1, loopback only, any bearer key accepted):
  GET  /v1/models
  POST /v1/chat/completions   stream true (SSE deltas) or false; first token after
                              `delay_s`, then `tokens_per_s` words per second
  POST /v1/embeddings         deterministic unit vectors after `embed_delay_s`
                              (0 / 1.7 / 2.5 s: the advisor retrieval-budget test)
  POST /_stub/config          JSON body: any of delay_s, tokens_per_s, embed_delay_s,
                              reply ("short"|"normal"|"long"), fixed_reply (str|null),
                              fail_next (int: answer the next N chat calls with HTTP 500)
  GET  /_stub/requests        the request log (JSON list)
  POST /_stub/reset           clear the log and the reply counter

Every reply is DIFFERENT and differs within its first 20 characters: DisPatch's
reply dedup treats two replies sharing a 60-character opening as one ("twins")
and would silently drop the second, which made earlier harnesses look like a
voice bug. Replies open with "Answer <n>." and rotate their sentences.

A client that disconnects mid-stream (a discarded speculation) is recorded as
`aborted: true` with the time it was noticed, so a test can assert the stream
was closed promptly.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Self

SENTENCES = (
    "You have two things this afternoon.",
    "The dentist is at three, and the school call is at four thirty.",
    "Traffic looks light on the way there.",
    "I can remind you ten minutes before each one.",
    "The package from the hardware store shipped this morning.",
    "It should arrive on Thursday before noon.",
    "The weather stays dry until the evening.",
    "There is a small chance of rain after seven.",
    "Your next free slot is tomorrow at ten.",
    "I moved the grocery list to the shared note.",
    "The garage quote came in a little under budget.",
    "The electrician can start the week after next.",
    "Nothing else needs your attention today.",
    "Want me to read the rest of the list?",
)
LENGTHS = {"short": 1, "normal": 3, "long": 12}


@dataclass
class StubConfig:
    delay_s: float = 1.2
    tokens_per_s: float = 40.0
    embed_delay_s: float = 0.0
    reply: str = "normal"
    fixed_reply: str | None = None
    fail_next: int = 0
    embed_dim: int = 64


@dataclass
class StubState:
    cfg: StubConfig = field(default_factory=StubConfig)
    requests: list[dict] = field(default_factory=list)
    counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def next_reply(self) -> str:
        with self.lock:
            self.counter += 1
            n = self.counter
            cfg = self.cfg
        if cfg.fixed_reply is not None:
            return f"Answer {n}. {cfg.fixed_reply}"
        k = LENGTHS.get(cfg.reply, 3)
        start = (n * 5) % len(SENTENCES)
        body = [SENTENCES[(start + i) % len(SENTENCES)] for i in range(k)]
        return f"Answer {n}. " + " ".join(body)


def embed(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding: same text, same unit vector."""
    out: list[float] = []
    i = 0
    while len(out) < dim:
        h = hashlib.sha256(f"{i}\0{text}".encode()).digest()
        out.extend((b - 127.5) / 127.5 for b in h)
        i += 1
    v = out[:dim]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _make_handler(state: StubState):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):          # quiet
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                # The caller hung up first (an Undo, a barge-in on an API bot): not an
                # error, and not a traceback in every e2e log (V2 #2).
                self.close_connection = True

        def _body(self) -> dict:
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                obj = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                obj = {}
            return obj if isinstance(obj, dict) else {}

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{"id": "stub", "object": "model"}]})
            if self.path == "/_stub/requests":
                with state.lock:
                    return self._json(200, list(state.requests))
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            body = self._body()
            if path == "/_stub/config":
                with state.lock:
                    for k, v in body.items():
                        if hasattr(state.cfg, k):
                            setattr(state.cfg, k, v)
                    cfg = dict(vars(state.cfg))
                return self._json(200, cfg)
            if path == "/_stub/reset":
                with state.lock:
                    state.requests.clear()
                    state.counter = 0
                return self._json(200, {"ok": True})
            if path in ("/v1/embeddings", "/embeddings"):
                return self._embeddings(body)
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat(body)
            return self._json(404, {"error": {"message": "not found"}})

        def _log(self, entry: dict) -> dict:
            entry["t"] = time.time()
            with state.lock:
                state.requests.append(entry)
            return entry

        def _embeddings(self, body: dict) -> None:
            texts = body.get("input")
            texts = [texts] if isinstance(texts, str) else list(texts or [])
            self._log({"path": "embeddings", "model": body.get("model"), "n": len(texts)})
            time.sleep(float(state.cfg.embed_delay_s))
            dim = int(state.cfg.embed_dim)
            self._json(200, {"object": "list", "model": body.get("model") or "stub-embed",
                             "data": [{"object": "embedding", "index": i, "embedding": embed(t, dim)}
                                      for i, t in enumerate(texts)],
                             "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)}})

        def _chat(self, body: dict) -> None:
            msgs = body.get("messages") or []
            users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            system = "\n".join(str(m.get("content")) for m in msgs
                               if isinstance(m, dict) and m.get("role") in ("system", "developer"))
            stream = bool(body.get("stream"))
            entry = self._log({"path": "chat", "model": body.get("model"), "stream": stream,
                               "n_messages": len(msgs), "system": system,
                               "last_user": users[-1].get("content") if users else None,
                               "aborted": False})
            with state.lock:
                fail = state.cfg.fail_next > 0
                if fail:
                    state.cfg.fail_next -= 1
            if fail:
                return self._json(500, {"error": {"message": "stub failure (fail_next)"}})
            reply = state.next_reply()
            entry["reply"] = reply
            time.sleep(float(state.cfg.delay_s))
            if not stream:
                return self._json(200, {
                    "id": f"stub-{state.counter}", "object": "chat.completion", "model": "stub",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": reply}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": len(reply.split()),
                              "total_tokens": 1 + len(reply.split())}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            words = reply.split(" ")
            gap = 1.0 / max(1e-3, float(state.cfg.tokens_per_s))
            try:
                for i, w in enumerate(words):
                    piece = w if i == 0 else " " + w
                    chunk = {"id": "stub", "object": "chat.completion.chunk", "model": "stub",
                             "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if i + 1 < len(words):
                        time.sleep(gap)
                done = {"id": "stub", "object": "chat.completion.chunk", "model": "stub",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except (ConnectionError, TimeoutError):
                entry["aborted"] = True
                entry["aborted_at"] = time.time()

    return H


class StubLLM:
    """In-process stub server on a loopback port (0 = any free port)."""

    def __init__(self, port: int = 0, host: str = "127.0.0.1", **cfg) -> None:
        self.state = StubState(cfg=StubConfig(**cfg))
        self.server = http.server.ThreadingHTTPServer((host, port), _make_handler(self.state))
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base_url = f"http://{host}:{self.port}/v1"
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        self._thread = threading.Thread(target=self.server.serve_forever, name="stub-llm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def configure(self, **kw) -> None:
        with self.state.lock:
            for k, v in kw.items():
                if not hasattr(self.state.cfg, k):
                    raise AttributeError(k)
                setattr(self.state.cfg, k, v)

    @property
    def requests(self) -> list[dict]:
        with self.state.lock:
            return list(self.state.requests)

    def chat_requests(self) -> list[dict]:
        return [r for r in self.requests if r["path"] == "chat"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stub OpenAI-compatible LLM for the voice harnesses.")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.2, help="seconds to the first token")
    ap.add_argument("--tps", type=float, default=40.0, help="words per second while streaming")
    ap.add_argument("--embed-delay", type=float, default=0.0)
    ap.add_argument("--reply", choices=sorted(LENGTHS), default="normal")
    a = ap.parse_args(argv)
    s = StubLLM(a.port, delay_s=a.delay, tokens_per_s=a.tps, embed_delay_s=a.embed_delay, reply=a.reply)
    print(json.dumps({"base_url": s.base_url}), flush=True)
    try:
        s.server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
