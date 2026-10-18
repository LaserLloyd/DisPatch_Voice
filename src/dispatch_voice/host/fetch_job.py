"""The in-app ``fetch-models``: one background job with progress and cancel.

``dispatch-voice fetch-models`` on a terminal was step 2 of 4 to get voice working. This is
the same code (:func:`host.download.fetch_all`) run from a thread inside DisPatch, so
Settings > Voice can offer a Download button with a progress bar and a Cancel button:

* ``POST /api/voice/models/fetch``   start (one job at a time; a second start while one runs
  is refused with the running job's status);
* ``GET  /api/voice/models/status``  state, phase, current file, percent, the last log lines;
* ``POST /api/voice/models/cancel``  stop at the next chunk (speech models) or file (Chatterbox).

Progress is weighted by approximate file sizes, so the bar moves with the bytes and not with
the file count (one Parakeet encoder is 650 MB of the 1.4 GB). A cancelled or failed job
removes its partial file; finished files stay, so a retry resumes at the next missing file.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

log = logging.getLogger("dispatch.voice")

#: Approximate sizes in MB, used only to weight the progress bar.
APPROX_MB = {"speech": {"silero_vad.onnx": 2, "smart-turn-v3.2-cpu.onnx": 9, "encoder-model.int8.onnx": 652,
                        "decoder_joint-model.int8.onnx": 18, "nemo128.onnx": 1, "vocab.txt": 1, "config.json": 1},
             "chatterbox": 720, "stock": 20}
STATES = ("idle", "running", "done", "failed", "cancelled")


class FetchJob:
    def __init__(self, run: Callable | None = None) -> None:
        self._run = run                       # tests inject a fake body; None = download.fetch_all
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self.state = "idle"
        self.phase = ""
        self.current = ""
        self.files_done = 0
        self.files_total = 0
        self.percent = 0.0
        self.error = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.only: str | None = None
        self.log: deque[str] = deque(maxlen=30)
        self.restart_required = False
        self._weight_done = 0.0
        self._weight_total = 1.0

    # ---- control -------------------------------------------------------- #
    def start(self, only: str | None = None, *, force: bool = False, stock: bool = True, wait: bool = False) -> bool:
        """Begin a job. False when one is already running."""
        if only not in (None, "speech", "chatterbox"):
            raise ValueError("only must be speech or chatterbox")
        with self._lock:
            if self.state == "running":
                return False
            self._reset()
            self._cancel.clear()
            self.state = "running"
            self.only = only
            self.started_at = time.time()
            total = 0.0
            if only in (None, "speech"):
                total += sum(APPROX_MB["speech"].values())
            if only in (None, "chatterbox"):
                total += APPROX_MB["chatterbox"] + (APPROX_MB["stock"] if stock else 0)
            self._weight_total = max(1.0, total)
            t = threading.Thread(target=self._body, args=(only, force, stock), name="voice-fetch-models", daemon=True)
            self._thread = t
            t.start()
        if wait:
            t.join()
        return True

    def cancel(self) -> bool:
        with self._lock:
            if self.state != "running":
                return False
            self._cancel.set()
            self.log.append("cancelling...")
            return True

    def join(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    # ---- the body ------------------------------------------------------- #
    def _echo(self, line: str) -> None:
        self.log.append(str(line)[:200])

    def _progress(self, ev: dict) -> None:
        """``{"phase", "file", "index", "total", "bytes", "bytes_total", "weight_mb"}`` from the downloaders."""
        with self._lock:
            self.phase = ev.get("phase", self.phase)
            self.current = ev.get("file", self.current)
            self.files_total = ev.get("total", self.files_total)
            w = ev.get("weight_mb") or 0.0
            frac = 0.0
            if ev.get("bytes_total"):
                frac = min(1.0, ev.get("bytes", 0) / ev["bytes_total"])
            elif ev.get("done"):
                frac = 1.0
            if ev.get("done"):
                self.files_done = ev.get("index", self.files_done)
                self._weight_done += w
            self.percent = round(min(99.0, 100.0 * (self._weight_done + (0 if ev.get("done") else w * frac))
                                     / self._weight_total), 1)

    def _body(self, only, force, stock) -> None:
        from . import download
        run = self._run or download.fetch_all
        try:
            rc = run(only=only, force=force, verify=False, stock=stock, echo=self._echo, progress=self._progress,
                     cancel=self._cancel)
        except download.Cancelled:
            self._finish("cancelled", "")
            return
        except Exception as e:      # noqa: BLE001 - shown to the operator, logged with its traceback
            log.exception("voice: model download failed")
            self._finish("failed", f"{type(e).__name__}: {str(e)[:160]}")
            return
        if self._cancel.is_set():
            self._finish("cancelled", "")
        elif rc:
            self._finish("failed", "some files could not be fetched (see the log lines)")
        else:
            self._finish("done", "")

    def _finish(self, state: str, error: str) -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.finished_at = time.time()
            if state == "done":
                self.percent = 100.0
                self.restart_required = self.only in (None, "chatterbox")
            self.log.append({"done": "finished", "cancelled": "cancelled", "failed": "failed"}[state])

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "phase": self.phase, "file": self.current, "files_done": self.files_done,
                    "files_total": self.files_total, "percent": self.percent, "error": self.error,
                    "only": self.only, "started_at": self.started_at, "finished_at": self.finished_at,
                    "log": list(self.log), "cancellable": self.state == "running",
                    "restart_required": self.restart_required}


JOB = FetchJob()
