"""``PowerHold``: the ``performance`` power profile is held while at least one
voice chat is open (SPEC §4.11; contracts §B.4, amendment A13).

Why: a screen-power daemon can park a mini-PC like this in ``power-saver`` whenever
the screen is off, which is exactly when someone is talking to it from the car, and that
costs 28 % on a spoken sentence and 47 % on a decoder chunk (measured). While
a session is open the add-on asks power-profiles-daemon (or tuned-ppd) for
``performance`` through the cooperative hold API, and gives it back when the
last session ends.

Mechanics: the hold lives in a child process on the SYSTEM interpreter
(``host/power_hold.py``, fixed argv), because ``gi`` is not in the add-on's
environment and a hold dies with its bus connection. Closing the child's
stdin ends the hold; so does DisPatch dying. The child re-holds when another
client's profile change drops the hold (a screen-power daemon on a screen transition),
at most once per 10 s. Fail-soft everywhere: no system python, no ``gi``, no
daemon, a refused hold -> logged once, ``state == "unavailable"``, and voice
chat carries on at whatever profile the box is in.
"""
from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("dispatch.voice")

#: The system interpreter that has ``gi`` (the uv venv does not): used on Linux when it exists.
SYSTEM_PYTHON = "/usr/bin/python3"
#: The hold talks to power-profiles-daemon over D-Bus: Linux only.
SUPPORTED = sys.platform.startswith("linux")


def holder_python() -> str:
    """The interpreter the holder child runs under: the system one on a Linux box that
    has it (it carries ``gi``; a uv venv does not), else this interpreter. Nothing is
    hard-wired to a path that exists on one OS only."""
    if SUPPORTED and os.access(SYSTEM_PYTHON, os.X_OK):
        return SYSTEM_PYTHON
    return sys.executable

HOLDER = Path(__file__).with_name("power_hold.py")
#: How long the first "held"/"unavailable" line may take (the bus call + GLib start).
START_TIMEOUT_S = 8.0
#: How long the child may take to release and exit after its stdin closes.
STOP_TIMEOUT_S = 3.0


def holder_argv(profile: str = "performance", *, python: str | None = None) -> list[str]:
    """The fixed argv of the holder child (no shell, ever)."""
    return [python or holder_python(), str(HOLDER), profile]


#: What the holder child is allowed to see of our environment: the DisPatch
#: process carries tokens and PINs (its unit drop-ins); the child is a
#: stdlib + gi script that needs none of them.
HOLDER_ENV_PASS = ("LANG", "LC_ALL", "DBUS_SYSTEM_BUS_ADDRESS")


def holder_env(source: dict | None = None) -> dict:
    """A minimal environment for the child: a fixed PATH, the locale, and the
    system-bus address only when the box sets one (names only, nothing secret)."""
    src = os.environ if source is None else source
    if os.name == "nt":
        # A Windows interpreter cannot start without its system folder; PATH and the
        # locale are not secrets. (The hold itself is Linux only: see SUPPORTED.)
        env = {k: src[k] for k in ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP") if src.get(k)}
    else:
        env = {"PATH": "/usr/bin:/bin"}
    for k in HOLDER_ENV_PASS:
        v = src.get(k)
        if v:
            env[k] = v
    env.setdefault("LANG", "C.UTF-8")
    return env


class PowerHold:
    """Reference-counted: ``acquire()`` on session start, ``release()`` on
    session end; the child runs from the first acquire to the last release.

    ``state``: "held" (the child reported the hold), "off" (no session, or the
    feature is off), "unavailable" (the last attempt failed; logged once).
    """

    def __init__(self, *, profile: str = "performance", argv: list[str] | None = None) -> None:
        self.profile = profile
        self._argv = argv
        self._count = 0
        self._proc: subprocess.Popen | None = None
        self._state = "off"
        self._lock = threading.Lock()
        self._warned = False
        self._reader: threading.Thread | None = None
        self.events: list[dict] = []          # the child's reports (for status/tests)
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=5)   # the child's last lines

    @property
    def state(self) -> str:
        return self._state

    @property
    def sessions(self) -> int:
        return self._count

    # -- lifecycle ---------------------------------------------------------- #

    def acquire(self, *, enabled: bool = True) -> str:
        """A session opened. Starts the holder when it is the first one and the
        feature is on. Blocking (reads the child's first line): call it off the
        event loop. Returns the state."""
        with self._lock:
            self._count += 1
            if not enabled:
                self._state = "off" if self._proc is None else self._state
                return self._state
            if self._proc is None or self._proc.poll() is not None:
                self._start_locked()
            return self._state

    def release(self) -> str:
        """A session ended. The last one stops the holder (= releases the profile)."""
        with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                self._stop_locked()
            return self._state

    def release_all(self) -> None:
        """Shutdown: whatever the count, end the hold now."""
        with self._lock:
            self._count = 0
            self._stop_locked()

    # -- internals ---------------------------------------------------------- #

    def _start_locked(self) -> None:
        if self._argv is None and not SUPPORTED:
            self._unavailable("the power-profile hold needs Linux (power-profiles-daemon); "
                              "this system runs voice at its current profile")
            return
        argv = self._argv or holder_argv(self.profile)
        if not os.access(argv[0], os.X_OK):
            self._unavailable(f"no interpreter at {argv[0]}")
            return
        self.stderr_tail.clear()
        try:
            self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, bufsize=1,
                                          env=holder_env())
        except OSError as e:
            self._unavailable(f"could not start the holder: {e}")
            return
        threading.Thread(target=self._drain_stderr, args=(self._proc,), name="voice-power-hold-err",
                         daemon=True).start()
        first = self._read_first_line()
        if first is None or first.get("event") not in ("held",):
            reason = (first or {}).get("reason") or "no answer from the holder"
            self._reap_locked()
            if self.stderr_tail:              # a crash says why on stderr; the log gets its last line
                reason = f"{reason}; holder said: {self.stderr_tail[-1]}"
            self._unavailable(reason)
            return
        self._state = "held"
        self._warned = False
        log.info("voice: holding the %s power profile while voice chat is open", self.profile)
        self._reader = threading.Thread(target=self._pump, args=(self._proc,), name="voice-power-hold",
                                        daemon=True)
        self._reader.start()

    def _read_first_line(self) -> dict | None:
        """The child's first report, within START_TIMEOUT_S (a thread, so a
        wedged child cannot hang the caller)."""
        proc = self._proc
        box: dict = {}

        def read():
            try:
                line = proc.stdout.readline() if proc and proc.stdout else ""
            except (OSError, ValueError):
                line = ""
            with contextlib.suppress(ValueError):
                box["line"] = json.loads(line) if line.strip() else None

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(START_TIMEOUT_S)
        if t.is_alive():
            return None
        line = box.get("line")
        if isinstance(line, dict):
            self.events.append(line)
        return line if isinstance(line, dict) else None

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep the child's stderr flowing (a full pipe would block it) and
        remember its last few lines for the `unavailable` warning."""
        try:
            for raw in proc.stderr:
                line = raw.strip()
                if line:
                    self.stderr_tail.append(line[:200])
        except (OSError, ValueError):
            pass

    def _pump(self, proc: subprocess.Popen) -> None:
        """Follow the child's later reports (re-holds, losses, exit)."""
        try:
            for raw in proc.stdout:
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                self.events.append(ev)
                kind = ev.get("event")
                if kind in ("held", "reheld"):
                    self._state = "held"
                    if kind == "reheld":
                        log.info("voice: re-held the %s power profile", self.profile)
                elif kind == "released":
                    log.info("voice: the %s power profile hold was released by another client; "
                             "re-holding shortly", self.profile)
                elif kind == "unavailable":
                    self._unavailable(ev.get("reason") or "refused")
        except (OSError, ValueError):
            pass
        finally:
            if self._proc is proc and proc.poll() is not None and self._state == "held":
                # The child died on its own (daemon gone): nothing is held any more.
                self._state = "unavailable"

    def _stop_locked(self) -> None:
        """The last session ended (or shutdown): end the hold; the state is "off"."""
        was_held = self._state == "held"
        self._reap_locked()
        if was_held:
            log.info("voice: released the %s power profile", self.profile)
        self._state = "off"

    def _reap_locked(self) -> None:
        """Close the child's stdin (= release) and wait for it; no state change."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin:
                proc.stdin.close()            # EOF: the child releases and exits
        try:
            proc.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(1.0)
        with contextlib.suppress(OSError, ValueError):
            if proc.stdout:
                proc.stdout.close()
        with contextlib.suppress(OSError, ValueError):
            if proc.stderr:
                proc.stderr.close()

    def _unavailable(self, reason: str) -> None:
        self._state = "unavailable"
        if not self._warned:
            self._warned = True
            log.warning("voice: power-profile hold unavailable (%s); voice chat runs at the "
                        "box's current profile", reason)


#: The process-wide hold the sessions share.
HOLD = PowerHold()


def acquire(settings) -> str:
    """Session start: hold unless ``rt.power_hold`` is off. Blocking; run it off the loop."""
    return HOLD.acquire(enabled=getattr(settings, "power_hold", "performance") == "performance")


def release() -> str:
    return HOLD.release()


def release_all() -> None:
    HOLD.release_all()


def state(settings=None) -> str:
    """For ``/api/voice/status.power_hold`` and ``ready.power_hold``."""
    if settings is not None and getattr(settings, "power_hold", "performance") != "performance":
        return "off"
    return HOLD.state


def platform_facts() -> dict:
    """``platform_profile`` and ``max_mhz`` for the metrics line / the harness
    (SPEC §4.11); empty values where the files do not exist."""
    out: dict = {"platform_profile": None, "max_mhz": None}
    with contextlib.suppress(OSError, ValueError):
        out["platform_profile"] = Path("/sys/firmware/acpi/platform_profile").read_text(encoding="utf-8").strip()
    with contextlib.suppress(OSError, ValueError):
        out["max_mhz"] = int(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text(encoding="utf-8")) // 1000
    return out
