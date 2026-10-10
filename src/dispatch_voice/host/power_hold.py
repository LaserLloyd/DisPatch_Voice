"""Child-process holder of the ``performance`` power profile (SPEC §4.11).

Run on the SYSTEM interpreter, by ``host/power.py``, with a fixed argv::

    /usr/bin/python3 <this file> performance [--fake-bus [refuse]]

Why a child process: a power-profiles hold lives exactly as long as the bus
connection that took it, ``gi`` (GLib/Gio) is not in the add-on's virtual
environment, and a hold must never outlive the voice chat. So this script
holds the profile on net.hadess.PowerProfiles (the system bus), keeps the
connection open, and ends the hold when its stdin closes: the parent closing
the pipe, or the parent dying, both release it. Nothing is ever left behind.

Measured behaviour on tuned-ppd (2026-10-10): ``balanced`` cannot be held;
``performance`` can; any ``Set ActiveProfile`` by another client (a
screen-power daemon on every screen or remote-desktop transition) drops the
hold at once and the daemon emits
``ProfileReleased(cookie)`` about a second later. The holder re-holds on that
signal, at most once per :data:`MIN_REHOLD_GAP_S` (never a fight with that
daemon), and a periodic check re-holds when the signal was missed.

Stdout carries one JSON object per line (``{"event": ...}``) for the parent:
``held``, ``reheld``, ``released`` (by someone else), ``unavailable`` (with a
``reason``), ``exit``. Standard library + ``gi`` only; nothing from this
package is imported (the system python cannot see it). ``--fake-bus`` (tests)
needs no ``gi`` at all.
"""
from __future__ import annotations

import json
import os
import select
import signal
import sys
import time

BUS = "net.hadess.PowerProfiles"
PATH = "/net/hadess/PowerProfiles"
APP_ID = "dispatch-voice"
REASON = "voice chat"
#: Fewest seconds between two holds (feasibility H1 / SPEC §4.11: "<= 1 per 10 s").
MIN_REHOLD_GAP_S = 10.0
#: How often the holder double-checks that its hold still exists.
CHECK_EVERY_S = 10.0
HOLDABLE = ("power-saver", "performance")


def emit(event: str, **fields) -> None:
    try:
        sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


class HoldRefused(Exception):
    """The daemon refused the hold (unknown profile, policy, no daemon)."""


class GioBus:
    """The real system bus through Gio (needs ``gi``)."""

    def __init__(self) -> None:
        import gi
        gi.require_version("Gio", "2.0")
        from gi.repository import Gio, GLib
        self.Gio, self.GLib = Gio, GLib
        self.bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)

    def hold(self, profile: str) -> int:
        try:
            return int(self.bus.call_sync(
                BUS, PATH, BUS, "HoldProfile", self.GLib.Variant("(sss)", (profile, REASON, APP_ID)),
                self.GLib.VariantType("(u)"), self.Gio.DBusCallFlags.NONE, 3000, None).unpack()[0])
        except Exception as e:
            raise HoldRefused(str(e)[:200]) from e

    def release(self, cookie: int) -> None:
        try:
            self.bus.call_sync(BUS, PATH, BUS, "ReleaseProfile", self.GLib.Variant("(u)", (cookie,)),
                               None, self.Gio.DBusCallFlags.NONE, 3000, None)
        except Exception:
            pass          # "No active hold": someone else released it already

    def holds(self) -> list[dict]:
        try:
            v = self.bus.call_sync(BUS, PATH, "org.freedesktop.DBus.Properties", "Get",
                                   self.GLib.Variant("(ss)", (BUS, "ActiveProfileHolds")),
                                   self.GLib.VariantType("(v)"), self.Gio.DBusCallFlags.NONE, 3000, None).unpack()[0]
            return list(v)
        except Exception:
            return []

    def subscribe_released(self, callback) -> None:
        def on_signal(conn, sender, path, iface, name, params):
            try:
                callback(int(params.unpack()[0]))
            except Exception:
                pass
        self.bus.signal_subscribe(BUS, BUS, "ProfileReleased", PATH, None,
                                  self.Gio.DBusSignalFlags.NONE, on_signal)


class FakeBus:
    """Tests only (``--fake-bus``): holds succeed, or are refused with
    ``FakeBus(refuse=True)``; nothing touches the system bus."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.held: dict[int, str] = {}
        self._next = 100
        self._cb = None

    def hold(self, profile: str) -> int:
        if self.refuse or profile not in HOLDABLE:
            raise HoldRefused(f"Only 'power-saver' and 'performance' profiles may be held ({profile})")
        self._next += 1
        self.held[self._next] = profile
        return self._next

    def release(self, cookie: int) -> None:
        self.held.pop(cookie, None)

    def holds(self) -> list[dict]:
        return [{"Profile": p, "Reason": REASON, "ApplicationId": APP_ID} for p in self.held.values()]

    def subscribe_released(self, callback) -> None:
        self._cb = callback

    def external_set(self) -> None:
        """What another client's Set ActiveProfile does: every hold goes, then the signal."""
        gone = list(self.held)
        self.held.clear()
        for c in gone:
            if self._cb:
                self._cb(c)


class Holder:
    """The policy, bus-agnostic and clock-injectable so it is unit-testable.

    ``hold()`` takes the hold; ``on_released(cookie)`` answers how many
    seconds to wait before re-holding (None = not ours, nothing to do);
    ``check()`` re-holds when the hold vanished without a signal;
    ``release()`` ends it. ``state`` is "held" | "lost" | "unavailable" | "off".
    """

    def __init__(self, bus, profile: str, *, now=time.monotonic, emit=emit) -> None:
        self.bus = bus
        self.profile = profile
        self.now = now
        self.emit = emit
        self.cookie: int | None = None
        self.state = "off"
        self._last_hold_at: float | None = None
        self.holds_taken = 0

    def _can_hold_in(self) -> float:
        if self._last_hold_at is None:
            return 0.0
        return max(0.0, MIN_REHOLD_GAP_S - (self.now() - self._last_hold_at))

    def hold(self) -> bool:
        if self.state == "held" and self.cookie is not None:
            return True
        try:
            self.cookie = self.bus.hold(self.profile)
        except HoldRefused as e:
            self.state = "unavailable"
            self.emit("unavailable", reason=str(e))
            return False
        first = self.holds_taken == 0
        self.holds_taken += 1
        self._last_hold_at = self.now()
        self.state = "held"
        self.emit("held" if first else "reheld", cookie=self.cookie, profile=self.profile)
        return True

    def on_released(self, cookie: int) -> float | None:
        """Someone else dropped our hold: re-hold after the rate-limit gap."""
        if cookie != self.cookie or self.state != "held":
            return None
        self.cookie = None
        self.state = "lost"
        self.emit("released", by="external")
        return self._can_hold_in()

    def check(self) -> float | None:
        """Periodic belt: the hold is gone (no signal seen) -> delay until a re-hold."""
        if self.state != "held":
            return None if self.state != "lost" else self._can_hold_in()
        try:
            mine = [h for h in self.bus.holds() if h.get("ApplicationId") == APP_ID]
        except Exception:
            return None
        if mine:
            return None
        self.cookie = None
        self.state = "lost"
        self.emit("released", by="missing")
        return self._can_hold_in()

    def release(self) -> None:
        cookie, self.cookie = self.cookie, None
        if cookie is not None:
            self.bus.release(cookie)
        self.state = "off"


def _run_glib(bus, profile: str) -> int:
    """The real loop: GLib so the bus signal arrives; hold, re-hold on
    release/check, end on stdin EOF or a signal."""
    import gi
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    holder = Holder(bus, profile)
    loop = GLib.MainLoop()
    pending = {"id": None}

    def schedule_rehold(delay: float | None) -> None:
        if delay is None or pending["id"] is not None:
            return

        def go():
            pending["id"] = None
            if holder.state == "lost":
                holder.hold()
            return False

        pending["id"] = GLib.timeout_add(int(delay * 1000) + 50, go)

    if not holder.hold():
        return 1
    bus.subscribe_released(lambda cookie: schedule_rehold(holder.on_released(cookie)))

    def periodic():
        schedule_rehold(holder.check())
        return True

    GLib.timeout_add(int(CHECK_EVERY_S * 1000), periodic)

    def quit_(*_a) -> bool:
        holder.release()
        emit("exit")
        loop.quit()
        return False

    def on_stdin(fd, cond) -> bool:
        try:
            data = os.read(fd, 4096)
        except OSError:
            data = b""
        if not data:          # EOF: the parent closed the pipe (or died)
            quit_()
            return False
        return True

    GLib.io_add_watch(sys.stdin.fileno(), GLib.PRIORITY_DEFAULT,
                      GLib.IO_IN | GLib.IO_HUP | GLib.IO_ERR, on_stdin)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        _signal_add(GLib, GLib.PRIORITY_HIGH, sig, quit_)
    try:
        loop.run()
    finally:
        holder.release()      # belt: never leave a hold behind, whatever ended the loop
    return 0


def _signal_add(GLib, priority: int, signum: int, handler) -> None:
    """``GLibUnix.signal_add`` where PyGObject has it (3.50+); the older
    ``GLib.unix_signal_add`` otherwise, without its deprecation line on stderr."""
    try:
        import gi
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix
    except Exception:                 # an older PyGObject: the deprecated name still works
        GLibUnix = None
    if GLibUnix is not None:
        GLibUnix.signal_add(priority, signum, handler)
        return
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GLib.unix_signal_add(priority, signum, handler)


def _run_select(bus, profile: str, *, tick_s: float = 0.2) -> int:
    """The same protocol without GLib (the fake bus has no signals): hold,
    periodic check, end on stdin EOF. Lets the pipe protocol be tested on
    any interpreter."""
    holder = Holder(bus, profile)
    if not holder.hold():
        return 1
    stopping = {"now": False}

    def on_signal(*_a):
        stopping["now"] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    fd = sys.stdin.fileno()
    next_check = time.monotonic() + CHECK_EVERY_S
    rehold_at: float | None = None
    try:
        while not stopping["now"]:
            ready, _, _ = select.select([fd], [], [], tick_s)
            if ready:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
            now = time.monotonic()
            if rehold_at is not None and now >= rehold_at:
                rehold_at = None
                if holder.state == "lost":
                    holder.hold()
            if now >= next_check:
                next_check = now + CHECK_EVERY_S
                delay = holder.check()
                if delay is not None and rehold_at is None:
                    rehold_at = now + delay
    finally:
        holder.release()
        emit("exit")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    profile = argv[0] if argv and not argv[0].startswith("--") else "performance"
    fake = "--fake-bus" in argv
    refuse = fake and "refuse" in argv
    if profile not in HOLDABLE:
        emit("unavailable", reason=f"profile {profile!r} cannot be held")
        return 1
    if fake:
        return _run_select(FakeBus(refuse=refuse), profile)
    try:
        bus = GioBus()
    except Exception as e:      # no gi, no system bus, no daemon
        emit("unavailable", reason=f"{type(e).__name__}: {str(e)[:160]}")
        return 1
    return _run_glib(bus, profile)


if __name__ == "__main__":
    sys.exit(main())
