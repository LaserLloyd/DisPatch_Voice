"""The power-profile hold (SPEC §4.11, contracts §B.4 A13): the policy with a
fake bus (hold, external release → re-hold after the gap, refused →
unavailable, a vanished hold noticed by the periodic check) and the real
child-process pipe protocol under ``--fake-bus`` (held while stdin is open,
released the moment it closes, reference-counted by sessions, fail-soft when
the interpreter or the hold is unavailable). The system-bus path itself was
measured on the box (ProfileReleased ≈ 1 s after a screen-power daemon's Set) and is
exercised by the model-backed session test; nothing here touches it."""
from __future__ import annotations

import sys

import pytest

from dispatch_voice.host import power, power_hold
from dispatch_voice.host.config import VoiceSettings


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def holder(bus, clock, events):
    return power_hold.Holder(bus, "performance", now=clock,
                             emit=lambda ev, **f: events.append((ev, f)))


def test_holder_holds_loses_it_to_another_client_and_reholds_after_the_gap():
    bus, clock, events = power_hold.FakeBus(), Clock(), []
    h = holder(bus, clock, events)
    assert h.hold() is True and h.state == "held"
    assert bus.holds() == [{"Profile": "performance", "Reason": "voice chat", "ApplicationId": "dispatch-voice"}]
    assert events[0][0] == "held" and events[0][1]["profile"] == "performance"
    assert h.hold() is True and h.holds_taken == 1              # idempotent while held
    # A screen-power daemon writes the profile: the daemon drops every hold, then signals ProfileReleased(cookie).
    delays = []
    bus.subscribe_released(lambda c: delays.append(h.on_released(c)))
    clock.t = 3.0
    bus.external_set()
    assert h.state == "lost" and delays == [7.0]                # 10 s gap minus the 3 s elapsed
    assert events[-1] == ("released", {"by": "external"})
    assert h.on_released(999) is None                           # someone else's cookie
    clock.t = 10.0
    assert h.hold() is True and h.state == "held" and events[-1][0] == "reheld"
    assert h.holds_taken == 2 and bus.holds()
    h.release()
    assert bus.holds() == [] and h.state == "off" and h.cookie is None


def test_holder_refused_is_unavailable_and_release_is_harmless():
    bus, clock, events = power_hold.FakeBus(refuse=True), Clock(), []
    h = holder(bus, clock, events)
    assert h.hold() is False and h.state == "unavailable"
    assert events == [("unavailable", {"reason": "Only 'power-saver' and 'performance' profiles may be "
                                                 "held (performance)"})]
    h.release()
    assert h.state == "off" and bus.holds() == []


def test_holder_check_notices_a_hold_that_vanished_without_a_signal():
    bus, clock, events = power_hold.FakeBus(), Clock(), []
    h = holder(bus, clock, events)
    h.hold()
    assert h.check() is None                                    # still there
    bus.held.clear()                                            # gone, no signal seen
    clock.t = 4.0
    assert h.check() == 6.0 and h.state == "lost" and events[-1] == ("released", {"by": "missing"})
    assert h.check() == 6.0                                     # lost: the remaining gap again
    clock.t = 12.0
    assert h.check() == 0.0


def test_only_holdable_profiles(capsys):
    assert power_hold.main(["balanced", "--fake-bus"]) == 1
    assert "cannot be held" in capsys.readouterr().out
    with pytest.raises(power_hold.HoldRefused):
        power_hold.FakeBus().hold("balanced")


def fake_argv(*extra):
    return [sys.executable, str(power.HOLDER), "performance", "--fake-bus", *extra]


def test_the_child_holds_until_its_stdin_closes():
    ph = power.PowerHold(argv=fake_argv())
    assert ph.state == "off" and ph.sessions == 0
    assert ph.acquire() == "held" and ph.events[0]["event"] == "held"
    proc = ph._proc
    assert proc is not None and proc.poll() is None
    assert ph.acquire() == "held" and ph.sessions == 2          # a second chat shares the hold
    assert ph.release() == "held" and proc.poll() is None       # one chat still open
    assert ph.release() == "off" and ph.sessions == 0
    assert proc.poll() == 0                                     # EOF -> released -> clean exit
    assert ph.events[-1]["event"] == "exit"


def test_the_child_refusing_is_unavailable_and_nothing_lingers():
    ph = power.PowerHold(argv=fake_argv("refuse"))
    assert ph.acquire() == "unavailable" and ph._proc is None
    assert ph.events[0]["event"] == "unavailable"
    assert ph.release() == "off"


def test_no_interpreter_is_unavailable(caplog):
    ph = power.PowerHold(argv=["/nonexistent/python3", str(power.HOLDER), "performance"])
    assert ph.acquire() == "unavailable" and ph._proc is None
    assert "unavailable" in caplog.text
    assert ph.release() == "off"


def test_disabled_never_starts_a_child():
    ph = power.PowerHold(argv=fake_argv())
    assert ph.acquire(enabled=False) == "off" and ph._proc is None and ph.sessions == 1
    assert ph.release() == "off"


def test_release_all_ends_the_hold_whatever_the_count():
    ph = power.PowerHold(argv=fake_argv())
    for _ in range(3):
        ph.acquire()
    proc = ph._proc
    ph.release_all()
    assert ph.state == "off" and ph.sessions == 0 and proc.poll() == 0


def test_holder_argv_is_fixed_and_points_at_the_system_python():
    argv = power.holder_argv()
    assert argv == ["/usr/bin/python3", str(power.HOLDER), "performance"]
    assert power.HOLDER.name == "power_hold.py" and power.HOLDER.exists()


def test_module_level_state_honours_the_setting():
    assert power.state(VoiceSettings(power_hold="off")) == "off"
    assert power.state(VoiceSettings()) == power.HOLD.state
    assert power.state() in ("off", "held", "unavailable")


def test_platform_facts_shape():
    facts = power.platform_facts()
    assert set(facts) == {"platform_profile", "max_mhz"}
    assert facts["max_mhz"] is None or isinstance(facts["max_mhz"], int)


# --------------------------------------------------------------------------- #
# Review fixes (review-T3s1 finding 4): the child's environment and stderr
# --------------------------------------------------------------------------- #

def test_holder_env_is_minimal_and_carries_no_secrets():
    src = {"PATH": "/weird:/usr/bin", "LANG": "en_GB.UTF-8", "DISPATCH_API_KEY": "nope",
           "EXAMPLE_SERVICE_PIN": "nope", "HOME": "/var/home/x", "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=/x"}
    assert power.holder_env(src) == {"PATH": "/usr/bin:/bin", "LANG": "en_GB.UTF-8",
                                     "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=/x"}
    assert power.holder_env({}) == {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    assert power.holder_env({"LANG": ""}) == {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def test_the_child_runs_with_the_minimal_env(monkeypatch):
    seen = {}
    real_popen = power.subprocess.Popen

    class Spy(real_popen):
        def __init__(self, argv, **kw):
            seen["env"] = kw.get("env")
            seen["stderr"] = kw.get("stderr")
            super().__init__(argv, **kw)

    monkeypatch.setattr(power.subprocess, "Popen", Spy)
    monkeypatch.setenv("SOME_TOKEN", "secret-value")
    ph = power.PowerHold(argv=fake_argv())
    assert ph.acquire() == "held"
    ph.release()
    assert seen["env"] == power.holder_env() and "SOME_TOKEN" not in seen["env"]
    assert seen["stderr"] == power.subprocess.PIPE


def test_a_crashing_child_reports_its_last_stderr_line(tmp_path, caplog):
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nprint('about to die', file=sys.stderr)\n"
                     "raise SystemExit('holder crashed: no bus')\n")
    ph = power.PowerHold(argv=[sys.executable, str(crash)])
    assert ph.acquire() == "unavailable" and ph._proc is None
    assert "holder said: holder crashed: no bus" in caplog.text
    assert list(ph.stderr_tail) == ["about to die", "holder crashed: no bus"]
    assert ph.release() == "off"


def test_a_noisy_child_never_blocks_on_a_full_stderr_pipe(tmp_path):
    noisy = tmp_path / "noisy.py"
    noisy.write_text("import sys, json\n"
                     "sys.stderr.write('x' * 300000); sys.stderr.flush()\n"     # > a pipe buffer
                     "print(json.dumps({'event': 'held', 'profile': 'performance'}), flush=True)\n"
                     "sys.stdin.read()\n"
                     "print(json.dumps({'event': 'exit'}), flush=True)\n")
    ph = power.PowerHold(argv=[sys.executable, str(noisy)])
    assert ph.acquire() == "held"
    assert ph.release() == "off" and ph.events[-1]["event"] == "exit"
    assert len(ph.stderr_tail) == 1 and len(ph.stderr_tail[0]) == 200    # capped tail
