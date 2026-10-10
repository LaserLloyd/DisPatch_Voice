"""Timing constants that live on both sides of the wire and must agree.

The acknowledgement line (server, ``ack_after_s`` after ``eot``) has to reach the
phone INSIDE the client's turn-end tone hold (``drive.js EOT_TONE_HOLD_MS``), or
every acknowledged turn makes two sounds — the tone, then the line. At 0.7 s
against a 650 ms hold both played, 50 ms apart, on every acked turn (FR-works 5).
The pair is pinned here with 100 ms of network headroom.
"""
import re
from pathlib import Path

from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.text.acks import TurnAcks

DRIVE_JS = Path(__file__).resolve().parents[2] / "src" / "dispatch_voice" / "static" / "drive.js"


def _hold_ms() -> int:
    m = re.search(r"export const EOT_TONE_HOLD_MS = (\d+);", DRIVE_JS.read_text(encoding="utf-8"))
    assert m, "drive.js no longer exports EOT_TONE_HOLD_MS"
    return int(m.group(1))


def test_the_ack_line_lands_inside_the_turn_end_tone_hold():
    ack_ms = VoiceSettings().ack_after_s * 1000
    assert ack_ms + 100 < _hold_ms(), (ack_ms, _hold_ms())


def test_the_pure_gate_default_is_no_later_than_the_host_default():
    # TurnAcks.ACK_AFTER_S is the gate's own default (tests and the standalone tools);
    # the session overrides it with the host setting. Neither may sit past the hold.
    assert TurnAcks.ACK_AFTER_S * 1000 + 100 < _hold_ms()
