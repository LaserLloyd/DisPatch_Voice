"""``rt.*`` settings (C-§F.1), validated with the ranges the contract lists.

The host (T3) reads ``<data>/voice.yaml``'s ``rt:`` mapping and hands the result
of :meth:`RtSettings.from_mapping` to the realtime engines; the engines never
read files themselves. Unknown keys are dropped, out-of-range values are
clamped, wrong types fall back to the default — so a hand-edited file can
never put the endpointer outside the measured envelope.

Every default is the measured one from SPEC §4 (see the field comments).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from ..host import settings_schema as _schema

STT_ENGINES = _schema.STT_ENGINES
SPECULATE_MODES = _schema.SPECULATE_MODES
POWER_HOLDS = _schema.POWER_HOLD_MODES

#: (lo, hi) for the numeric keys, straight from C-§F.1: generated from the settings table
#: (``host/settings_schema.py``), which is the one place a range is written.
RANGES: dict[str, tuple[float, float]] = _schema.rt_ranges()


@dataclass(frozen=True)
class RtSettings:
    stt_engine: str = "parakeet"            # parakeet (tail mode) | moonshine (optional extra)
    speculate: str = "advisor"              # advisor | all | off (hardening M7: metered bots opt in)
    speculate_max_per_turn: int = 2         # ≤ 2 speculations per turn
    speculate_min_gap_s: float = 3.0        # ≥ 3 s between two starts in one turn
    preemptive_tts: bool = False            # a wasted first chunk costs ≈ 1 s of decoder time
    decoder_prompt_s: float = 4.0           # reference prompt the decoder re-decodes per chunk (2–10 s)
    first_chunk_tokens: int = 25            # 25 with one decoder session; 15 with two
    decoder_sessions: int = 1               # 2 only after the §4.10 contention run passes on that box
    stt_spinning: bool = False              # ORT intra-op spinning for STT/Smart Turn (lead-measured: off)
    power_hold: str = "performance"         # performance | off (T3 owns the hold itself)
    probe_silence_s: float = 0.25           # pause that triggers Smart Turn + at_pause()
    hold_min_s: float = 0.3                 # learned-hold clamp
    hold_max_s: float = 1.2
    addendum_s: float = 3.0                 # "oh, and…" merge window × patience (session)
    echo_guard_s: float = 1.5               # no optimistic duck in a reply's first 1.5 s (client)
    false_interrupt_timeout_s: float = 2.0  # resume after this much silence following a duck
    stt_threads_while_speaking: int = 4     # STT threads while the decoder streams
    #: Hard-silence end of turn (seconds × patience). This is the top-level
    #: ``hard_silence_s`` of voice.yaml; the host copies it in so the policy
    #: has every number it needs without reading the host's settings.
    hard_silence_s: float = 1.4
    #: Server-side bound on an idle voice chat (FR-hardening L3): with no utterance and
    #: no reply sentence for this long, the session stops keeping the unlocked
    #: session alive and ends the chat (close 4002 "quiet"). Above the longest
    #: device "Hang up when quiet" choice (30 min + its 20 s grace), so the polite
    #: client check normally comes first; "Never" on a device is bounded here.
    quiet_max_s: float = 2400.0
    #: Per-bot conversation defaults (``bots.<id>.{backchannel, follow_up}``).
    bots: dict[str, dict[str, bool]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, hard_silence_s: float | None = None) -> RtSettings:
        """Validate a ``rt:`` mapping; unknown keys dropped, bad values → default, numbers clamped."""
        base = cls()
        if hard_silence_s is not None:
            base = replace(base, hard_silence_s=_clamp("hard_silence_s", hard_silence_s, base.hard_silence_s))
        if not isinstance(raw, Mapping):
            return base
        out: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw or f.name == "bots":
                continue
            v = raw[f.name]
            default = getattr(base, f.name)
            if f.name == "stt_engine":
                out[f.name] = v if v in STT_ENGINES else default
            elif f.name == "speculate":
                out[f.name] = v if v in SPECULATE_MODES else default
            elif f.name == "power_hold":
                out[f.name] = v if v in POWER_HOLDS else default
            elif isinstance(default, bool):
                out[f.name] = v if isinstance(v, bool) else default
            elif isinstance(default, int):
                out[f.name] = int(_clamp(f.name, v, default)) if _is_num(v) else default
            elif isinstance(default, float):
                out[f.name] = float(_clamp(f.name, v, default)) if _is_num(v) else default
        bots = raw.get("bots")
        if isinstance(bots, Mapping):
            clean: dict[str, dict[str, bool]] = {}
            for bot_id, conf in bots.items():
                if not isinstance(bot_id, str) or not isinstance(conf, Mapping):
                    continue
                entry = {k: conf[k] for k in ("backchannel", "follow_up") if isinstance(conf.get(k), bool)}
                if entry:
                    clean[bot_id] = entry
            out["bots"] = clean
        s = replace(base, **out)
        # Two decoder sessions want the 15-token first chunk unless the file says otherwise.
        if s.decoder_sessions == 2 and "first_chunk_tokens" not in raw:
            s = replace(s, first_chunk_tokens=15)
        if s.hold_min_s > s.hold_max_s:
            s = replace(s, hold_min_s=s.hold_max_s)
        return s

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["bots"] = {k: dict(v) for k, v in self.bots.items()}
        return d

    def bot(self, bot_id: str | None, *, backend: str | None = None) -> dict[str, bool]:
        """Effective per-bot defaults: backchannel off; follow_up on unless the backend is the advisor."""
        conf = self.bots.get(bot_id or "", {})
        return {"backchannel": bool(conf.get("backchannel", False)),
                "follow_up": bool(conf.get("follow_up", backend != "advisor"))}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clamp(name: str, v: Any, default: float) -> float:
    if not _is_num(v):
        return default
    lo, hi = RANGES.get(name, (float("-inf"), float("inf")))
    return min(hi, max(lo, float(v)))
