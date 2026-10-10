"""The opt-in voice tools the add-on registers with DisPatch (contracts §B.1
``register_tools`` / ``ToolSpec``; amendment T0-2: DisPatch reads these by
attribute, never ``isinstance``, so the mirror below is the add-on's own).

A tool row is a ``tools.yaml`` builtin (``kind: builtin``): off until the
operator switches it on in Settings → Tools, never listed for Safe Mode. The
``probe`` answers "is the feature installed on this box" (never "switched
on"), which is what the Tools tab shows next to a switch that is on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ToolSpec:
    id: str                       # tools.yaml builtin id, [a-z0-9-]{1,40}
    feature: str                  # the feature key the frontend tile maps to
    title: str
    icon: str                     # emoji or rail icon id (display only)
    opt_in: bool                  # enabled=false by default in a fresh manifest
    probe: Callable[[], bool]     # installed on this box


#: The three opt-in voice tools, by id (the order the Tools tab lists them in).
VOICE_TOOLS = ("voice-cloning", "voice-builtin", "voice-lab")


def cloning_installed() -> bool:
    """The cloned-voice engine is this package: installed whenever the add-on
    is (whether its models are fetched is what ``/api/voice/status.missing``
    says, and the tool's own pane explains)."""
    return True


def builtin_installed() -> bool:
    """A built-in voice provider (e.g. the optional Kokoro plugin) registered
    itself through the add-on's plugin seam."""
    from . import registry
    return registry.has_builtin_voices()


def lab_installed() -> bool:
    return True


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("voice-cloning", "voice_cloning", "Voice cloning", "🎙️", True, cloning_installed),
        ToolSpec("voice-builtin", "voice_builtin", "Built-in voices", "🔊", True, builtin_installed),
        ToolSpec("voice-lab", "voice_lab", "Voice lab", "🧪", True, lab_installed),
    ]
