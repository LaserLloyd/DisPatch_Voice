"""The settings object the engines read, for a host that has no settings class of its own.

The engines (``stt``, ``tts``, ``vad_turn``, ``providers``, the rig client) take a duck-typed
``settings`` argument and read the attributes listed below with ``getattr``. DisPatch Voice passes its
own ``VoiceSettings`` (a superset); a host that does not want to write one builds an
:class:`EngineSettings` from its own configuration::

    from forge_voice_core.engine_settings import EngineSettings
    cfg = EngineSettings(threads=4, stt_provider="cpu", models_dir=models.speech_dir())

Every field has the default the shipped engines were tuned with.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path


def _models_dir() -> Path:
    from . import models
    return models.speech_dir()


def _rt():
    from .rt.settings import RtSettings
    return RtSettings()


def _rig():
    from .rig import RigSettings
    return RigSettings()


@dataclass(frozen=True)
class EngineSettings:
    models_dir: Path = field(default_factory=_models_dir)
    # speech to text
    stt_engine: str = "parakeet"          # parakeet | moonshine
    stt_provider: str = "cpu"             # cpu | openai | a plugin's id
    stt_url: str = ""
    stt_model: str = ""
    stt_timeout_s: float = 20.0
    # text to speech
    tts_engine: str = "auto"
    tts_provider: str = "auto"
    builtin_voice: str = ""
    speed: float = 1.05
    minimax_model: str = "speech-2.8-turbo"
    minimax_voice: str = "Wise_Woman"
    minimax_base_url: str = "https://api.minimax.io"
    hq_url: str = ""
    hq_model: str = ""
    hq_timeout_s: float = 4.0
    # endpointing (seconds unless noted)
    vad_threshold: float = 0.5
    min_speech_s: float = 0.25
    turn_probe_silence_s: float = 0.25
    turn_hard_silence_s: float = 1.4
    no_smart_turn_silence_s: float = 0.8
    max_utterance_s: float = 30.0
    smart_turn_threshold: float = 0.5
    # compute
    threads: int = 6                      # onnxruntime intra-op threads, speech recognition
    tts_threads: int = 8                  # onnxruntime threads, a local TTS engine
    # the realtime pipeline, the optional speech rig, the network allow-list, plugins
    rt: object = field(default_factory=_rt)
    rig: object = field(default_factory=_rig)
    rig_allow: tuple = ()
    extensions: tuple = ()
    sources: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def silero_path(self) -> Path:
        return self.models_dir / "silero_vad.onnx"

    @property
    def smart_turn_path(self) -> Path:
        return self.models_dir / "smart-turn-v3.2-cpu.onnx"

    @property
    def parakeet_dir(self) -> Path:
        return self.models_dir / "parakeet-tdt-0.6b-v2"

    @property
    def switched_on(self) -> bool:
        """A host without an on/off switch is always on."""
        return True

    @property
    def forced(self) -> str | None:
        """"on"/"off" when the host's environment decides the switch; a host without a switch: None."""
        return None

    def missing(self) -> list[str]:
        """Human-readable list of what stops voice from working (empty = ready)."""
        from . import models, netpolicy, registry
        out: list[str] = []
        for mod in ("numpy", "onnxruntime"):
            if importlib.util.find_spec(mod) is None:
                out.append(f"python package {mod}")
        if not self.silero_path.exists():
            out.append(f"model {self.silero_path.name}")
        if self.stt_provider == "openai":
            if not self.stt_url:
                out.append("a transcription server address (stt_url)")
            else:
                v = netpolicy.check_url_policy(self.stt_url, tuple(self.rig_allow), schemes=("http", "https"))
                if not v.ok:
                    out.append(f"a transcription server address the network policy allows (stt_url: {v.code})")
        elif self.stt_engine in ("parakeet", "moonshine"):         # Moonshine falls back to Parakeet
            if importlib.util.find_spec("onnx_asr") is None:
                out.append("python package onnx-asr")
            if not models.parakeet_complete(self.parakeet_dir):
                out.append("model parakeet-tdt-0.6b-v2")
        out += self._tts_missing(registry)
        return out

    def _tts_missing(self, registry) -> list[str]:
        from .engines import chatterbox
        name = self.tts_engine
        if name == "minimax":
            return [] if os.environ.get("MINIMAX_API_KEY") else ["MINIMAX_API_KEY (DISPATCH_VOICE_TTS=minimax)"]
        if name in ("chatterbox", "auto"):
            if chatterbox.available():
                return []
            if name == "chatterbox" or not any(s.ready(self) for s in registry._engines.values()):
                return chatterbox.missing()
            return []
        spec = registry.engine_spec(name)
        if spec is None:
            return [f"TTS engine {name!r} is not installed"]
        return spec.missing_for(self)
