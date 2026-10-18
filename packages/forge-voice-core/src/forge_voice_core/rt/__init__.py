"""Realtime engines: streaming STT, turn policy, speculation, pipelined TTS,
backchannels and the heard map.

Rules of this package:

* It never imports a host (``forge_voice_core.hostlink`` is the only seam); the session hub of the
  application is the only caller and the only place the pieces are wired together.
* Nothing here loads a model at import time. ``onnxruntime``, ``onnx_asr``
  and the Chatterbox engine are imported inside the functions that need them,
  so the pure parts (policy, speculator, classifier, heard map) run in any
  test without the speech stack.
* Every number is the measured one; the settings type (:class:`RtSettings`)
  carries the defaults and the ranges.
"""
from .settings import RtSettings

__all__ = ["RtSettings"]
