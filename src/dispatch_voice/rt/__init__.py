"""Realtime engines: streaming STT, turn policy, speculation, pipelined TTS,
backchannels and the heard map (C-§B.3, owner T4).

Rules of this package:

* It never imports ``dispatch_voice.host``; the session hub (T3) is the only
  caller and the only place the pieces are wired together.
* Nothing here loads a model at import time. ``onnxruntime``, ``onnx_asr``
  and the Chatterbox engine are imported inside the functions that need them,
  so the pure parts (policy, speculator, classifier, heard map) run in any
  test without the speech stack.
* Every number is the measured one from SPEC §4; the settings type
  (:class:`RtSettings`) carries the defaults and the contract's ranges.
"""
from .settings import RtSettings

__all__ = ["RtSettings"]
