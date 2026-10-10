"""Speech engines the host registers.

``chatterbox`` (this package's Chatterbox-Turbo ONNX, cloned voices + the
stock voice) is registered by ``profiles_bridge.install`` when its models are
fetched; plugins add more through ``registry.register_engine`` (the optional
dispatch-voice-kokoro plugin adds ``kokoro``). Track T4's pipelined engine
replaces the adapter in ``chatterbox.py`` at Release C with the same name.
"""
