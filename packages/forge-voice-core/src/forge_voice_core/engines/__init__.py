"""Speech engines.

``chatterbox`` (the Chatterbox-Turbo ONNX engine: cloned voices + the stock voice) is registered when
its models are on disk (``HostLink.ensure_engine``); plugins add more through
``registry.register_engine``; ``rig_tts`` speaks through the optional GPU speech rig.
"""
