"""forge-voice-core: the host-independent voice engines.

CPU speech-to-text (Parakeet, Moonshine), voice-activity and turn detection (Silero, Smart Turn),
text-to-speech (Chatterbox-Turbo ONNX), OpenAI-compatible STT/TTS clients, the provider interfaces, the
model registry and download job, the network address policy and the optional SpeechForge (sf.1) rig
client. No web framework, no application: a host talks to the package through
:mod:`forge_voice_core.hostlink` (see ``docs/HOST_API.md``).

Importing the package imports nothing heavy; each engine imports numpy/onnxruntime when it is used.
"""
__version__ = "0.1.0"
