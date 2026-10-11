"""Local voice cloning and speech for DisPatch Chat, and the DisPatch Voice add-on.

This module is metadata only and imports nothing heavy: DisPatch reads
``EXTENSION`` during discovery (entry point group ``dispatch.extensions``) and
imports ``setup`` only after the API handshake passes.
"""
__version__ = "1.1.0"

EXTENSION = {
    "id": "voice",                        # [a-z][a-z0-9-]{1,31}, unique
    "name": "DisPatch Voice",
    "version": __version__,
    "api": [1, 1],                        # [min, max] of DisPatch's EXT_API this add-on accepts
    "setup": "dispatch_voice.host:setup",  # imported only after the handshake passes
    "env": ["DISPATCH_SPEECH_MODELS", "DISPATCH_VOICE_MODELS"],   # names copied into ctx.env
}
