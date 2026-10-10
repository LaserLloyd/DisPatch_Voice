"""Conversation text: spoken-reply instructions, acknowledgements, speech text and role-play.

* :mod:`.persona` — ``build(...) -> VoiceInstructions`` (system section + gateway trailer).
* :mod:`.acks` / :mod:`.ack_lines` — intent, pools, selector, timing gate, tool
  status lines, background render queue, ``voice-acks.yaml``.
* :mod:`.speech_text` — ``SentenceSplitter`` (source :class:`Span` s),
  ``normalise``, ``segment``, ``rp_render``.

Pure Python (numpy only inside the render queue, imported lazily). Never imports
``host/`` or DisPatch; the session (host/) is the only caller. Docs:
``docs/voice-chat/persona-acks-rp.md``.
"""
from .speech_text import Span  # noqa: F401  (rt.heard and the session use text.Span)
