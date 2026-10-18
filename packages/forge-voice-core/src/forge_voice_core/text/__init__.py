"""Spoken-text logic that does not depend on a host: ``speech_text`` (``SentenceSplitter`` with source
:class:`Span` s, ``normalise``, ``segment``, ``rp_render``). Pure Python; never imports a host."""
from .speech_text import Span  # noqa: F401  (rt.heard and the sessions use text.Span)
