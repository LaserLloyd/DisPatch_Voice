"""The spoken-reply instructions a voice turn hands DisPatch.

Layered on whatever the bot already is, and never stored on the user's row.
They travel as a :class:`VoiceInstructions` (contracts §B.1; amendment T0-2:
DisPatch reads ``.system`` / ``.trailer`` by attribute): DisPatch places
``system`` in the system slot of the backends that have one (advisor, API
bots) and appends ``trailer`` to the composed user text of gateway agents —
never a prefix (SPEC §6.1). The wording is T5's ``text/persona.py`` builder;
this module only gathers what the builder needs from the host: the bot and
its backend (through the extension context), the bot's ``voice.yaml``
defaults, whether the target voice renders expressive tags, an extension's
own persona (``registry.set_persona_provider``), the driver's reply-length and
follow-up choices (the ``start`` frame) and, on the one turn after a barge-in,
the heard/unheard note (SPEC §4.8).
"""
from __future__ import annotations

from types import SimpleNamespace

from ..text import persona as _persona
from ..text.persona import VoiceInstructions
from . import context, registry
from .config import REPLY_STYLES

__all__ = ["VoiceInstructions", "follow_up_default", "instructions_for", "tags_capable"]


def tags_capable(profile_id: str | None, settings, tts_engine=None) -> bool:
    """Does the voice that will speak render ``[laugh]``/``[sigh]``-style tags?

    Decided by the engine that ACTUALLY speaks ``profile_id`` (FR-works 3): "no
    voice" is first resolved to the box's default voice (``routes.effective_voice``
    — on a box whose default is a built-in Kokoro voice that is the plugin, not
    the cloned engine behind it), then the chain is followed to that voice's
    engine (``tts.speaker``) and its ``supports_tags`` decides. A cloned
    (Chatterbox) voice says True while ``expressive_tags`` is on; a built-in
    voice never does unless its engine declares ``supports_tags = True``.
    Without an engine (tests, the CLI) a non-built-in voice counts as the cloned
    engine when that is installed."""
    from .tts import builtin_id, speaker
    if settings is not None and not getattr(settings, "expressive_tags", True):
        return False
    voice = profile_id
    if not voice and settings is not None:
        try:
            from .routes import effective_voice
            voice = effective_voice(settings, None)
        except Exception:
            voice = None
    if tts_engine is not None:
        flag = getattr(speaker(tts_engine, voice), "supports_tags", None)
        return flag is True
    if builtin_id(voice):
        return False
    try:
        from .engines.chatterbox import available
        return available()
    except Exception:
        return False


def follow_up_default(bot_id: str | None, settings, backend: str | None = None) -> bool:
    """The bot's own follow-up default (``voice.yaml: bots.<id>.follow_up``, else
    on unless the backend is the advisor — SPEC §6.1 L6)."""
    backend = backend if backend is not None else context.bot_backend(bot_id)
    conf = None
    if settings is not None:
        conf = settings.bot(bot_id, backend=backend) if hasattr(settings, "bot") else None
    return _persona.default_follow_up(backend, conf)


def instructions_for(bot_id: str | None, profile_id: str | None = None, *,
                     style: str = "normal", follow_up: bool | None = None,
                     interrupted: dict | None = None, settings=None,
                     tts_engine=None, backend: str | None = None,
                     tags_ok: bool | None = None, langs=("en",)) -> VoiceInstructions:
    """The instructions for one voice turn (T5's wording, placed per backend).

    ``follow_up`` None = the bot's default; ``interrupted`` =
    ``{"heard": …, "unheard": …}`` on the turn right after a barge-in, else
    None. ``settings`` (a ``VoiceSettings``) supplies ``expressive_tags``,
    ``gateway_trailer`` and the per-bot defaults; without it the add-on's
    defaults apply. ``backend``/``tags_ok`` override what the context and the
    engine say (tests, the standalone CLI)."""
    style = style if style in REPLY_STYLES else "normal"
    bot = context.bot(bot_id) or SimpleNamespace(id=bot_id or "", name="", backend=None)
    if backend is None:
        backend = context.bot_backend(bot_id)
    if follow_up is None:
        follow_up = follow_up_default(bot_id, settings, backend)
    if tags_ok is None:
        tags_ok = tags_capable(profile_id, settings, tts_engine)
    trailer_mode = getattr(settings, "gateway_trailer", "full") if settings is not None else "full"
    note = None
    if isinstance(interrupted, dict) and (interrupted.get("heard") or interrupted.get("unheard")):
        note = {"heard": str(interrupted.get("heard") or ""),
                "unheard": str(interrupted.get("unheard") or "")}
    # DisPatch ≥ 2.2.0 tells the add-on how a persona names its user (the
    # advisor's ``owner``) and whether the bot has a persona of its own at all
    # (an API bot on the default system prompt has none): additive BotView
    # fields, read by attribute so an older host simply leaves them out
    # (Amendment FINAL-FIX-1; FR-works 12).
    owner = getattr(bot, "owner", None)
    has_persona = getattr(bot, "has_persona", None)
    return _persona.build(bot, backend=backend, profile_tags=bool(tags_ok), style=style,
                          follow_up=bool(follow_up), interrupted=note,
                          extension_persona=registry.persona_for(bot_id, profile_id),
                          gateway_trailer=trailer_mode if trailer_mode in ("full", "tag") else "full",
                          langs=tuple(langs) or ("en",),
                          owner=owner if isinstance(owner, str) and owner.strip() else None,
                          has_persona=has_persona if isinstance(has_persona, bool) else None)
