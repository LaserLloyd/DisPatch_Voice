"""The add-on's OWN plugin seam: extra TTS engines, built-in voices, hooks.

Voice chat itself only needs "speak this text in voice X". Everything about
WHERE a built-in voice comes from (the optional, GPL-licensed Kokoro plugin
``dispatch-voice-kokoro``) lives in a separate package that plugs in here,
without this package importing it by name. This seam is unchanged by the v3
split (contracts §A: ``ExtensionAPI.api_version`` stays 3); only how a
plugin's router reaches DisPatch changed, see :meth:`ExtensionAPI.include_router`.

Wiring:

* ``DISPATCH_VOICE_EXTENSIONS=pkg.module:setup[,other.module:setup]``. At
  ``host.setup`` time :func:`mount_extensions` imports each target and calls
  ``setup(api)`` with an :class:`ExtensionAPI`. A broken plugin is logged and
  skipped, and never stops the add-on (nor DisPatch) booting.
* ``api.include_router(router, prefix="")`` mounts the plugin's FastAPI
  router under ``/api/voice`` through DisPatch's ``ctx.include_router``: the
  operator gate (unlocked tier only, Safe Mode 403, never the machine tier)
  and the Safe-Mode block list apply to every route. The router's paths are
  RELATIVE to ``/api/voice``; a router that already spells ``/api/voice`` in
  its paths is refused, because the gate's prefix cannot be stripped from a
  built router.
* ``api.register_engine(name, factory, *, role="primary", missing=None,
  sample_rate=None)`` adds a TTS engine selectable with
  ``DISPATCH_VOICE_TTS=<name>``. ``factory(settings)`` returns an object with
  the :class:`host.tts.TTSEngine` shape. ``role="fallback"`` marks an engine
  that may sit BEHIND the primary one (it speaks a sentence the primary
  cannot: a voice it does not know, or a failure before audio).
  ``missing(settings) -> list[str]`` says what the engine still needs (empty =
  ready); ``sample_rate`` lets the host pair a fallback with a primary of the
  same rate without building either. Optional engine attributes used when
  present: ``cache_tag(profile_id) -> str | None`` (opt-in disk cache of
  acknowledgements and spoken notices: return a tag only when the output is a
  pure function of the text for that voice), ``apply_settings(settings)``
  (the in-app voice/speed preferences changed) and ``warm()`` (start-up prewarm).
* ``api.set_builtin_voices_provider(list_fn, preview=None, info=None)``: the
  "built-in voices" a plugin brings. ``list_fn(settings) -> [{id:
  "builtin:<x>", name, lang, default?}]``; ``preview(settings, voice_id,
  text) -> (int16 pcm, rate)`` renders the Settings preview. ``info``
  (API 3) describes the provider for the Built-in voices tool, so the host
  names no plugin itself: ``{name, licence, install: [{where, cmd}],
  models: [{where, cmd}]}`` (``where`` is ``plugin``, ``env`` or
  ``backend``). ``builtin:<x>`` ids are opaque here. With no provider the
  list is empty and a saved ``builtin:`` voice degrades to the default
  cloned/stock voice.
* ``api.add_release_hook(fn)`` (API 3): ``fn()`` is called when voice is
  switched off, after the host dropped its own engines, so a plugin can drop
  the models it caches in its own module (switching off frees memory for
  plugins too).
* ``api.set_profile_resolver(fn)``: ``fn(bot_id) -> profile_id | None``. It
  picks the voice for a thread's bot, and the id is handed to the engine
  untouched. (``profiles_bridge`` installs the package's own.)
* ``api.set_ack_provider(fn)``: ``fn(profile_id) -> list[int16 ndarray]``
  returns short acknowledgements already rendered in that voice, at the
  engine's sample rate.
* ``api.set_persona_provider(fn)``: ``fn(bot_id, profile_id) -> str | None``
  returns the spoken-reply instruction for that voice. ``None`` means the
  built-in default (persona.py).
"""
from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("dispatch.voice")


@dataclass(frozen=True)
class EngineSpec:
    name: str
    factory: Callable
    role: str = "primary"                     # "primary" | "fallback"
    missing: Callable | None = None           # settings -> list[str]
    sample_rate: int | None = None

    def missing_for(self, settings) -> list[str]:
        if self.missing is None:
            return []
        try:
            return list(self.missing(settings) or [])
        except Exception as e:
            log.exception("voice: engine %s readiness check failed", self.name)
            return [f"TTS engine {self.name} check failed ({type(e).__name__})"]

    def ready(self, settings) -> bool:
        return not self.missing_for(settings)


_engines: dict[str, EngineSpec] = {}
_builtin_list: Callable | None = None
_builtin_preview: Callable | None = None
_builtin_info: dict = {}
_release_hooks: list[Callable] = []
_profile_resolver: Callable[[str | None], str | None] | None = None
_ack_provider: Callable[[str | None], list] | None = None
_persona_provider: Callable[[str | None, str | None], str | None] | None = None


def register_engine(name: str, factory: Callable, *, role: str = "primary",
                    missing: Callable | None = None, sample_rate: int | None = None) -> None:
    if role not in ("primary", "fallback"):
        raise ValueError(f"role must be 'primary' or 'fallback', not {role!r}")
    key = name.strip().lower()
    _engines[key] = EngineSpec(key, factory, role, missing, sample_rate)


def engine_spec(name: str) -> EngineSpec | None:
    return _engines.get((name or "").strip().lower())


def engine_factory(name: str) -> Callable | None:
    spec = engine_spec(name)
    return spec.factory if spec else None


def engine_names() -> list[str]:
    return sorted(_engines)


def primary_engines() -> list[EngineSpec]:
    """Registered primary engines, in registration order."""
    return [e for e in _engines.values() if e.role == "primary"]


def fallback_for(settings, primary: str, sample_rate: int | None = None) -> EngineSpec | None:
    """The first ready fallback engine that can sit behind ``primary``: not
    the same engine, and (when both declare one) the same sample rate."""
    for spec in _engines.values():
        if spec.role != "fallback" or spec.name == (primary or "").strip().lower():
            continue
        if sample_rate and spec.sample_rate and spec.sample_rate != sample_rate:
            continue
        if spec.ready(settings):
            return spec
    return None


def set_builtin_voices_provider(list_fn: Callable, preview: Callable | None = None,
                                info: dict | None = None) -> None:
    global _builtin_list, _builtin_preview, _builtin_info
    _builtin_list, _builtin_preview = list_fn, preview
    _builtin_info = _clean_info(info)


def _clean_info(info) -> dict:
    """Only the documented keys, as short strings: it is shown to the client."""
    if not isinstance(info, dict):
        return {}
    out: dict = {}
    for k in ("name", "licence"):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()[:80]
    for k in ("install", "models"):
        steps = []
        for st in info.get(k) or []:
            if (isinstance(st, dict) and st.get("where") in ("plugin", "env", "backend")
                    and isinstance(st.get("cmd"), str) and st["cmd"].strip()):
                steps.append({"where": st["where"], "cmd": st["cmd"].strip()[:300]})
        if steps:
            out[k] = steps[:6]
    return out


def builtin_info() -> dict:
    """What the provider says about itself ({} without a provider or info)."""
    return dict(_builtin_info)


def add_release_hook(fn: Callable) -> None:
    _release_hooks.append(fn)


def run_release_hooks() -> None:
    for fn in _release_hooks:
        try:
            fn()
        except Exception:
            log.exception("voice: a release hook failed")


def has_builtin_voices() -> bool:
    return _builtin_list is not None


def builtin_voices(settings) -> list[dict]:
    """The provider's built-in voices; [] when no provider is installed or it fails."""
    if _builtin_list is None:
        return []
    try:
        out = _builtin_list(settings) or []
        return [dict(v) for v in out if isinstance(v, dict) and isinstance(v.get("id"), str)]
    except Exception:
        log.exception("voice: built-in voice provider failed")
        return []


def builtin_preview():
    """The provider's preview renderer, or None."""
    return _builtin_preview


def set_profile_resolver(fn) -> None:
    global _profile_resolver
    _profile_resolver = fn


def set_ack_provider(fn) -> None:
    global _ack_provider
    _ack_provider = fn


def set_persona_provider(fn) -> None:
    global _persona_provider
    _persona_provider = fn


def resolve_profile(bot_id: str | None, voice: str | None = None) -> str | None:
    """The profile that speaks for ``bot_id``. ``voice`` (a locked device's
    override, already checked to exist) takes the place of the bot's saved voice."""
    if _profile_resolver is None:
        # No profile store (Chatterbox models not fetched): a bot can still
        # pick a built-in voice. The bot's saved voice comes from DisPatch
        # through the extension context.
        try:
            from . import context
            from .tts import builtin_id
            want = voice or context.bot_voice(bot_id)
            # A saved built-in voice only means something while a provider
            # is installed; without one the engine's default voice speaks.
            return want if builtin_id(want) and has_builtin_voices() else None
        except Exception:
            return None
    try:
        if voice:
            try:
                return _profile_resolver(bot_id, voice=voice) or None
            except TypeError:                      # a resolver that predates the override
                pass
        return _profile_resolver(bot_id) or None
    except Exception:
        log.exception("voice: profile resolver failed; using the default voice")
        return None


def acks_for(profile_id: str | None) -> list | None:
    """Pre-rendered acks for this profile, or None when no provider has any."""
    if _ack_provider is None:
        return None
    try:
        acks = _ack_provider(profile_id)
        return list(acks) if acks else None
    except Exception:
        log.exception("voice: ack provider failed")
        return None


def persona_for(bot_id: str | None, profile_id: str | None) -> str | None:
    if _persona_provider is None:
        return None
    try:
        p = _persona_provider(bot_id, profile_id)
        return p.strip() if isinstance(p, str) and p.strip() else None
    except Exception:
        log.exception("voice: persona provider failed")
        return None


def reset() -> None:
    """Tests only."""
    global _profile_resolver, _ack_provider, _persona_provider, _builtin_list, _builtin_preview
    global _builtin_info
    _engines.clear()
    _profile_resolver = _ack_provider = _persona_provider = None
    _builtin_list = _builtin_preview = None
    _builtin_info = {}
    _release_hooks.clear()


PLUGIN_PREFIX = "/api/voice"


@dataclass
class ExtensionAPI:
    """Handed to a plugin's ``setup(api)``."""
    #: The DisPatch ExtensionContext (its ``include_router`` is the only way a
    #: router reaches the app). None outside DisPatch: routers then raise.
    ctx: object | None = None
    #: Kept for the v2/v3 shape (``ExtensionAPI(app, require_full)``); unused.
    require_full: Callable | None = None
    #: 2 = role/missing/sample_rate on register_engine + the built-in voices
    #: provider. 3 = ``info=`` on set_builtin_voices_provider and
    #: add_release_hook. A plugin can check it and refuse politely, or pass
    #: the newer arguments only when they exist.
    api_version: int = 3

    def include_router(self, router, *, prefix: str = "") -> None:
        """Mount ``router`` under ``/api/voice<prefix>`` behind DisPatch's gate.

        The router's own paths are relative to that prefix. A router whose
        paths already begin with ``/api/voice`` (the pre-v3 way, where the
        plugin mounted on the app itself) is refused with ValueError: the gate
        adds the prefix and nothing can strip it from a built router."""
        if self.ctx is None or not hasattr(self.ctx, "include_router"):
            raise RuntimeError("include_router needs the DisPatch extension context (only inside DisPatch)")
        if prefix and not prefix.startswith("/"):
            prefix = "/" + prefix
        for r in getattr(router, "routes", []):
            path = getattr(r, "path", "")
            if path.startswith(PLUGIN_PREFIX):
                raise ValueError(f"plugin router paths are relative to {PLUGIN_PREFIX}; got {path!r}")
        self.ctx.include_router(router, prefix=PLUGIN_PREFIX + prefix.rstrip("/"))

    register_engine = staticmethod(register_engine)
    set_builtin_voices_provider = staticmethod(set_builtin_voices_provider)
    add_release_hook = staticmethod(add_release_hook)
    set_profile_resolver = staticmethod(set_profile_resolver)
    set_ack_provider = staticmethod(set_ack_provider)
    set_persona_provider = staticmethod(set_persona_provider)


def mount_extensions(ctx=None, spec: str | None = None) -> list[str]:
    """Load every ``module:callable`` in DISPATCH_VOICE_EXTENSIONS. Returns
    the targets that loaded. ``ctx`` is the DisPatch extension context (None
    for the CLI: plugins then register engines but cannot mount routers)."""
    import os
    spec = spec if spec is not None else os.environ.get("DISPATCH_VOICE_EXTENSIONS", "")
    loaded: list[str] = []
    for target in (t.strip() for t in spec.split(",")):
        if not target:
            continue
        mod_name, _, attr = target.partition(":")
        try:
            fn = getattr(importlib.import_module(mod_name), attr or "setup")
            fn(ExtensionAPI(ctx))
            loaded.append(target)
            log.info("voice: extension %s loaded", target)
        except Exception:
            log.exception("voice: extension %s failed to load; skipped", target)
    return loaded
