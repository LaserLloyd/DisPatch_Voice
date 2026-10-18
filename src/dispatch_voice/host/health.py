"""``GET /api/voice/status`` ``health``: what is wrong, in words, with the fix.

Every entry is ``{"id", "state", "reason", "fix", "code"}``; ``state`` is one of
``ok`` / ``degraded`` / ``unconfigured`` / ``down`` (the same words a provider's
``health()`` uses). Reasons and fixes are plain sentences a person can act on; ``code`` is
stable for scripts and for the UI's own wording. Nothing here contains a path, a key or a
token.

The report is cheap (file stats, a PATH lookup, no network, no model load).
"""
from __future__ import annotations

from . import models, netpolicy, sysreq
from . import providers as P
from .providers import Health

FETCH_HINT = "Settings > Voice > Models > Download, or run `dispatch-voice fetch-models`."


def _entry(id_: str, h: Health) -> dict:
    return {"id": id_, **h.to_dict()}


def report(settings, *, rig_token: bool | None = None) -> list[dict]:
    """The health of every part of voice for ``settings`` (a ``VoiceSettings``)."""
    out: list[dict] = []
    forced = settings.forced
    if forced == "off":
        out.append(_entry("switch", Health("down", "Voice is switched off by the server's DISPATCH_VOICE setting.",
                                           "Unset DISPATCH_VOICE (or set it to auto) and restart DisPatch.",
                                           "forced_off")))
    elif not settings.switched_on:
        out.append(_entry("switch", Health("unconfigured", "Voice chat is switched off.",
                                           "Turn on Settings > Voice > Voice chat.", "switch_off")))
    else:
        out.append(_entry("switch", Health("ok")))

    # models and engines
    miss = settings.missing()
    if miss:
        models_missing = [m for m in miss if m.startswith(("model", "dispatch-voice models"))]
        if models_missing:
            out.append(_entry("models", Health("unconfigured", "Speech models are missing: " + ", ".join(models_missing) + ".",
                                               FETCH_HINT, "models_missing")))
        for m in miss:
            if m in models_missing:
                continue
            if m.startswith("python package"):
                out.append(_entry("python", Health("down", f"{m} is not installed.",
                                                   "Reinstall dispatch-voice: `uv pip install -U dispatch-voice`.",
                                                   "package_missing")))
            else:
                out.append(_entry("engine", Health("unconfigured", m[0].upper() + m[1:] + ".", "", "engine_missing")))
    else:
        out.append(_entry("models", Health("ok")))
    if models.legacy_present():
        out.append(_entry("model_folders", Health(
            "degraded", "Speech models are still in the old two-folder layout.",
            "Settings > Voice > Models > Move to the new folder, or `dispatch-voice migrate-models --apply`.",
            "models_legacy_layout")))

    # operating-system programs
    for r in sysreq.check():
        if r["ok"]:
            continue
        out.append(_entry(r["id"], Health("degraded" if r["optional"] else "unconfigured",
                                          f"{r['id']} is not installed ({r['required_for']} needs it).", r["fix"],
                                          f"{r['id']}_missing")))

    # the rig
    rs = settings.rig
    if rs.enabled or rs.url:
        from . import rig
        token = rig.read_token() if rig_token is None else None
        has_token = bool(token.value) if token is not None else rig_token
        allow = tuple(settings.rig_allow)
        if not rs.url:
            out.append(_entry("rig", Health("unconfigured", "The rig is switched on but no address is set.",
                                            "Enter the rig's ws:// address in Settings > Voice.", "rig_url_missing")))
        else:
            v = netpolicy.check_url_policy(rs.url, allow)
            if not v.ok:
                out.append(_entry("rig", Health("down", v.reason, v.fix, v.code)))
            elif not has_token:
                why = {"permissions": "its file is readable by other users"}.get(getattr(token, "problem", None), "")
                out.append(_entry("rig", Health(
                    "unconfigured", "The rig token is not set" + (f" ({why})." if why else "."),
                    "Paste the token in Settings > Voice, or set SPEECHFORGE_TOKEN.", "rig_token_missing")))
            elif v.warning:
                out.append(_entry("rig", Health("degraded", v.warning, "Use wss:// or keep the rig on a VPN.", v.code)))
            else:
                out.append(_entry("rig", Health("ok")))

    # remote speech servers
    hq = settings.hq_url if settings.tts_provider != "cpu" else ""
    for name, url, code in (("stt_server", settings.stt_url if settings.stt_provider == "openai" else "", "stt_url"),
                            ("tts_server", hq, "hq_url")):
        if not url:
            if name == "stt_server" and settings.stt_provider == "openai":
                out.append(_entry(name, Health("unconfigured", "No transcription server address is set.",
                                               "Enter it in Settings > Voice (stt_url).", "stt_url_missing")))
            continue
        v = netpolicy.check_url_policy(url, tuple(settings.rig_allow), schemes=("http", "https"))
        if not v.ok:
            out.append(_entry(name, Health("down", v.reason, v.fix, v.code)))
        elif v.warning:
            out.append(_entry(name, Health("degraded", v.warning, "Use https:// or a VPN.", v.code)))
        else:
            out.append(_entry(name, Health("ok")))

    # providers the user chose
    chosen = {"stt": settings.stt_provider}
    for kind, pid in chosen.items():
        if pid not in P.REGISTRY.ids(kind):
            out.append(_entry(f"{kind}_provider", Health(
                "down", f"The {kind.upper()} provider {pid!r} is not installed.",
                "Choose another provider in Settings > Voice, or install the plugin.", "provider_missing")))
    return out


def overall(entries: list[dict]) -> str:
    order = {"ok": 0, "degraded": 1, "unconfigured": 2, "down": 3}
    return max((e["state"] for e in entries), key=lambda s: order.get(s, 0), default="ok")
