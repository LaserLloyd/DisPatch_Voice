"""``host.setup(ctx)``: the add-on wires itself into DisPatch in the order the
contract fixes (§B.4), with the right arguments, and nothing else."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from fakes import FakeContext, ToolSpec
from fastapi.testclient import TestClient

from dispatch_voice import EXTENSION, __version__
from dispatch_voice.host import STATIC_DIR, context, routes, setup, ws
from dispatch_voice.host.engines import chatterbox
from dispatch_voice.host.tools import VOICE_TOOLS



def rt_settings_out(**over):
    """The ``rt`` tree GET/PATCH /api/voice/settings returns: every C-§F.1 key (defaults)
    plus the hold, with ``over`` applied."""
    from dispatch_voice.rt.settings import RtSettings
    d = RtSettings().to_dict()
    d.pop("bots", None)
    d.pop("hard_silence_s", None)
    d.update(over)
    return d


def rt_status_out(power_hold="performance"):
    """The ``rt`` block of /api/voice/status (C-§D)."""
    from dispatch_voice.rt.settings import RtSettings
    r = RtSettings()
    return {"stt": r.stt_engine, "speculate": r.speculate, "decoder_prompt_s": r.decoder_prompt_s,
            "decoder_sessions": r.decoder_sessions, "addendum_s": r.addendum_s, "power_hold": power_hold}

def test_setup_wires_the_context_in_contract_order(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    setup(host_env)
    names = [n for n, _, _ in host_env.calls]
    assert names == ["serve_static", "include_router", "thread_websocket",
                     "set_bot_field_validator", "set_bot_field_validator",
                     "register_tools", "on_startup", "on_shutdown"]
    st = host_env.static
    assert (st["directory"], st["entry"], st["css"], st["locales"]) == (STATIC_DIR, "index.js", ["voice.css"], True)
    assert routes.ASSETS_VERSION == st["assets_version"] and st["not_served"] == []
    (_router, prefix, limits), = host_env.routers
    assert prefix == "/api/voice"
    assert limits == {"/api/voice/profiles": 20 * 1024 * 1024 + 64 * 1024,
                      "/api/voice/profiles/analyse": 20 * 1024 * 1024 + 64 * 1024,
                      "/api/voice/lab/test": 320_000}
    reg = host_env.websockets["voice"]
    assert reg["handler"] is ws.serve and reg["max_total"] == 3 and reg["max_per_session"] == 2
    assert set(host_env.validators) == {"voice", "voice_rp"}
    assert [t.id for t in host_env.tools] == list(VOICE_TOOLS)
    assert all(t.opt_in for t in host_env.tools)
    probes = {t.id: t.probe() for t in host_env.tools}
    assert probes == {"voice-cloning": True, "voice-builtin": False, "voice-lab": True}
    assert host_env.startup == [(routes.prewarm_start, True)]
    assert host_env.shutdown == [routes.shutdown]
    assert routes._LOADED == []                       # no Chatterbox models: no cloned voices
    assert context.get_context() is host_env


def test_setup_registers_a_locked_status_when_the_host_offers_it(host_env, monkeypatch):
    """DisPatch ≥ 2.3.4 lets a locked device use voice on the bots it can see;
    what that device may read is ``{enabled, ready}`` — two booleans, nothing
    that names an engine, a path or the rig."""
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    got = []
    host_env.locked_status = got.append
    setup(host_env)
    (fn,) = got
    st = fn()
    assert set(st) == {"enabled", "ready"}
    assert all(isinstance(v, bool) for v in st.values())
    from dispatch_voice.host.config import load_settings
    s = load_settings()
    assert st == {"enabled": bool(s.enabled), "ready": not s.missing()}


def test_validators_registered_by_setup(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    setup(host_env)
    host_env.validate_bot_field("voice", "")
    with pytest.raises(ValueError, match="Unknown voice"):
        host_env.validate_bot_field("voice", "who-knows")
    for ok in ("", "all", "skip_actions", "dialogue", "dialogue_narrator"):
        host_env.validate_bot_field("voice_rp", ok)
    with pytest.raises(ValueError, match="voice_rp"):
        host_env.validate_bot_field("voice_rp", "quotes")
    with pytest.raises(ValueError):
        host_env.validate_bot_field("voice_rp", None)


def test_max_sessions_from_voice_yaml_reaches_the_gate(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    host_env.paths.voice_yaml.parent.mkdir(parents=True, exist_ok=True)
    host_env.paths.voice_yaml.write_text("max_sessions: 5\n", encoding="utf-8")
    setup(host_env)
    assert host_env.websockets["voice"]["max_total"] == 5


def test_setup_mounts_the_profiles_router_when_the_chatterbox_models_are_present(host_env, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    monkeypatch.setattr(chatterbox.LazyEngine, "_get",
                        lambda self: SimpleNamespace(dtype="fake", watermarking=False))
    monkeypatch.setenv("DISPATCH_VOICE_MODELS", str(host_env.paths.models_dir / "cb"))
    setup(host_env)
    assert routes._LOADED == ["dispatch-voice"]
    c = TestClient(host_env.app)
    r = c.get("/api/voice/profiles")
    assert r.status_code == 200 and r.json()["voices"] == [] and r.json()["default"] is None
    assert c.get("/api/voice/profiles/consent-kinds").status_code == 200
    # Contracts §D / A12: no HTML under /api/voice; the page lives in `dispatch-voice serve`.
    assert c.get("/api/voice/profiles/ui").status_code == 404
    assert c.get("/api/voice/profiles/ui/voices.js").status_code == 404
    # /api/voices is gone, no alias (C11).
    assert c.get("/api/voices").status_code == 404
    host_env.operator = False
    assert c.get("/api/voice/profiles").status_code == 403
    host_env.operator = True
    st = c.get("/api/voice/status").json()
    assert "dispatch-voice" in st["extensions"] and st["stock_voice"] == "stock-host"
    assert st["plugins"]["cloning"] == {"installed": True, "ready": True, "missing": []}


def test_plugin_seam_runs_inside_setup_and_mounts_under_api_voice(host_env, monkeypatch):
    from fastapi import APIRouter
    monkeypatch.setattr(chatterbox, "available", lambda: False)

    def plug(api):
        assert api.api_version == 3
        r = APIRouter(prefix="/kokoro")

        @r.get("/voices")
        def voices():
            return {"voices": ["af_heart"]}

        api.include_router(r)
        api.register_engine("kok", lambda s: None, role="fallback", missing=lambda s: [], sample_rate=24000)
        api.set_builtin_voices_provider(lambda s: [{"id": "builtin:af_heart", "name": "Heart"}])

    mod = types.ModuleType("fake_voice_plugin_ok")
    mod.setup = plug
    monkeypatch.setitem(sys.modules, "fake_voice_plugin_ok", mod)
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", "fake_voice_plugin_ok:setup")
    setup(host_env)
    assert routes._LOADED == ["fake_voice_plugin_ok:setup"]
    prefixes = [p for _, p, _ in host_env.routers]
    assert prefixes == ["/api/voice", "/api/voice"]              # the plugin's router rides the same gate
    c = TestClient(host_env.app)
    assert c.get("/api/voice/kokoro/voices").json() == {"voices": ["af_heart"]}
    host_env.operator = False
    assert c.get("/api/voice/kokoro/voices").status_code == 403
    host_env.operator = True
    probes = {t.id: t.probe() for t in host_env.tools}
    assert probes["voice-builtin"] is True                      # a provider registered itself


def test_a_plugin_router_with_absolute_paths_is_refused_and_the_plugin_skipped(host_env, monkeypatch, caplog):
    from fastapi import APIRouter
    monkeypatch.setattr(chatterbox, "available", lambda: False)

    def plug(api):
        r = APIRouter(prefix="/api/voice/old")

        @r.get("")
        def old():
            return {}

        api.include_router(r)

    mod = types.ModuleType("fake_voice_plugin_abs")
    mod.setup = plug
    monkeypatch.setitem(sys.modules, "fake_voice_plugin_abs", mod)
    monkeypatch.setenv("DISPATCH_VOICE_EXTENSIONS", "fake_voice_plugin_abs:setup")
    setup(host_env)                                  # never raises: a broken plugin is skipped
    assert routes._LOADED == []
    assert [p for _, p, _ in host_env.routers] == ["/api/voice"]
    assert "failed to load" in caplog.text


def test_status_payload_has_the_v3_fields(voice_app):
    c, ctx = voice_app
    st = c.get("/api/voice/status").json()
    for k in ("enabled", "switch", "forced", "ready", "missing", "default_voice", "builtin_voices",
              "profiles", "speed", "voice_turn_thinking", "warm", "stock_voice", "plugins", "extensions"):
        assert k in st, k
    assert st["langs"] == ["en"]
    assert st["sessions"] == {"open": 0, "max": 3}
    assert st["power_hold"] in ("off", "held", "unavailable")
    assert st["extension"] == {"name": EXTENSION["name"], "version": __version__}
    assert st["assets_version_at_setup"] == ctx.static["assets_version"] and "assets_version" not in st
    assert st["max_sessions"] == 3 and st["rt"] == rt_status_out()
    assert st["voices_ui"] is None and "tools" not in st
    ctx.operator = False
    assert c.get("/api/voice/status").status_code == 403


def test_standalone_router_keeps_the_ui_routes_only_when_asked(tmp_path):
    from dispatch_voice.profiles import ProfileStore
    from dispatch_voice.router import build_router
    store = ProfileStore(tmp_path / "v", None)
    paths = lambda r: {getattr(x, "path", "") for x in r.routes}
    assert {"/ui", "/ui/voices.js"} <= paths(build_router(store, allow_open=True))
    assert not ({"/ui", "/ui/voices.js"} & paths(build_router(store, allow_open=True, ui=False)))


def test_shutdown_hook_releases_the_power_hold_and_the_models(monkeypatch):
    from dispatch_voice.host import power
    calls = []
    monkeypatch.setattr(power, "release_all", lambda: calls.append("power"))
    monkeypatch.setattr(routes, "release_models", lambda **kw: calls.append(("models", kw)))
    routes.shutdown()
    assert calls == ["power", ("models", {"why": "shutdown"})]   # the journal line names the reason


def test_setup_with_a_second_context_replaces_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    a, b = FakeContext(tmp_path / "a"), FakeContext(tmp_path / "b")
    try:
        setup(a)
        setup(b)
        assert context.get_context() is b
        assert context.paths().voice_yaml == b.paths.voice_yaml
    finally:
        context.clear_context()


def test_tool_specs_follow_the_contract_shape():
    from dispatch_voice.host.tools import tool_specs
    rows = tool_specs()
    assert [r.id for r in rows] == ["voice-cloning", "voice-builtin", "voice-lab"]
    assert [r.feature for r in rows] == ["voice_cloning", "voice_builtin", "voice_lab"]
    for r in rows:
        assert isinstance(ToolSpec(r.id, r.feature, r.title, r.icon, r.opt_in, r.probe), ToolSpec)
        assert r.title and r.icon and callable(r.probe)


def test_the_client_entry_reports_the_package_version():
    """The client's `EXT.version` (static/index.js) is the package's version (T7b note: it
    still said 0.1.0 at 1.0.0); a release bumps both, this keeps them together."""
    import re
    src = (STATIC_DIR / "index.js").read_text(encoding="utf-8")
    m = re.search(r"export const EXT = \{[^}]*version: '([0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?)'", src)
    assert m is not None and m.group(1) == __version__
