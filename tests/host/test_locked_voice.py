"""Per-bot "Voice on locked devices" (add-on 1.3.4): a locked (Safe-Mode) device
hears the bot's ``locked_voice`` when one is set; everything else hears the
bot's standard voice. The choice is the SERVER's, from the socket's locked flag."""
from __future__ import annotations

import asyncio

import pytest

from fakes import BotView

from dispatch_voice.host import config as vconfig
from dispatch_voice.host import registry, routes
from dispatch_voice.host import ws as ws_mod

from .test_routes import BUILTINS, app_voice  # noqa: F401  (fixture)
from .test_ws_serve import THREAD, served, settle  # noqa: F401  (fixture)

STD, LOCKED = "builtin:af_heart", "builtin:am_michael"


@pytest.fixture
def two_voices(app_voice, monkeypatch):  # noqa: F811
    c, ctx, eng = app_voice
    ctx.bots._bots["sage"] = BotView("sage", "Sage", voice=STD, backend="advisor")
    monkeypatch.setattr(registry, "_builtin_list", lambda settings: BUILTINS)
    return c, ctx, eng


def _set(c, **conf):
    r = c.patch("/api/voice/settings", json={"bots": {"sage": conf}})
    assert r.status_code == 200, r.text
    return r.json()


def test_unlocked_uses_standard_even_with_an_override_set(two_voices):
    c, ctx, _ = two_voices
    _set(c, locked_voice=LOCKED)
    s = vconfig.load_settings()
    assert routes.voice_for_bot(s, None, "sage") == STD            # unlocked: standard


def test_locked_uses_the_override(two_voices):
    c, ctx, _ = two_voices
    _set(c, locked_voice=LOCKED)
    s = vconfig.load_settings()
    assert routes.voice_for_bot(s, None, "sage", locked=True) == LOCKED


def test_empty_override_is_the_standard_voice(two_voices):
    c, ctx, _ = two_voices
    s = vconfig.load_settings()
    assert routes.voice_for_bot(s, None, "sage", locked=True) == STD     # never set
    _set(c, locked_voice=LOCKED)
    body = _set(c, locked_voice="")                                       # cleared
    assert "locked_voice" not in body["bots"]["sage"]
    assert vconfig.load_settings().locked_voice("sage") == ""
    assert routes.voice_for_bot(vconfig.load_settings(), None, "sage", locked=True) == STD


def test_an_override_naming_a_vanished_voice_falls_back_to_standard(two_voices, monkeypatch):
    c, ctx, _ = two_voices
    _set(c, locked_voice=LOCKED)
    monkeypatch.setattr(registry, "_builtin_list", lambda settings: [BUILTINS[0]])
    assert routes.voice_for_bot(vconfig.load_settings(), None, "sage", locked=True) == STD


def test_override_is_validated_and_other_bots_untouched(two_voices):
    c, ctx, _ = two_voices
    assert c.patch("/api/voice/settings", json={"bots": {"sage": {"locked_voice": "builtin:nope"}}}).status_code == 422
    assert c.patch("/api/voice/settings", json={"bots": {"sage": {"locked_voice": 5}}}).status_code == 422
    body = _set(c, locked_voice=LOCKED)
    assert body["bots"]["sage"]["locked_voice"] == LOCKED
    assert "locked_voice" not in body["bots"]["nova"]               # unset rows are unchanged
    _set(c, follow_up=True)                                          # a flag PATCH keeps it
    assert vconfig.load_settings().locked_voice("sage") == LOCKED


def test_settings_writes_and_reads_are_refused_when_locked(two_voices):
    c, ctx, _ = two_voices
    _set(c, locked_voice=LOCKED)
    ctx.operator = False
    assert c.patch("/api/voice/settings", json={"bots": {"sage": {"locked_voice": ""}}}).status_code == 403
    assert c.get("/api/voice/settings").status_code == 403
    assert c.get("/api/voice/status").status_code == 403
    ctx.operator = True
    assert vconfig.load_settings().locked_voice("sage") == LOCKED    # the refused write changed nothing


def test_the_locked_payload_never_includes_it(two_voices):
    c, ctx, _ = two_voices
    _set(c, locked_voice=LOCKED)
    assert set(routes.locked_status()) == {"enabled", "ready"}
    assert LOCKED not in repr(routes.locked_status())


async def test_serve_picks_the_voice_from_the_sockets_locked_flag(served, fake_engines, monkeypatch):
    ctx, _ = served
    seen = []

    def vfb(settings, tts, bot_id, locked=False):
        seen.append(locked)
        return LOCKED if locked else None

    monkeypatch.setattr(routes, "voice_for_bot", vfb)
    for locked in (False, True):
        ws = fake_engines.FakeWS()
        sock = ctx.thread_socket(THREAD, ws=ws, locked=locked)
        task = asyncio.create_task(ws_mod.serve(ws, sock))
        await settle(15)
        assert ws.json_of("ready")
        ws.hang_up()
        await asyncio.wait_for(task, 5)
    assert seen == [False, True]


async def test_only_a_literal_true_flag_means_locked(served, fake_engines, monkeypatch):
    ctx, _ = served
    seen = []
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b, locked=False: seen.append(locked))
    ws = fake_engines.FakeWS()
    sock = ctx.thread_socket(THREAD, ws=ws)
    sock.locked = "yes"                  # only a literal True counts
    task = asyncio.create_task(ws_mod.serve(ws, sock))
    await settle(15)
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert seen == [False]
