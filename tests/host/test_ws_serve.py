"""``ws.serve(ws, sock)`` on the new seam: the add-on's two refusals (the switch,
the models) are accept-then-1013 through ``sock.close_unavailable``; a live
session goes through the FakeContext (submit_turn / subscribe_frames /
abort / retract / replies_after), persists the user row with
``metadata.voice``, speaks the scripted reply, and cleans up after itself.
DisPatch's gate (1008 pre-accept, slots, busy) is DisPatch's to test."""
from __future__ import annotations

import asyncio

import pytest

from dispatch_voice.host import power, routes, vad_turn
from dispatch_voice.host import ws as ws_mod
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.host.engines import chatterbox

THREAD = "t-voice-1"


async def settle(n=30):
    for _ in range(n):
        await asyncio.sleep(0.005)


async def wait_json(ws, t, timeout=5.0):
    """The frames of type ``t`` once at least one arrived (polls; a fixed settle()
    raced the event loop under load and made the reconnect test order-dependent)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not ws.json_of(t):
        if loop.time() > end:
            raise AssertionError(f"no {t!r} frame within {timeout} s")
        await asyncio.sleep(0.005)
    return ws.json_of(t)


async def wait_until(cond, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met within %s s" % timeout)
        await asyncio.sleep(0.005)


@pytest.fixture
def served(host_env, monkeypatch, fake_engines, no_models):
    """setup(ctx) with voice forced on, fake engines, a one-turn detector and a
    recording power hold. Returns (ctx, holds)."""
    from dispatch_voice.host import setup
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setenv("DISPATCH_VOICE_MERGE_WINDOW", "0")

    async def fake_warm(settings):
        return fake_engines.stt, fake_engines.tts

    monkeypatch.setattr(routes, "warm_models", fake_warm)
    monkeypatch.setattr(vad_turn, "build_detector", lambda s: fake_engines.OneTurnDetector())
    monkeypatch.setattr(routes, "voice_for_bot", lambda s, e, b: None)
    holds: list[str] = []
    monkeypatch.setattr(power, "acquire", lambda settings: holds.append("acquire") or "held")
    monkeypatch.setattr(power, "release", lambda: holds.append("release") or "off")
    setup(host_env)
    return host_env, holds


async def test_switch_off_is_accept_then_1013_naming_the_switch(host_env, monkeypatch, fake_engines, no_models):
    from dispatch_voice.host import setup
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    setup(host_env)                                  # DISPATCH_VOICE unset, voice.yaml absent: off
    ws = fake_engines.FakeWS()
    await host_env.connect("voice", THREAD, ws)
    assert ws.accepted and ws.closed_with == 1013 and ws.close_reason == routes.OFF_SETTINGS
    assert not host_env.calls_to("subscribe_frames") and ws_mod.open_count() == 0


async def test_forced_off_names_the_server(host_env, monkeypatch, fake_engines, no_models):
    from dispatch_voice.host import setup
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    monkeypatch.setenv("DISPATCH_VOICE", "0")
    setup(host_env)
    ws = fake_engines.FakeWS()
    await host_env.connect("voice", THREAD, ws)
    assert ws.accepted and ws.closed_with == 1013 and "on this server" in ws.close_reason


async def test_models_missing_is_1013(host_env, monkeypatch, fake_engines):
    from dispatch_voice.host import setup
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    monkeypatch.setenv("DISPATCH_VOICE", "1")
    monkeypatch.setattr(VoiceSettings, "missing", lambda self: ["model silero_vad.onnx"])
    setup(host_env)
    ws = fake_engines.FakeWS()
    await host_env.connect("voice", THREAD, ws)
    assert ws.accepted and ws.closed_with == 1013 and ws.close_reason == routes.MODELS_MISSING


async def test_spoken_turn_lands_in_the_thread_and_the_reply_is_spoken(served, fake_engines):
    ctx, holds = served
    ctx.reply_for = lambda text: "Two meetings today. Want details?"
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    await settle(10)
    assert ws.accepted and ws.json_of("ready") and ws_mod.open_count() == 1
    ws.push_json({"type": "start"})
    ws.push_audio()
    await settle(40)
    await ctx.drain()
    await settle(40)
    # The host contract: text, metadata.voice, both instruction placements, the thinking level.
    ((tid, text), kw), = ctx.calls_to("submit_turn")
    assert (tid, text) == (THREAD, "what's the weather")
    assert kw["metadata"] == {"voice": True} and kw["thinking"] == "off"
    # sage is the advisor: the instructions go in the system slot only (T5's wording).
    assert kw["instructions"].system.startswith("## Spoken replies") and kw["instructions"].trailer == ""
    assert "podcast host" not in kw["instructions"].system
    assert ws.json_of("ready")[0]["say"] is True and len(ws.json_of("ready")[0]["sid"]) == 16
    # `say` precedes the first audio of every reply sentence (C.5 #1).
    says = ws.json_of("say")
    assert says and says[0]["kind"] == "reply" and says[0]["text"].startswith("Two meetings today")
    first_audio = next(i for i, m in enumerate(ws.sent) if isinstance(m, dict) and m.get("type") == "audio")
    first_say = next(i for i, m in enumerate(ws.sent) if isinstance(m, dict) and m.get("type") == "say")
    assert first_say < first_audio and ws.json_of("say_end")
    rows = ctx.db().messages(THREAD)
    assert rows[0].role == "user" and rows[0].metadata == {"voice": True}
    assert rows[1].role == "assistant" and rows[1].content.startswith("Two meetings")
    # Frames came through subscribe_frames under the session token, and were spoken.
    ((sub_tid,), sub_kw), = ctx.calls_to("subscribe_frames")
    assert sub_tid == THREAD and sub_kw["token"] == "tok-unlocked"
    assert ws.json_of("transcript")[0]["text"] == "what's the weather"
    reply_hdrs = [m for m in ws.json_of("audio") if m["kind"] == "reply"]
    assert reply_hdrs and reply_hdrs[0]["text"].startswith("Two meetings today")
    assert holds == ["acquire"]
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert ws.closed_with == 1000 and ws_mod.open_count() == 0
    assert holds == ["acquire", "release"]
    assert ctx._subs.get(THREAD) == []                # unsubscribed


async def test_abort_retract_and_replies_after_reach_the_context(served, fake_engines):
    ctx, _ = served
    ctx.add_thread("t-2", "nova")                                # a gateway bot: the trailer
    sock = ctx.thread_socket("t-2")
    host = ws_mod.build_host(ctx, sock, bot_id="nova")
    mid = await host.submit("t-2", "hello", {"reply_style": "short", "follow_up": False})
    assert mid and ctx.db().get_message(mid).metadata == {"voice": True}
    (_, kw), = ctx.calls_to("submit_turn")
    assert kw["instructions"].system == "" and kw["instructions"].trailer.startswith("[voice chat]")
    assert "No closing question" in kw["instructions"].trailer
    assert await host.abort("t-2") is None                       # keep_partial defaults to False
    assert ctx.calls_to("abort_turn")[-1][1] == {"keep_partial": False}
    assert await host.note_heard("t-2", "nope", heard_upto=1, heard_text="a", unheard_text="b") is False
    assert await host.replies_after("t-2", mid) == []
    await host.retract("t-2", mid)
    assert ctx.db().get_message(mid) is None
    assert ctx.calls_to("retract_message")[-1][1] == {"only_metadata_key": "voice"}
    assert host.session_live() is True
    host.touch()
    assert ctx.touched == ["tok-unlocked"]


async def test_lapsed_session_from_the_socket_locks_the_session(served, fake_engines, monkeypatch):
    from dispatch_voice.host import session as vs
    monkeypatch.setattr(vs, "LIVE_CHECK_S", 0.0)
    ctx, _ = served
    live = {"ok": True}
    ws = fake_engines.FakeWS()
    sock = ctx.thread_socket(THREAD, ws=ws, live=lambda: live["ok"])
    task = asyncio.create_task(ws_mod.serve(ws, sock))
    await settle(10)
    assert ws.json_of("ready")
    live["ok"] = False
    ws.push_json({"type": "ping"})
    await asyncio.wait_for(task, 5)
    assert ws.json_of("locked") and ws.closed_with == 1008


async def test_switching_off_ends_the_open_session_with_1013(served, fake_engines, monkeypatch):
    ctx, _ = served
    monkeypatch.setattr(routes, "release_models", lambda: None)
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    await settle(10)
    assert ws_mod.open_count() == 1
    await routes.apply_switch(True, VoiceSettings(switch=False))
    await asyncio.wait_for(task, 5)
    assert ws.closed_with == 1013 and ws.close_reason == routes.OFF_SETTINGS
    assert ws_mod.open_count() == 0


async def test_model_load_failure_closes_1013_after_accept(served, fake_engines, monkeypatch):
    ctx, holds = served

    async def boom(settings):
        raise RuntimeError("onnx gone")

    monkeypatch.setattr(routes, "warm_models", boom)
    ws = fake_engines.FakeWS()
    await ctx.connect("voice", THREAD, ws)
    assert ws.accepted and ws.closed_with == 1013 and "failed to load" in ws.close_reason
    assert holds == [] and not ctx.calls_to("subscribe_frames")


async def test_serve_without_a_context_raises(fake_engines, monkeypatch):
    from dispatch_voice.host import context
    context.clear_context()
    ws = fake_engines.FakeWS()
    with pytest.raises(RuntimeError, match="context"):
        await ws_mod.serve(ws, object())


async def test_voice_profile_lookup_failure_falls_back_to_the_default_voice(served, fake_engines, monkeypatch):
    ctx, _ = served

    def boom(s, e, b):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(routes, "voice_for_bot", boom)
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    await settle(10)
    assert ws.json_of("ready")[0]["voice_acks"] == 0
    ws.hang_up()
    await asyncio.wait_for(task, 5)


# --------------------------------------------------------------------------- #
# Review fixes (review-T3s1 findings 1, 2, 3, 8)
# --------------------------------------------------------------------------- #

async def test_ready_does_not_wait_for_the_power_hold(served, fake_engines, monkeypatch):
    """Finding 1: the hold only makes the reply faster, so a slow (or wedged)
    power daemon must not delay `ready`. The acquire runs beside the session;
    the clean-up still waits for it before it releases, so the two never cross."""
    import threading
    import time
    ctx, holds = served
    gate = threading.Event()

    def slow_acquire(settings):
        gate.wait(5)                              # the daemon answers only when the test lets it
        holds.append("acquire")
        return "held"

    monkeypatch.setattr(power, "acquire", slow_acquire)
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    await settle(10)
    assert ws.json_of("ready") and holds == []      # ready went out while the acquire was pending
    ws.hang_up()                                    # hang up BEFORE the hold exists
    await settle(10)
    assert holds == [] and not task.done()          # the release waits for the acquire
    t0 = time.monotonic()
    gate.set()
    await asyncio.wait_for(task, 5)
    assert holds == ["acquire", "release"]          # exactly once each, in order
    assert time.monotonic() - t0 < 2.0
    assert ws.closed_with == 1000 and ws_mod.open_count() == 0 and ctx._subs.get(THREAD) == []


async def test_a_raising_power_acquire_leaks_nothing(served, fake_engines, monkeypatch, caplog):
    """Finding 2: an exploding holder must not take the subscription or the
    session entry with it; the chat still runs and is cleaned up."""
    ctx, holds = served

    def boom(settings):
        holds.append("acquire")
        raise RuntimeError("holder exploded")

    monkeypatch.setattr(power, "acquire", boom)
    ws = fake_engines.FakeWS()
    task = asyncio.create_task(ctx.connect("voice", THREAD, ws))
    await settle(10)
    assert ws.json_of("ready") and ws_mod.open_count() == 1
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert holds == ["acquire", "release"]          # PowerHold counted the session: give it back
    assert ws.closed_with == 1000 and ws_mod.open_count() == 0 and ctx._subs.get(THREAD) == []
    assert "power hold acquire failed" in caplog.text


async def test_a_failing_subscribe_runs_the_cleanup(served, fake_engines, monkeypatch):
    """The whole wiring sits inside the try: a context that refuses the frame
    subscription leaves no hold and no session entry behind."""
    ctx, holds = served

    async def boom(*a, **k):
        raise RuntimeError("frames refused")

    monkeypatch.setattr(ctx, "subscribe_frames", boom)
    ws = fake_engines.FakeWS()
    await asyncio.wait_for(ctx.connect("voice", THREAD, ws), 5)
    assert holds == [] and ws_mod.open_count() == 0 and ws.closed_with == 1000


async def test_an_evicted_socket_ends_its_session_at_once(served, fake_engines):
    """Finding 3 (amendment T1-1): when DisPatch's slot evicts this socket for
    a newer one of the same session, the add-on ends the session itself instead
    of waiting for a half-open phone's close handshake."""
    ctx, holds = served
    ws = fake_engines.FakeWS()
    sock = ctx.thread_socket(THREAD, ws=ws)
    task = asyncio.create_task(ws_mod.serve(ws, sock))
    await settle(10)
    assert ws.json_of("ready") and len(sock.replaced_callbacks) == 1
    await sock.evict()                              # callbacks, then DisPatch's own close(4001)
    await asyncio.wait_for(task, 5)                 # the receive loop never had to wake
    assert ws.closed_with == 4001 and ws.close_reason == "replaced"
    assert holds == ["acquire", "release"] and ws_mod.open_count() == 0 and ctx._subs.get(THREAD) == []


async def test_a_socket_without_on_replaced_still_works(served, fake_engines):
    ctx, _ = served
    ws = fake_engines.FakeWS()
    sock = ctx.thread_socket(THREAD, ws=ws)
    sock.on_replaced = None                         # a host that predates T1-1 (nothing callable)
    task = asyncio.create_task(ws_mod.serve(ws, sock))
    await settle(10)
    assert ws.json_of("ready")
    ws.hang_up()
    await asyncio.wait_for(task, 5)
    assert ws.closed_with == 1000


async def test_detector_load_failure_closes_1013_after_accept(served, fake_engines, monkeypatch):
    """Finding 8: a corrupt VAD/Smart Turn model is the add-on's refusal
    (1013 + reason), not a bare 1011 from DisPatch's handler guard."""
    ctx, holds = served

    def boom(settings):
        raise RuntimeError("silero_vad.onnx corrupt")

    monkeypatch.setattr(vad_turn, "build_detector", boom)
    ws = fake_engines.FakeWS()
    await ctx.connect("voice", THREAD, ws)
    assert ws.accepted and ws.closed_with == 1013 and ws.close_reason == "voice models failed to load"
    assert holds == [] and not ctx.calls_to("subscribe_frames") and ws_mod.open_count() == 0


# --------------------------------------------------------------------------- #
# review-T3s4 #4: a reconnect replaces the older session of the same voice chat
# --------------------------------------------------------------------------- #

V3_QUERY = {"v": "3", "d": "dev-abcdef12", "s": "str-0123456789ab", "up": "pcm16", "down": "pcm16"}


async def test_a_reconnect_of_the_same_voice_chat_replaces_the_older_session(served, fake_engines):
    """A phone that changed networks reconnects with the SAME device and stream ids while
    its old socket is half-open (it never closed). The old session ends at once (4001
    "replaced", as DisPatch's slot eviction does) instead of synthesising the reply into a
    dead socket for up to 30 s; one voice chat holds one session slot. A second tab — a
    different stream id — is a different voice chat and is left alone."""
    ctx, _holds = served
    assert not ws_mod._BY_STREAM and ws_mod.open_count() == 0    # the fixture's clean slate
    ws_a = fake_engines.FakeWS(query=dict(V3_QUERY))
    task_a = asyncio.create_task(ctx.connect("voice", THREAD, ws_a))
    assert (await wait_json(ws_a, "ready"))[0]["proto"] == 3
    await wait_until(lambda: ws_mod.open_count() == 1)
    ws_b = fake_engines.FakeWS(query=dict(V3_QUERY))
    task_b = asyncio.create_task(ctx.connect("voice", THREAD, ws_b))
    await asyncio.wait_for(task_a, 5)                 # A ended without its socket ever closing
    assert ws_a.closed_with == ws_mod.REPLACED and ws_a.close_reason == "replaced"
    await wait_json(ws_b, "ready")
    assert ws_b.closed_with is None and ws_mod.open_count() == 1
    ws_c = fake_engines.FakeWS(query={**V3_QUERY, "s": "str-another-tab-1"})
    task_c = asyncio.create_task(ctx.connect("voice", THREAD, ws_c))
    await wait_json(ws_c, "ready")
    await wait_until(lambda: ws_mod.open_count() == 2)
    assert ws_b.closed_with is None
    for ws, task in ((ws_b, task_b), (ws_c, task_c)):
        ws.hang_up()
        await asyncio.wait_for(task, 5)
    assert ws_mod.open_count() == 0 and not ws_mod._BY_STREAM


async def test_a_v2_socket_or_one_without_a_stream_id_is_never_replaced(served, fake_engines):
    ctx, _holds = served
    first = fake_engines.FakeWS()                                  # v2: no stream id
    t1 = asyncio.create_task(ctx.connect("voice", THREAD, first))
    await settle(10)
    second = fake_engines.FakeWS(query={**V3_QUERY, "s": "bad"})  # malformed `s`: no key
    t2 = asyncio.create_task(ctx.connect("voice", THREAD, second))
    await settle(10)
    assert ws_mod.open_count() == 2 and first.closed_with is None and second.closed_with is None
    for ws, task in ((first, t1), (second, t2)):
        ws.hang_up()
        await asyncio.wait_for(task, 5)
