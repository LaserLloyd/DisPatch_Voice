"""host/acks.py: the pools bound to the bots, the shared render queue, the
per-session kit, status/rebuild/release — with a fake engine (no models)."""
from __future__ import annotations

import threading

import numpy as np

from dispatch_voice.host import acks as hacks
from dispatch_voice.host import tts as vtts
from dispatch_voice.host.config import VoiceSettings
from dispatch_voice.text.acks import RenderQueue


class Eng:
    """Renders every line as 1 s of noise; `render_line` records the seed."""
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def cache_tag(self, profile_id):
        return f"fake-{profile_id or 'default'}"

    def render_line(self, text, profile_id=None, *, seed):
        self.calls.append((text, profile_id, seed))
        return np.full(24000, 300, "<i2")

    def synth(self, text, profile_id=None):
        self.calls.append((text, profile_id, None))
        return np.full(24000, 300, "<i2"), 24000


def test_pools_bind_the_context_bots_and_hot_reload(host_env):
    p = hacks.pools()
    assert p.names == {"sage": "Sage", "swift": "Swift", "nova": "Nova", "alpha": "Alpha"}
    assert hacks.lines_for("sage", None)[0] in p.all_texts()
    # Every voice gets the notices and the tool lines on top of its pools.
    lines = hacks.lines_for("nova", "builtin:af_heart")
    assert set(hacks.NOTICE_TEXTS) <= set(lines) and set(hacks.TOOL_LINES) <= set(lines)
    # An operator file is picked up (hot reload is PoolsFile's; the path is the context's).
    host_env.paths.acks_yaml.write_text("default:\n  question: ['Right, let me check.']\n")
    hacks.reset()
    assert "Right, let me check." in hacks.pools().default["question"]


def test_kit_renders_through_the_shared_queue_and_reads_back(host_env):
    eng = Eng()
    st = VoiceSettings(flag="1")
    kit = hacks.kit_for(st, eng, bot_id="sage", profile_id="host", mode="often", name="Host")
    assert isinstance(kit.queue, RenderQueue) and kit.voice == "host" and kit.mode == "often"
    assert kit.queue.wait_idle(10)
    texts = {t for t, _, _ in eng.calls}
    assert "Sorry, that didn't go through." in texts and "Searching" in texts
    assert all(pid == "host" and isinstance(seed, int) for _, pid, seed in eng.calls)   # seeded render_line
    assert kit.has_rendered() and kit.ready_count() > 10
    assert kit.pcm("Searching").dtype == np.dtype("<i2") and kit.clip_s("Searching") == 1.0
    assert kit.pcm("never rendered") is None and kit.clip_s("never rendered") is None
    # The disk cache lives under <data>/voice-cache/<engine tag>/ with the host's key.
    d = host_env.paths.cache_dir / "fake-host"
    assert d.is_dir() and list(d.glob("*.npy"))
    # The same queue serves a second voice; status lists both, by key, with names.
    hacks.kit_for(st, eng, bot_id="nova", profile_id=None, name=None)
    assert hacks.current_queue() is kit.queue
    assert kit.queue.wait_idle(10)
    s = hacks.status()
    assert s["host"]["name"] == "Host" and s["host"]["failed"] == 0 and s["host"]["ready"] == s["host"]["total"]
    assert "default" in s and "name" not in s["default"]
    # rebuild: known voice → True (re-queued), unknown → False.
    n = len(eng.calls)
    assert hacks.rebuild("host") is True and hacks.rebuild("ghost") is False
    assert kit.queue.wait_idle(10) and len(eng.calls) > n
    hacks.release()
    assert hacks.current_queue() is None and hacks.status() == {}


def test_queue_waits_while_a_chat_is_open_or_a_reply_synthesises(host_env, monkeypatch):
    """A render cannot be interrupted once started, so none starts while a voice
    chat is open (measured: a render in flight cost a turn up to ~3 s of first
    audio) nor while a reply is being synthesised; it resumes after the hang-up."""
    from dispatch_voice.host import ws
    eng = Eng()
    ws._SESSIONS[1] = object()                                          # an open chat, idle or not
    try:
        kit = hacks.kit_for(VoiceSettings(flag="1"), eng, bot_id=None, profile_id=None)
        assert not kit.queue.wait_idle(0.6) and eng.calls == []
        ws._SESSIONS.pop(1, None)                                       # hang-up
        vtts.REPLY_ACTIVITY.enter()
        assert not kit.queue.wait_idle(0.6) and eng.calls == []        # a reply is being synthesised
        vtts.REPLY_ACTIVITY.exit()
        assert kit.queue.wait_idle(10) and eng.calls
        monkeypatch.setenv("DISPATCH_VOICE_ACK_RENDER", "0")
        assert hacks._busy() is True                                    # the operator switch
    finally:
        ws._SESSIONS.pop(1, None)
        vtts.REPLY_ACTIVITY._n = 0
        hacks.release()


def test_kit_is_fail_soft_without_a_queue():
    kit = hacks.AckKit.silent(mode="off")
    assert kit.mode == "off" and kit.pcm("x") is None and kit.clip_s("x") is None
    assert not kit.has_rendered() and kit.ready_count() == 0
    kit.set_mode("often")
    assert kit.mode == "often"
    kit.observe(ttft_s=0.4, first_chunk_s=0.8)
    assert kit.estimator.n == {"ttft": 1, "chunk": 1}


def test_eta_prior_carries_over_to_the_next_session(host_env):
    eng = Eng()
    st = VoiceSettings(flag="1")
    kit = hacks.kit_for(st, eng, bot_id=None, profile_id="host")
    assert kit.estimator.prior["chunk"] == hacks.FIRST_CHUNK_PRIOR_S["cloned"]
    kit.observe(first_chunk_s=0.5)
    kit.observe(first_chunk_s=0.5)
    nxt = hacks.kit_for(st, eng, bot_id=None, profile_id="host")
    assert abs(nxt.estimator.prior["chunk"] - 0.5) < 1e-6
    assert hacks.kit_for(st, eng, bot_id=None, profile_id="builtin:x").estimator.prior["chunk"] == \
        hacks.FIRST_CHUNK_PRIOR_S["builtin"]
    hacks.release()


def test_release_stops_the_render_thread(host_env):
    eng = Eng()
    kit = hacks.kit_for(VoiceSettings(flag="1"), eng, bot_id=None, profile_id=None)
    assert kit.queue.wait_idle(10)
    names_before = {t.name for t in threading.enumerate()}
    hacks.release()
    import time
    for _ in range(50):
        if not any("render" in n.lower() or "ack" in n.lower() for n in {t.name for t in threading.enumerate()} - names_before):
            break
        time.sleep(0.02)
    assert hacks.current_queue() is None


# ---- FR-works 4: the kit is keyed by the voice that actually speaks ----------- #

def test_kit_for_the_default_voice_on_a_builtin_box_is_the_builtin_voice(host_env, monkeypatch):
    """The live box's default voice is Kokoro af_heart: the kit's key, its word-line
    filter and its ETA prior follow that voice, not the bare "" ("Huh, okay." was
    rendered by Kokoro and heard as "Hello K.")."""
    from dispatch_voice.host import registry, routes
    from dispatch_voice.text.acks import is_word_line
    voices = [{"id": "builtin:af_heart", "name": "Heart", "lang": "en-us", "default": True}]
    registry.set_builtin_voices_provider(lambda settings=None: voices)
    monkeypatch.setattr(routes, "_dv", lambda: None)
    try:
        eng = Eng()
        st = VoiceSettings(flag="1")
        kit = hacks.kit_for(st, eng, bot_id="sage", profile_id=None, mode="often")
        assert kit.voice == "builtin:af_heart"
        assert kit.estimator.prior["chunk"] == hacks.FIRST_CHUNK_PRIOR_S["builtin"]
        lines = hacks.lines_for("sage", kit.voice)
        pool_lines = [x for x in lines if x not in hacks.NOTICE_TEXTS and x not in hacks.TOOL_LINES]
        assert pool_lines and all(is_word_line(x) for x in pool_lines)
        assert not any(x in ("Mm.", "Huh, okay.", "Mm, let me see.") for x in lines)
        assert kit.queue.wait_idle(10)
        assert {pid for _, pid, _ in eng.calls} == {"builtin:af_heart"}
    finally:
        registry.reset()


def test_prewarm_ensures_every_bot_in_its_resolved_voice(host_env, monkeypatch):
    """ensure_all keys every voice the way the sessions do, and queues each bot's own
    flavour lines at start-up (not only on its first chat)."""
    from dispatch_voice.host import registry, routes
    voices = [{"id": "builtin:af_heart", "name": "Heart", "lang": "en-us", "default": True}]
    registry.set_builtin_voices_provider(lambda settings=None: voices)
    monkeypatch.setattr(routes, "_dv", lambda: None)
    host_env.paths.acks_yaml.write_text("bots:\n  nova:\n    flavour: gruff\n")
    hacks.reset()
    try:
        eng = Eng()
        hacks.ensure_all(VoiceSettings(flag="1"), eng, host_env.bots.all(), default_profile=None)
        st = hacks.status()
        assert set(st) == {"builtin:af_heart"}, st
        rq = hacks.current_queue()
        assert rq.wait_idle(10)
        from dispatch_voice.text import ack_lines
        gruff = {x for lines in ack_lines.FLAVOUR["gruff"].values() for x in lines}
        rendered = {t for t, _, _ in eng.calls}
        assert rendered & gruff, "nova's flavour lines are rendered at prewarm"
    finally:
        registry.reset()
