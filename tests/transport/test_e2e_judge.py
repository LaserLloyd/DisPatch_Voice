"""The harness's DisPatch judge (review-T6 M1): login + cookie, the 1008 gate, the messages
API, and the judging rules — proven against the reference server's DisPatch-shaped mode."""
import asyncio
import json

import e2e_lowbw as E
import pytest

from dispatch_voice.transport import opus


# ------------------------------------------------------------------ rules #
def test_full_sentence_rule_catches_a_lost_head_and_tolerates_one_slip():
    ref = "What's on my calendar for this afternoon?"
    assert E.full_sentence_text("What's on my calendar for this afternoon?", ref)
    assert E.full_sentence_text("what's on my calender for this afternoon", ref)       # 1/7 = 0.14
    assert not E.full_sentence_text("my calendar for this afternoon?", ref)             # the head is gone
    assert not E.full_sentence_text("", ref)
    assert E.wer("a b c", "a b c") == 0.0 and E.wer("", "a b") == 1.0


def test_spoken_words_follow_the_session_normaliser():
    w = E.spoken_words("**The dentist** costs $5. It is at 3 pm.")
    assert w == ["the", "dentist", "costs", "five", "dollars", "it", "is", "at", "three", "p", "m"]


# ---------------------------------------------------- against the ref server #
@pytest.fixture
def ref_dispatch(tmp_path, port_pair):
    server_port, proxy_port = port_pair
    pin_file = tmp_path / "pin"
    pin = E.write_pin(pin_file)
    assert oct(pin_file.stat().st_mode & 0o777) == "0o600"
    clip = E.load_clip()
    proc = E.start_refserver(server_port, tmp_path / "state", "short", "oracle", tmp_path / "refserver.log",
                             ("--dispatch-api", "--pin-file", str(pin_file), "--oracle-text", E.known_text(),
                              "--oracle-voiced", str(round(E.voiced_seconds(clip), 3))))
    try:
        yield server_port, proxy_port, pin, clip
    finally:
        proc.terminate()
        proc.wait(10)


async def test_gate_cookie_and_messages_api(ref_dispatch):
    import websockets
    from websockets.exceptions import InvalidStatus
    server_port, _proxy, pin, _clip = ref_dispatch
    base = f"http://127.0.0.1:{server_port}"
    j = E.DispatchJudge(base, pin="00000000" if pin != "00000000" else "11111111", bot="refbot")
    with pytest.raises(RuntimeError, match="unlock refused"):
        await asyncio.to_thread(j.login)
    j = E.DispatchJudge(base, bot="refbot")
    with pytest.raises(RuntimeError):                       # no cookie: no threads
        await asyncio.to_thread(j.new_thread)
    j = E.DispatchJudge(base, pin=pin, bot="refbot")
    await asyncio.to_thread(j.login)
    assert j.cookie and j.ws_headers() == {"Cookie": f"lc_session={j.cookie}"}
    tid = await asyncio.to_thread(j.new_thread)
    assert await asyncio.to_thread(j.messages, tid) == []
    url = f"ws://127.0.0.1:{server_port}/ws/voice/{tid}?v=3&d=device-0001&s=stream-0001"
    # no cookie, or a foreign Origin: refused before accept (the handshake fails)
    for headers in (None, {**j.ws_headers(), "Origin": "http://evil.example"}):
        with pytest.raises(InvalidStatus):
            async with websockets.connect(url, additional_headers=headers, proxy=None, open_timeout=5):
                pass
    async with websockets.connect(url, additional_headers=j.ws_headers(), proxy=None, open_timeout=5) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert ready["type"] == "ready" and ready["proto"] == 3


@pytest.mark.skipif(not opus.available(), reason="libopus not installed")
async def test_baseline_end_to_end_with_the_dispatch_judge(ref_dispatch):
    server_port, proxy_port, pin, clip = ref_dispatch
    judge = E.DispatchJudge(f"http://127.0.0.1:{server_port}", pin=pin, bot="refbot")
    r = await E.run_one("baseline", up="opus", down="opus", compression="deflate", server_port=server_port,
                        proxy_port=proxy_port, thread=None, server_http=None, clip=clip, timeout_s=60,
                        judge=judge)
    assert r["judge"] == "dispatch"
    assert all(r["assertions"].values()), (r["assertions"], E.summary_line(r))
    assert r["rows"][0]["text"] == E.known_text()
    assert r["reply_words"]["expected"] == r["reply_words"]["heard"] > 10


def test_check_fails_a_doubled_or_missing_sentence():
    base = {"profile": "baseline", "judge": "dispatch", "rows": [{"text": E.known_text()}],
            "known_text": E.known_text(), "transcripts": [{}], "heard_twice": [], "timed_out": False,
            "drops_mid_reply": 0, "link_sent": [], "latency_s": {}, "reconnects": 0}
    ok = E.check({**base, "heard_all": True, "heard_in_order": True})
    assert all(ok.values())
    bad = E.check({**base, "heard_all": False, "heard_in_order": False})
    assert not bad["reply_complete_once_in_order"]
    two_rows = E.check({**base, "rows": base["rows"] * 2, "heard_all": True, "heard_in_order": True})
    assert not two_rows["one_user_row"]
    stall = E.check({**base, "profile": "stall-5s", "heard_all": True, "heard_in_order": True,
                     "drops_mid_reply": 1, "reply_sids": 1})
    assert not stall["rejoined_mid_reply"]


def test_a_stall_run_whose_reply_arrived_before_the_stall_is_not_exercised_not_failed():
    """review-T3s4 #6: the whole reply reached the phone before the stall (no drop mid-reply):
    the rejoin was not exercised — reported apart, never a pass and never a failure."""
    base = {"profile": "stall-5s", "judge": "dispatch", "rows": [{"text": E.known_text()}],
            "known_text": E.known_text(), "transcripts": [{}], "heard_twice": [], "timed_out": False,
            "link_sent": [], "latency_s": {}, "reconnects": 1, "heard_all": True, "heard_in_order": True}
    r = {**base, "drops_mid_reply": 0, "reply_sids": 1}
    a = E.check(r)
    assert "rejoined_mid_reply" not in a and all(a.values())
    assert r["not_exercised"] == ["rejoined_mid_reply"]
    r2 = {**base, "drops_mid_reply": 1, "reply_sids": 2}
    assert E.check(r2)["rejoined_mid_reply"] and "not_exercised" not in r2


# ------------------------------------------------- scratch server settling (V1) #
def test_settle_waits_for_the_prewarm_and_every_ack_render(monkeypatch):
    """A matrix's first run must not overlap the fresh server's prewarm and ack renders
    (V1: pongs ~2 s late -> text mode -> "not exercised")."""
    answers = iter([
        {"warm": False, "acks": {}},
        {"warm": True, "acks": {"default": {"ready": 10, "total": 62, "failed": 0}}},
        {"warm": True, "acks": {"default": {"ready": 61, "total": 62, "failed": 1}, "v2": {"ready": 3, "total": 4, "failed": 0}}},
        {"warm": True, "acks": {"default": {"ready": 61, "total": 62, "failed": 1}, "v2": {"ready": 4, "total": 4, "failed": 0}}},
    ])
    seen = []

    def fake_req(self, method, path, body=None):
        assert self.cookie == "c0", "the status read is the unlocked browser's"
        st = next(answers)
        seen.append(st)
        return 200, {}, st

    def fake_login(self):
        self.cookie = "c0"

    monkeypatch.setattr(E.DispatchJudge, "_req", fake_req)
    monkeypatch.setattr(E.DispatchJudge, "login", fake_login)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    st = E.settle_scratch("http://127.0.0.1:1", "0000", timeout_s=60)
    assert len(seen) == 4 and st["acks"]["v2"]["ready"] == 4


def test_settle_gives_up_at_its_deadline(monkeypatch):
    monkeypatch.setattr(E.DispatchJudge, "_req", lambda self, m, p, b=None: (200, {}, {"warm": False}))
    monkeypatch.setattr(E.DispatchJudge, "login", lambda self: None)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(E.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    assert E.settle_scratch("http://127.0.0.1:1", "0000", timeout_s=30) == {"warm": False}
