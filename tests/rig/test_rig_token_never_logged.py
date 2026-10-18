"""The rig token must not appear in any log line: not on a good connection, not on a refused one
(401), not when the rig is unreachable, not in the repr of the objects that hold it."""
from __future__ import annotations

import logging

from dispatch_voice.host import rig

from .conftest import make_conn, proven, wait_for

SECRET = "tok-DO-NOT-LOG-0123456789"  # scrub-ok: test fixture


def client_log(caplog) -> str:
    """Everything logged by the add-on and its websocket CLIENT (the fake rig's own server side
    runs in this process and legitimately sees the header; the real one is another machine)."""
    return chr(10).join(r.getMessage() for r in caplog.records if r.name != "websockets.server")


async def test_a_good_connection_logs_no_token(fake_rig, caplog):
    caplog.set_level(logging.DEBUG)
    conn = make_conn(fake_rig, token=fake_rig.token)
    conn.start()
    assert await proven(conn)
    await conn.close()
    assert fake_rig.token not in client_log(caplog)
    assert "Authorization: Bearer ***" in client_log(caplog)         # websockets DEBUG, redacted
    assert fake_rig.token not in repr(conn.summary()) and fake_rig.token not in repr(rig.status.__doc__)


async def test_a_refused_token_logs_the_reason_not_the_token(fake_rig, caplog):
    caplog.set_level(logging.DEBUG)
    fake_rig.refuse = (401, {})
    conn = make_conn(fake_rig, token=SECRET)
    conn.start()
    assert await wait_for(lambda: conn.state in ("down", "cooling", "misconfigured") or conn.reason == "token")
    await conn.close()
    assert SECRET not in client_log(caplog) and fake_rig.token not in client_log(caplog)
    assert "401" in client_log(caplog) or "refused the token" in client_log(caplog)


async def test_an_unreachable_rig_logs_no_token(caplog):
    caplog.set_level(logging.DEBUG)
    rs = rig.RigSettings(enabled=True, url="ws://127.0.0.1:1/v1/realtime")      # nothing listens on port 1
    conn = rig.RigConnection(rs, SECRET, version="test")
    conn.start()
    assert await wait_for(lambda: conn.state != "connecting" or conn.reason is not None, timeout=5)
    await conn.close()
    assert SECRET not in client_log(caplog)


def test_the_redacting_filter_rewrites_the_header_and_nothing_else():
    rec = logging.LogRecord("websockets.client", logging.DEBUG, "x", 1, "> Authorization: Bearer %s", (SECRET,), None)
    assert rig._RedactTokens().filter(rec) and rec.getMessage() == "> Authorization: Bearer ***"
    rec = logging.LogRecord("websockets.client", logging.DEBUG, "x", 1, "> Host: %s", ("10.0.0.1",), None)  # scrub-ok: log fixture
    rig._RedactTokens().filter(rec)
    assert rec.getMessage() == "> Host: 10.0.0.1"  # scrub-ok: log fixture


def test_the_token_objects_never_show_the_value():
    t = rig.Token(SECRET, "env")
    assert SECRET not in repr(t) and SECRET not in str(t)
    s = rig.RigSettings(enabled=True, url="ws://127.0.0.1:1/x")
    conn = rig.RigConnection(s, SECRET, version="test")
    assert SECRET not in repr(conn.summary())


def test_the_fingerprint_that_latches_a_401_is_a_hash():
    fp = rig._fingerprint("ws://127.0.0.1:1/x", SECRET)
    assert SECRET not in fp and len(fp) == 16
