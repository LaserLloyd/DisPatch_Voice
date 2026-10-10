"""LinkState (applied at sentence boundaries), probes, and the Tailscale relay indicator."""
import json
import zlib

import pytest

from dispatch_voice.transport.link import (
    LinkState,
    ProbeLimiter,
    RelayIndicator,
    _parse_ping,
    is_tailnet,
    peer_address,
    probe_payload,
)


# ----------------------------------------------------------------- LinkState #
def test_request_waits_for_the_boundary():
    ls = LinkState()
    ls.request({"mode": "text", "why": "underrun"}, now=10.0)
    assert ls.mode == "audio" and not ls.text_mode and ls.pending       # current sentence finishes as it is
    assert ls.at_boundary(now=11.0) == {"mode": "text"}
    assert ls.text_mode and ls.why == "underrun" and ls.degraded_since == 10.0
    assert ls.at_boundary(now=12.0) == {}                                # nothing pending
    ls.request({"mode": "audio", "why": "recovered"}, now=50.0)
    assert ls.at_boundary(now=51.0) == {"mode": "audio"}
    assert ls.degraded_since is None and ls.changes == 2


def test_rungs_and_validation():
    ls = LinkState()
    ls.request({"mode": "audio", "down_kbps": 16, "up_kbps": 12, "why": "goodput"})
    assert ls.up_kbps == 12                    # the client's own encoder: recorded at once
    assert ls.down_kbps == 24
    assert ls.at_boundary() == {"down_kbps": 16}
    assert ls.degraded_since is not None       # a lower rung counts as degraded
    for junk in ({"mode": "fax"}, {"down_kbps": 8}, {"up_kbps": 64}, {"why": "boredom"}, {}):
        ls.request(junk)
    assert ls.at_boundary() == {} and (ls.mode, ls.down_kbps, ls.up_kbps) == ("audio", 16, 12)
    assert ls.why == "goodput"                 # an unknown `why` never overwrites a known one


def test_later_request_wins_field_by_field():
    ls = LinkState()
    ls.request({"mode": "text", "why": "rtt"})
    ls.request({"down_kbps": 16, "why": "goodput"})
    assert ls.at_boundary() == {"mode": "text", "down_kbps": 16}
    assert ls.why == "goodput"


def test_decoder_failure_means_pcm_from_the_next_sentence():
    ls = LinkState()
    ls.request({"mode": "audio", "why": "decoder"})
    assert ls.at_boundary() == {"pcm_fallback": True}
    ls2 = LinkState()
    ls2.request({"mode": "audio", "down_kbps": None, "why": "user"})
    assert ls2.at_boundary()["pcm_fallback"] is True


def test_slow_link_always_phone_forces_text():
    ls = LinkState()
    assert ls.set_slow_link("always_phone") == "always_phone" and ls.text_mode
    assert ls.set_slow_link("nonsense") == "phone" and not ls.text_mode
    assert ls.metrics(now=0.0)["mode"] == "audio"


def test_metrics_shape():
    ls = LinkState()
    ls.request({"mode": "text", "why": "stall"}, now=100.0)
    ls.at_boundary(now=100.5)
    m = ls.metrics(now=112.0)
    assert m == {"mode": "text", "down_kbps": 24, "up_kbps": 16, "why": "stall", "degraded_s": 12.0}


# --------------------------------------------------------------------- probe #
def test_probe_payload_is_capped_random_and_incompressible():
    a, b = probe_payload(16384), probe_payload(16384)
    assert len(a) == 16384 and a != b
    assert len(probe_payload(10**9)) == 32768
    assert probe_payload(-5) == b"" and probe_payload("x") == b""
    # permessage-deflate (raw deflate) cannot shrink it
    co = zlib.compressobj(wbits=-15)
    assert len(co.compress(a) + co.flush()) >= len(a)


def test_probe_limiter_one_per_20s():
    t = [0.0]
    lim = ProbeLimiter(clock=lambda: t[0])
    assert lim.allow()
    t[0] = 19.9
    assert not lim.allow()
    t[0] = 20.0
    assert lim.allow()
    assert lim.refused == 1


# --------------------------------------------------------------- peer address #
def test_peer_address_reads_forwarded_for_only_from_loopback():
    h = {"X-Forwarded-For": "6.6.6.6, 100.101.102.103"}  # scrub-ok: fictional fixture address
    assert peer_address("127.0.0.1", h) == "100.101.102.103"           # the entry tailscaled added  # scrub-ok: fictional fixture address
    assert peer_address("::1", {"x-forwarded-for": "100.64.0.9"}) == "100.64.0.9"  # scrub-ok: fictional fixture address
    assert peer_address("10.0.0.20", h) == "10.0.0.20"           # a LAN client cannot spoof it  # scrub-ok: fictional fixture address
    assert peer_address("127.0.0.1", {}) == "127.0.0.1"
    assert peer_address("127.0.0.1", {"X-Forwarded-For": "garbage"}) == "127.0.0.1"
    assert peer_address(None, h) is None


def test_is_tailnet():
    assert is_tailnet("100.101.102.104") and is_tailnet("fd7a:115c:a1e0::1")  # scrub-ok: fictional fixture address
    assert not is_tailnet("10.0.0.4") and not is_tailnet("100.128.0.1") and not is_tailnet("x")  # scrub-ok: fictional fixture address


# ------------------------------------------------------------ relay indicator #
PHONE = "100.70.1.2"  # scrub-ok: fictional fixture address


def status(**peer):
    base = {"HostName": "phone", "TailscaleIPs": [PHONE, "fd7a:115c:a1e0::2"], "Relay": "tok",
            "CurAddr": "", "PeerRelay": "", "Active": False, "RxBytes": 0, "TxBytes": 0}
    base.update(peer)
    return {"Self": {"TailscaleIPs": ["100.70.0.1"]}, "Peer": {"nodekey:abc": base}}  # scrub-ok: fictional CGNAT fixture address


class Runner:
    def __init__(self, status_obj, ping_out="", rc=0):
        self.status_obj = status_obj
        self.ping_out = ping_out
        self.rc = rc
        self.argv = []

    async def __call__(self, argv, timeout):
        self.argv.append(argv)
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)   # fixed argv, no shell
        if argv[1] == "status":
            return 0, json.dumps(self.status_obj)
        return self.rc, self.ping_out


def indicator(runner, t=None):
    clock = t or [0.0]
    return RelayIndicator(tailscale="/usr/bin/tailscale", runner=runner, clock=lambda: clock[0]), clock


async def test_idle_peer_never_reads_as_relay():
    r = Runner(status())                                  # no traffic: CurAddr empty, Relay always set
    ind, _ = indicator(r)
    assert await ind.lookup(PHONE) == {"path": "idle"}
    assert RelayIndicator.relayed({"path": "idle"}) is None
    assert all(a[1] == "status" for a in r.argv)          # no ping without probe=True


async def test_traffic_decides_the_path():
    for peer, want in [({"Active": True, "CurAddr": "203.0.113.5:41641"}, {"path": "direct"}),
                       ({"Active": True}, {"path": "relay", "region": "tok"}),
                       ({"RxBytes": 1000, "PeerRelay": "100.70.9.9:7777"}, {"path": "peer_relay"})]:  # scrub-ok: fictional CGNAT fixture address
        ind, _ = indicator(Runner(status(**peer)))
        assert await ind.lookup(PHONE) == want


async def test_probe_runs_tailscale_ping_once_per_30s():
    r = Runner(status(), ping_out="pong from phone (100.70.1.2) via DERP(tok) in 182ms\n")  # scrub-ok: fictional CGNAT fixture address
    ind, clock = indicator(r)
    assert await ind.lookup(PHONE, probe=True) == {"path": "relay", "region": "tok"}
    assert r.argv[-1] == ["/usr/bin/tailscale", "ping", "-c", "1", "--timeout", "3s", PHONE]
    clock[0] = 10.0
    assert await ind.lookup(PHONE, probe=True) == {"path": "relay", "region": "tok"}   # cached
    assert sum(1 for a in r.argv if a[1] == "ping") == 1
    clock[0] = 31.0
    r.ping_out = "pong from phone (100.70.1.2) via 203.0.113.5:41641 in 40ms\n"  # scrub-ok: fictional CGNAT fixture address
    assert await ind.lookup(PHONE, probe=True) == {"path": "direct"}
    assert sum(1 for a in r.argv if a[1] == "ping") == 2
    # the status itself is cached 30 s
    assert sum(1 for a in r.argv if a[1] == "status") == 2


async def test_failed_ping_stays_idle():
    r = Runner(status(), ping_out='ping "100.70.1.2" timed out\nno reply\n', rc=1)  # scrub-ok: fictional CGNAT fixture address
    ind, _ = indicator(r)
    assert await ind.lookup(PHONE, probe=True) == {"path": "idle"}


async def test_non_tailnet_unknown_and_unavailable():
    ind, _ = indicator(Runner(status()))
    assert await ind.lookup("10.0.0.20") == {"path": "local"}  # scrub-ok: fictional fixture address
    assert await ind.lookup("127.0.0.1") == {"path": "local"}
    assert await ind.lookup("100.70.0.1") == {"path": "local"}             # this node itself  # scrub-ok: fictional CGNAT fixture address
    assert await ind.lookup("100.99.99.99") == {"path": "unknown"}  # scrub-ok: fictional CGNAT fixture address
    assert await ind.lookup(None) == {"path": "unknown"}
    none = RelayIndicator(tailscale="", runner=Runner(status()))
    assert await none.lookup(PHONE) == {"path": "unavailable"}

    async def broken(argv, timeout):
        return 1, ""
    assert await RelayIndicator(tailscale="/x/tailscale", runner=broken).lookup(PHONE) == {"path": "unavailable"}


async def test_region_codes_are_sanitised():
    ind, _ = indicator(Runner(status(Active=True, Relay="<script>")))
    assert await ind.lookup(PHONE) == {"path": "relay"}


def test_parse_ping_forms():
    assert _parse_ping("pong from b (100.1.2.3) via DERP(fra) in 80ms") == ("relay", "fra")
    assert _parse_ping("pong from b (100.1.2.3) via peer-relay(100.9.9.9:7777:vni:5) in 9ms") == ("peer_relay", None)
    assert _parse_ping("pong from b (100.1.2.3) via 10.0.0.4:41641 in 4ms") == ("direct", None)  # scrub-ok: fictional fixture address
    assert _parse_ping("no reply") is None


@pytest.mark.skipif(not __import__("shutil").which("tailscale"), reason="no tailscale binary")
async def test_real_tailscale_status_is_parsed_without_a_probe():
    """Read-only against this box's tailscaled: never probes, never fails."""
    ind = RelayIndicator()
    info = await ind.lookup("100.64.0.1")  # scrub-ok: fictional CGNAT fixture address
    assert info["path"] in ("unknown", "idle", "relay", "direct", "peer_relay", "local", "unavailable")
    assert all(a[1] == "status" for a in ind.calls)
