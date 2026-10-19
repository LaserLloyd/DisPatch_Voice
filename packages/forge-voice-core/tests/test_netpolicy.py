"""Which addresses the rig may be (host/netpolicy.py): allowed ranges, the refusals that no
allow-list can lift, the plain-ws:// warning, and DNS names that resolve somewhere forbidden."""
from __future__ import annotations

import socket

import pytest

from forge_voice_core import netpolicy as N


@pytest.mark.parametrize("host, kind", [
    ("127.0.0.1", "loopback"), ("[::1]", "loopback"), ("localhost", "loopback"), ("x.localhost", "loopback"),
    ("10.1.2.3", "private"), ("172.16.0.9", "private"), ("172.31.255.1", "private"), ("192.168.77.37", "private"),  # scrub-ok: address-range fixture
    ("fd12:3456::1", "private"),
    ("100.64.0.1", "tailnet"), ("100.127.255.254", "tailnet"), ("100.100.100.200", "metadata"),  # scrub-ok: address-range fixture
    ("169.254.169.254", "metadata"), ("169.254.170.2", "metadata"), ("fd00:ec2::254", "metadata"),
    ("169.254.1.1", "link_local"), ("fe80::1", "link_local"),
    ("0.0.0.0", "invalid"), ("::", "invalid"), ("224.0.0.1", "invalid"),
    ("8.8.8.8", "public"), ("192.0.2.7", "public"), ("172.32.0.1", "public"), ("100.128.0.1", "public"),
    ("rig", "lan_name"), ("rig.local", "lan_name"), ("box.lan", "lan_name"), ("box.tail1234.ts.net", "lan_name"),  # scrub-ok: address-range fixture
    ("rig.example.com", "public_name"), ("::ffff:169.254.169.254", "metadata"),
])
def test_classify_host(host, kind):
    assert N.classify_host(host) == kind


def verdict(url, allow=()):
    return N.check_url_policy(url, tuple(allow))


@pytest.mark.parametrize("url", ["ws://127.0.0.1:1236/v1/realtime", "wss://10.0.0.5/x", "ws://192.168.1.2:1236/v1/realtime",  # scrub-ok: address-range fixture
                                 "ws://100.64.1.1:1236/x", "ws://rig.local:1236/x", "ws://rig:1236/x",  # scrub-ok: address-range fixture
                                 "wss://box.ts.net/v1/realtime", "ws://[::1]:1236/x"])  # scrub-ok: address-range fixture
def test_private_loopback_tailnet_and_lan_names_are_allowed(url):
    v = verdict(url)
    assert v.ok and v.code in ("ok", "url_warn_plaintext")


@pytest.mark.parametrize("url, code", [
    ("ws://169.254.169.254/latest/meta-data", "url_blocked_metadata"),
    ("wss://169.254.169.254/", "url_blocked_metadata"),
    ("ws://100.100.100.200/", "url_blocked_metadata"),  # scrub-ok: address-range fixture
    ("ws://[fd00:ec2::254]/", "url_blocked_metadata"),
    ("ws://169.254.7.7/", "url_blocked_link_local"),
    ("ws://[fe80::1]/", "url_blocked_link_local"),
    ("ws://0.0.0.0:1236/", "url_invalid"),
    ("http://10.0.0.1/", "url_invalid"),                 # not ws/wss  # scrub-ok: address-range fixture
    ("ws://user:pw@10.0.0.1/", "url_invalid"),  # scrub-ok: address-range fixture
    ("ws://10.0.0.1/?token=x", "url_invalid"),  # scrub-ok: address-range fixture
    ("ws://10.0.0.1:99999/", "url_invalid"),  # scrub-ok: address-range fixture
    ("ws://8.8.8.8:1236/", "url_not_allowed"),
    ("wss://rig.example.com/v1/realtime", "url_not_allowed"),
])
def test_refusals_carry_a_code_a_reason_and_a_fix(url, code):
    v = verdict(url)
    assert not v.ok and v.code == code and v.reason and v.fix


def test_metadata_and_link_local_cannot_be_allow_listed():
    for url in ("ws://169.254.169.254/", "ws://169.254.5.5/", "ws://100.100.100.200/"):  # scrub-ok: address-range fixture
        assert not verdict(url, ["169.254.0.0/16", "100.64.0.0/10", "*", "169.254.169.254"]).ok


def test_a_public_host_needs_the_admin_allow_list():
    assert not verdict("wss://rig.example.com/x").ok
    assert verdict("wss://rig.example.com/x", ["rig.example.com"]).ok
    assert verdict("wss://a.rig.example.com/x", ["*.example.com"]).ok
    assert not verdict("wss://example.org/x", ["*.example.com"]).ok
    assert verdict("ws://8.8.8.8/x", ["8.8.8.0/24"]).ok
    assert not verdict("ws://8.8.9.8/x", ["8.8.8.0/24"]).ok
    assert N.parse_allow("a.com, 10.0.0.0/8 ,,") == ("a.com", "10.0.0.0/8") and N.parse_allow(None) == ()  # scrub-ok: address-range fixture


def test_plain_ws_off_the_machine_warns_but_works_and_loopback_does_not():
    v = verdict("ws://192.168.1.2:1236/x")  # scrub-ok: address-range fixture
    assert v.ok and v.warning and v.code == "url_warn_plaintext"
    assert verdict("wss://192.168.1.2:1236/x").warning == ""  # scrub-ok: address-range fixture
    assert verdict("ws://127.0.0.1:1236/x").warning == ""


def test_plain_ws_on_the_tailnet_does_not_warn():
    """WireGuard already encrypts a tailnet hop, and the warning's own advice is "keep it on a VPN or
    tailnet": a tailnet rig (the 100.64/10 range or a MagicDNS name) is not degraded (ops check
    2026-10-19: the live rig showed "degraded" and warned on every voice chat)."""
    for url in ("ws://100.64.1.1:1236/v1/realtime", "ws://box.tail1234.ts.net:1236/v1/realtime"):  # scrub-ok: address-range fixture
        v = verdict(url)
        assert v.ok and v.warning == "" and v.code == "ok", url
    assert verdict("ws://10.0.0.5:1236/x").warning  # scrub-ok: address-range fixture
    assert verdict("ws://rig.lan:1236/x").warning


def test_http_schemes_for_the_speech_servers():
    assert N.check_url_policy("http://10.0.0.5:8000", (), schemes=("http", "https")).ok  # scrub-ok: address-range fixture
    assert not N.check_url_policy("ws://10.0.0.5:8000", (), schemes=("http", "https")).ok  # scrub-ok: address-range fixture
    assert N.check_url_policy("https://api.example.com", ("api.example.com",), schemes=("http", "https")).ok


def test_a_name_that_resolves_to_a_metadata_address_is_refused(monkeypatch):
    def fake_getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    v = N.check_resolved("ws://rig.local:1236/x")
    assert not v.ok and v.code == "url_blocked_resolved"
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.4", 1))])  # scrub-ok: address-range fixture
    assert N.check_resolved("ws://rig.local:1236/x").ok
    assert N.check_resolved("ws://127.0.0.1:1/x").ok           # an IP literal is not resolved at all

    def boom(*a, **k):
        raise OSError("no dns")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert N.check_resolved("ws://nope.local:1/x").ok          # unreachable is the connection's business


@pytest.mark.parametrize("host", ["2852039166", "0xa9fea9fe", "0251.0376.0251.0376", "169.254.43518", "0xa9.0xfe.0xa9.0xfe"])
def test_numeric_shorthand_for_an_ipv4_address_is_canonicalised(host):
    """The OS resolver (inet_aton) reads a single decimal, a hex word or octal octets as an IPv4
    address; the policy must see the same address, not a 'LAN name' with no dot in it."""
    assert N.classify_host(host) == "metadata"
    v = verdict(f"ws://{host}/x")
    assert not v.ok and v.code == "url_blocked_metadata"
    assert not N.check_url_policy(f"http://{host}/", schemes=("http", "https")).ok


def test_numeric_shorthand_for_an_allowed_address_keeps_its_kind():
    assert N.classify_host("2130706433") == "loopback"            # 127.0.0.1
    assert N.classify_host("167772161") == "private"              # the first host of the 10/8 range  # scrub-ok: address-range fixture
    assert N.classify_host("rig") == "lan_name" and N.classify_host("cafe") == "lan_name"
