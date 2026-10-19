"""Which addresses the rig (and an HQ speech server) may be.

``rig.url`` receives the bearer token and the live microphone audio, and the HQ server
receives the reply text. An address typed by any operator is therefore a place data goes, so
it is checked before anything is sent (1.5.0 accepted every host, link-local and cloud
metadata addresses included):

* **always refused**: link-local addresses (IPv4 and IPv6), the cloud metadata addresses
  (``METADATA`` below), unspecified addresses and multicast;
* **allowed**: loopback, the private ranges (``PRIVATE_NETS``), the CGNAT/tailnet range
  (``TAILNET_NETS``) and plain LAN names (a single label, or one of ``LAN_SUFFIXES``);
* **everything else** (a public address or domain) only when the ADMIN allow-list names it:
  ``DISPATCH_VOICE_RIG_ALLOW`` / ``rig.allow``, entries being a host, ``*.suffix`` or a CIDR;
* plain ``ws://`` / ``http://`` to anything but loopback or the tailnet (the CGNAT range or a
  MagicDNS name: WireGuard encrypts that hop) is allowed with a warning (the audio and the token
  cross the network unencrypted).

A hostname is also resolved just before connecting (:func:`check_resolved`): a name that
points at a link-local or metadata address is refused (DNS rebinding).

Pure functions, stdlib only. Every refusal carries a stable ``code`` plus a ``reason`` and a
``fix`` string for the status page.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from .settings_schema import check_url

_V4 = ipaddress.IPv4Network
_V6 = ipaddress.IPv6Network

METADATA = frozenset(ipaddress.ip_address(a) for a in (
    "169.254.169.254", "169.254.170.2", "100.100.100.200", "192.0.0.192", "fd00:ec2::254"))  # scrub-ok: cloud metadata addresses
PRIVATE_NETS = (_V4("10.0.0.0/8"), _V4("172.16.0.0/12"), _V4("192.168.0.0/16"), _V6("fc00::/7"))  # scrub-ok: RFC1918 ranges
TAILNET_NETS = (_V4("100.64.0.0/10"),)
LINK_LOCAL_NETS = (_V4("169.254.0.0/16"), _V6("fe80::/10"))
LAN_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal", ".ts.net", ".localhost")  # scrub-ok: suffix list, no host


@dataclass(frozen=True)
class Verdict:
    ok: bool
    kind: str = ""              # loopback | private | tailnet | lan_name | public | public_name | link_local | metadata | invalid
    code: str = ""              # stable machine code: url_invalid, url_blocked_metadata, ...
    reason: str = ""
    fix: str = ""
    warning: str = ""           # set when ok but worth saying (plain ws:// off the machine)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "code": self.code, "reason": self.reason,
                "fix": self.fix, "warning": self.warning}


#: A host made only of digits, dots and hex letters with an ``0x`` / leading-zero prefix: the
#: shorthand IPv4 forms the OS resolver accepts (``2852039166``, ``0xa9fea9fe``,
#: ``0251.0376.0251.0376`` are all 169.254.169.254) that :mod:`ipaddress` does not.
_NUMERIC_V4 = re.compile(r"\A(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}\Z", re.I)


def _addr(host: str):
    h = host.strip("[]")
    try:
        a = ipaddress.ip_address(h)
    except ValueError:
        a = None
    if a is None and _NUMERIC_V4.match(h):
        # Canonicalise the shorthand the way the connector will (inet_aton), so a metadata
        # address written as one decimal number cannot pass as a "LAN name".
        try:
            a = ipaddress.IPv4Address(socket.inet_aton(h))
        except (OSError, ValueError):
            return None
    if a is None:
        return None
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        return a.ipv4_mapped
    return a


def classify_host(host: str) -> str:
    """The kind of ``host`` (see the module docstring)."""
    if not host:
        return "invalid"
    a = _addr(host)
    if a is not None:
        if a in METADATA:
            return "metadata"
        if a.is_unspecified or a.is_multicast:
            return "invalid"
        if any(a in n for n in LINK_LOCAL_NETS if a.version == n.version):
            return "link_local"
        if a.is_loopback:
            return "loopback"
        if any(a in n for n in PRIVATE_NETS if a.version == n.version):
            return "private"
        if any(a in n for n in TAILNET_NETS if a.version == n.version):
            return "tailnet"
        return "public"
    name = host.lower().rstrip(".")
    if name == "localhost" or name.endswith(".localhost"):
        return "loopback"
    if "." not in name or name.endswith(LAN_SUFFIXES):
        return "lan_name"
    return "public_name"


def _allowed(host: str, allow: tuple[str, ...]) -> bool:
    h = host.strip("[]").lower().rstrip(".")
    a = _addr(host)
    for raw in allow:
        item = raw.strip().lower()
        if not item:
            continue
        if "/" in item and a is not None:
            try:
                if a in ipaddress.ip_network(item, strict=False):
                    return True
            except ValueError:
                continue
        elif item.startswith("*."):
            if h.endswith(item[1:]):
                return True
        elif item == h:
            return True
    return False


def parse_allow(raw) -> tuple[str, ...]:
    """The allow-list from a comma string or a list."""
    if isinstance(raw, str):
        raw = raw.split(",")
    return tuple(x.strip() for x in (raw or ()) if isinstance(x, str) and x.strip())


def check_url_policy(url: str, allow: tuple[str, ...] = (), *, schemes: tuple[str, ...] = ("ws", "wss")) -> Verdict:
    """Is ``url`` an address the rig may be at?"""
    if not check_url(url, schemes=schemes):
        want = " or ".join(f"{s}://" for s in schemes)
        return Verdict(False, "invalid", "url_invalid",
                       f"The address is not a {want} address with a host and no credentials, query or fragment.",
                       f"Use the form {schemes[0]}://host:port/path (the token is sent in a header, never in the URL).")
    u = urlsplit(url)
    host = u.hostname or ""
    kind = classify_host(host)
    if kind == "metadata":
        return Verdict(False, kind, "url_blocked_metadata",
                       "That is a cloud metadata address; sending the token and audio there would leak credentials.",
                       "Use the address of your speech server.")
    if kind == "link_local":
        return Verdict(False, kind, "url_blocked_link_local",
                       "Link-local addresses (169.254.x.x, fe80::) are never used for the rig.",
                       "Use the server's LAN, tailnet or loopback address.")
    if kind == "invalid":
        return Verdict(False, kind, "url_invalid", "That host cannot be connected to.",
                       "Use the server's LAN, tailnet or loopback address.")
    if kind in ("public", "public_name") and not _allowed(host, allow):
        return Verdict(False, kind, "url_not_allowed",
                       "That is a public address and it is not on this server's allow-list.",
                       "Use a private, tailnet or loopback address, or ask the administrator to add the host to "
                       "rig.allow (or DISPATCH_VOICE_RIG_ALLOW).")
    warning = ""
    # A tailnet hop is already encrypted (WireGuard): the warning's own advice is "keep it on a VPN or
    # tailnet", so a tailnet address or MagicDNS name is not warned about.
    on_tailnet = kind == "tailnet" or host.lower().rstrip(".").endswith(".ts.net")  # scrub-ok: suffix, no host
    if u.scheme in ("ws", "http") and kind != "loopback" and not on_tailnet:
        secure = "wss://" if u.scheme == "ws" else "https://"
        warning = (f"This address is not encrypted ({u.scheme}://): your voice and the token cross the network in "
                   f"the clear. Use {secure} or keep it on a VPN or tailnet.")
    return Verdict(True, kind, "url_warn_plaintext" if warning else "ok", "", "", warning)


def check_resolved(url: str, allow: tuple[str, ...] = ()) -> Verdict:
    """For a host NAME: resolve it and refuse when any answer is link-local, metadata or
    otherwise never-allowed (a name can be pointed anywhere). IP literals were checked by
    :func:`check_url_policy`; a resolution failure is not this module's business (the
    connection reports unreachable)."""
    host = urlsplit(url).hostname or ""
    if not host or _addr(host) is not None:
        return Verdict(True, classify_host(host), "ok")
    u = urlsplit(url)
    port = u.port or (443 if u.scheme in ("wss", "https") else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return Verdict(True, classify_host(host), "ok")
    for info in infos:
        kind = classify_host(info[4][0])
        if kind in ("metadata", "link_local", "invalid"):
            return Verdict(False, kind, "url_blocked_resolved",
                           f"{host} resolves to an address that is never used for the rig ({kind.replace('_', '-')}).",
                           "Fix the DNS record or use the server's IP address.")
    return Verdict(True, classify_host(host), "ok")
