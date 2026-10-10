"""Server side of the link: `link`/`probe` frames, link metrics, the Tailscale relay indicator.

The client decides when the link is bad (``link-monitor.js``: RTT, underruns,
goodput, first-audio deadlines; never ``bufferedAmount``) and tells the server
with ``link {mode, down_kbps?, up_kbps?, why}``. :class:`LinkState` validates that
and holds it until the next sentence boundary (contracts C-§C.3, invariant 6): a
sentence already on its way finishes on its path, so the voice never changes
mid-sentence. ``up_kbps`` is only recorded (the client reconfigures its own
encoder). ``down_kbps: null`` or ``why: "decoder"`` means the client's Opus decoder
failed: the downlink falls back to PCM16 from the next sentence.

:func:`probe_payload` makes the incompressible burst a ``probe`` asks for
(``os.urandom``: permessage-deflate cannot shrink it); :class:`ProbeLimiter`
allows one per 20 s per session.

:class:`RelayIndicator` reports whether the phone reaches this box directly or
through a Tailscale relay (DERP or a peer relay), from ``tailscale status --json``
(fixed argv, cached 30 s) and, when asked, one ``tailscale ping -c 1`` (at most
one per peer per 30 s). An idle peer has no current path, so it never reads as
"relay" until traffic has flowed or a probe ran (hardening L5). It only ever
reports the path kind and the DERP region code, never addresses.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import shutil
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

log = logging.getLogger("dispatch_voice.transport.link")

MODES = ("audio", "text")
DOWN_KBPS = (24, 16)
UP_KBPS = (16, 12)
WHYS = ("underrun", "rtt", "goodput", "deadline", "stall", "user", "recovered", "decoder", "encoder")
SLOW_LINK = ("phone", "wait", "text", "always_phone")

PROBE_MAX = 32_768
PROBE_EVERY_S = 20.0


def probe_payload(n: int) -> bytes:
    """``os.urandom(min(n, 32768))`` (negative or junk sizes -> empty)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return b""
    return os.urandom(max(0, min(n, PROBE_MAX)))


class ProbeLimiter:
    """One probe per ``every_s`` per session (the frame still spends a control token)."""

    def __init__(self, every_s: float = PROBE_EVERY_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.every_s = every_s
        self._clock = clock
        self._last: float | None = None
        self.refused = 0

    def allow(self) -> bool:
        now = self._clock()
        if self._last is not None and now - self._last < self.every_s:
            self.refused += 1
            return False
        self._last = now
        return True


@dataclass
class LinkState:
    """What the client asked for, applied at sentence boundaries.

    ``mode`` is the server's speaking path: ``"text"`` = send ``say`` only, never
    call the TTS. ``slow_link: "always_phone"`` (from ``start``) forces text for
    the whole socket. ``codec_out`` mirrors the transport's downlink codec."""
    mode: str = "audio"
    down_kbps: int = 24
    up_kbps: int = 16
    degraded_since: float | None = None
    why: str = ""
    slow_link: str = "phone"
    pcm_fallback: bool = False          # the client's decoder failed: PCM16 down from the next sentence
    changes: int = 0
    _pending: dict = field(default_factory=dict, repr=False)

    @property
    def text_mode(self) -> bool:
        """True when the next sentence must not be synthesised."""
        return self.mode == "text" or self.slow_link == "always_phone"

    def set_slow_link(self, value) -> str:
        """``start.slow_link``; unknown -> ``"phone"`` (the default), silently."""
        self.slow_link = value if value in SLOW_LINK else "phone"
        return self.slow_link

    def request(self, msg: Mapping, now: float | None = None) -> dict:
        """Validate one ``link`` frame and hold it for the next boundary.

        Returns the normalised request (what will be applied). Unknown values fall
        back to the current ones; a later request overrides an earlier pending one
        field by field."""
        now = time.monotonic() if now is None else now
        req: dict = {}
        mode = msg.get("mode")
        if mode in MODES:
            req["mode"] = mode
        if "down_kbps" in msg:
            v = msg.get("down_kbps")
            if v is None:
                req["pcm_fallback"] = True
            elif v in DOWN_KBPS:
                req["down_kbps"] = v
        why = msg.get("why")
        why = why if why in WHYS else ""
        if why == "decoder":
            req["pcm_fallback"] = True
        v = msg.get("up_kbps")
        if v in UP_KBPS:
            self.up_kbps = v          # the client's own encoder: nothing to wait for
        req["why"] = why
        self._pending.update(req)
        self._pending["_at"] = now
        return {k: v for k, v in req.items()}

    @property
    def pending(self) -> bool:
        return any(k != "_at" for k in self._pending)

    def at_boundary(self, now: float | None = None) -> dict:
        """Apply what is pending (call between sentences). Returns the fields that changed."""
        if not self.pending:
            return {}
        now = time.monotonic() if now is None else now
        p, self._pending = self._pending, {}
        changed: dict = {}
        if "mode" in p and p["mode"] != self.mode:
            self.mode = p["mode"]
            changed["mode"] = self.mode
        if "down_kbps" in p and p["down_kbps"] != self.down_kbps:
            self.down_kbps = p["down_kbps"]
            changed["down_kbps"] = self.down_kbps
        if p.get("pcm_fallback") and not self.pcm_fallback:
            self.pcm_fallback = True
            changed["pcm_fallback"] = True
        if p.get("why"):
            self.why = p["why"]
        degraded = self.mode == "text" or self.down_kbps != DOWN_KBPS[0]
        if degraded and self.degraded_since is None:
            self.degraded_since = p.get("_at", now)
        elif not degraded:
            self.degraded_since = None
        if changed:
            self.changes += 1
        return changed

    def metrics(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        out = {"mode": "text" if self.text_mode else "audio", "down_kbps": self.down_kbps,
               "up_kbps": self.up_kbps}
        if self.why:
            out["why"] = self.why
        if self.degraded_since is not None:
            out["degraded_s"] = round(now - self.degraded_since, 1)
        return out


# --------------------------------------------------------------------------- #
# Tailscale relay indicator
# --------------------------------------------------------------------------- #

_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_REGION = re.compile(r"^[a-z0-9-]{1,16}$")
PATHS = ("direct", "relay", "peer_relay", "idle", "local", "unknown", "unavailable")


def _ip(s: str | None):
    if not s:
        return None
    try:
        return ipaddress.ip_address(s.strip().strip("[]"))
    except ValueError:
        return None


def is_tailnet(addr) -> bool:
    a = _ip(addr) if isinstance(addr, str) else addr
    if a is None:
        return False
    if a.version == 6 and a.ipv4_mapped:
        a = a.ipv4_mapped
    return (a.version == 4 and a in _TAILNET_V4) or (a.version == 6 and a in _TAILNET_V6)


def peer_address(client_host: str | None, headers: Mapping[str, str] | None = None) -> str | None:
    """The address of the phone behind this socket.

    A direct connection gives it as the socket peer. Through Tailscale Serve the
    peer is loopback and ``tailscaled`` appends the real one to ``X-Forwarded-For``:
    the header is read ONLY when the peer is loopback, and only its last entry
    (the one the local proxy added), so a remote client cannot spoof it. A client
    ON this box (a loopback peer that is not Serve) can still name any address and
    steer the reading; it only feeds the Voice lab's relay indicator, never a
    security decision (review-T6 L4, accepted)."""
    host = _ip(client_host)
    if host is None:
        return None
    if host.is_loopback and headers is not None:
        xff = None
        for k, v in headers.items():
            if k.lower() == "x-forwarded-for":
                xff = v
        if xff:
            last = _ip(xff.split(",")[-1])
            if last is not None:
                return str(last)
    return str(host)


def _parse_ping(out: str) -> tuple[str, str | None] | None:
    """`pong from x (ip) via DERP(tok) in 80ms` -> ("relay", "tok"); `via peer-relay(...)` ->
    ("peer_relay", None); `via 203.0.113.4:41641` -> ("direct", None); no pong -> None."""
    for line in out.splitlines():
        m = re.search(r"\bvia\s+(\S+)", line)
        if not line.lower().startswith("pong") or not m:
            continue
        via = m.group(1)
        d = re.match(r"DERP\(([^)]*)\)", via)
        if d:
            region = d.group(1).lower()
            return "relay", region if _REGION.match(region) else None
        if "relay" in via.lower():
            return "peer_relay", None
        return "direct", None
    return None


Runner = Callable[[list[str], float], Awaitable[tuple[int, str]]]


async def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    """Fixed-argv subprocess (never a shell); output capped at 4 MB; killed on timeout."""
    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, ""
    return proc.returncode or 0, (out or b"")[:4 << 20].decode("utf-8", "replace")


@dataclass
class _Probe:
    at: float
    path: str
    region: str | None


class RelayIndicator:
    """``await lookup(addr, probe=...)`` -> ``{"path": ..., "region"?: str}``.

    ``path``: ``direct`` | ``relay`` (DERP, ``region`` = its code) | ``peer_relay`` |
    ``idle`` (a tailnet peer with no traffic yet and no probe) | ``local`` (not a
    tailnet address: LAN or loopback) | ``unknown`` (a tailnet address this node
    does not list) | ``unavailable`` (no ``tailscale`` binary or it failed).
    """

    def __init__(self, *, tailscale: str | None = None, runner: Runner | None = None,
                 cache_s: float = 30.0, probe_every_s: float = 30.0, timeout_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.binary = tailscale if tailscale is not None else shutil.which("tailscale")
        self._run = runner or _run
        self.cache_s = cache_s
        self.probe_every_s = probe_every_s
        self.timeout_s = timeout_s
        self._clock = clock
        self._status: dict | None = None
        self._status_at = float("-inf")
        self._status_lock = asyncio.Lock()
        self._probes: dict[str, _Probe] = {}
        self.calls: deque[list[str]] = deque(maxlen=32)   # the last 32 argv (tests, Voice lab)

    async def _status_json(self) -> dict | None:
        async with self._status_lock:
            now = self._clock()
            if self._status is not None and now - self._status_at < self.cache_s:
                return self._status
            argv = [self.binary, "status", "--json"]
            self.calls.append(argv)
            try:
                rc, out = await self._run(argv, self.timeout_s)
                st = json.loads(out) if rc == 0 and out else None
            except (OSError, ValueError) as e:
                log.info("voice: tailscale status failed (%s)", e)
                st = None
            self._status = st if isinstance(st, dict) else {}
            self._status_at = now
            return self._status or None

    @staticmethod
    def _find_peer(status: dict, addr: str) -> dict | None:
        for p in (status.get("Peer") or {}).values():
            if not isinstance(p, dict):
                continue
            if addr in (p.get("TailscaleIPs") or []):
                return p
        return None

    async def lookup(self, addr: str | None, *, probe: bool = False) -> dict:
        a = _ip(addr)
        if a is None:
            return {"path": "unknown"}
        if not is_tailnet(a):
            return {"path": "local"}
        if not self.binary:
            return {"path": "unavailable"}
        status = await self._status_json()
        if not status:
            return {"path": "unavailable"}
        peer = self._find_peer(status, str(a))
        if peer is None:
            self_ips = (status.get("Self") or {}).get("TailscaleIPs") or []
            return {"path": "local"} if str(a) in self_ips else {"path": "unknown"}
        trafficked = bool(peer.get("Active")) or bool(peer.get("RxBytes")) or bool(peer.get("TxBytes"))
        if trafficked:
            if peer.get("CurAddr"):
                return {"path": "direct"}
            if peer.get("PeerRelay"):
                return {"path": "peer_relay"}
            region = str(peer.get("Relay") or "").lower()
            return {"path": "relay", **({"region": region} if _REGION.match(region) else {})}
        key = str(a)
        now = self._clock()
        last = self._probes.get(key)
        if last is not None and now - last.at < self.probe_every_s:
            return {"path": last.path, **({"region": last.region} if last.region else {})}
        if not probe:
            return {"path": "idle"}
        argv = [self.binary, "ping", "-c", "1", "--timeout", "3s", key]
        self.calls.append(argv)
        try:
            _, out = await self._run(argv, self.timeout_s)
            parsed = _parse_ping(out)
        except OSError as e:
            log.info("voice: tailscale ping failed (%s)", e)
            parsed = None
        path, region = parsed if parsed else ("idle", None)
        self._probes[key] = _Probe(now, path, region)
        if len(self._probes) > 64:
            oldest = min(self._probes, key=lambda k: self._probes[k].at)
            self._probes.pop(oldest, None)
        return {"path": path, **({"region": region} if region else {})}

    @staticmethod
    def relayed(info: Mapping) -> bool | None:
        """``metrics.link.relay``: True/False once known, None while idle/unknown."""
        p = info.get("path")
        if p in ("relay", "peer_relay"):
            return True
        if p in ("direct", "local"):
            return False
        return None
