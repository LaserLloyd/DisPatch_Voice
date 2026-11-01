"""Who may use, see, change and download which cloned voice (DisPatch 4.0 and later).

Every voice has an owner: a profile id (a personal clone) or ``"household"``
(everyone). A profile.json without an owner key is a household voice, so the
voices made before per-profile clones existed need no migration. The host
(DisPatch) gates only the coarse permission (``use_voice`` / ``voice_lab`` /
``manage_plugins``); the ownership rules live here:

=========================  ====================================================
Action                     Rule
=========================  ====================================================
list / picker              household voices + the caller's own clones. Another
                           profile's clone is never listed, not even to Admins.
create                     needs ``voice_lab``; the clone belongs to the caller.
                           ``household=true`` only for Admins (422 otherwise).
rename / delete / say      the owner; Admins only for household clones. Anyone
                           else gets 404.
household <-> personal     Admins only.
default voice              a household voice, set by an Admin.
download recordings        the owner; Admins only for household clones.
speak (a bot's voice)      household voices for any bot; a personal clone only
                           for a personal bot owned by the same profile.
=========================  ====================================================

Compatibility: a host without ``ctx.caller`` (DisPatch before 4.0) gets the old
single-tenant behaviour, unchanged (:data:`LEGACY`). A host whose ``ctx.caller``
returns None (nobody signed in) reads household voices only and may not write.

The host's ``Caller`` is used by duck typing: ``profile_id``, ``is_admin`` and
``can(perm)``. Nothing here imports DisPatch.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import HTTPException

from . import context

log = logging.getLogger("dispatch.voice")

HOUSEHOLD = "household"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


class _Marker:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:      # pragma: no cover
        return f"<{self.name}>"

    def __bool__(self) -> bool:
        return False


#: ``caller_of`` answer on a host that has no ``ctx.caller``: behave as 1.5.x did.
LEGACY: Any = _Marker("legacy")
#: A view that sees household voices only (the Admin-only settings routes, the Lab picker).
HOUSEHOLD_ONLY: Any = _Marker("household-only")
#: ``voice_for_bot(bot_owner=...)`` default: do not check ownership at all.
NO_CHECK: Any = _Marker("no-check")


# --------------------------------------------------------------------------- #
# The host
# --------------------------------------------------------------------------- #

def tenant_mode() -> bool:
    """True when the host knows profiles (it offers ``ctx.caller``)."""
    ctx = context.get_context()
    return ctx is not None and callable(getattr(ctx, "caller", None))


def declares_voice_access() -> bool:
    ctx = context.get_context()
    return ctx is not None and callable(getattr(ctx, "register_voice_access", None))


def caller_of(request) -> Any:
    """The signed-in caller, ``None`` (nobody / a machine), or :data:`LEGACY`."""
    ctx = context.get_context()
    fn = getattr(ctx, "caller", None) if ctx is not None else None
    if not callable(fn):
        return LEGACY
    try:
        return fn(request)
    except Exception:
        log.exception("voice: ctx.caller failed; treating the request as signed out")
        return None


def is_legacy(caller) -> bool:
    return caller is LEGACY


def _pid(caller) -> str | None:
    if caller is None or caller is LEGACY or caller is HOUSEHOLD_ONLY:
        return None
    p = getattr(caller, "profile_id", None)
    return p if isinstance(p, str) and p else None


def is_admin(caller) -> bool:
    if caller is None or caller is HOUSEHOLD_ONLY:
        return False
    if caller is LEGACY:
        return True
    return bool(getattr(caller, "is_admin", False))


def _can(caller, perm: str) -> bool:
    if caller is LEGACY:
        return True
    if caller is None or caller is HOUSEHOLD_ONLY:
        return False
    try:
        return bool(caller.can(perm))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Owners on disk (read-only: never creates the voices directory)
# --------------------------------------------------------------------------- #

#: Owner of a voice folder whose metadata is unreadable: matches no profile and is not household.
UNKNOWN_OWNER = "\x00unknown"


def owner_of_meta(meta: dict) -> str:
    o = meta.get("owner") if isinstance(meta, dict) else None
    return o if isinstance(o, str) and o.strip() and o != HOUSEHOLD else HOUSEHOLD


def owner_of_voice(voice_id) -> str:
    """The owner of a voice id. Built-in voices, the stock voice and unknown ids are household."""
    if not isinstance(voice_id, str) or not _ID_RE.match(voice_id):
        return HOUSEHOLD
    folder = context.paths().voices_dir / voice_id
    try:
        meta = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # No such voice: household (nothing to protect). A voice folder whose metadata cannot be
        # read is of unknown ownership: nobody matches, so it is never treated as household.
        return UNKNOWN_OWNER if folder.is_dir() else HOUSEHOLD
    if not isinstance(meta, dict):
        return UNKNOWN_OWNER
    return owner_of_meta(meta)


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #

def can_use(voice_id, caller, *, owner: str | None = None) -> bool:
    """May this caller hear the voice (the preview, the picker)?"""
    if caller is LEGACY:
        return True
    owner = owner_of_voice(voice_id) if owner is None else owner
    return owner == HOUSEHOLD or (_pid(caller) is not None and _pid(caller) == owner)


def can_manage(pid, caller, *, owner: str | None = None) -> bool:
    """Rename, delete, try a line. The owner; Admins for household voices."""
    if caller is LEGACY:
        return True
    if caller is None or caller is HOUSEHOLD_ONLY:
        return False
    owner = owner_of_voice(pid) if owner is None else owner
    if owner == HOUSEHOLD:
        return is_admin(caller)
    return _pid(caller) == owner


def can_download(pid, caller, *, owner: str | None = None) -> bool:
    """The recordings: the owner of a personal clone; Admins only for a household one."""
    return can_manage(pid, caller, owner=owner)


def visible(profiles: list[dict], caller) -> list[dict]:
    """The rows this caller may see (each row carries ``owner``)."""
    if caller is LEGACY:
        return list(profiles)
    me = _pid(caller)
    return [p for p in profiles
            if owner_of_meta(p) == HOUSEHOLD or (me is not None and owner_of_meta(p) == me)]


def may_publish(caller) -> bool:
    """Make a clone household, or a household clone personal: Admins only."""
    return caller is LEGACY or is_admin(caller)


def may_create(caller) -> bool:
    return _can(caller, "voice_lab")


def voice_access(voice_id: str, bot_owner: str | None) -> bool:
    """The rule the host calls (``ctx.register_voice_access``): may a bot owned by
    ``bot_owner`` (None = a household bot) speak with ``voice_id``?"""
    owner = owner_of_voice(voice_id)
    if owner == HOUSEHOLD:
        return True
    return bool(bot_owner) and bot_owner == owner


def bot_owner_of(bot, sock=None) -> str | None:
    """The owner to hand ``voice_access`` for a bot: its profile when it is a personal bot, else None."""
    if bot is None or getattr(bot, "scope", "household") != "personal":
        return None
    return (getattr(bot, "owner_profile", "") or getattr(sock, "profile_id", None) or None)


# --------------------------------------------------------------------------- #
# Route guard used by dispatch_voice.router.build_router(access=...)
# --------------------------------------------------------------------------- #

class Access:
    """Wraps one ProfileStore with the rules above. The router calls these with the request;
    each raises the HTTPException the route should answer."""

    def __init__(self, store):
        self.store = store

    # -- who ---------------------------------------------------------------- #
    def caller(self, request):
        return caller_of(request)

    def reader(self, request):
        """Anyone may read (a None caller reads household voices only)."""
        return caller_of(request)

    def writer(self, request, perm: str = "voice_lab"):
        c = caller_of(request)
        if c is LEGACY:
            return c
        if c is None:
            raise HTTPException(403, "Sign in to manage voices.")
        if not _can(c, perm):
            raise HTTPException(403, "Your profile can't clone voices.")
        return c

    # -- rows --------------------------------------------------------------- #
    def annotate(self, row: dict, caller) -> dict:
        """Add what the UI needs and drop the owner's profile id."""
        if caller is LEGACY:
            return row
        owner = owner_of_meta(row)
        out = {k: v for k, v in row.items() if k != "owner"}
        out["household"] = owner == HOUSEHOLD
        out["mine"] = owner != HOUSEHOLD and owner == _pid(caller)
        out["can_manage"] = can_manage(row.get("id"), caller, owner=owner)
        out["can_download"] = can_download(row.get("id"), caller, owner=owner)
        return out

    def list_rows(self, rows: list[dict], caller) -> list[dict]:
        return [self.annotate(r, caller) for r in visible(rows, caller)]

    def row_or_404(self, pid: str, caller, need: str = "read") -> dict:
        """The voice's row, or 404 when this caller may not do ``need`` with it
        (``read`` | ``manage`` | ``download``). A missing voice is the same 404."""
        try:
            row = self.store.get(pid)
        except Exception:
            raise HTTPException(404, "Unknown voice.") from None
        owner = owner_of_meta(row)
        ok = {"read": can_use, "manage": can_manage, "download": can_download}[need](pid, caller, owner=owner)
        if not ok:
            raise HTTPException(404, "Unknown voice.")
        return row

    # -- create / patch ----------------------------------------------------- #
    def owner_for_new(self, caller, household: bool) -> str | None:
        """The ``owner`` to give a new clone (None = household)."""
        if caller is LEGACY:
            return None
        if household:
            if not may_publish(caller):
                raise HTTPException(422, "Only an Admin can make a household voice.")
            return None
        return _pid(caller)             # an open (unclaimed) install has no profile: household

    def plan_patch(self, row: dict, caller, body: dict) -> dict:
        """Validate a PATCH body against the rules; returns ``{"owner": ..., "name": ..., "default": ...}``
        with only the keys to apply (``owner`` is a profile id or None for household)."""
        plan: dict = {}
        if caller is LEGACY:
            return {k: body[k] for k in ("name", "default") if k in body}
        owner = owner_of_meta(row)
        target_owner = owner
        if "household" in body:
            h = body["household"]
            if not isinstance(h, bool):
                raise HTTPException(422, "household must be true or false")
            if not may_publish(caller):
                raise HTTPException(403, "Only an Admin can change who a voice belongs to.")
            if h:
                target_owner = HOUSEHOLD
            elif owner == HOUSEHOLD:
                me = _pid(caller)
                if me is None:
                    raise HTTPException(422, "Sign in as a profile to keep a voice private.")
                target_owner = me
            if target_owner != owner:
                plan["owner"] = None if target_owner == HOUSEHOLD else target_owner
        if "name" in body and body["name"] is not None:
            plan["name"] = body["name"]
        if body.get("default") is not None:
            if body["default"]:
                if not may_publish(caller):
                    raise HTTPException(403, "Only an Admin can set the default voice.")
                if target_owner != HOUSEHOLD:
                    raise HTTPException(422, "A private voice can't be the default.")
            plan["default"] = body["default"]
        return plan


# --------------------------------------------------------------------------- #
# Profile data hooks (ctx.register_profile_data)
# --------------------------------------------------------------------------- #

def owned_ids(profile_id: str) -> list[str]:
    """Ids of the clones this profile owns (read from disk; household never matches)."""
    if not isinstance(profile_id, str) or not profile_id or profile_id == HOUSEHOLD:
        return []
    root = context.paths().voices_dir
    out: list[str] = []
    try:
        dirs = sorted(root.iterdir())
    except OSError:
        return []
    for d in dirs:
        if d.is_dir() and _ID_RE.match(d.name):
            try:
                meta = json.loads((d / "profile.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if owner_of_meta(meta) == profile_id:
                out.append(d.name)
    return out


def profile_count(profile_id: str) -> dict:
    """``{"voices": n}``: counts only, never names."""
    return {"voices": len(owned_ids(profile_id))}


def profile_purge(profile_id: str) -> int:
    """Delete every clone the profile owns: its directory, the rig's copy of the conditionals,
    the acknowledgement lines. Returns how many went. Reset does not call this."""
    from . import profiles_bridge
    ids = owned_ids(profile_id)
    if not ids:
        return 0
    st = profiles_bridge._RigAwareStore()
    done = 0
    for pid in ids:
        try:
            st.delete(pid)            # ack lines, directory, rig conditionals (rig.delete_voice_async)
            done += 1
        except Exception:
            log.exception("voice: deleting a removed profile's voice failed")
    log.info("voice: removed %d cloned voice(s) of a deleted profile", done)
    return done
