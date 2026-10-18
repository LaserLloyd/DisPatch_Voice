"""Owner-only files on POSIX and on Windows.

POSIX answers with the mode bits (``0600`` for a file, ``0700`` for a directory). A Windows
file has no group/other bits: ``os.chmod`` only toggles the read-only flag and ``st_mode``
always reads back ``0o666`` / ``0o777``, so a mode test either rejects every file or proves
nothing. On Windows the protection is the file's ACL, so:

* :func:`restrict` removes inherited access and grants the current user alone (plus the
  SYSTEM account Windows itself needs), through ``icacls`` with SIDs (language independent);
* :func:`others_can_read` reads the DACL (``advapi32`` via ctypes) and reports whether any
  well-known "other people" SID (Everyone, Users, Authenticated Users, Guests, ...) is
  granted access.

When the ACL cannot be read (a filesystem without ACLs: FAT/exFAT, a network share, a
sandbox) :func:`others_can_read` returns ``None`` ("cannot tell") and the caller logs a
warning and goes on, instead of refusing every file (the 1.5.0 behaviour on Windows).
"""
from __future__ import annotations

import logging
import os
import stat
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("dispatch.voice")

IS_WINDOWS = sys.platform == "win32"

#: SIDs that mean "people other than the owner": Everyone, Users, Authenticated Users,
#: Guests, Anonymous, Interactive, Network, Domain Users (S-1-5-21-*-513), Domain Guests.
OTHER_SIDS = frozenset({"S-1-1-0", "S-1-5-32-545", "S-1-5-11", "S-1-5-32-546", "S-1-5-7",
                        "S-1-5-4", "S-1-5-2", "S-1-5-32-547"})
OTHER_SID_SUFFIXES = ("-513", "-514")        # Domain Users / Domain Guests
SYSTEM_SID = "S-1-5-18"


def restrict(path: Path | str, *, directory: bool = False) -> bool:
    """Make ``path`` private to the current user. True when it is now, False when it
    could not be (logged; the file stays as it was)."""
    p = Path(path)
    if not IS_WINDOWS:
        try:
            os.chmod(p, 0o700 if directory else 0o600)
            return True
        except OSError as e:
            log.warning("voice: could not restrict %s (%s)", p.name, e)
            return False
    sid = current_user_sid()
    if not sid:
        log.warning("voice: could not find your Windows account to restrict %s", p.name)
        return False
    grant = "(OI)(CI)F" if directory else "F"
    try:
        # /reset drops every explicit entry (a stray "Everyone" grant included), then the
        # inherited ones are cut and only the owner and SYSTEM are granted.
        subprocess.run(["icacls", str(p), "/reset"], capture_output=True, text=True, timeout=20, check=False)
        r = subprocess.run(
            ["icacls", str(p), "/inheritance:r", "/grant:r", f"*{sid}:{grant}", "/grant:r", f"*{SYSTEM_SID}:{grant}"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("voice: could not run icacls on %s (%s)", p.name, e)
        return False
    if r.returncode != 0:
        log.warning("voice: icacls failed on %s", p.name)
        return False
    return True


def ensure_private(path: Path | str, *, directory: bool = False) -> bool:
    """:func:`restrict` unless ``path`` is already private. Cheap when it is (a stat on
    POSIX, a DACL read on Windows), so it can follow every write of a small file."""
    if not IS_WINDOWS:
        return restrict(path, directory=directory)       # chmod is as cheap as the stat
    if others_can_read(path) is False:
        return True
    return restrict(path, directory=directory)            # also when the ACL cannot be read: try


def mode_is_private(path: Path | str, *, directory: bool = False) -> bool | None:
    """True/False for a POSIX mode check (no group/other bits), None on Windows."""
    if IS_WINDOWS:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return not (st.st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def others_can_read(path: Path | str) -> bool | None:
    """Can anybody except the owner (and the system) read ``path``?  True: refuse it.
    False: private. None: cannot tell on this platform / filesystem."""
    if not IS_WINDOWS:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return bool(st.st_mode & (stat.S_IRWXG | stat.S_IRWXO))
    sids = _dacl_sids(str(path))
    if sids is None:
        return None
    for sid in sids:
        if sid in OTHER_SIDS or sid.endswith(OTHER_SID_SUFFIXES):
            return True
    return False


def describe() -> str:
    """One line for docs and logs: how privacy is decided here."""
    return ("the file's ACL grants nobody but you (and the system) access" if IS_WINDOWS
            else "the file's mode has no group or other bits (0600)")


# --------------------------------------------------------------------------- #
# Windows plumbing (ctypes; never imported elsewhere)
# --------------------------------------------------------------------------- #

def current_user_sid() -> str | None:
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TOKEN_QUERY = 0x0008
        h = wintypes.HANDLE()
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
        if not adv.OpenProcessToken(k32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(h)):
            return None
        try:
            n = wintypes.DWORD()
            adv.GetTokenInformation(h, 1, None, 0, ctypes.byref(n))     # TokenUser
            buf = ctypes.create_string_buffer(n.value)
            if not adv.GetTokenInformation(h, 1, buf, n, ctypes.byref(n)):
                return None
            sid_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
            return _sid_to_string(adv, k32, sid_ptr)
        finally:
            k32.CloseHandle(h)
    except Exception:           # noqa: BLE001 - any ctypes problem means "cannot tell"
        return None


def _sid_to_string(adv, k32, sid_ptr) -> str | None:
    import ctypes
    from ctypes import wintypes
    s = wintypes.LPWSTR()
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    if not adv.ConvertSidToStringSidW(sid_ptr, ctypes.byref(s)):
        return None
    try:
        return s.value
    finally:
        k32.LocalFree.argtypes = [ctypes.c_void_p]
        k32.LocalFree(ctypes.cast(s, ctypes.c_void_p))


def _dacl_sids(path: str) -> list[str] | None:
    """SIDs with an ACCESS_ALLOWED ACE in the file's DACL; None when it cannot be read.
    A file with NO DACL (a NULL DACL grants everyone) reports Everyone."""
    try:
        import ctypes
        from ctypes import wintypes
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        class ACE_HEADER(ctypes.Structure):
            _fields_ = [("AceType", ctypes.c_ubyte), ("AceFlags", ctypes.c_ubyte), ("AceSize", ctypes.c_ushort)]
        DACL_SECURITY_INFORMATION = 0x4
        SE_FILE_OBJECT = 1
        pdacl = ctypes.c_void_p()
        psd = ctypes.c_void_p()
        adv.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        rc = adv.GetNamedSecurityInfoW(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None,
                                       ctypes.byref(pdacl), None, ctypes.byref(psd))
        if rc != 0:
            return None
        try:
            if not pdacl.value:
                return ["S-1-1-0"]
            class ACL_SIZE_INFORMATION(ctypes.Structure):
                _fields_ = [("AceCount", wintypes.DWORD), ("AclBytesInUse", wintypes.DWORD),
                            ("AclBytesFree", wintypes.DWORD)]
            info = ACL_SIZE_INFORMATION()
            adv.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD]
            if not adv.GetAclInformation(pdacl, ctypes.byref(info), ctypes.sizeof(info), 2):   # AclSizeInformation
                return None
            out: list[str] = []
            adv.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
            for i in range(info.AceCount):
                ace = ctypes.c_void_p()
                if not adv.GetAce(pdacl, i, ctypes.byref(ace)):
                    return None
                hdr = ACE_HEADER.from_address(ace.value)
                if hdr.AceType != 0:                       # ACCESS_ALLOWED_ACE_TYPE only
                    continue
                sid_ptr = ace.value + 8                    # header (4) + ACCESS_MASK (4)
                sid = _sid_to_string(adv, k32, ctypes.c_void_p(sid_ptr))
                if sid:
                    out.append(sid)
            return out
        finally:
            k32.LocalFree.argtypes = [ctypes.c_void_p]
            k32.LocalFree(psd)
    except Exception:           # noqa: BLE001
        return None
