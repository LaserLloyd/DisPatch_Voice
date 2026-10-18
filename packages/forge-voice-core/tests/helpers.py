"""Small test helpers."""
from forge_voice_core import fsperm


def assert_private(path, *, directory: bool = False) -> None:
    """POSIX: exactly 0600 (0700 for a directory). Windows: the mode bits mean nothing there
    (always 0o666/0o777), so the ACL is checked instead: nobody but the owner and the system
    may be granted access. A filesystem whose ACL cannot be read is skipped, not failed."""
    import os
    import stat
    if fsperm.IS_WINDOWS:
        got = fsperm.others_can_read(path)
        if got is None:
            import pytest
            pytest.skip("cannot read the ACL on this filesystem")
        assert got is False, f"{path} grants access to other users"
        return
    want = 0o700 if directory else 0o600
    assert stat.S_IMODE(os.stat(path).st_mode) == want, oct(stat.S_IMODE(os.stat(path).st_mode))
