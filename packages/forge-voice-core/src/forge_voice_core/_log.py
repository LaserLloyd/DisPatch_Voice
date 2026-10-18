"""Logger names.

Every module in the package logs through one logger family so a host decides where the voice
logs go. The default family is ``forge_voice``. A host that already routes a logger name (DisPatch
routes ``dispatch.voice``) calls :func:`set_name` BEFORE importing the engines, and its logs stay
where they were. Child loggers whose historical name is not ``<family>.<key>`` are mapped in
:data:`OVERRIDES` (DisPatch maps ``opus`` and ``websockets`` to the names it always used).

Stdlib only.
"""
from __future__ import annotations

import logging

NAME = "forge_voice"
OVERRIDES: dict[str, str] = {}


def set_name(name: str, overrides: dict[str, str] | None = None) -> None:
    """Rename the logger family (call before importing the engine modules; they bind at import)."""
    global NAME
    NAME = name
    if overrides is not None:
        OVERRIDES.clear()
        OVERRIDES.update(overrides)


def get(key: str | None = None) -> logging.Logger:
    """The family logger, or its child ``key`` (``OVERRIDES[key]`` when the host names it)."""
    if key is None:
        return logging.getLogger(NAME)
    return logging.getLogger(OVERRIDES.get(key, f"{NAME}.{key}"))
