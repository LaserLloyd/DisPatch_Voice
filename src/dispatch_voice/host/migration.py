"""One-shot move of ``voice.yaml`` into a settings store.

DisPatch is gaining a settings store (a key/value store with namespaces). When it lands, the
host calls :func:`migrate_voice_yaml` once at start with a ``store_writer(key, value)`` that
writes under the add-on's ``voice.*`` namespace. Nothing in this package calls it yet, and
nothing here knows what the store is: the only contract is the callable.

* **Keys** are the schema ids under ``voice.``: ``voice.speed``, ``voice.rt.power_hold``,
  ``voice.rig.url``, ``voice.bots`` (a whole map). 2.1 names (``kokoro_speed``) arrive under
  their canonical id, and aliases (a top-level ``power_hold``) under the one definition.
* **Dry run is the default**: nothing is written, the report lists every ``(key, value)`` that
  would be, and the keys the file has that the schema refuses (``dropped``).
* **Idempotent**: after a real run a marker file ``voice.yaml.migrated`` records it and a second
  call reports ``already_migrated`` without writing (``force=True`` runs it again).
* **Reversible**: the file is copied to ``voice.yaml.bak`` first and is NOT deleted. The add-on
  keeps reading ``voice.yaml`` as the lowest-priority source for one release (environment >
  store > voice.yaml > default).
* **Secrets** are never in voice.yaml (the rig token is a credential file, not a setting), so
  none are moved here.
* A ``store_writer`` that raises stops the run: no marker is written, the error is in the report,
  and a retry starts over (writing the same values again is harmless).
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import settings_schema as schema

log = logging.getLogger("dispatch.voice")

NAMESPACE = "voice"
MARKER_SUFFIX = ".migrated"
BACKUP_SUFFIX = ".bak"

#: Top-level keys of voice.yaml that are written as one value (a map), not flattened.
WHOLE_VALUE = frozenset({"bots"})


def flatten(prefs: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The cleaned tree as ``[(voice.<id>, value)]``, sorted by key; ``rt`` / ``rig`` flattened one
    level, ``bots`` and lists kept whole."""
    out: list[tuple[str, Any]] = []
    for k, v in prefs.items():
        if k in schema.TREES and isinstance(v, Mapping):
            for k2, v2 in v.items():
                out.append((f"{NAMESPACE}.{k}.{k2}", v2))
        else:
            out.append((f"{NAMESPACE}.{k}", v))
    return sorted(out, key=lambda kv: kv[0])


def _kept(cleaned: Mapping[str, Any], name: str) -> bool:
    """Did the file key ``name`` (dotted, any alias or 2.1 name) survive into ``cleaned``?"""
    if name == "bots":
        return "bots" in cleaned
    legacy = schema.LEGACY_FILE_KEYS.get(name)
    s = schema.BY_NAME.get(legacy or name)
    if s is None:
        return False
    return schema._dig(cleaned, s.parts) is not schema.MISSING


def _dropped(raw: Any, cleaned: Mapping[str, Any]) -> list[str]:
    """The names in the file that did not make it: unknown keys, and known keys whose value the
    schema refused (a wrong type, out of range). Names only, never values."""
    if not isinstance(raw, Mapping):
        return []
    lost: list[str] = []
    for k, v in raw.items():
        if k in schema.TREES:
            if isinstance(v, Mapping):
                lost += [f"{k}.{k2}" for k2 in v if not _kept(cleaned, f"{k}.{k2}")]
            else:
                lost.append(str(k))
        elif not _kept(cleaned, str(k)):
            lost.append(str(k))
    return sorted(lost)


def migrate_voice_yaml(store_writer: Callable[[str, Any], None], dry_run: bool = True, *,
                       path: Path | str | None = None, force: bool = False) -> dict:
    """Move ``voice.yaml`` into the store through ``store_writer(key, value)``.

    Returns ``{"dry_run", "source", "items": [{"key", "value"}], "dropped": [...], "written": n,
    "backup": path|None, "marker": path|None, "status": ...}`` where ``status`` is one of
    ``nothing_to_migrate`` (no file or an empty one), ``already_migrated``, ``dry_run``,
    ``migrated`` or ``error`` (``error`` carries the reason, never a value)."""
    if path is None:
        from .config import prefs_path
        path = prefs_path()
    p = Path(path)
    marker = p.with_name(p.name + MARKER_SUFFIX)
    report: dict = {"dry_run": dry_run, "source": p.name, "items": [], "dropped": [], "written": 0,
                    "backup": None, "marker": None, "status": "nothing_to_migrate"}
    if not p.exists():
        return report
    if marker.exists() and not force:
        report["status"] = "already_migrated"
        report["marker"] = marker.name
        return report
    try:
        import yaml
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:      # noqa: BLE001 - reported, not raised
        report.update(status="error", error=f"voice.yaml could not be read ({type(e).__name__})")
        return report
    cleaned = schema.clean_prefs(raw)
    items = flatten(cleaned)
    report["items"] = [{"key": k, "value": v} for k, v in items]
    report["dropped"] = _dropped(raw, cleaned)
    if not items:
        return report
    if dry_run:
        report["status"] = "dry_run"
        return report
    backup = p.with_name(p.name + BACKUP_SUFFIX)
    try:
        for k, v in items:
            store_writer(k, v)
            report["written"] += 1
        shutil.copy2(p, backup)
        marker.write_text(json.dumps({"at": int(time.time()), "keys": len(items)}), encoding="utf-8")
    except Exception as e:      # noqa: BLE001
        log.warning("voice: voice.yaml migration stopped after %d key(s) (%s)", report["written"], type(e).__name__)
        report.update(status="error", error=f"{type(e).__name__} after {report['written']} key(s)")
        return report
    report.update(status="migrated", backup=backup.name, marker=marker.name)
    log.info("voice: voice.yaml moved into the settings store (%d keys; backup %s kept)", len(items), backup.name)
    return report
