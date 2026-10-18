"""The provider catalog as the host builds it."""
from __future__ import annotations

import json

from dispatch_voice.host import providers as P


def test_the_builtin_catalog_lists_cpu_rig_and_openai(host_env):
    from dispatch_voice.host.config import load_settings
    cat = P.REGISTRY.catalog(load_settings())
    ids = {(d["kind"], d["id"]) for d in cat}
    assert {("stt", "cpu"), ("stt", "rig"), ("stt", "openai"), ("tts", "cpu"), ("tts", "rig"),
            ("tts", "openai"), ("vad", "cpu")} <= ids
    assert all(d["health"]["state"] in P.STATES and "capabilities" in d for d in cat)
    json.dumps(cat)
