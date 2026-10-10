"""The low-bandwidth e2e on the `baseline` profile, as CI runs it (SPEC §12: "the baseline shaper
profile"): reference v3 server + shaper + the modelled phone client, Opus both ways, a short
reply. The full seven-profile matrix is `tests/e2e_lowbw.py` (about 15 minutes)."""
import e2e_lowbw as E
import pytest

from dispatch_voice.transport import opus


@pytest.mark.skipif(not opus.available(), reason="libopus not installed")
async def test_baseline_profile_end_to_end(tmp_path, port_pair):
    server_port, proxy_port = port_pair
    proc = E.start_refserver(server_port, tmp_path / "state", "short", "fake", tmp_path / "refserver.log")
    try:
        r = await E.run_one("baseline", up="opus", down="opus", compression="deflate", server_port=server_port,
                            proxy_port=proxy_port, thread="t-ci", server_http=f"http://127.0.0.1:{server_port}",
                            clip=E.load_clip(), timeout_s=60)
    finally:
        proc.terminate()
        proc.wait(10)
    assert all(r["assertions"].values()), (r["assertions"], E.summary_line(r))
    assert r["reply_sentences"] == 3 and r["paths"] == {"server": 3, "phone": 0}
    assert r["rows"][0]["voiced_s"] == pytest.approx(r["clip_voiced_s"], abs=0.2)
    # the wire really was v3 Opus: ~16 kbps up, ~24 kbps down (+ framing), nothing near PCM's 256/384
    assert r["bytes"]["up"] < 12_000
    assert r["latency_s"]["eot_to_first_say"] < 2.0
