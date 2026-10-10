# Voice chat on a slow connection

Voice chat is meant to work from a phone on cellular, reached over Tailscale: weak
signal, a relay in the path, tunnels and handovers. This page covers the server side of
that: the v3 wire, Opus, sample offsets, the commit cursor, the link state and the test
harness that drives all of it through a simulated bad network. The browser half (codecs,
link monitor, phone voice, uplink ring) is in [audio-modules.md](audio-modules.md); the
normative frame list is [protocol.md](protocol.md).

The code lives in `src/dispatch_voice/transport/`. It is pure apart from libopus, never
imports `host/` or DisPatch, and the session is its only caller.

| Module | What it does |
|---|---|
| `transport/__init__.py` | `Transport`: negotiation, the session's only send path, uplink decode |
| `transport/wire3.py` | the v3 binary framing, both directions (pure) |
| `transport/opus.py` | a ctypes binding to the system libopus, fail-soft |
| `transport/offsets.py` | `OffsetTracker`: gaps, overlaps and the zero-fill budget |
| `transport/cursor.py` | `CommitCursor`: what has been submitted, persisted across restarts |
| `transport/link.py` | `LinkState`, probes, the Tailscale relay indicator |

## What changes on the wire

A client that asks for it (`?v=3&up=opus,pcm16&down=opus,pcm16&s=<stream>`) gets protocol 3;
anything else gets exactly the v2 wire it had before (a JSON `audio` header and raw PCM).

| | v2 (today) | v3, Opus | v3, no Opus |
|---|---|---|---|
| Uplink | 640 B PCM every 20 ms, ≈ 313 kbps on the wire | 5 × 20 ms Opus packets per 100 ms message at 16 kbps, ≈ 28 kbps | PCM with offsets, no JSON |
| Downlink | 157 B JSON header + PCM per 40 ms, ≈ 470 kbps | up to 10 packets per message at 24 kbps, ≈ 30 kbps | PCM, one 40 ms chunk per message |
| Text | — | a `say` frame (120–230 B) before every sentence | same |

Every sentence is announced by its `say` frame first, so the phone can always fall back
to reading it aloud itself (phone voice) or showing it, whatever happens to the audio.

### Opus

`opus.py` loads the system libopus at run time (`ctypes`, like `soundfile` loads
libsndfile); nothing is compiled or bundled. libopus is BSD-3-Clause (Xiph.Org and
contributors). When it is missing, or `DISPATCH_VOICE_OPUS=off` / `voice.yaml: opus: off`,
`opus.available()` is False and both directions negotiate `pcm16`; nothing else changes.

Measured on this box (libopus 1.6): encoding a 20 ms frame takes 0.28 ms at 24 kHz /
24 kbps and 0.22 ms at 16 kHz / 16 kbps; decoding takes 0.03 to 0.06 ms. Packets
average 59 B at 24 kbps, 40 B at 16 kbps and 29 B at 12 kbps. Parakeet transcribes the
Opus uplink exactly like PCM. Tested on the seven speech fixture clips at 16 and 12 kbps,
through the real decode and offset path: word error rate 0.0 against the PCM transcript
(`tests/transport/test_models_opus.py`).

Downlink details:

- Each sentence is its own Opus stream (a fresh encoder at the current bitrate rung), so a
  rung change lands at the next sentence by construction, and a client may reset its
  decoder per `seq`.
- The first message of a sentence goes out at once with whatever packets exist, so first
  audio is never delayed. After that, packets go out in full 10-packet messages, and a
  partial batch waits at most 100 ms.
- Ending a sentence (`end_sentence`, or `last=True`) pads the last partial frame plus the
  encoder lookahead (6.5 ms) with silence, so the tail is never lost. It then sends
  `audio_end {gen, seq, kind}`.
- The binary `t` is the transport's clock at the moment the message is written. The link
  monitor's test needs send time: if `arrival − t` keeps growing, the link is slow; if it
  stays flat, the server's TTS is.

## Offsets: gaps, replays and a budget

Every v3 uplink message carries `off`, the 16 kHz sample index of its first sample. The
mic gate leaves holes while nobody talks, and a reconnect replays audio, so
`OffsetTracker.feed(off, pcm)` turns the messages back into the continuous signal the turn
detector needs:

- **In order:** the audio passes through as it is.
- **A gap of up to 5 s:** that many zeros are fed first. The end-of-turn logic has to
  *hear* the silence; 5 s costs at most 75 ms of Silero.
- **A larger gap, a jump back of more than 60 s, or a gap the budget cannot pay for:**
  `GapTooLarge`. The session resets the detector and feeds `err.frames`; the tracker has
  already re-anchored.
- **An overlap (a replay of audio already fed):** the overlapping samples are dropped.
- **The budget:** zero-fill is a token bucket refilled at twice realtime and holding at
  most 10 s. A client whose `off` races ahead gets resets, never 50× realtime of synthetic
  silence fed into Silero.
- **`rx_off`:** the end of the furthest message received. The server sends it in every
  `pong` so the client can measure its uplink backlog.

## The commit cursor: no lost and no doubled turns

`CommitCursor` records, per `(slot, stream)`, the offset where the last submitted
utterance ended, and keeps the transcript alongside it. It lives in
`ctx.paths.state_dir/cursor.sqlite`:

- WAL mode, one row write per submit (well under 1 ms).
- Mode 0600 inside the 0700 state dir.
- 256 entries, LRU; 10 minutes TTL.
- It is a file because every backend deploy restarts the process, and a restart between
  `transcript` and the next `pong` must not submit an utterance twice.
- An unreadable file is moved aside. An unwritable directory falls back to memory with one
  log line.

On reconnect the client sends `start.cursor {stream, from}` and the server answers
`cursor {from}` with `resume_from(key, client_from)`:

- **Normally:** `max(client_from, committed)`. The client replays its ring from there.
  Anything already submitted is never replayed, and a transcript the client never saw is
  sent again with `replayed: true`.
- **An utterance in progress (Amendment T6-1):** the session marks the start of an
  utterance when the detector starts hearing it (`mark_open`) and unmarks it when it is
  discarded (`clear_open`); a submit clears it too. If the socket drops mid-utterance,
  the answer is that utterance's start, even when that is below the client's `from`.
  The old session's half-heard audio died with it, so the new session must hear the whole
  utterance again.
  - Measured on `blackout-20s`: without the mark, the one row carried 1.26 s of the clip's
    1.88 s of speech.
  - With the mark, it carried 1.82 s, the same as with no blackout.

## The link: text first, switches at sentence boundaries

The client decides when the link is bad and sends `link {mode, down_kbps?, up_kbps?, why}`.
`LinkState` validates the request (unknown values fall back, silently) and holds it until
the next sentence: a sentence already on its way finishes on its path, so the voice never
changes mid-sentence.

- `mode: "text"` means the server sends `say` and never calls the TTS. That frees the CPU
  for everyone else.
- `down_kbps` changes the rung for the next sentence's encoder.
- `up_kbps` is recorded only; the client reconfigures its own encoder.
- `down_kbps: null` or `why: "decoder"` means the client's Opus decoder failed. The
  downlink is PCM from the next sentence on.
- `start.slow_link: "always_phone"` forces text mode for the whole socket.
- `probe {bytes}` gets one binary message of up to 32 KB of `os.urandom` bytes;
  permessage-deflate cannot shrink it. One probe is allowed per 20 s.

### Relay indicator

The Voice lab shows whether the phone reaches this box directly or through a Tailscale
relay. `RelayIndicator.lookup(addr)` builds the answer as follows:

- It reads `tailscale status --json` (fixed argv, cached for 30 s).
- Behind Tailscale Serve the socket peer is loopback, and the phone's address is the last
  `X-Forwarded-For` entry. That header is read only when the peer is loopback.
- An idle peer has no current path, so the answer stays `idle` until traffic has flowed,
  or until one `tailscale ping -c 1` probe has run (at most one per peer per 30 s). It
  never reads as `relay` just because the status names a DERP region.
- The answer is one of `direct`, `relay` (plus the DERP region code), `peer_relay`,
  `idle`, `local`, `unknown` or `unavailable`. It never contains an address.

## For the session (T3): where each call goes

```python
tr = Transport.negotiate(ws, ws.query_params, opus_mode=cfg.opus, opus_down_kbps=cfg.opus_down_kbps,
                         opus_up_kbps=cfg.opus_up_kbps, rate_out=tts.sample_rate,
                         hard_silence_s=cfg.hard_silence_s, hold_max_s=cfg.rt.hold_max_s)
ready = {..., **tr.ready_fields()}                 # proto, codecs, rungs, rx_off, gate, say
# receive loop
if bytes: if Transport.too_big(b): bad_frame (counted, rate-limited); continue   # BEFORE queueing
          queue it (byte-bounded, 64,000 B)
# audio worker
got = tr.decode_up(b)                              # None -> v2 raw PCM; ValueError -> bad_frame
off, flags, pcm = got
try:    parts = tr.offsets.feed(off, pcm)
except GapTooLarge as e: detector.reset(); cursor.clear_open(key); parts = e.frames
# the cursor key: (device or session, tr.stream). tr.stream is None on v2 and when the
# client's `s` failed the check: then there is NO cursor (skip every cursor call below);
# never fall back to a shared key, or one client's submit swallows another's utterance
key = (slot_id, tr.stream) if tr.stream is not None else None
# detector events
SpeechStart -> cursor.mark_open(key, start_off)    Discarded -> cursor.clear_open(key)
submit      -> cursor.set(key, upto, meta={"text": ..., "message_id": ...})
# speaking one sentence
tr.at_sentence_boundary(); seq += 1
await tr.send_say({...})                           # v3 only; returns False on v2
if not tr.link.text_mode:
    for chunk in tts: await tr.send_audio(kind=..., gen=..., seq=..., pcm16=chunk, t_ms=tr.now_ms(),
                                          text=words if n == 0 else None, message_id=mid, role=role, chunk=n)
    await tr.end_sentence(kind=..., gen=..., seq=...)    # tail + audio_end
# end_sentence(N) BEFORE send_say(N+1): a say ends the open sentence, and a late chunk of
# an ended sentence is dropped (stats["late_audio_dropped"], one warning), never re-opened
# barge-in:   await tr.drop_older_gens(new_gen)    # later audio of older gens is dropped too
# controls:   ping -> tr.send(tr.pong(msg.get("t")))   # pong() echoes only a finite number
#             link -> tr.link.request(msg); probe -> tr.send_probe(n)
#             start.slow_link -> tr.link.set_slow_link(v); start.cursor -> send cursor {from: cursor.resume_from(...)}
# metrics.link: tr.link_metrics(await relay.lookup(peer_address(ws.client.host, ws.headers)))
# on close:   speaking.cancel(); await asyncio.gather(speaking, return_exceptions=True)
#             await tr.close()     # idempotent; waits (<= 2 s) for an encode in a worker thread
```

Two rules about lifetime and the receive loop:

- `tr.close()` takes the audio lock before it frees the Opus codecs, and the codecs
  carry their own locks, so a socket that drops while a chunk is being encoded in a
  worker thread can no longer corrupt the process's heap (review-T6 H1). Still await
  the cancelled speaking task first: it ends cleanly instead of finding a closed
  transport mid-send.
- A stalled downlink holds the transport's send lock. Do not await `send_probe`,
  `pong` or any other transport send inline in the receive loop where it could delay
  `drop_older_gens` on a barge-in; hand them to a task.

`tests/transport/refserver.py` is a reference session written exactly this way, around
deliberately simple stand-ins (an energy detector, a fake STT, a tone TTS). Read it as a
worked example.

## The harness

`tests/shaper.py` is a root-free TCP proxy that models a thin, slow, stalling pipe. It
shapes capacity with a per-packet overhead (134 B: TCP/IP, TLS and WireGuard), one-way
delay with jitter, and a modem-sized queue that pushes back like TCP. It supports
`stall(s)`, `cut()` and `blackout(s)`; during a blackout, new connections are refused and
what a closed socket had queued is lost. CDP throttling is not used: it let a WebSocket
run at 19.6 Mbps under a 250 kbps setting. Its profiles are the single source of the e2e
matrix:

| Profile | Up / down kbps | One-way delay | Event |
|---|---|---|---|
| baseline | — | 0 | none (the harness itself) |
| lte-edge | 600 / 1500 | 125 ± 20 ms | none |
| derp-busy | 1000 / 1000 | 75 ± 50 ms | none |
| 3g-weak | 250 / 250 | 250 ± 40 ms | none |
| edge | 120 / 120 | 450 ± 80 ms | none |
| blackout-20s | lte-edge | | 20 s without signal, 1.2 s into the utterance |
| stall-5s | lte-edge | | a 5 s stall 4.5 s into the reply, after which the old path is gone: the client rejoins while the reply still streams |

`tests/e2e_lowbw.py` is the wire client that plays the phone. It models:

- a gated mic with offsets and Opus or PCM uplink;
- the 30 s ring with `cursor` replay;
- the gap-free player that reports `playback_done`;
- seq claims keyed `(sid, seq)`;
- the link monitor and the phone voice;
- 2 s pings with the dead-socket rule;
- reconnect with backoff and an immediate try on `online`;
- `after_partial`, counting sentences whose audio is complete.

By default it runs against the reference server and reads the outcome from that
server's `/rows` debug view. The `dispatch` judge reads only what DisPatch exposes:
it unlocks with the PIN (`--pin-file` or `DISPATCH_PIN`, never on the command line),
sends the `lc_session` cookie on the socket (without it the gate closes the socket
1008 before accept), makes one thread per run (`--bot`) and reads
`/api/threads/{id}/messages`. There, "the whole utterance" is the user row's text
against the clip's known words (WER at most 0.15, and the first word present, which
is what a cut-off head loses), and "every sentence once, in order" is the heard `say`
texts spelling exactly the assistant row's words after the sentence splitter and
`normalise`. The mid-reply rejoin is judged by the phone: the drop came mid-reply and
the reply's sentences came from two sessions.

- `--ref-dispatch`: the reference server in its DisPatch-shaped mode (PIN, cookie
  gate, threads and messages API, an oracle STT that loses the head in proportion to
  the voiced audio lost), judged like DisPatch. This proves the judge.
- `--server ws://127.0.0.1:PORT --bot ID`: a running DisPatch with the add-on.
- `--start-dispatch BACKEND`: starts DisPatch from that checkout with scratch data,
  one API bot on the stub LLM's long reply and a scratch PIN, then runs the matrix.
  `--kokoro DIR` also overlays the Kokoro plugin exactly like the planned live unit
  (SPEC §11 step 2: entry-point discovery, `voice.yaml: enabled: true`, the unit's
  uvicorn flags); `--online` lets `uv` resolve the overlay (the default is `--offline`).
  `--bot-kind advisor` makes the bot an advisor streamed as the stub LLM writes it (so a
  mid-reply rejoin meets an open stream: the late-joiner case; the default API bot's reply is
  stored before a ~2.5 s simulated stream); `--power-hold off|performance` (default `off`: a
  network run must not hold the box's performance profile); `--llm-port` pins the stub LLM's
  port. The client sends `start.after` (the user row the cut reply answers) on a reconnect, as
  drive.js does; `--no-after` models a client that does not. `--fixed-reply TEXT|@FILE|split`
  makes the stub LLM answer with that reply (`split`: one whose first sentence is cut at its
  comma while it streams — the resume must still skip exactly what was heard). A socket that
  never gets `ready` is a failed connection, retried with backoff like a refused one; a
  `stall-5s` run whose whole reply arrived before the stall did not exercise the rejoin and is
  reported "N/E" (not exercised: re-run it), never as a pass or a failure.

Its assertions:

- **Every run:** exactly one user row, carrying the whole utterance; every reply sentence
  heard in full exactly once, in order; the transcript reached the phone.
- **3g-weak with a PCM downlink:** text mode was entered.
- **edge:** the first `say` arrived within `eot` + 6 s.
- **blackout-20s:** the phone reconnected, and there is still one full row.
- **stall-5s:** the server resumed the reply while the agent's stream was still open, and
  only unplayed sentences followed.

One known window is not tested (review-T6 L3, accepted): `audio_end` is a text frame
sent after a sentence's last binary message. If the socket dies exactly between the
two, the phone has all of that sentence's audio but no end, so `after_partial` does
not count it and the sentence is sent again and heard twice. The window is one
socket write wide; closing it would need the end flag inside the binary frame, a wire
change not worth making for it.

```sh
uv run python tests/e2e_lowbw.py                         # the matrix (about 15 minutes)
uv run python tests/e2e_lowbw.py edge --compression raw  # one profile
python tests/shaper.py --profile edge --listen 19056 --target 19055   # for a browser run; stdin: stall N | cut | blackout N | stats
```

CI runs only the `baseline` profile (`tests/transport/test_e2e_baseline.py`, about 10 s).

### Results on this box (reference server, 2026-10-10)

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | opus/opus | deflate | 0.584 s | 1.201 s | 2.1 s | 10 / 0 of 10 | audio | 0 | 0 | 1 / 40 | 9 / 112 KB | pass |
| baseline | opus/opus | raw | 0.583 s | 1.201 s | 2.095 s | 10 / 0 of 10 | audio | 0 | 0 | 1 / 2 | 9 / 112 KB | pass |
| lte-edge | opus/opus | deflate | 0.861 s | 1.196 s | 2.374 s | 10 / 0 of 10 | audio | 0 | 0 | 272 / 290 | 9 / 112 KB | pass |
| lte-edge | opus/opus | raw | 0.869 s | 1.196 s | 2.381 s | 10 / 0 of 10 | audio | 0 | 0 | 273 / 295 | 9 / 112 KB | pass |
| derp-busy | opus/opus | deflate | 0.8 s | 1.174 s | 2.323 s | 10 / 0 of 10 | audio | 0 | 0 | 214 / 245 | 9 / 112 KB | pass |
| derp-busy | opus/opus | raw | 0.781 s | 1.228 s | 2.309 s | 10 / 0 of 10 | audio | 0 | 0 | 204 / 263 | 9 / 112 KB | pass |
| 3g-weak | opus/opus | deflate | 1.122 s | 1.219 s | 2.661 s | 10 / 0 of 10 | audio | 0 | 0 | 553 / 583 | 9 / 112 KB | pass |
| 3g-weak | opus/opus | raw | 1.128 s | 1.198 s | 2.672 s | 10 / 0 of 10 | audio | 0 | 0 | 553 / 579 | 9 / 112 KB | pass |
| edge | opus/opus | deflate | 1.6 s | 1.222 s | 3.176 s | 10 / 0 of 10 | text | 0 | 0 | 1023 / 1090 | 9 / 112 KB | pass |
| edge | opus/opus | raw | 1.57 s | 1.234 s | — (phone voice) | 0 / 10 of 10 | text | 0 | 0 | 1019 / 1063 | 9 / 3 KB | pass |
| blackout-20s | opus/opus | deflate | 19.904 s | 1.198 s | — (phone voice) | 0 / 10 of 10 | text | 1 | 0 | 275 / 292 | 19 / 3 KB | pass |
| blackout-20s | opus/opus | raw | 19.942 s | 1.2 s | — (phone voice) | 0 / 10 of 10 | text | 1 | 0 | 279 / 293 | 19 / 3 KB | pass |
| stall-5s | opus/opus | deflate | 0.871 s | 1.196 s | 2.385 s | 1 / 9 of 10 | text | 1 | 0 | 276 / 5268 | 9 / 25 KB | pass |
| stall-5s | opus/opus | raw | 0.858 s | 1.214 s | 2.392 s | 10 / 0 of 10 | audio | 1 | 0 | 278 / 320 | 9 / 121 KB | pass |
| 3g-weak | opus/pcm16 | deflate | 1.145 s | 1.207 s | 2.749 s | 1 / 9 of 10 | text | 0 | 84 | 558 / 5014 | 9 / 200 KB | pass |
| 3g-weak | opus/pcm16 | raw | 1.121 s | 1.201 s | 2.73 s | 1 / 9 of 10 | text | 0 | 89 | 559 / 5524 | 9 / 200 KB | pass |
| edge | opus/pcm16 | deflate | 1.576 s | 1.159 s | 3.191 s | 0 / 10 of 10 | text | 1 | 0 | 1010 / 2560 | 10 / 154 KB | pass |
| edge | opus/pcm16 | raw | 1.579 s | 1.243 s | 3.284 s | 0 / 10 of 10 | text | 1 | 0 | 1022 / 2582 | 10 / 149 KB | pass |

Every run passed. Reading the table:

- **Opus fits every shaped link.** On lte-edge, derp-busy and 3g-weak the reply streamed in
  server audio with no underrun. The bytes are about 9 KB up and 112 KB down for a 2.4 s
  question and a 37 s answer.
- **A PCM downlink on 3g-weak underruns** (84–89 underruns). The link monitor switches to
  the phone voice after the first sentence.
- **edge:** its RTT (≈ 1 s with a 450 ms one-way delay) is on the 1 s threshold, so the
  monitor goes to the phone voice there too, either before the reply or at its end. With a
  PCM downlink, the backlog also trips the dead-socket rule once; the reconnect costs
  nothing.
- **blackout-20s:** the reconnect replays the utterance from its start (Amendment T6-1).
  The one row carries 1.82 s of the clip's 1.88 s of speech, as at baseline. The pongs
  were missing for 20 s, so the reply came in the phone voice.
- **stall-5s:** the server resumed from sentence 1 while the agent's stream was still open.
  Sentence 2, cut by the drop, came again from its start. In the deflate run, the
  5 s-late pong also moved it to the phone voice.
- **These numbers measure the transport and the client model, not the speech engines.**
  The reference server's TTS is a tone at 0.6× realtime and its STT is a fake. The
  end-of-turn and first-audio times of the real pipeline are T4's latency harness.

### Results against the real session (T3 slice 4, scratch DisPatch, 2026-10-10)

`e2e_lowbw.py --start-dispatch <dispatch-chat>/backend --kokoro <dispatch-voice-kokoro>` — the
real DisPatch 2.2.0 branch with the add-on and the Kokoro plugin as the planned live unit runs
them (`--offline`, `--ws-ping-timeout 60 --ws-max-size 262144`, `rt.power_hold: off`), real
Silero, Smart Turn and Parakeet, Kokoro speaking, the stub LLM's long reply (12 sentences,
about 100 words), the `dispatch` judge. Under the build's 10-thread CPU cap and `nice 15`, so the
times are pessimistic. One API bot (the reply is stored, then streamed in about 2.5 s):

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | opus/opus | deflate | 0.361 s | 1.277 s | 2.163 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 2 / 73 | 9 / 87 KB | pass |
| baseline | opus/opus | raw | 0.328 s | 1.308 s | 2.26 s | 12 / 0 of 12 (101/101 words) | audio | 0 | 0 | 1 / 41 | 9 / 89 KB | pass |
| lte-edge | opus/opus | deflate | 0.619 s | 1.293 s | 2.74 s | 12 / 0 of 12 (103/103 words) | audio | 0 | 0 | 276 / 319 | 9 / 92 KB | pass |
| lte-edge | opus/opus | raw | 0.47 s | 1.284 s | 2.373 s | 12 / 0 of 12 (102/102 words) | audio | 0 | 0 | 283 / 317 | 9 / 88 KB | pass |
| derp-busy | opus/opus | deflate | 0.507 s | 1.291 s | 2.343 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 202 / 241 | 9 / 88 KB | pass |
| derp-busy | opus/opus | raw | 0.361 s | 1.306 s | 2.181 s | 12 / 0 of 12 (99/99 words) | audio | 0 | 0 | 211 / 250 | 9 / 87 KB | pass |
| 3g-weak | opus/opus | deflate | 0.886 s | 1.348 s | 2.811 s | 12 / 0 of 12 (104/104 words) | audio | 0 | 0 | 566 / 858 | 9 / 90 KB | pass |
| 3g-weak | opus/opus | raw | 0.86 s | 1.296 s | 2.733 s | 12 / 0 of 12 (101/101 words) | audio | 0 | 0 | 556 / 950 | 9 / 87 KB | pass |
| edge | opus/opus | deflate | 1.146 s | 1.28 s | — (phone voice) | 0 / 12 of 12 (98/98 words) | text | 0 | 0 | 993 / 1061 | 9 / 4 KB | pass |
| edge | opus/opus | raw | 1.34 s | 1.264 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 0 | 0 | 1015 / 1098 | 9 / 4 KB | pass |
| blackout-20s | opus/opus | deflate | 20.011 s | 1.53 s | — (phone voice) | 0 / 12 of 12 (104/104 words) | text | 1 | 0 | 277 / 305 | 19 / 4 KB | pass |
| blackout-20s | opus/opus | raw | 20.056 s | 1.593 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 1 | 0 | 280 / 319 | 19 / 4 KB | pass |
| stall-5s | opus/opus | deflate | 0.611 s | 1.303 s | 2.507 s | 3 / 9 of 12 (101/101 words) | text | 1 | 0 | 276 / 317 | 9 / 34 KB | pass |
| stall-5s | opus/opus | raw | 0.627 s | 1.234 s | 2.371 s | 12 / 0 of 12 (103/103 words) | text | 1 | 0 | 278 / 327 | 9 / 96 KB | pass |
| 3g-weak | opus/pcm16 | deflate | 0.893 s | 1.251 s | 2.774 s | 1 / 11 of 12 (100/100 words) | text | 0 | 53 | 565 / 9626 | 9 / 362 KB | pass |
| 3g-weak | opus/pcm16 | raw | 0.919 s | 1.257 s | 2.842 s | 1 / 11 of 12 (101/101 words) | text | 0 | 68 | 568 / 12511 | 9 / 389 KB | pass |
| edge | opus/pcm16 | deflate | 1.39 s | 1.244 s | — (phone voice) | 0 / 12 of 12 (103/103 words) | text | 0 | 0 | 1026 / 1065 | 9 / 4 KB | pass |
| edge | opus/pcm16 | raw | 1.307 s | 1.329 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 0 | 0 | 1003 / 1053 | 9 / 4 KB | pass |

The open-stream late joiner — an advisor bot, streamed as the stub writes it (8 words/s), so the
`stall-5s` rejoin meets the reply STILL STREAMING (`--bot-kind advisor`; the rejoin came 1.3–1.5 s
before the stream ended, `after_partial` named the provisional `run:` id):

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stall-5s | opus/opus | deflate | 0.61 s | 2.354 s | 13.571 s | 0 / 12 of 12 (100/100 words) | text | 1 | 0 | 274 / 1881 | 9 / 11 KB | pass |
| stall-5s | opus/opus | raw | 0.619 s | 2.588 s | 3.755 s | 4 / 8 of 12 (101/101 words) | text | 1 | 0 | 280 / 322 | 9 / 43 KB | pass |
| blackout-20s | opus/opus | deflate | 20.018 s | 3.091 s | — (phone voice) | 0 / 13 of 12 (103/103 words) | text | 1 | 0 | 278 / 312 | 19 / 4 KB | pass |
| blackout-20s | opus/opus | raw | 20.072 s | 2.538 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 1 | 0 | 278 / 324 | 19 / 4 KB | pass |
| baseline | opus/opus | deflate | 0.314 s | 2.474 s | 3.367 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 2 / 42 | 9 / 88 KB | pass |
| baseline | opus/opus | raw | 0.31 s | 2.349 s | 3.192 s | 12 / 0 of 12 (99/99 words) | audio | 0 | 0 | 2 / 42 | 9 / 87 KB | pass |

A client that sends no `after` on the reconnect (`--no-after`; the server anchors the stored reply
on the cursor's record of the last submitted utterance):

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stall-5s | opus/opus | deflate | 0.615 s | 1.265 s | 2.437 s | 4 / 8 of 12 (100/100 words) | text | 1 | 0 | 279 / 480 | 9 / 40 KB | pass |
| stall-5s | opus/opus | raw | 0.451 s | 1.269 s | 2.266 s | 4 / 8 of 12 (101/101 words) | text | 1 | 0 | 280 / 335 | 9 / 43 KB | pass |
| 3g-weak | opus/opus | deflate | 0.881 s | 1.246 s | 2.981 s | 12 / 0 of 12 (103/103 words) | audio | 0 | 0 | 561 / 755 | 9 / 92 KB | pass |
| 3g-weak | opus/opus | raw | 0.877 s | 1.249 s | 2.69 s | 12 / 0 of 12 (102/102 words) | audio | 0 | 0 | 564 / 707 | 9 / 88 KB | pass |
| 3g-weak | opus/pcm16 | deflate | 0.679 s | 1.349 s | 2.596 s | 1 / 11 of 12 (100/100 words) | text | 0 | 54 | 567 / 8812 | 9 / 342 KB | pass |
| 3g-weak | opus/pcm16 | raw | 0.786 s | 1.301 s | 2.68 s | 1 / 11 of 12 (99/99 words) | text | 0 | 56 | 563 / 11982 | 9 / 378 KB | pass |

Reading these beside the reference-server table above:

- **Before slice 4 was finished the real session failed 3 of 18**: both `stall-5s` runs and
  3g-weak with a PCM downlink (deflate) heard 87–95 of ~100 words and timed out. The API bot's
  reply is STORED before its (simulated) stream, so after the drop it could only be resumed from
  the row, which the session looked up only after `start.after` — and this harness sent none.
  The client now sends `after` like drive.js does, and the server anchors on its cursor when a
  client does not.
- **The advisor late joiner first failed too** (122 of 100 words, the reply from its first
  sentence again): the subscribe snapshot of the still-open stream was spoken before the
  session had read the `start` carrying `after_partial`. Reply frames now wait for the first
  `start` (Amendment T3s4-1).
- **3g-weak with a PCM downlink still reads RTT spikes of 9–12 s** (3.1–4.5 s after the slice-4
  review: the PCM lead is now 1.25 s, see below). PCM is 384 kbps on a
  250 kbps link: the client switches to the phone voice after the first sentence (underruns),
  but the audio already written stays in the pipe. Send-ahead pacing (4 s) bounds it; before
  pacing the deflate run's pongs were late enough to trip the dead-socket rule mid-reply. Opus
  (what every WebCodecs browser negotiates) streams the same reply in 87–92 KB with no underrun.

### Results after the review of slice 4 (T3-hub-fix, scratch DisPatch, 2026-10-11)

The same setup as above (the real DisPatch branch, the add-on and the Kokoro plugin, `--offline`,
`rt.power_hold: off`, the 10-thread cap and `nice 15`, so the times are pessimistic), on the tree
with the slice-4 review applied: canonical sentence splitting, pacing that restarts at a cut and caps
a PCM16 downlink at 1.25 s, positions from the detector, and the harness fixes (the shaper releases
what an aborted connection had queued; a socket that never gets `ready` is retried, not fatal; a
`stall-5s` run whose reply all arrived before the stall is reported "N/E", not exercised).

**The full matrix: 18/18** (twice: before and after the last follow-up change; the final run):

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | opus/opus | deflate | 0.347 s | 1.29 s | 2.142 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 1 / 66 | 9 / 88 KB | pass |
| baseline | opus/opus | raw | 0.134 s | 1.331 s | 2.075 s | 12 / 0 of 12 (101/101 words) | audio | 0 | 0 | 1 / 41 | 9 / 90 KB | pass |
| lte-edge | opus/opus | deflate | 0.623 s | 1.289 s | 2.391 s | 13 / 0 of 13 (103/103 words) | audio | 0 | 0 | 276 / 315 | 9 / 93 KB | pass |
| lte-edge | opus/opus | raw | 0.604 s | 1.255 s | 2.381 s | 12 / 0 of 12 (102/102 words) | audio | 0 | 0 | 274 / 328 | 9 / 89 KB | pass |
| derp-busy | opus/opus | deflate | 0.545 s | 1.289 s | 2.476 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 206 / 332 | 9 / 89 KB | pass |
| derp-busy | opus/opus | raw | 0.539 s | 1.302 s | 2.363 s | 12 / 0 of 12 (99/99 words) | audio | 0 | 0 | 211 / 272 | 9 / 88 KB | pass |
| 3g-weak | opus/opus | deflate | 0.881 s | 1.257 s | 2.795 s | 12 / 0 of 12 (104/104 words) | audio | 0 | 0 | 555 / 859 | 9 / 91 KB | pass |
| 3g-weak | opus/opus | raw | 0.88 s | 1.308 s | 2.757 s | 12 / 0 of 12 (101/101 words) | audio | 0 | 0 | 556 / 783 | 9 / 88 KB | pass |
| edge | opus/opus | deflate | 1.277 s | 1.331 s | — (phone voice) | 0 / 12 of 12 (98/98 words) | text | 0 | 0 | 1016 / 1081 | 9 / 4 KB | pass |
| edge | opus/opus | raw | 1.164 s | 1.342 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 0 | 0 | 1024 / 1058 | 9 / 4 KB | pass |
| blackout-20s | opus/opus | deflate | 19.897 s | 1.475 s | — (phone voice) | 0 / 12 of 12 (104/104 words) | text | 1 | 0 | 277 / 310 | 19 / 4 KB | pass |
| blackout-20s | opus/opus | raw | 19.906 s | 1.555 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 1 | 0 | 278 / 316 | 19 / 4 KB | pass |
| stall-5s | opus/opus | deflate | 0.632 s | 1.226 s | 2.453 s | 12 / 0 of 12 (101/101 words) | text | 1 | 0 | 276 / 318 | 9 / 98 KB | pass |
| stall-5s | opus/opus | raw | 0.603 s | 1.265 s | 2.437 s | 3 / 9 of 12 (103/103 words) | text | 1 | 0 | 282 / 329 | 9 / 37 KB | pass |
| 3g-weak | opus/pcm16 | deflate | 0.914 s | 1.286 s | 2.772 s | 1 / 11 of 12 (100/100 words) | text | 0 | 49 | 558 / 3082 | 9 / 155 KB | pass |
| 3g-weak | opus/pcm16 | raw | 0.905 s | 1.286 s | 2.881 s | 1 / 11 of 12 (101/101 words) | text | 0 | 73 | 552 / 4528 | 9 / 173 KB | pass |
| edge | opus/pcm16 | deflate | 1.18 s | 1.31 s | — (phone voice) | 0 / 13 of 13 (103/103 words) | text | 0 | 0 | 1022 / 1068 | 9 / 4 KB | pass |
| edge | opus/pcm16 | raw | 1.346 s | 1.263 s | — (phone voice) | 0 / 12 of 12 (102/102 words) | text | 0 | 0 | 1011 / 1080 | 9 / 4 KB | pass |

**A reply whose first sentence is clause-cut while it streams** (`--fixed-reply split`: "The dentist
is at three this afternoon, and the school call is at four thirty. …", 13 pieces, 105 words; the
review measured 98/105 words — "It should arrive on Thursday before noon." lost — in 3 of 3 rejoins).
API bot (stored, then streamed), then the advisor (an open stream at the rejoin); each after a
`baseline` warm-up run:

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stall-5s | opus/opus | deflate | 0.425 s | 1.322 s | 2.28 s | 13 / 0 of 13 (105/105 words) | text | 1 | 0 | 280 / 325 | 9 / 99 KB | pass |
| stall-5s | opus/opus | raw | 0.531 s | 1.286 s | 2.433 s | 3 / 10 of 13 (105/105 words) | text | 1 | 0 | 279 / 324 | 9 / 37 KB | pass |
| stall-5s | opus/opus | deflate | 0.466 s | 2.709 s | 3.803 s | 13 / 0 of 13 (105/105 words) | text | 1 | 0 | 280 / 322 | 9 / 99 KB | pass |
| stall-5s | opus/opus | raw | 0.6 s | 2.717 s | 3.831 s | 3 / 10 of 13 (105/105 words) | text | 1 | 0 | 280 / 331 | 9 / 37 KB | pass |

A client with no `after` (`--no-after`, API bot):

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stall-5s | opus/opus | deflate | 0.608 s | 1.279 s | 2.352 s | 3 / 10 of 13 (103/103 words) | text | 1 | 0 | 278 / 320 | 9 / 37 KB | pass |
| stall-5s | opus/opus | raw | 0.55 s | 1.259 s | 2.331 s | 3 / 9 of 12 (102/102 words) | text | 1 | 0 | 281 / 334 | 9 / 37 KB | pass |

PCM16 both ways on good links (`--codecs pcm16`): the 1.25 s PCM lead must not starve the player:

| Profile | Codec up/down | Compression | eot after speech end | eot → first `say` | speech end → first audio | Heard (server / phone) | Mode at end | Reconnects | Underruns | RTT median / max ms | Bytes up / down | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | pcm16/pcm16 | deflate | 0.32 s | 1.267 s | 2.1 s | 12 / 0 of 12 (100/100 words) | audio | 0 | 0 | 1 / 400 | 168 / 1290 KB | pass |
| baseline | pcm16/pcm16 | raw | 0.321 s | 1.38 s | 2.234 s | 12 / 0 of 12 (101/101 words) | audio | 0 | 0 | 1 / 42 | 168 / 1323 KB | pass |
| lte-edge | pcm16/pcm16 | deflate | 0.598 s | 1.284 s | 2.321 s | 13 / 0 of 13 (103/103 words) | audio | 0 | 0 | 280 / 392 | 168 / 1372 KB | pass |
| lte-edge | pcm16/pcm16 | raw | 0.608 s | 1.258 s | 2.54 s | 12 / 0 of 12 (102/102 words) | audio | 0 | 0 | 282 / 669 | 168 / 1306 KB | pass |

Reading these:

- **No sentence is lost on a rejoin any more**, also when the reply's first sentence was cut at its
  comma while it streamed: every exercised `stall-5s` run heard every word once (105/105 with the
  split reply, API bot and advisor). The pieces now depend on the reply's text alone. (The advisor
  late joiner and blackout, run on the first version of this round's tree, passed 4/4 too.)
- **3g-weak with a PCM downlink: RTT max 3.1 / 4.5 s instead of 9.6–12.6 s**, and 155–173 KB sent
  instead of 362–389 KB: the PCM lead is 1.25 s, in 0.25 s slices, so far less audio the client
  then drops sits in the pipe ahead of its pongs. EDGE with a PCM downlink passes both ways (the
  review's one failure was the shaper starving the reconnect, now fixed). On good links PCM both
  ways streams with 0 underruns.
- **"N/E"** happened only on the first run after the scratch server started: its pongs waited ~2 s
  while the fresh process loaded its models for the first turn (RTT max 1.97–1.98 s), the client
  handed the reply to the phone voice before it began, and all of it arrived before the stall.
  After a warm-up turn every case was exercised and passed.

## Tests

| File | What it proves |
|---|---|
| `tests/transport/test_wire3.py` | the shared vectors (`tests/fixtures/wire3.json`, also read by the JS tests) pack and unpack byte for byte; bad frames raise |
| `tests/transport/test_opus.py` | round trips (waveform corr ≥ 0.9 for the downlink mode, envelope ≥ 0.98 for SILK), bitrate rungs, bad input, fail-soft without libopus |
| `tests/transport/test_models_opus.py` | Parakeet on Opus 16/12 kbps == PCM (models; run under the lock) |
| `tests/transport/test_offsets.py` | zero-fill ≤ 5 s, reset beyond, overlaps, the 2× realtime budget, the u32 wrap |
| `tests/transport/test_cursor.py` | LRU/TTL, never backwards, survives a process that dies without closing, 0600, corrupt file, open utterances |
| `tests/transport/test_link.py` | requests wait for the boundary, validation, probes, the relay indicator (fixed argv, idle never "relay") |
| `tests/transport/test_transport.py` | v2 adjacency under concurrency, v3 batching and padded tails, `say` ordering, `audio_end`, PCM fallback, uplink decode |
| `tests/transport/test_shaper.py` | the proxy's rate, delay, stall, cut and blackout |
| `tests/transport/test_e2e_baseline.py` | the e2e on `baseline` (CI) |

Regenerate the shared vectors with `uv run python tests/transport/gen_wire3.py` (`--check`
compares).
