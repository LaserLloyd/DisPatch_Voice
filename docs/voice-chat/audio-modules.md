# Voice chat: browser audio modules

<!-- Owner: track T8. Contracts: v3 CONTRACTS C-§E.2 (worklet messages, module APIs),
     C.2 (binary frames), C.5 invariants 1 and 9–11, §G (wire3.json); amendment T8-1. -->

Seven ES modules in `src/dispatch_voice/static/` move the audio and watch the link. They have
no DOM code and no build step. Every clock, port and browser class can be injected, so each one
runs under `node --test`. The client app (`drive.js`, track T7) reaches them through
`bridge.js`.

| File | What it is | Runs on |
|---|---|---|
| `voice-worklet.js` | mic processor `drive-mic`: 16 kHz frames with offsets, energy gate, speech edges | audio thread (+ main-thread helpers) |
| `voice-player-worklet.js` | reply player `drive-player`: one queue, pause/resume/rewind, played/underrun/idle | audio thread (+ main-thread client) |
| `opus-codec.js` | v3 framing (`wire3`), WebCodecs Opus up/down, PCM16 fallback | main thread |
| `uplink-ring.js` | the last 30 s of uplink for replay after a reconnect; `cursorFrom()` | main thread |
| `seq-claims.js` | which path (server audio, phone voice, text) owns each sentence | main thread |
| `link-monitor.js` | is the network keeping up: degraded/recovered with hysteresis | main thread |
| `local-voice.js` | the phone's own voice (`speechSynthesis`) for a slow link | main thread |

The two worklet files register their processor only when `registerProcessor` exists. That
means the main thread can import them for their helpers (`createPlayerClient`,
`createOptimisticDuck`, `heardPlayedMs`, `createEchoBias`, ...) without side effects.

## The audio path

```
mic ─► drive-mic (16 kHz, off, gate) ─► createUplink (Opus 5×20 ms | PCM 640 B) ─► wire3 ─► socket
                                              └─► uplink ring (30 s) ─► replay after reconnect
socket ─► createDownlink (wire3; Opus → AudioDecoder 48 kHz | PCM 24 kHz) ─► player client
      ─► drive-player (resample to the context rate, queue, priorities) ─► duck gain ─► speakers
say frames ─► seq claims ─► server audio | local voice | caption
pongs, chunks, underruns, says ─► link monitor ─► `link {mode, down_kbps, up_kbps, why}`
```

## Mic worklet (`voice-worklet.js`, processor `drive-mic`)

**Messages from the worklet to the main thread**

- `{off, buf}`: one 20 ms frame. `buf` is an ArrayBuffer of 320 Int16 (640 bytes) and is
  transferred. `off` is the 16 kHz sample index of its first sample since the stream began. It
  is a u32 and wraps.
- `{level}`: the RMS of the last 100 ms, 0..1, for the ring.
- `{gate: 'open'|'closed', off}`: gate edges.
  - On `open`, the 300 ms pre-roll frames follow at once with their own offsets.
  - On `closed`, `off` is the offset just past the last frame sent, which is the uplink's cue to
    flush its encoder.
- `{speech: true|false}`: energy edges, used by the optimistic duck.

**Messages from the main thread to the worklet**

- `{enabled}`: when false, nothing is posted but the clock keeps counting. A pause therefore
  shows up as an offset gap, and the server zero-fills gaps up to 5 s.
- `{gate: {on, preRollMs, hangoverS}}`: `gateFromReady(ready.gate)` builds this. The gate is off
  for v2 servers.
- `{localSpeaking}` and `{echoBiasDb}`: apply the echo bias.
- `{hold}` (amendment T8-1): the server reports `hearing`/`transcribing`, so the gate must not
  close.
- `{restart: true}` (amendment T8-1): a new stream, so offsets return to 0.

`processorOptions` takes the same `gate`, `enabled` and `echoBiasDb`.

**Rates**

- At 16 kHz and above, a 31-tap Blackman-windowed sinc (cutoff 7 kHz) is evaluated only at the
  output instants.
- Below 16 kHz (Bluetooth SCO/HFP at 8 kHz), the worklet upsamples linearly and does not
  decimate.

In both cases `off` counts 16 kHz samples.

**Gate (SPEC §5.6)**

- **Noise floor:** each frame's energy in dBFS is compared with a noise floor. The floor starts
  at the minimum of the first 300 ms. After that it follows a slow EWMA (τ ≈ 2 s, faster
  downwards) of frames under floor + 6 dB, and never drops below −70 dBFS.
  - If no frame has been quiet for 5 s, the floor climbs toward the quietest frame of the last
    3 s. A louder cabin (engine, open window) therefore cannot hold the gate open forever.
- **Speech edges:** speech starts after 60 ms (3 frames) above floor + 9 dB. It ends after 200 ms
  under floor + 6 dB, a −3 dB hysteresis. Both thresholds add `echoBiasDb` while
  `localSpeaking` is set.
- **Opening and closing:**
  - The gate opens on a speech start and sends the pre-roll, so at least 220 ms from before the
    onset reaches the server.
  - It closes `hangoverS` after the last frame above the lower threshold, never while `hold` is
    set.
  - The hangover comes from `ready.gate.hangover_s`, so the server always hears the silence its
    own end-of-turn decision needs.

**Helpers**

- `isBluetoothOutput({label, outputLatency})`: true for a Bluetooth-like label, or an output
  latency above 100 ms.
- `createEchoBias({bluetooth})`: starts at +6 dB on Bluetooth and 0 otherwise. Each
  `echo_ignored` during local speech adds 3 dB, each clean turn subtracts 1 dB, and the value
  stays within 0..+9.

## Player worklet (`voice-player-worklet.js`, processor `drive-player`)

Create it with `new AudioWorkletNode(ctx, 'drive-player', {numberOfInputs: 0,
outputChannelCount: [1]})`. Then connect it to the AnalyserNode and the duck gain node, as
before. `createPlayerClient(node.port, {onPlayed, onUnderrun, onIdle, onLevel})` wraps the
port, transfers the PCM, and keeps the last `played` report in `client.last`.

**Messages from the main thread to the worklet**

- `push {gen, seq, kind, rate, pcm}`: `pcm` is a transferred Float32Array, or Int16Array
  (the worklet divides it by 32768; `createPlayerClient.push` passes an Int16Array through
  as is). Optional `totalMs` gives the clip's full length, which matters for acks. An empty
  `pcm` is ignored. A chunk for a sentence that already finished, or that is fading out after a
  cut, is dropped, so a straggler never plays after the next sentence (the player remembers the
  last 64 finished `(gen, seq)` pairs of the current gen; `reset` forgets them). Input arrives at
  24 kHz (PCM) or 48 kHz (decoded Opus) and is resampled on push by a streaming windowed sinc
  (32 taps when upsampling). The resampler has no start-up delay and keeps its state across
  chunks.
- `pause`: a 20 ms ramp, then the clock stops. The ramp plays 959 more samples.
- `resume {rewind}`: a 20 ms ramp up. With `rewind: true` the current seq restarts from its first
  sample. Without a pause, `resume` is ignored, with one exception: a `resume {rewind: true}`
  that lands within 1.5 s of an in-place resume of the same sentence still rewinds it. So the
  server's `duck off` and `resume {rewind}` work in either order.
- `flush {gen}`: everything older than `gen` fades over 100 ms. If the player is paused into
  silence, those sentences are dropped at once, so an interrupted reply never fades back in. A
  flush also ends the pause.
- `gain {value, rampMs}`, `prebuffer {ms}`.
- `end {gen, seq}` and `reset` (amendment T8-1), described below.

**Messages from the worklet to the main thread**

- `played {gen, seq, ms, leadMs}`: every 100 ms on the sample clock while a sentence is current.
  It keeps arriving while paused, with `ms` frozen.
- `underrun {gen, seq, gapMs}`: one per hole inside a sentence, sent when the audio resumes,
  with the exact gap.
- `idle {gen, seq, ended}`: the queue drained. `ended` is true when the seq was known to be
  complete.
- `level {v}`.

**`end` and `idle`.** Send `end {gen, seq}` when the next `say` (or `say_end`) arrives. The
server sends the next sentence's `say` only after this one's last audio, so `end` tells the
player that a drain is the end, not an underrun. It also flushes the resampler tail, and lets a
prebuffered sentence start. Without `end`, a sentence still counts as complete as soon as a
later seq is queued. `playback_done` follows `idle {ended: true}` for the reply's last seq.

On a v3 socket the server also sends `audio_end {gen, seq, kind}` after a sentence's last audio
(amendment T6-2). Send `end {gen, seq}` on it at once: it arrives even when a stall holds up the
next `say`. Count the reply's sentences whose `audio_end` arrived (audio complete, not only
played) for `start.after_partial.sentences` on a resume; the player keeps playing what it holds
across a reconnect, so asking for those again would play them twice. A second `end` for the same
seq (from the later `say`) is harmless.

`idle {ended: false}` is only the queue running dry mid-sentence (or before `end` arrived): never
send `playback_done` on it. A drain followed by `end` posts `idle {ended: true}` and no
`underrun`; the resampler's tail is the end of the sentence, not a hole.

**Priorities.** The order is notice > reply > backchannel > ack. The rule lives in the worklet,
which owns the queue (amendment T8-1).

- A filler (ack or backchannel) never queues behind a reply or notice, and never interrupts one.
  It is dropped.
- A reply never cuts an ack of 1.3 s or less (`ACK_UNCUT_MS`). The ack's length is `totalMs`, or
  what has arrived so far, and the reply queues behind it. The wire carries no duration, so
  **the server must send a pre-rendered ack or wait-tier line in one burst before any reply
  audio** (T3): a tier line streamed slowly would be judged by its first 600 ms and played in
  full.
- A reply cuts a longer ack (a wait-tier line), a notice line or a backchannel with a 50 ms fade.
- A notice that arrives during a reply waits its turn.

**gen.** The wire carries a u16, compared with serial arithmetic: 0 after 65 535 is newer. A push
with a newer gen flushes the older ones. `reset` (a new socket) forgets the gens.

**Main-thread helpers**

- `heardPlayedMs(playedMs, ctx, sentenceMs)` gives `heard.played_ms` for `source: "server"`:
  `played` minus `outputLatency`, falling back to `baseLatency` and then 0, clamped to the
  sentence.
- `createOptimisticDuck({apply, notify, rtt, now, setTimer, clearTimer, enabled})` is SPEC §4.6
  layer 1.
  - On a `{speech: true}` edge while a reply plays, it calls `apply(true)`: ramp the duck gain
    to 0.15 over 40 ms and `pause`. It also calls `notify(true)`, so the app sends
    `duck_local`.
  - It restores after `duckRestoreMs(rtt)` = max(700, 400 + RTT EWMA) ms, unless the server's
    `duck` or `barge_in` arrives first.
  - It never ducks in the first 1.5 s of a reply, ducks at most twice per reply, and turns off
    for the voice chat after two `echo_ignored` in a row. Pass `enabled: () => false` while the
    phone's voice speaks.
  - A restore by timeout rewinds when the delay is over 800 ms (the same rule as the server's
    `resume`), so a long optimistic pause never resumes mid-word.
  - When the phone paused first, the server's `duck off` / `resume` also rewinds once THIS
    side's pause (`REWIND_AFTER_MS`, 800) has passed: the server counts its pause from its own
    later `duck` and came back with `rewind: false` after a 1.08 s pause (FINAL-FIX-4). A
    pause the server started stays the server's call.
  - Feed it `onReplyStart`, `onReplyEnd`, `onServerDuck(on)`, `onResume(rewind)`,
    `onBargeIn()`, `onEchoIgnored()` and `onTurnAccepted()`. `onResume` always reaches the
    player (`apply(false, {rewind})`), even after a `duck off` already restored it.
  - Call `reset()` on every `ready`. A socket that drops during a server `duck {on}` leaves a
    duck the new session will never end; `reset()` restores it (`apply(false, {flushed:
    true})`, so the gain goes back to 1) and clears the reply state. The echo verdicts stand.

**Notes for the app (T7)**

- `player.flush(gen)` clears `client.last`. The `heard` sent after a `barge_in` (C.3) must read
  `client.last` **before** flushing.
- Setting the mic `{enabled: false}` while the gate is open posts no `{gate: 'closed'}`. Call
  `uplink.flush()` when you pause the mic, or up to 80 ms waits in the uplink until the next
  speech. Likewise send `{hold: false}` on a socket close, or the gate stays open.
- **On a socket close, `downlink.reset()`** (or close it and make a new one) **before** the next
  `ready`'s `player.reset()`. After `reset` the first push sets the player's gen floor, so a
  decoder output from the old socket (gen 5, say) arriving after it would make the new
  session's gen-0 sentences "older" and silently drop them.
- **On every `ready`:** `player.reset()`, `claims.reset()`, `localVoice.reset()`,
  `duck.reset()`, then re-send `monitor.resendFrame()` if it is not null.
- Inject one `now` into both `createDownlink` and `createLinkMonitor`, or leave both on the
  default `monoNow`.

## Codecs and framing (`opus-codec.js`)

**Framing**

- `wire3.packUp/unpackUp/packDown/unpackDown/isV3` implement C.2 byte for byte.
- The validation rules are exactly `transport/wire3.py`'s, and the shared
  `tests/fixtures/wire3.json` holds both sides to them:
  - 1..50 packets up and at least 1 down;
  - every packet 1..65 535 B;
  - PCM16 packets even-length (a probe's random bytes are exempt);
  - known codecs and kinds only;
  - no trailing bytes;
  - uplink frames of at most 64 KiB.
- A bad frame throws `WireError`.

**Support check.** `opusSupport()` resolves to `{up, down}` from `isConfigSupported`, and
`codecQuery()` turns that into the `up=`/`down=` URL values. The configurations are the measured
ones:

- encode at 16 kHz mono, `voip`/`voice`, complexity 5, 16 or 12 kbps;
- decode at 48 kHz.

**Uplink: `createUplink({codec, kbps, onMessage, ring, flags, onError, codecs})`**
(`createOpusUplink` is the same with `codec: 'opus'`)

- `push(frame, off)` takes each worklet frame. A message carries 5 packets (100 ms) and is always
  contiguous: a gate gap starts a new message.
- `flags()` is sampled per frame when it is pushed, so `local_speaking` reflects the moment of
  capture. A message carries the OR of its frames' flags.
- Call `flush()` on `{gate: 'closed'}`. `setKbps()` is the ladder rung.
- On an encoder error, the uplink switches to PCM16 for good and re-sends the frames the encoder
  still held, so none is lost. It calls `onError('encoder')` once.
- An encoder that stalls silently (no output and no error, a known WebCodecs failure after a
  reconfigure or backgrounding) is handled the same way. The trigger is 25 or more held frames
  (a healthy encoder holds one or two) with the oldest waiting more than 2 s of wall time
  (`ENCODER_STALL_MS`), 500 held frames, or `flush()` not finishing within 2 s. It needs both
  the count and the wall time, so a main-thread hitch that delivers a burst of queued frames is
  not a stall. `now`, `setTimer` and `clearTimer` are injectable.
- `onError` can run **synchronously inside `createUplink`** (no WebCodecs at all). The handler
  must not use the uplink variable, which is not assigned yet.

**Downlink: `createDownlink({player, onChunk, onProbe, onError, codecs, now, setTimer,
clearTimer})`** (`createOpusDownlink` is the same)

- `feed(buf, arrival)` takes each binary frame.
- `onChunk` reports `{gen, seq, kind, t, arrival, durationMs, bytes}` on receipt, for the link
  monitor. A probe goes to `onProbe` and is never played.
- `flush()` waits for the decoder, and `reset()` drops what it holds after a barge-in.
- A decoder error calls `onError('decoder')` once. The app then sends `link {why: "decoder"}` and
  reconnects with `down=pcm16`.
- A decoder that stops answering without an error is the same failure (`onError('decoder',
  Error('decoder stalled'))`). It is suspected when it holds 25 or more packets and the oldest
  was fed over 2 s ago (`DECODER_STALL_MS`), or when it holds 500. It is confirmed 250 ms later
  only if no output arrived meanwhile, so a big pre-rendered burst or a main-thread hitch is not
  a stall. A `flush()` that does not settle within 2 s fails at once. Without this, a mute
  decoder meant silence for every later reply with nothing noticing: `onChunk` fires on
  receipt, so the first-audio deadline is met.
- **One clock.** `now` defaults to `monoNow` (monotonic epoch-ms), exported by both
  `opus-codec.js` and `link-monitor.js` as the same function. The downlink's `arrival` feeds
  `monitor.onChunk`, so if you inject a `now`, inject the same one into both.

## Replay ring (`uplink-ring.js`)

**`createUplinkRing({seconds: 30, maxBytes, stream})`**

- Keeps every uplink message: 30 s of Opus is about 59 KB. For PCM, use `seconds: 10, maxBytes:
  320000`.
- `replayFrom(off, {nowOff, maxAgeS: 20})` returns re-packed messages:
  - each carries the `replay` flag;
  - Opus messages hold at most 50 packets and PCM messages exactly one;
  - runs split at gate gaps and at `local_speaking` changes;
  - the packet that contains `off` is included whole.
- Staleness is judged per burst (a run of offset-contiguous packets, one stretch the gate let
  through) by the burst's last sample (amendment T6-1). A burst that ended more than 20 s before
  `nowOff` is skipped whole and `ring.cutOff` turns true; the app then says "That got cut off.
  Say it again." A burst that ended inside the window is replayed whole from `off`, even when it
  began earlier, so the server never gets a sentence's tail without its head. With the gate off
  the stream is one long burst and nothing is judged stale (the server's answer bounds it).
- Replay from the server's `cursor {from}` answer, which may lie **below** the client's own
  `from` (an utterance the old socket heard only half of, T6-1).
- `trim(off)`, `oldest()`, `newest()`. Trim only up to what the server has committed (the last
  transcript or end-of-turn offset), never up to the last rx offset, or the head of an
  unfinished utterance is gone when the answer asks for it.
- Packet lengths come from the Opus TOC byte, so real libopus packets of any duration are
  measured correctly.

**`createReplayPump({send, factor: 4})`** sends a replay at 4× realtime. Ten seconds of PCM (320
KB) would otherwise overflow the server's 64,000-byte audio queue.

**`cursorFrom({lastRxOff, lastTranscriptUpto, lastEotUpto})`** returns the largest of the three on
the wrapping u32 clock (C.5 #10). `drive-settings.js` (T7) has a plain `Math.max` version.
Delegating to this one also covers the wrap.

## Sentence claims (`seq-claims.js`)

`createSeqClaims()` provides `claim(sid, seq, path)` and `reset()`.

- The first of `server`, `phone` and `text` to claim a sentence wins. The same path keeps
  re-claiming it, chunk after chunk.
- Claims are keyed `(sid, seq)` and cleared on every `ready` (C.5 #9). A new server session's
  seq 1 is therefore never confused with the old one's.
- At most 512 claims are kept.

## Link monitor (`link-monitor.js`)

`createLinkMonitor({now, thresholds = LINK_THRESHOLDS, onChange, codec: {down, up}})`.

**Inputs**

- `onPong(rttMs, rxOff, sentOff)`
- `onSay(seq, tServer, kind)`
- `onSayEnd()`: also clears the reply's first-audio deadlines and its goodput window.
- `onReplyCut()`: the reply is over without a `say_end` (the server sends none for a cut reply):
  a barge-in, a tap-stop, an undo, a failed turn (`turn_error` other than `tts_failed`) or a new
  turn (a `transcript` with text). It does what `onSayEnd()` does, including applying a held
  predictive change now. Without it `replyActive` stays set and `wantProbe()` never becomes true,
  so a call that fell back to the phone voice could not switch back after one interrupted reply.
- `onTurnError(seq?)`: call it on `turn_error` (for example `tts_failed`). It clears that seq's
  deadline, or every armed one without a seq, so a TTS fault is never blamed on the network.
- `onChunk(seq, arrival, tServer, durationMs, leadMs, bytes)`: take `arrival` from a monotonic
  clock (`performance.timeOrigin + performance.now()`), as the monitor's own default `now` does.
- `onUnderrun(seq, gapMs)`
- `onStall()`
- `onReady()`, `onClose()`
- `onMetrics({first_chunk_ms})`
- `onProbe(bytes, elapsedMs)`
- `probeFailed()`: call it when the probe was refused (`error {rate_limited}`). A probe that is
  never answered is written off after 30 s (`probeTimeoutMs`); a late answer is measured but
  does not recover.
- `tick()`: call it about every 250 ms.

**Outputs**

- `onChange({mode, downKbps, upKbps, why})`. Send it as `link` with `linkFrame(e)`.
- `onChange({backlogS, slow})`.
- `wantProbe()`: when true, send `probe {bytes: 16384}`.
- `resendFrame()`: the `link` frame to re-send after a reconnect's `ready`, or null.
- `setPinned('text')`: "Always use the phone's voice".
- `setCodec()`: after negotiation or a codec fallback.
- `stats()`: for the Voice lab.
- `deadAfterMs`: the heartbeat timeout, 8 s, or 4 × the RTT EWMA once that exceeds 1.5 s.

**Rules.** The numbers are SPEC §5.3's, all in `LINK_THRESHOLDS`.

- **Underruns count only when the link is behind.** The skew is `arrival − t_server`, measured
  against its minimum **within the current reply** (it starts over at each reply). Chatterbox's
  own 1.1–1.35 s hole inside a long sentence keeps the skew flat and is never blamed on the
  network. A settled offset, such as a cell handover or an NTP step, is not the link falling
  behind.
- **Goodput** counts only the last 2 s of the current reply (the window is trimmed on every tick
  and cleared at `say` and `say_end`). It needs the skew both 300 ms over the reply's best and
  still rising (by 100 ms or more) inside the window. A realtime TTS never delivers 1.3 × the
  need, so the ratio alone means nothing.
- **First-audio deadlines** are armed for reply sentences in audio mode only. An ack or tier line
  is one pre-rendered burst, and a notice ("Still there?") has no `say_end`, so a lost one is a
  TTS fault, not worth a mode switch. They are not armed for a `say` arriving within max(1.5 s,
  2 × RTT) of a switch back to audio, since that may be a text-mode sentence.
- **First step down is the codec ladder:** 24→16 kbps down and 16→12 kbps up. A second trigger
  more than 10 s later (the rung needs a sentence boundary to apply) switches to text. A missed
  first-audio deadline or a stall switches to text at once.
- **Timing of the switch:**
  - Predictive causes (`rtt`, `goodput`) are held until `say_end`, so the voice never changes
    mid-answer.
  - Reactive causes (`underrun`, `deadline`, `stall`) are sent at once.
- **Recovery** goes one step at a time: text → 16 kbps → 24 kbps. The dwell gates the first
  step only; the next probe (20 s later) can finish the climb. Each step needs all of the
  following:
  - the dwell has passed;
  - 30 s without a bad sign;
  - no reply in progress;
  - RTT EWMA under 500 ms;
  - a probe at 2 × or more of the next step's need, at most one per 20 s.
- **Dwell** starts at 60 s, doubles for each flap within 5 min, and caps at 10 min. A flap is
  any new degraded spell within 5 min of a recovery step, including a text → 16 kbps step whose
  rung then fails. The ladder → text step inside one spell restarts the dwell without doubling.
- `startMode(effectiveType, {bluetooth})` starts a `2g`/`slow-2g` connection in text mode. `3g`
  starts in audio: Chrome reports it for any RTT of 270 ms or more, where the measured LTE and
  relay medians sit, and 28 kbps Opus carries fine there. Over SCO the downlink starts at the
  16 kbps rung.

## Phone voice (`local-voice.js`)

**Choosing a voice**

- `pickVoice(voices, lang, preferredURI, {android})`:
  - matches the UI language's primary subtag and prefers an exact `lang` (with `en_US` read as
    `en-US`);
  - honours the stored URI while it is still offered;
  - applies the `localService` filter on desktop only, because Chrome on Android lists one entry
    per locale, all "local";
  - avoids `-network`/`online` variants when an offline one exists;
  - returns null when nothing matches, and the device's fallback is then Text.
- `voiceOptions()` builds picker rows labelled from `lang` ("English (United States)"). The voice
  name is added only where two rows share a lang.
- `waitForVoices(synth)` waits up to 1 s for `voiceschanged`.

**`createLocalVoice({synth, rate, voice, lang, onDone, onFail, onSpeaking, now, setTimer, clearTimer, Utterance})`**

- `enqueue(say)` speaks one utterance per `say` from our own queue. A text over 260 characters
  is split at sentence, clause or word boundaries.
- The watchdog cancels and counts a failure when there is no `onstart` within 1.5 s, or no
  `onend` within chars/12 + 3 s. After two failures it calls `onFail(reason)` and the voice
  chat goes text-only. Three clean sentences in a row forgive a failure, because a cold Android
  engine is slow to start once after being evicted.
- `reset()`: call it on every `ready`, beside `claims.reset()`, `player.reset()` and
  `duck.reset()`. seq restarts
  at 1 and gen at 0 on a new socket, so a stale `say_end` would otherwise mark the new reply's
  first sentence final.
- The utterance's `lang` is the voice's lang in BCP 47 (`en_US` becomes `en-US`).
- `dropOlderThan(gen)` handles `barge_in`: cancel and drop. There is no pause, because Android's
  `pause()` is a stop.
- `respeak()` handles `resume {rewind}`.
- `playedMs()` is the time since `onstart`, clamped to the sentence at rate × 15 chars/s. It
  feeds `heard {source: "phone"}`.
- `onDone({gen, seq, final, why})` fires after each sentence. `final` means send `playback_done`
  now. It is set on the last sentence's `onend` after `sayEnd({gen, last_seq})`, or 400 ms after
  the queue drains with no `say` pending.
- `onSpeaking(bool)` drives the uplink's `local_speaking` flag and the worklet's
  `{localSpeaking}`. Those only feed the server's own echo rule (a backup now).
- Half-duplex mic (`drive-audio.js`): speechSynthesis gives no echo-cancellation reference, so
  while the phone voice speaks, and for `PHONE_VOICE_MUTE_TAIL_MS` (400 ms) after `onSpeaking(false)`,
  every mic frame is zeroed in `micMessage` before it reaches the uplink, the pre-ready queue or a
  v2 server's `sendBinary`. The frame keeps its length and `off`, so the uplink offsets stay
  contiguous and the replay ring stays consistent. `{speech}` edges are not forwarded to the UI
  during that time (no "hearing you" for the phone's own voice); `{level}` reports still flow.
  `api.micHalfDuplex` exposes the state. The user's Mute is a separate mechanism.
- After a `cancel()`, the next `speak()` waits 60 ms, because Chrome can drop a `speak()` issued
  in the same task.

## Tests and the real-browser check

```
node --test "tests/js/*.test.js"      # 8 T8 suites: worklet, player, opus, link-monitor,
                                      # local-voice, uplink-ring, seq-claims, wire3
.venv/bin/python tests/js/browser/audio_modules_check.py --port 19080 \
    --out .e2e-out/audio-modules            # add --stt under the model lock:
flock "$VOICE_LOCK" .venv/bin/python tests/js/browser/audio_modules_check.py ... --stt
```

The browser check runs the shipped files unchanged in headless Chromium (driven over CDP, no
Playwright needed) and compares them against the server's own libopus and wire3. Results on
2026-10-10, HeadlessChrome 151:

| Check | Result |
|---|---|
| WebCodecs Opus | supported both ways |
| Mic worklet at 48, 44.1 and 8 kHz | contiguous offsets; waveform correlation with the source 0.985, 0.981, 0.982 |
| Browser uplink at 16 and 12 kbps | 17.3 and 13.3 kbps on the wire, about 270 µs per frame; the server's `unpack_up` and libopus decode every packet |
| Replay ring loaded through `?v=` (as `bridge.js` does) | its `opus-codec.js` comes through the same query (one instance); a replay of the browser's 74 live messages is 8 messages carrying the same 367 packets with the replay flag |
| Parakeet transcripts | the mic-worklet PCM, both Opus uplinks and the 8 kHz SCO path transcribe word for word like the fixture |
| Server Opus through `createDownlink` and the real `drive-player` | equal to the server's own 48 kHz decode (correlation 1.000000, lag 0, max error 0); `idle {ended: true}`, no underruns |
| PCM16 24 kHz path | resampled to 48 kHz with correlation 1.000000 |
| Pause, resume, then `resume {rewind}` on the real worklet | 0.0 error against the continuous render at the next sample and at the seq's first sample |

The things only a phone can tell are on the Pixel checklist (SPEC §12): the `getVoices()` dump,
`ctx.sampleRate` and `outputLatency` with the car kit, the echo level that sets the initial echo
bias, and `rate` from a background tab.
