# Voice chat: how the add-on plugs into DisPatch

DisPatch Voice is an add-on. DisPatch (2.2.0 and later) ships a small, stable
extension host; this package plugs into it and brings the whole voice feature:
the speech pipeline, the session, the voices, the settings routes and its own
frontend assets. DisPatch runs fully without it. This page is the host side:
what `setup(ctx)` wires, which routes exist, where the files live, how the
switch and the models behave, and the power-profile hold.

## The handshake

| Layer | Who declares | Value |
|---|---|---|
| DisPatch ↔ add-on, Python | DisPatch `app/extensions.py: EXT_API`; add-on `dispatch_voice.EXTENSION["api"]` | `1`; the add-on accepts `[1, 1]` |
| DisPatch ↔ add-on, JS | `shell.EXT_API`; `static/index.js: EXT.api` | `1`; `[1, 1]` |
| Add-on ↔ its plugins (Kokoro) | `dispatch_voice.host.registry.ExtensionAPI.api_version` | `3` (unchanged since DisPatch 2.1) |
| Release numbers | `dispatch-voice 1.1.0`, `dispatch-voice-kokoro 0.1.1` | informational, never compared |

DisPatch discovers the add-on through the `dispatch.extensions` entry point
(`voice = dispatch_voice:EXTENSION`), or `DISPATCH_EXTENSIONS=dispatch_voice:EXTENSION`
for a checkout that is on `PYTHONPATH` but not installed. `dispatch_voice`'s top
module is metadata only (no numpy, no onnxruntime); `dispatch_voice.host:setup`
is imported only after the API check passes. A raising `setup` is DisPatch's
state `failed`, with the routes rolled back; DisPatch still boots.

`dispatch-voice doctor` prints the handshake (`EXT_API` imported from
DisPatch when this environment can see it, else read from
`--backend DIR/app/extensions.py` without importing anything), the models, the
engines and plugins, the running server's add-on version against the checkout,
and (with `--backend DIR`) whether the add-on is installed in DisPatch's venv
and whether the `--with-editable` overlay resolves offline.

## `setup(ctx)`, in order

`src/dispatch_voice/host/__init__.py` (contracts §B.4):

1. `ctx.serve_static(static/, entry="index.js", css=["voice.css"], locales=True)` —
   DisPatch serves `static/` at `/ext/voice/…` (operator only, asset types only,
   `?v=<assets hash>` immutable) and returns the hash of that moment
   (`/api/voice/status.assets_version_at_setup`; the client keys its cache on
   `opts.assetsVersion` from `/api/extensions`, which DisPatch recomputes when a
   static file changes in an editable checkout).
2. `ctx.include_router(router, prefix="/api/voice", body_limits=…)` — every
   route below; DisPatch adds its operator dependency (unlocked tier only,
   Safe Mode 403, never the machine tier) and the prefix joins the Safe-Mode
   block list. The profiles router rides inside it at `/profiles` when the
   Chatterbox models are fetched.
3. `ctx.thread_websocket("voice", ws.serve, max_total=max_sessions)` —
   `/ws/voice/{thread_id}`. DisPatch owns the gate (Origin == Host, a live
   unlocked session, the thread exists: 1008 before accept (a `safe` bot's
   thread is allowed since DisPatch 2.3.1 — voice is unlocked-only, never Safe
   Mode, and that is decided by the device's session, not the bot's flag); the
   per-session slot: 4001 "replaced"; `max_total`: accept-then-1013 "busy").
   `ws.serve` only ever sees a gated socket.
4. `ctx.set_bot_field_validator("voice", …)` and `("voice_rp", …)` — the Bots
   save answers 422 for a voice that is neither `""`, a profile id nor a known
   `builtin:<id>`, and for a `voice_rp` outside `"" | all | skip_actions |
   dialogue | dialogue_narrator`.
5. `ctx.register_tools([...])` — the three opt-in `tools.yaml` builtins
   (`voice-cloning`, `voice-builtin`, `voice-lab`), each with an "installed on
   this box" probe. Their switch decides whether the Tools tab shows a tile.
6. `ctx.on_startup(prewarm, thread=True)` — the models load in the background
   after boot when the switch is on, so the first chat is fast.
7. `ctx.on_shutdown(shutdown)` — the power hold is released and the models dropped.
8. `registry.mount_extensions(ctx)` — the add-on's **own** plugin seam
   (`DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup`), where the
   optional Kokoro plugin registers its engine and built-in voices. A plugin's
   `api.include_router(router)` mounts under `/api/voice` through the same
   DisPatch gate; its router's paths are relative to that prefix.

Everything the add-on needs from DisPatch arrives through the context
(`ctx.paths`, `ctx.bots`, `submit_turn`, `abort_turn`, `retract_message`,
`replies_after`, `subscribe_frames`, `socket_count`). The add-on never imports
`app.*`; `host/context.py` holds the context, and outside DisPatch (the CLI,
`dispatch-voice serve`, tests) the paths come from `DISPATCH_DATA_DIR` instead.

## A voice chat, end to end

```
browser ──/ws/voice/{thread}──▶ DisPatch gate ──▶ ws.serve(ws, sock)
                                                   │ switch off / models missing → sock.close_unavailable (1013)
                                                   │ accept → warm models + detector (fail: 1013 "voice models failed to load")
                                                   │ VoiceSession (ready) ‖ power.acquire on a thread beside it
mic PCM ──▶ TurnDetector (Silero + Smart Turn) ──▶ Parakeet ──▶ ctx.submit_turn(text, instructions, thinking, metadata={"voice": true})
       └──▶ the ack gate (text/acks.TurnAcks): one rendered line at +0.55 s when the reply will take a while, wait lines, tool lines
reply frames ◀── ctx.subscribe_frames(thread) ◀── DisPatch persists, broadcasts, runs the agent (progress {tool} → a status line)
       └──▶ SentenceSplitter (source spans) ──▶ speech text + role-play (text/speech_text) ──▶ one `seq` per piece
              ──▶ `say` the instant its hold releases ──▶ TTS (Chatterbox | Kokoro | …) ──▶ audio frames ──▶ `say_end`
              (text mode: `say` only; the phone reads it)
every probed pause ──▶ Smart Turn ‖ StreamingSTT.at_pause ──▶ TurnPolicy: end | hold (eot_soft) | reprobe | listen (backchannel?)
       └──▶ end/hold with text: Speculator.start (advisor) ‖ final() ──▶ commit (similar ≥ 0.95, head unchanged) ──▶ submit(prefetched)
interruption ──▶ barge_in ──▶ ctx.abort_turn(keep_partial=True) ──▶ the exact `heard` (≤ 1.5 s) ──▶ ctx.note_heard(row, upto, heard, unheard)
hang-up ──▶ session ends ──▶ power.release (after the acquire), unsubscribe, close (shielded: never skipped by a cancellation)
evicted (a newer socket of the same session) ──▶ sock.on_replaced ──▶ session.end(4001 "replaced") at once
reconnect of the same voice chat (same device + stream id, v3) ──▶ the older session.end(4001 "replaced") at once
```

The power hold is taken on a thread beside the session, not on the way to
`ready`: it only makes the reply faster, so a slow or wedged power daemon never
delays the ready tone. `ready.power_hold` may read `off` for the first ~0.3 s
after `ready`; `/api/voice/status.power_hold` catches up (the e2e polls it for
up to 2 s). The clean-up waits for the acquire to finish before it releases, so
the two can never cross.

The transcript goes through DisPatch's normal send path (`submit_turn`), so a
spoken message lands in the thread like a typed one, tagged `metadata.voice:
true`, and any bot type answers. The spoken-reply instructions are built per
turn by T5's `text/persona.py` through `host/persona.instructions_for`: a
`VoiceInstructions(system, trailer)` where `system` (the "Spoken replies"
section) goes to the advisor and API bots and `trailer` (the short app note,
or the bare `[voice chat]` tag under `gateway_trailer: tag`) is appended to a
gateway agent's text — never a prefix. The host supplies the bot's backend,
its `voice.yaml` follow-up default (device override on top), whether the voice
renders expressive tags, an extension's own persona, and — on the one turn
after an interruption — the heard/unheard note. Merges and undo use
`abort_turn(keep_partial=False)` / `retract_message`; a reconnect uses
`replies_after`. The session's own protocol (frames, states, close codes) is
in [protocol.md](protocol.md); the pieces this host adds are:

* **`seq` at enqueue, `say` / `say_end`** — every spoken piece gets its `seq`
  when it is queued and a `say {gen, seq, text, message_id, kind, role, t}`
  the instant its hold (merge window, duck) releases, before its first audio;
  `say_end {gen, message_id, last_seq}` once the reply's last sentence was
  announced. A reply cut by a `barge_in` gets no `say_end`. `message_id` is
  the id the stream frames carried (the provisional `run:…` id until
  `stream_done` names the row), exactly as the v2 audio headers did.
* **Text mode** — `link {mode: "text"}` (or `start.slow_link: always_phone`,
  which pins it for the session) makes the next sentence and every one after it
  a `say` with no audio: the phone reads it, the engine is never called,
  `_recent_said` still learns the words, and `playback_done {gen, seq}` ends
  `speaking` as before. `link {mode: "audio"}` switches back at the next
  sentence; a sentence already synthesising finishes on its path.
* **Speech text and role-play** — each sentence is `normalise`d (markdown →
  words, numbers read out, Chatterbox tags kept only for a tag-capable voice
  with `expressive_tags` on — decided by the engine that ACTUALLY speaks the
  voice: "no voice" is first resolved to the box's default voice the way
  `/api/voice/status.default_voice` is, and on a box whose default is a built-in
  Kokoro voice that is the plugin, which declares `supports_tags = False`, not
  the cloned engine behind it; `tts.speaker()` follows a `FallbackTTS` chain
  to the engine of a voice) and rendered per the bot's `voice_rp` (else the
  server's `rp_speech`): `all`, `skip_actions`, `dialogue`, `dialogue_narrator`.
  A narrator segment is its own `seq` with `role: narrator`, spoken in
  `narrator_voice` when that voice exists (else the character voice, logged).
* **Acknowledgements** (`host/acks.py`) — the pools of `text/acks.py` (and the
  operator's `<data>/voice-acks.yaml`, hot-reloaded, bound to DisPatch's bot
  list; its `tools` rules map install-specific tool names to spoken status
  lines) are keyed by the RESOLVED voice (`acks.resolved_voice`: "no voice"
  becomes the box's default, e.g. `builtin:af_heart`), so a built-in voice's
  pool keeps word lines only and its ETA prior is the built-in one; they are
  rendered by one background `RenderQueue` per process into
  `<data>/voice-cache/<engine tag>/` — one engine call at a time, seeded per
  line, only while no reply is being synthesised and no voice chat is open at
  all (a render cannot be interrupted, and one in flight cost a turn up to ~3 s
  of first audio when it was allowed during an idle chat; `DISPATCH_VOICE_ACK_RENDER=0`
  switches background rendering off). A brand-new voice's first chat therefore
  acks with the earcon; the lines render at prewarm and after hang-ups. Per
  turn, `TurnAcks` decides: one line at
  `ack_after_s` (0.55 s; inside the client's 800 ms turn-end tone hold, so an acknowledged turn makes one sound) when no reply sentence is queued and the ETA of first
  audio (an EMA per session, primed from the lab's first-chunk figure) is
  ≥ 1.2 s and the clip ends 0.3 s before it; the 5 s / 15 s wait lines; and a
  spoken status per distinct gateway tool call (`progress {item.kind: "tool"}`
  → "Searching", ≤ 2 per turn, in place of the 5 s line). None of them plays while the
  listener is talking (an utterance confirmed since the `eot` and still open, or a merge
  hold — a premature end at a comma used to ack over the second clause; review-T3s3 #3).
  Acks go out as
  `say {kind: ack}` + one audio frame, wait/tool lines as `kind: notice`; in
  text / phone-voice mode an ack is the `ack` earcon frame and the others are a
  `say` alone. Reactions `off` (`start.acks`) silences all of it, earcon
  included. The earcon stands in exactly when the gate wanted a line but none
  can play — this voice's lines are not rendered yet, none fits the time left,
  or there are no lines at all (`AckSelector.last_outcome` `none_fit` /
  `no_lines`); a Reactions roll that skipped stays silent (Amendment T3s3-1).
  Every line spoken goes into `_recent_said`; the earcon (nothing spoken) does not. The old
  per-profile `fillers/<n>.wav` are neither played nor rendered inside DisPatch (the
  `/api/voice/profiles` router is mounted with `fillers=False`; only the standalone page
  renders them); a plugin's `set_ack_provider` is kept for API compatibility but unused by
  the session. Start-up (`ensure_all`) queues the default voice's lines and then every
  bot's lines in the voice that bot speaks, its character flavour included.
* **Heard tracking** — every spoken piece registers its source span
  (`rt/heard.HeardMap`), audio length added as chunks go out. The client's
  `heard {gen, seq, played_ms, source}` frames (standalone, or inside
  `stop_playback` / `cancel` / `undo`) locate the cut. On a real interruption
  (words over the reply): `barge_in` → `abort_turn(keep_partial=True)` (the
  direct backends persist the spoken prefix and return its row id) → the exact
  post-`barge_in` `heard` is waited for, at most 1.5 s, but only from a client
  that sends `heard` frames at all (a 2.1 client pays no wait; the last periodic
  one, else the piece being sent, stands in) → `note_heard(row, upto, heard,
  unheard)`. With no row id yet (a gateway bot, a provisional id) the note is
  deferred to the `stream_done` / row that names it. The note feeds the next
  turn's instructions once. A tap-to-stop never aborts the turn (the answer
  finishes into the chat) but is noted; `cancel` while a reply plays counts as
  an interruption (`keep_partial=True`), while still thinking it cancels
  outright; merges, undo and echoes write nothing. Every piece of a reply is
  noted under its STREAM id, so the heard/unheard texts span the provisional →
  row id switch; a periodic `heard` stands in for the exact one only when it names
  the reply being cut (its gen and one of its pieces); an exact `heard` that lands
  after the fallback re-notes the row and updates the pending note; a hang-up
  during the wait ends the session at once (the cancellation propagates). The
  tap, `cancel` and `undo` come in on the receive loop — the same loop that reads
  the `heard` the wait is for — so their abort and wait run in the background
  (`undo` answers `undone` at once and retracts after the abort). The pending note
  lives in the session: a reconnect between the interruption and the next turn
  loses it (the row keeps its `voice_heard`, and the advisor/API history is cut at
  `upto` either way).
* **The realtime turn (slice 3; [turn-taking.md](turn-taking.md))** — with a
  streaming STT (`realtime_options`: Parakeet tail mode over the shared recogniser,
  spinning off, or Moonshine) the detector runs in probe mode and hands every
  utterance frame to it. At every pause ≥ the learned probe silence the session
  runs Smart Turn and `at_pause()` in parallel and asks `TurnPolicy.decide`:
  `end` (eot now, no hold), `hold` (`eot_soft {p, hold_ms}` then eot; the reply is
  synthesised but held for the learned hold × patience, and speech inside it
  merges), `reprobe` (Smart Turn again 0.4 s later, end at 1.0 s), `listen` (the
  hard silence × patience ends it; a backchannel may play). On an `end`/`hold` row
  with text the `Speculator` starts the advisor's run (`rt.speculate`: `advisor` by
  default, `all` adds API bots, never a gateway bot; `start.rt.speculate: false`
  on the device) while `final()` finishes (bounded by `rt.hold_max_s`); the run is
  adopted when the final text matches (≥ 0.95, end-weighted) and the thread head
  the frames show has not moved since it started (a typed message or the previous
  reply landing makes it stale; DisPatch's own adoption check is the law and a
  refusal is counted `stale`). `metrics` carries `p_done`, `hold_ms`,
  `speculated`, `speculation_hit` and the Speculator's counters. Turn ends run off
  the audio worker (the mic keeps flowing; a step that fails is logged and the chat
  goes back to listening with an error frame, never a silent lost turn) and are
  serialised; speech that starts
  while one is being submitted holds every reply and merges with it (a false end,
  learned as one) — also when it ENDS before that turn end finished. Each end keeps its own
  utterance's state (`_Utt`: the stop cut, the verdict, "began over our voice", the hold's
  onset), taken when the end is decided, so the next utterance's SpeechStart never changes
  what a running or queued end does. The streaming STT is one stream: the detector marks
  every utterance end in its frame sink, and until the end that owns the mark has read the
  stream (`final()`) and reset it, the next utterance's frames wait and then go into the
  fresh stream — nothing is lost or filed under the previous turn (review-T3s3 #1, #2).
  A stop cut's run is aborted beside `final()` (and always before the speculative start).
  A merge hold no speech is left for (none open, no end in progress) is released after 4 s. The **"oh, and…" addendum**: a second utterance within
  `rt.addendum_s` × patience (3 s) while nothing of the first reply was AUDIBLE yet
  when the driver STARTED talking (the onset, i.e. the VAD's confirmation minus its
  0.25 s — a reply whose first audio went out inside that quarter second did not
  interrupt them) — no audio sent, or with the phone's voice no `say` — merges: one user row (the
  first retracted, its run aborted without a partial), `transcript {merged: true}`,
  one reply; once a reply sentence is audible, more speech is an interruption.
  **Barge-in**: speech over a reply ducks it (after 0.25 s of VAD speech over the
  server's audio, 0.6 s over the phone's voice — `start.slow_link: always_phone`,
  text mode or the uplink's `local_speaking` flag — with the Silero margin at 0.65
  either way); the words decide (`classify_interruption` at the first pause, or
  from a tick partial of ≥ 3 words, or at the end of an utterance the hard silence
  ended): **echo** (a contiguous run of a sentence our voice may be playing: the
  piece the listener is at per the latest `heard` and its neighbours, else every
  piece of this gen sent since the last drain, plus the last two said; only for
  an utterance that began while our voice played — an acknowledgement, notice or
  backchannel clip counts, and raises the thresholds like a reply) → `echo_ignored`
  + `resume`;
  (an echo verdict from a tick only resumes the reply — the utterance stays open and its
  words at the pause or the end decide, so a listener who starts by quoting the bot is
  heard; review-T3s3 #4)
  **backchannel** or no words → `resume`; **stop** (a command word, a stop
  phrase, a second backchannel within 3 s — counted over pause verdicts only, never
  an early tick's) → `barge_in`, nothing becomes a turn, the run is not aborted (a
  stop word in a 0.5 s tick — or a short tick of ≤ 3 words carrying a command word
  our voice is not itself saying, "Actually, wait" — cuts at once while the utterance
  goes on: a bare stop makes nothing, more words abort the run then and become the
  turn); **real** → `barge_in` + `abort_turn(keep_partial=True)`
  + the heard note, and the words become the next turn (the abort runs before the
  Speculator starts on them; `metrics.interrupt_to_submit_ms`). Words over a reply
  that nobody has heard yet (still synthesising) replace it instead: the turn is
  abandoned without a partial row and no note is written. `resume {gen, rewind}`
  rewinds the sentence when the pause lasted > 0.8 s; a duck with no verdict
  resumes after `rt.false_interrupt_timeout_s` (2 s) — except over a merge hold whose
  speech is still open (the addendum's own end decides). Over the phone's voice the
  duck is informational: it holds no `say`, and `rewind` is never set (the phone
  never paused, so it must never re-speak — feasibility H4); VAD alone never
  cancels in any mode. **Backchannels** (`rt/backchannel.BackchannelGate`): on a
  `listen` row, only with Interrupt-by-talking on, `start.rt.backchannels` (the
  device's Reactions "Often") and the bot opted in (`voice.yaml bots.<id>.backchannel`,
  default off); one rendered `statement` line at −6 dB, `say {kind: backchannel}`,
  no gen bump, ≤ 3 per utterance, ≥ 6 s apart. **Partials**: `partial {committed,
  tail, stable, t}` at every pause and ≤ 2/s from `tick()` while the listener talks
  — only with `start.rt.partials` (captions); with captions off a tick runs only
  over a reply (it is a recogniser call). **Learned endpointing**: a false end
  (speech back within 1 s of an eot, past the hold) raises the hold floor and the
  probe silence, clean turns decay them; the values go out in `ready.rt.learned` (null
  while this server has none of its own for the device, so the client keeps its blob),
  are kept per device (`?d=`, in memory) and win over the device blob's
  `start.rt.learned`, which seeds a server that has none (once per session);
  `start.rt.learned_reset: true` forgets them (the Voice lab's Reset sends it once).
  A voice's sliced prompt and LM prefix are prepared in the background at the first
  `start` (`tts.prepare`). The first-audio `metrics` carry `platform_profile` and
  `max_mhz`, so every timing is read beside the power profile it ran in.
* **The wire (slice 4; [low-bandwidth.md](low-bandwidth.md), Amendments V1-1, T3s4-1)** — `ws.serve`
  negotiates a `transport.Transport` from the socket's query and the session sends every
  frame through it: a client that asks for `v=3` gets binary frames with sample offsets and
  Opus both ways (when libopus is present and `opus` is not `off`), anything else gets the
  v2 wire unchanged. Per piece: `say` (it ends the previous sentence's audio first) →
  audio → `audio_end`; a pending `link` rung or decoder fallback applies at the piece
  boundary; a `barge_in` drops the cut gens' audio still pending in the transport.
  **Send-ahead pacing (v3):** reply audio goes out at most 4 s (`SEND_AHEAD_S`) ahead of
  real-time playback, counted from the first audio after the client drained (`playback_done`),
  a hold not counted. The engine speaks 3-10x faster than real time; unpaced, a whole reply
  sat in the socket's buffers and everything after it (the phone voice's `say` frames after
  a switch to text, `pong`, `barge_in`) queued behind seconds of audio the client then
  dropped. The clock starts over at a `barge_in` (the client flushed everything older) and
  at a `replay_last`, and a piece dropped while it waited is never counted — otherwise the
  reply after a tap, and "Tap to continue", waited seconds for audio nobody would play
  (review-T3s4 #2). Paced audio goes out in 0.25 s slices (`PACE_SLICE_S`: Kokoro yields a
  whole sentence as one chunk), and on a PCM16 downlink (a client without WebCodecs Opus)
  the lead is 1.25 s (`PCM_SEND_AHEAD_S`): it is the lead's BYTES that hold pongs back — 4 s
  of 384 kbps PCM is ~13 s of airtime on EDGE (review-T3s4 #7). The uplink goes `decode_up`
  (a coalesced batch bigger than a live message or two is decoded in a worker thread, never
  on DisPatch's event loop) → `OffsetTracker` (mic-gate holes zero-filled, replayed overlap
  dropped; a hole too large resets the detector and drops the open utterance with its merge
  hold, its open mark and its duck); `eot.upto` / `transcript.upto` and `pong.rx_off` share
  that sample clock. The detector is fed ≤ 1 s at a time with the offset just past each
  piece and puts the offset of the frame that decided on every event (`pos`, and the
  utterance's first frame on a `SpeechStart`), so `eot.upto`, the commit and the open mark
  are the detector's own positions — never the end of a coalesced batch, which was seconds
  late inside a replay burst (review-T3s4 #3). The commit cursor (`<state_dir>/cursor.sqlite`,
  keyed by device and stream id) is advanced to the eot's `upto` before the submit is
  awaited, marks an utterance open at `SpeechStart`, and answers `start.cursor` with
  `cursor {from}` (plus the transcript again, `replayed: true`, when the phone missed it).
  As a belt to that answer (C.5 #8), an utterance that ends inside REPLAYED uplink audio at
  or below the committed offset is dropped — never a second row, even from a client that
  replayed from below the answer (one that stopped waiting for it on a slow link); live
  audio is never judged by it. `start.after_partial` skips the reply pieces the phone heard
  on the old socket; a reply this session saw no `stream_start` (or snapshot) for is spoken
  once, from its final text (the late-joiner rule, hardening H4). Reply frames wait for the
  client's first `start`: the subscribe snapshot of a reply still streaming is queued
  before the receive loop reads that frame (the client sends it on open), so spoken any
  earlier the cut reply restarted from its first sentence (measured: an advisor bot,
  stall-5s, 122 words heard of 100). The first `start` applies `after_partial`, then lets
  the queued frames through, then looks up stored replies. A STORED reply (an API
  bot's row exists before its stream; an advisor's once its stream is done) is found only
  after something: `start.after` anchors it (drive.js sends the user row the cut reply
  answers), and a client that names the cut reply in `after_partial` without an `after` gets
  the rest of exactly that reply, anchored on the user row the cursor recorded for the last
  submitted utterance. When `after_partial` names the reply by a provisional `run:…` id this
  session never saw (the drop came before its `say_end`), the count applies to the stored
  reply that answers `after`; a reply still streaming at the reconnect is never also spoken
  from its stored row, and a reply spoken from its row keeps a late `stream_done` of its
  stream silent. A reply's pieces depend on its text alone: the text goes into the splitter
  in a canonical order (one push per whitespace character, `feed_canonical`), so a new
  session that splits the snapshot or the stored row whole makes exactly the pieces the old
  one made from deltas and `after_partial.sentences` names the same ones (review-T3s4 #1: the
  splitter's first-sentence clause cut fires only while the full stop has not arrived, and a
  reply whose first sentence was cut at its comma lost a sentence on every resume). At most
  8 `after_partial` entries are kept, ids ≤ 64 characters. A reconnect of the same voice
  chat (the same device and stream id: one chat, by definition) replaces the older session
  at once with 4001 "replaced": a half-open socket kept synthesising the reply for up to
  30 s, beside the new session resuming it (review-T3s4 #4). The uplink queue is bounded at
  64,000 bytes: live audio that finds it full is dropped; while replayed audio (the client's
  ring) is queued the bound is 512 KB for it and for the live audio behind it — nothing
  ever waits on the receive loop (a full queue used to block it, pings and `heard`
  included, for up to 5 s per message). `ping` has its own rate bucket, so a burst of
  control frames never costs the heartbeat its `pong`; the submit is shielded, so a session
  that ends while it runs (a hang-up, a replacement) never leaves a committed utterance
  half-submitted.
* **A rewritten reply (review-T3s4 #5)** — a `stream_chunk {replace}` that does not extend
  what the reply received (the advisor's provider fallback, a sanitised text, a gateway
  buffer that changed what it sent) or a final content that differs from the deltas is
  never spoken as `full[consumed:]` (that spliced two texts mid-word). The new text is split
  from scratch and lined up with the reply's pieces by their SPOKEN words: a change of
  markup or whitespace changes nothing audible (only the heard map's source spans move to
  the new offsets); from the first changed word, queued pieces not yet said are dropped,
  words already said are expected again (not repeated), and the rest of the new text is
  spoken — at most the sentence that changed while it was being said is heard twice.
* **`replay_last` (SPEC §4.6 item 4; `ready.replay_last: true`)** — no LLM turn. After a cut
  (a tap, a stop word, an interruption) younger than 2 min with no other reply said since,
  "Tap to continue" (the client's `replay_last`) continues the cut reply from the start of
  the sentence the listener was in (their exact post-`barge_in` `heard`, else the tap's own,
  else the piece being sent; one they had heard to its end is skipped — located by its
  SOURCE OFFSET in the reply, never by a count of pieces), takes the cut's
  note back from the next turn, follows a run still streaming into the chat (a tap never
  aborts it), and notes the row heard in full once the last sentence drains (the next
  turn's history is no longer cut at the tap). Otherwise — and for the utterance "say that
  again" / "repeat that" (the whole utterance, while nothing is said or pending; no row)
  — the last reply (≤ 2 min) is said again; "repeat the last part" says its last sentence
  only. The reply is rebuilt from its text (kept even
  while muted) under the current gen with new seqs; a sentence the engine made before goes
  out from its kept audio (the last reply only, ≤ 90 s) — also the very sentence a tap cut
  while its paced slices were going out, when the engine had finished it (Kokoro makes a
  sentence in one go) — the rest is synthesised; in
  phone/text mode the `say` frames alone. The heard map keeps one copy of the reply, so a
  cut during the replay is noted at the right sentence. Ignored while a reply is being said,
  a turn is ending or pending, and while the mic is paused (C.3: "while listening").
* **`ready`** carries `sid` (16 hex, random per session), `say: true`,
  `langs: ["en"]`, `acks` (the mode before the first `start` is read, i.e. the
  default `sometimes`), `power_hold` and the effective `follow_up`; a
  `start.follow_up: null` means the bot's own default.

## Staying unlocked during a chat, and the quiet bound

A voice chat makes no touches, and DisPatch locks an idle PIN session (its
`lock_timeout_seconds`, and the shell's own idle timer on the device). Two halves
keep a chat that is in use from locking at the wheel, and one bound keeps a chat
nobody uses from holding the session open:

* **The server's session window** — while the chat is attached with the microphone
  on and a client frame (pings count) arrived in the last 10 s, the session calls
  `ThreadSocket.touch` at most every 30 s (SPEC §5.6, hardening M8). A half-open
  socket stops touching within 10 s and dies at 30 s.
* **The shell's idle timer** — the client calls DisPatch's `shell.activity()`
  (DisPatch ≥ 2.2.0, feature-detected) on every conversation frame (`state`,
  `partial`, `transcript`, `say`, `eot`, `barge_in`) and every 20 s while the chat
  is open, listening and visible. Before this the shell locked itself
  mid-conversation (FR-works 1).
* **The quiet bound** — with no utterance (`hearing`) and no reply sentence for
  `rt.quiet_max_s` (default 2400 s, range 600–14400) the session stops touching and
  ends the chat with close code **4002 "quiet"**; the client hangs up as "Hang up
  when quiet" does, without reconnecting. It sits above the longest device setting
  (30 min plus its 20 s grace), so the polite "Still there?" normally comes first;
  a device set to "Never" is bounded here (FR-hardening L3).

## Routes (`/api/voice`, unlocked tier only)

| Route | What |
|---|---|
| `GET /status` | what the Settings page and the mic button draw from: the switch, `ready`/`missing`, the voices, `warm`, `plugins`, `langs`, `sessions {open, max}`, `power_hold`, `extension {name, version}`, `assets_version_at_setup`, `rp_speech`, `narrator_voice`, `expressive_tags`, `gateway_trailer`, `bots: {<id>: {backchannel, follow_up}}` (effective, every non-Safe bot), `acks: {<voice or "default">: {ready, total, failed, name?}}` |
| `GET/PATCH /settings` | the switch and the server preferences, persisted to `<data>/voice.yaml` and applied live: `enabled`, `default_voice`, `speed`, `voice_turn_thinking`, `merge_window_s`, `hard_silence_s`, `max_sessions`, `rt: {power_hold}`, `rp_speech`, `narrator_voice` (a known voice or ""; 422 otherwise), `expressive_tags`, `gateway_trailer`, `bots: {<id>: {backchannel?, follow_up?}}` (known non-Safe bots only; merged per bot per key), `reset: [...]`. `max_sessions` alone takes effect at the next restart (the gate's `max_total` is sealed at setup): the response carries `max_sessions_live` and `restart_required: true` while the saved and live values differ |
| `POST /acks/rebuild?voice=` | re-render one voice's acknowledgement lines (the Voice lab's Rebuild; `default` or "" = the default voice; 404 when nothing was ever queued for it) |
| `GET /preview?voice=&text=` | a WAV sample of a voice (503 while voice is off, 422 unknown voice / too long) |
| `GET /lab`, `POST /lab/test` | the Voice lab: engines, timings, a hear-and-say-back test (320 000 B cap) |
| `/profiles[...]` | the profile API: list, create (multipart, 20 MiB), analyse, PATCH default/rename, `/{id}/say`, DELETE, `/{id}/files/…`. Moved here from DisPatch 2.1's `/api/voices` (no alias). The standalone Voices page is `dispatch-voice serve` only; nothing under `/api/voice` serves HTML. |

## Files and settings

| What | Where |
|---|---|
| Server preferences | `<data>/voice.yaml` (`ctx.paths.voice_yaml`); hot-reloaded on mtime, a hand edit of `enabled` applies within 5 s to open chats too |
| Voice profiles (cloned voices, the stock voice) | `<data>/voices/` (`ctx.paths.voices_dir`, mode 0700) |
| Render cache (acknowledgements, notices, tool lines — every voice with a `cache_tag`, cloned ones included) | `<data>/voice-cache/<engine tag>/<sha256(text)[:24]>.npy` |
| Acknowledgement pools (operator overrides, hot-reloaded) | `<data>/voice-acks.yaml` (`ctx.paths.acks_yaml`; format in [persona-acks-rp.md](persona-acks-rp.md)) |
| Speech models (Silero, Smart Turn, Parakeet) | `DISPATCH_SPEECH_MODELS`, else `<data>/speech-models` |
| Chatterbox models + the stock voice tensors | `DISPATCH_VOICE_MODELS`, else `~/.local/share/dispatch-voice/models` |
| Add-on state | `<data>/ext/voice/` (0700): `cursor.sqlite` (0600), the commit cursor — what each voice chat's stream has submitted, so a reconnect or a restart never submits an utterance twice (LRU 256, 10 min) |

`voice.yaml` keys: `enabled`, `builtin_voice`, `speed` (0.8–1.3), `voice_turn_thinking`
(`off|low|medium`), `merge_window_s` (0.5–4), `hard_silence_s` (0.8–3),
`max_sessions` (1–8, default 3; read at start-up for the gate's `max_total` —
a change is saved at once and applies after a restart, `restart_required` in
the settings response says so),
`rt: {power_hold: performance|off, speculate: advisor|all|off, stt_engine, decoder_prompt_s,
first_chunk_tokens, decoder_sessions, probe_silence_s, hold_min_s, hold_max_s, addendum_s,
false_interrupt_timeout_s, stt_spinning, quiet_max_s, …}` (contracts §F.1, validated and clamped by
`rt/settings.py`), `rp_speech` (`all|skip_actions|dialogue|
dialogue_narrator`, default `all`; a bot's own `voice_rp` overrides it),
`narrator_voice` (a profile id or `builtin:<id>`, "" = the character voice
narrates), `expressive_tags` (default true), `gateway_trailer` (`full|tag`),
`opus` (`auto|off`, default `auto`; `DISPATCH_VOICE_OPUS=off` also turns it off),
`opus_down_kbps` (`24|16`) and `opus_up_kbps` (`16|12`), the first Opus rung each way,
`bots: {<bot id>: {backchannel: bool (false), follow_up: bool (true, false for
the advisor backend)}}` — the per-bot conversation defaults, add-on-owned,
never new DisPatch bot fields. The 2.1 names `kokoro_voice`/`kokoro_speed`
still read. Unknown keys are dropped on save. A hand-written `power_hold: off`
is YAML's boolean `false` and still means off.

Environment, read by the add-on: `DISPATCH_VOICE` (`auto`/`1`/`0`),
`DISPATCH_SPEECH_MODELS`, `DISPATCH_VOICE_MODELS`, `DISPATCH_VOICE_STT`,
`DISPATCH_VOICE_TTS`, `DISPATCH_VOICE_THREADS`, `DISPATCH_VOICE_TTS_THREADS`,
`DISPATCH_VOICE_EXTENSIONS`, `DISPATCH_VOICE_MAX_SESSIONS`, `DISPATCH_VOICE_POWER_HOLD`,
`DISPATCH_VOICE_PREWARM`. DisPatch reads only `DISPATCH_EXTENSIONS`.

## The switch and the models

Voice chat is off on a fresh install, even with everything installed
(`enabled` in `voice.yaml`); `DISPATCH_VOICE=1`/`0` forces it and then the API
refuses to move it (409). Off → on starts the prewarm; on → off ends every open
chat with 1013 and drops the loaded models (and hands the heap back to the OS).
Until the models are fetched, `/api/voice/status.missing` lists them with the
command to run; the socket refuses with 1013 "voice models missing".

`dispatch-voice fetch-models` fetches every core model: Silero VAD, Smart
Turn v3.2 and Parakeet TDT 0.6B (int8) into the speech-model directory
(sha256-checked, pinned revisions), Chatterbox-Turbo ONNX (q4) into the
Chatterbox directory, and builds the stock voice's tensors. `--verify` only
checks what is on disk; `--only speech|chatterbox` fetches one set.

## The power-profile hold

While at least one voice chat is open the add-on holds the `performance`
power profile through `net.hadess.PowerProfiles.HoldProfile` on the system bus
(`rt.power_hold: performance`, the default; `off` disables it). On a box whose
idle daemon parks it in `power-saver` whenever the screen is off, that is a
28 % faster spoken sentence and a 47 % faster decoder chunk (measured).

Mechanics (`host/power.py`, `host/power_hold.py`): a child process on the
system interpreter (`/usr/bin/python3`, which has `gi`; the uv venv does not)
holds the bus connection, because a hold lives exactly as long as its
connection. It starts with a fixed argv and a minimal environment (`PATH`, the
locale and `DBUS_SYSTEM_BUS_ADDRESS` when set — none of DisPatch's tokens), and
its stderr is kept so a crash is logged with its last line, not as a bare "no
answer". Closing the child's stdin ends the hold, so DisPatch dying ends it
too. The hold is acquired beside the session (never before `ready`) and
released after the last session ends. When another client changes the profile (an idle daemon on a screen
transition) the daemon drops the hold and emits `ProfileReleased`; the holder
re-holds at most once per 10 s, and a periodic check covers a missed signal.
tuned-ppd refuses to hold `balanced`; only `power-saver` and `performance` can
be held. Fail-soft everywhere: no system python, no `gi`, no daemon or a
refused hold is logged once and `status.power_hold` reads `unavailable`; the
chat carries on at the box's current profile. The session's clean-up releases
the hold even if the handler task is cancelled, and `on_shutdown` releases
whatever is left.

## Plugins (the add-on's own seam)

`DISPATCH_VOICE_EXTENSIONS=pkg.module:setup[,…]`. Each `setup(api)` gets an
`ExtensionAPI` (`api_version` 3): `register_engine`, `set_builtin_voices_provider`,
`add_release_hook`, `set_profile_resolver`, `set_ack_provider`, `set_persona_provider`
and `include_router(router, prefix="")`. The router mounts under `/api/voice<prefix>`
behind DisPatch's operator gate, with paths relative to it; a router that spells
`/api/voice` in its own paths is refused (the gate adds the prefix and nothing
can strip it from a built router). A broken plugin is logged and skipped. The
optional, GPL-licensed `dispatch-voice-kokoro` package (installed separately) is
the reference plugin: the `kokoro` engine behind the cloned-voice engine and
the `builtin:<id>` voices.

## Running DisPatch with the add-on

Two uvicorn flags on the unit are load-bearing for voice chat (SPEC §11 step 2):

* `--ws-max-size 262144` (hardening M5). uvicorn's default WebSocket message cap is 16 MiB;
  the add-on drops any uplink binary message over 65,536 B before queueing it and bounds its
  audio queue at 64,000 B, but the cap is what stops a buggy or hostile authorised client from
  making the server buffer 16 MiB per message first. DisPatch's own `/ws` caps a `send` at
  65,536 characters (≤ 196 KB of UTF-8), so 262,144 B never cuts a real message.
* `--ws-ping-timeout 60` (feasibility L4, design-lowbw §2.2). uvicorn's protocol keepalive
  (`--ws-ping-interval 20 --ws-ping-timeout 20` by default) closed the socket with 1011 on
  the `edge` profile once 20 s of uplink backlog built up; with 60 s a 20–60 s cellular
  handover stall is judged by the voice chat's own rules instead (the client's 2 s pings and
  dead-socket rule, the server's 30 s `DEAD_AFTER_S`), and the reconnect replays from the
  commit cursor. It applies to the main `/ws` too, which is harmless. (`--ws auto` keeps it.)

The scratch harnesses start DisPatch with both (`tests/e2e_voice_roundtrip.py`
`UNIT_UVICORN_FLAGS`).

Dev / a server (an editable overlay; `--offline` is load-bearing so a reboot
without network still boots DisPatch). **On a server, point the overlay at a
release checkout nobody develops in** — whatever is in the overlay's folder goes
live at the next restart, and a work-in-progress dependency in it would make the
`--offline` overlay fail to resolve:

```sh
git -C <dispatch-voice> worktree add --detach <release-dir> v1.1.0     # a new release: checkout --detach <tag> in it
```

DisPatch's `scripts/local-chat-launch` is the recommended `ExecStart`: it proves the
overlay resolves and imports (the host package, numpy, onnxruntime) in a throwaway
process first, and starts DisPatch WITHOUT the add-on when it does not — a crash of a
broken native wheel included — instead of a restart loop. Run
`dispatch-voice doctor --unit <unit>` to check with the unit's own voice settings
(`DISPATCH_SPEECH_MODELS`, `DISPATCH_VOICE_EXTENSIONS`, its working directory):

```sh
cd <dispatch-chat>/backend
uv run --with-editable <dispatch-voice> python -c "import dispatch_voice"    # warm the overlay once, online
DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup \
  uv run --offline --with-editable <dispatch-voice> [--with-editable <dispatch-voice-kokoro>] \
  uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-ping-timeout 60 --ws-max-size 262144
```

Bare metal: `uv pip install --python .venv/bin/python "dispatch-voice @ git+https://github.com/LaserLloyd/DisPatch_Voice@v1"`
(the moving `v1` tag follows the latest 1.x release). A later `uv sync` removes
a pip-installed add-on; `dispatch-voice doctor --backend <dir>` says so.

## Tests

`tests/host/` (setup, serve, the switch, routes, profiles, the engine split,
the registry, power, the CLI), `tests/session/` (the session with fakes —
`test_session_s3.py` drives the realtime turn with a probe-mode detector double and
a scripted streaming STT; `test_session_v3_wire.py` the v3 wire; `test_session_resume.py`
reconnects, the late joiner, the commit-cursor belt, `replay_last` and the send-ahead pacing;
`test_session_review_s3.py` and `test_session_review_s4.py` the slice-3 and slice-4 reviews'
findings (the canonical split's property test over golden texts and random chunkings, the
resumes, the continue by offset, pacing after a cut, positions inside a batch, rewrites, the
queue, the bounds); the model-backed
`test_speaks_real_engines.py`),
`tests/contract/` (this add-on against a real DisPatch checkout with the extension
host: `DISPATCH_BACKEND`), `tests/e2e_voice_roundtrip.py` (a scratch server, the
fixture clips, the wire: `--host fake` needs no DisPatch; `--backend DIR` runs the
real thing; `--v3` speaks protocol v3 like the real client and adds the "Tap to continue"
and reconnect-replaces phases), `tests/latency_ws.py` (the §12 latency harness through the real
session on a scratch DisPatch with an advisor on the stub LLM: speech end → first
audio per clip, barge-in, a false interruption and the addendum, idle and under
load, per power-profile column).
