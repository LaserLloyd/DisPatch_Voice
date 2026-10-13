# Voice chat wire protocol (v3)

Generated on day 0 from section C of the v3 interface contracts (revision r2, frozen).
This page is normative: the server side is owned by the session hub (`host/`), the client
side by the client app (`static/`). Change it only through a contract amendment, then
regenerate. Tags such as `(A9)` or `feasibility H4` name the design-review finding that
shaped a rule.

## WebSocket protocol v3 (`/ws/voice/{thread_id}`)

### C.1 Negotiation
URL query: `?d=<device id>&s=<stream id>&v=3&up=opus,pcm16&down=opus,pcm16`.
`d` and `s` match `[A-Za-z0-9-]{8,64}`; `s` is random per voice chat (the replay-ring key).
`up`/`down` list what the client can do, best first; `opus` only when
`AudioEncoder/AudioDecoder.isConfigSupported` both said yes. No `v=3` → the server speaks v2
exactly as today (JSON `audio` header + raw PCM, no `say`, no offsets).

### C.2 Binary frames
Uplink (client→server), v3: `0x03 | codec u8 (0 pcm16, 1 opus) | flags u8 | off u32 LE |
count u8 | count × (len u16 LE + bytes)`. `off` = 16 kHz sample index of the first sample,
counted from stream start (wraps at 2³²). flags bit0 `local_speaking` (the phone's own voice is
playing), bit1 `replay` (from the ring). Opus: 5 packets (100 ms) per live message, ≤ 50 per
replay message; PCM: 1 × 640 B. A frame not starting with `0x03` is v2 raw PCM16.
Server limits: a binary message > 65,536 B is dropped BEFORE it is enqueued (`bad_frame`,
counted, rate-limited); > 50 packets → `error {code:"bad_frame"}` and dropped; the session's
audio queue is byte-bounded at 64,000 B (2 s of 16 kHz PCM), not item-bounded (A11).

Downlink (server→client), v3: `0x03 | codec u8 | kind u8 (0 reply, 1 ack, 2 notice, 3 probe,
4 backchannel) | gen u16 LE | seq u32 LE | t u32 LE (server monotonic ms) | count u8 |
count × (len u16 LE + bytes)`. Opus: ≤ 10 packets (200 ms) per message, sent every ≤ 100 ms;
PCM: one 40 ms chunk. The sentence's text/role/message_id come from the `say` frame that
precedes its first audio (C.4). Decoded Opus is 48 kHz on every browser; PCM is 24 kHz.

### C.3 Client → server JSON frames

| Frame | Fields | Rules |
|---|---|---|
| `start` | `patience: "quick"\|"normal"\|"patient"\|float` · `barge_in: bool` · `reply_style: "short"\|"normal"\|"longer"` · `follow_up: bool` · `after: id\|null` · `after_ids: [≤8]` · `acks: "off"\|"sometimes"\|"often"` · `slow_link: "phone"\|"wait"\|"text"\|"always_phone"` · `cursor: {stream, from}` · `after_partial: {message_id, sentences}` · `rt: {partials, speculate, backchannels, optimistic_duck}` (bools) | re-sendable; absent keys keep their value (B4); `after`/`cursor`/`after_partial` re-read only when their value changes; never resets the detector |
| `pause` | — | mic ignored, speaking continues — the Pause button sends ONLY this (never `stop_playback`); the UI says "Paused — still speaking" until `playback_done` |
| `stop_playback` | `heard?` (C.3a) | tap-to-stop (the picture tap while speaking) → `barge_in {gen+1}` |
| `typed` | `text` (1–4000 chars after trimming) | 1.2.0. A message typed on the voice chat screen: the turn a transcript makes — submitted through the host with the voice persona (`metadata: {voice: true, typed: true}`), answered with speech, Undo and the acks as usual — with no audio, STT, hold or speculative start. The server echoes `transcript {text, typed: true, message_id, merged: false}` and `state: thinking`. Typed over a playing reply it cuts the reply first (`barge_in {gen+1}`, the run aborted as a spoken barge-in would); typed while the last answer is pending it replaces that question. The microphone's state is left alone (the client mutes it when the Type button opens the box). Empty, non-string or over-long text is a `bad_frame` |
| `quiet_check` | — | ≤ 1 per 60 s; the client's "Hang up when quiet" timer fired: the server speaks the notice line "Still there?" (`say {kind: notice}` + audio, or `say` alone in phone/text mode). If no `hearing` follows within 20 s the client hangs up and shows "Voice chat ended." (A9) |
| `replay_last` | — | optional (SPEC §4.6), answered when `ready.replay_last` is true (Amendment T3s4-1): no LLM turn. After a cut (a tap, a stop word, an interruption) younger than 2 min with no other reply said since, it CONTINUES the cut reply from the start of the sentence the listener was in (one they had heard to its end is skipped) — "Tap to continue" — following the run on if it is still streaming; otherwise it says the last reply (≤ 2 min old) again — "say that again". New seqs under the current gen; kept audio is re-sent, the rest synthesised; phone/text mode: the `say` frames. The continue's start is located by the piece's SOURCE OFFSET in the reply, never by a count of pieces (T3-hub-fix-1). Ignored while a reply is being said, a turn is ending or pending, while the mic is paused (not `listening`), or when nothing is kept. The utterance "say that again" / "repeat that" (the whole utterance) does the same as a replay, with no row; "repeat the last part" says the last sentence only |
| `playback_done` | `gen, seq?` | drain report; `seq` older than the last reply seq is ignored (B6). In phone-voice mode sent on the local `onend` of the last queued sentence after `say_end`, or 400 ms after the queue drains with no `say` pending |
| `cancel` | `heard?` | abort the pending/running turn |
| `undo` | — | within 8 s of the last `transcript`, and only before any reply sentence was queued |
| `ping` | `t` | every **2 s** while open; the server answers at once — from its own rate bucket (10, then 2/s), so a burst of other control frames never costs a `pong` (T3-hub-fix-1) |
| `heard` | `gen, seq, played_ms: int, source: "server"\|"phone"` | the sentence playing (or next to play) and how much of it the LISTENER has heard; also sent alone every 1 s while a reply plays. `source: server`: worklet clock minus `AudioContext.outputLatency` (fallback `baseLatency`, else 0), clamped to the sentence length. `source: phone`: ms since `onstart`, clamped; the server maps elapsed → chars at `rate × 15 chars/s`. After a `barge_in` the client sends one `heard` with the NEW gen immediately; the server waits ≤ 1.5 s for it before `note_heard` (L7) |
| `link` | `mode: "audio"\|"text"` · `down_kbps?: 24\|16` · `up_kbps?: 16\|12` · `why: "underrun"\|"rtt"\|"goodput"\|"deadline"\|"stall"\|"user"\|"recovered"\|"decoder"\|"encoder"` | applied at the next sentence boundary, never mid-sentence. The CLIENT defers a predictive switch (`rtt`, `goodput`) until after `say_end`, so the voice never changes mid-reply; reactive causes (`underrun`, `deadline`, `stall`) are sent at once |
| `probe` | `bytes ≤ 32768` | ≤ 1 per 20 s per session; shares the 40-frame bucket |
| `duck_local` | `on: bool` | informational (the phone ducked itself) |

C.3a `heard` as a sub-object: `stop_playback`, `cancel` and `undo` may carry `heard: {gen, seq, played_ms, source}`.

### C.4 Server → client JSON frames

| Frame | Fields | Rules |
|---|---|---|
| `ready` | `sample_rate_in: 16000, sample_rate_out, sample_format, streaming_tts, smart_turn, patience, voice_acks, barge_in, reply_style, follow_up` (existing) **+** `proto: 3, sid: <16 hex, random per VoiceSession>, langs: ["en"], codec_in, codec_out, opus_up_kbps, opus_down_kbps, rx_off, gate: {pre_roll_ms, hangover_s}, say: true, acks, power_hold: "held"\|"off"\|"unavailable", replay_last: true, rt: {stt: "parakeet"\|"moonshine", speculate: bool, learned: {probe_ms, hold_ms}\|null (null: the server has nothing of its own for this device — the client keeps its blob, T3s4-1), decoder_prompt_s, backchannels: bool (this bot's effective value)}` | once, when models are loaded; the client plays the ready tone, flushes the pre-ready mic buffer and **drops every seq claim** (claims are keyed `(sid, seq)`, A9) |
| `state` | `state: listening\|hearing\|transcribing\|thinking\|speaking\|paused` | unchanged |
| `partial` | `committed, tail, stable: int, t: float` | ≤ 2/s, only with `start.rt.partials`; never a `message_id`; never reaches the agent except via speculation |
| `eot_soft` | `p: float\|null, hold_ms: int` | the policy holds; UI shows "Got it — go on?"; the earcon waits for `eot`. `p` is null when no Smart Turn model ran (the words alone held the turn; T3-hub-fix-1) |
| `eot` | `reason, upto: int` | end of turn; `upto` = sample offset of the utterance end |
| `transcript` | `text, merged: bool, message_id: id\|null, upto: int, replayed?: true` | empty text = nothing intelligible |
| `echo_ignored` | `text` | unchanged |
| `ack` | — | earcon fallback when no rendered line exists |
| `say` | `gen, seq, text, message_id: id\|null, kind: "reply"\|"ack"\|"notice"\|"backchannel", role: "character"\|"narrator", intent?: str, t: int, lang?: str` | ONE per `seq`, sent the instant the sentence's hold releases — i.e. before (or instead of) its first audio; `text` is exactly the speakable text the TTS gets; `kind: ack` carries the line text |
| `say_end` | `gen, message_id, last_seq` | the reply has no more sentences |
| `audio_end` | `gen, seq, kind` | v3 only (Amendment T6-2): after the LAST audio message of a sentence that sent audio (`Transport.end_sentence`, `last=True`, or the next sentence starting); never on v2, never for a `say` without audio (phone/text mode). The client marks that sentence's audio complete; `start.after_partial` counts complete sentences |
| `audio` (v2 only) | `gen, seq, chunk, sample_rate, text, message_id, kind, role, intent?` followed by one binary PCM frame | adjacency pairing as today |
| `duck` | `on: bool, gen` | server audio: hold/pause playback (on) or restore (off); a real interruption is followed by `barge_in`. **Phone voice (`local_speaking`): informational only** — the phone keeps speaking; it cancels solely on `barge_in` (word-confirmed) and never re-speaks (feasibility H4) |
| `resume` | `gen, rewind: bool` | continue the PAUSED reply (false interruption, echo, backchannel); `rewind: true` when the pause lasted > 800 ms → the player restarts the current `seq` from its first sample (phone voice: `respeak()` of that sentence), else in place; only ever for the current gen; a `barge_in` supersedes a pending `resume` (A9). The server measures the pause from its own `duck`; when the PHONE paused first (`duck_local`), the client also rewinds when its own pause passed 800 ms — the server's clock under-counts by the detection delay (FINAL-FIX-4) |
| `barge_in` | `gen` | drop every queued/playing chunk with gen < this; also ends a duck |
| `turn_error` | `message_id?, reason: agent_failed\|no_speech_in_reply\|timeout\|tts_failed, detail?` | a spoken notice line follows when possible |
| `error` | `code: undo_expired\|stt_failed\|bad_frame\|rate_limited\|unavailable, message` | plain English |
| `undone` | `message_id` | |
| `pong` | `t, rx_off: int, srv_t: int` | `rx_off` = highest uplink `off` received |
| `cursor` | `from: int` | answer to `start.cursor`: replay the ring from here (`max(client_from, committed)`) |
| `locked` | — | the socket closes 1008 next |
| `metrics` | any of `eot_to_transcript_ms, eot_to_first_audio_ms, first_chunk_ms, utterance_ms, reason, speculated, speculation_hit, speculation: {started, adopted, discarded, stale, prompt_chars}, duck_to_confirm_ms, ack_played, ack_eta_ms, ack_tier, tool_lines, platform_profile, max_mhz, p_done, hold_ms, cut_settle_ms, interrupt_to_submit_ms, link: {rtt_ms, mode, down_kbps, up_kbps, underruns, backlog_s, relay?, out_ms}` | the Voice lab timings line; never shown otherwise. `cut_settle_ms`: the `barge_in` that cut a reply → its heard note settled (abort + exact `heard`); `interrupt_to_submit_ms`: the end of the words that cut it (`eot`) → their submit (T3s4-1) |

Close codes: **1008** locked/refused (pre-accept, or after `locked`) · **1013** unavailable or
busy (accept-then-close, authorised callers only; reason text says which) · **4001** replaced
(DisPatch's per-session slot, or — v3 — a newer socket of the SAME voice chat: the same `d` and
`s`, i.e. a reconnect; the older session ends at once, T3-hub-fix-1) ·
**4002** quiet (nothing said either way for `rt.quiet_max_s`, default 40 min: the server ended
the chat and stopped keeping the unlocked session alive — FINAL-FIX-2) ·
**1001** heartbeat lost (30 s) · **1011** internal error (after `error {code:"unavailable"}`).
Client rules (drive-contract.md) unchanged: 1013 → `stopped` with the reason ("Voice isn't set
up on this server" / "Voice is busy on this server (3 chats open)"); 4001 → close quietly;
4002 → hang up with "Voice chat ended." and never reconnect;
1001/1011/1006 → reconnect with backoff; after 60 s offline → `waiting` (up to 15 min, visible
page only — SPEC §5.5); 1008 after `locked` → no reconnect. `gen` on the wire is u16; the
client compares `gen & 0xFFFF`.

### C.5 Invariants
1. `say` for a `seq` precedes that seq's first audio on the same socket; a replayed reply after a
   reconnect may arrive with its audio claimed already — the client's `seqClaims` drops the loser.
2. `say` is never sent for a speculative run that has not been committed, nor while a hold is open.
3. `gen` bumps only on `barge_in`; `resume` never changes `gen`; backchannels/acks never bump it.
4. `playback_done` for an ack/notice/backchannel never ends `speaking` (seq rule).
5. Ack, notice and backchannel text goes into `_recent_said` so its echo is never a turn.
6. A `link {mode}` change applies at the next sentence; the current sentence finishes on its path.
7. `transcript.upto`/`eot.upto` and `pong.rx_off` share the uplink sample clock.
8. The server never submits one utterance twice (commit cursor) and never plays one `seq` twice.
   The cursor advances to the `eot`'s `upto` before the submit is awaited (V1-1); as a belt, an
   utterance that ends inside REPLAYED uplink audio (flags bit1) at or below the committed offset
   is dropped (a client that replayed from below the `cursor` answer). Live audio is never judged
   by it (T3s4-1).
9. `seqClaims` are keyed `(sid, seq)` and cleared on every `ready`; cross-session dedup is
   `after_partial`'s job, never `seq`'s (A9).
10. On reconnect the client sends `start.cursor.from = max(lastRxOff, lastTranscriptUpto,
    lastEotUpto)`; the server answers `max(client_from, committed)` from a persisted cursor (A10).
11. A `stream_chunk` for an id with no prior `stream_start` in THIS session marks the reply
    late: its deltas are dropped and it is spoken from the `stream_done`/`message` content,
    minus the sentences `start.after_partial` named — never a spliced or doubled tail (A5).
    A reply's pieces depend on its text alone (the server feeds its splitter canonically), so
    `after_partial.sentences` names the same pieces in every session; a `replace` (or final
    content) that rewrites the text is re-split and lined up by its spoken words — never
    spoken as a splice (T3-hub-fix-1).
12. A `say` for an ack is never sent in phone-voice mode (the `ack` earcon frame is); `say`
    carries tier and notice lines only (experience L5).
