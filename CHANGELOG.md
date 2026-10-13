# Changelog

All notable changes to DisPatch Voice.

## [1.3.0] — 2026-10-14

### Added
- **Output, on the voice chat screen.** A fourth round button, like a phone's
  call-screen route: it lists where replies can play (the system default, a
  headset, a Bluetooth device, the speakers, by the OS's own names), the
  button's word says which is on ("Speaker", "Headset", "Bluetooth"), and a
  pick applies at once and is remembered (the same `sinkId` Settings › Voice ›
  Advanced › Speaker uses). Only browsers whose `AudioContext` has `setSinkId`
  can do this — desktop Chrome and Edge; on Android and iOS the browser has no
  way to move audio between the earpiece, the speaker and Bluetooth (the OS
  routes it), so the button is not shown there. A native app could.

## [1.2.0] — 2026-10-14

A small feature release: typing on the voice chat screen, and a call-screen row of
buttons. Same extension API (1); the wire gains one client frame (`typed`).

### Added
- **Type on the voice chat screen.** A **Type** button next to Mute mutes the
  microphone and shows a text box above the buttons: what you type is your turn, the
  reply is spoken as usual (captions, Undo, the acks). For a headset or a quiet room.
  Enter sends, Shift+Enter is a new line, Esc closes the box; Unmute puts it away.
  The quick sheet and Settings › Listening gain **Type instead of talking**: every
  chat opens muted with the box out, and the box stays when you unmute.
- Wire: the `typed {text}` client frame (C.3). The server submits it through the host
  with the voice persona (`metadata.typed: true` on the row), echoes a `transcript
  {typed: true}`, and treats it as a spoken turn would be: typed over a playing reply
  it cuts the reply first, typed while an answer is pending it replaces the question.
  Nothing touches the uplink cursor (no audio is behind it).

### Changed
- **The buttons are a call screen's row:** three round icon buttons with a word under
  each — Mute (its circle lights while muted), Type, Hang up (red, the handset face
  down) — centred, the same upright and on a phone's side (smaller there), and sized
  down with the picture when the on-screen keyboard takes the lower half. A stopped
  chat shows Try again / Back to chat in the same circles.

## [1.1.0] — 2026-10-11

A feature release: the optional GPU rig. Same extension API (1); `voice.yaml` gains an
optional `rig:` tree. Nothing changes until it is set up.

### Added
- **Optional GPU rig (SpeechForge).** A machine running SpeechForge (the `sf.1`
  session protocol) can take the speech recognition, the end-of-turn score and the
  reply voice, so replies start sooner and this computer does far less work during a
  voice chat. It is **off by default** and stays off until `rig.enabled`, `rig.url`
  (in `voice.yaml`) and `SPEECHFORGE_TOKEN` (the environment, or the 0600 file
  `~/.config/local-chat/speechforge.env`) are all present. Without them nothing
  changes: no socket is opened and the session is the CPU one. The token is never
  stored in settings, logged or sent to the browser.
- This computer stays in charge of the conversation and takes over the speech work
  whenever the rig is slow, busy or away: a turn starts on the rig only when it
  answered recently, a dead rig is noticed in under a second, the words in flight
  replay into the local recogniser, and a reply sentence with no rig audio within
  800 ms is rendered here instead. One reply is never split across engines mid-word.
- The rig's states are all handled: a rig serving without a GPU lease is used (and
  Settings and the Voice lab say so), a rig still loading its models leaves the turns
  here until it is ready, a rig giving its GPU back finishes the reply it is speaking
  and is not tried again before the time it announced, and a `503` with
  `Retry-After` is honoured. An engine error from the rig, or two "busy" answers in a
  row, count as a failure, so the next replies do not pay the same round trip.
- Playback credit follows the rig's own rule: the playback position is re-sent every
  250 ms while a reply is open, and a sentence is never counted late while the rig may
  be holding back on purpose. A long reply stays on the rig: past the rig's per-reply
  text limit (2000 characters) it continues in a fresh context with the same voice.
  A reply's rig context closes as soon as its last sentence is out, so a rig asked to
  give its GPU back is never held up by an idle chat.
- A late final transcript moves only the recognition to this computer, not the reply
  sentence the rig is already speaking; a fresh rig session never reuses the old
  session's transcripts; the voice uploads ignore proxy settings.
- Session hooks for the rig: `cancel{gen}` before the host abort on a barge-in,
  pause/resume with the duck, and a word-accurate heard cut from the rig's word
  timings.
- Only voice conditionals go to the rig (never the reference clip); deleting a voice
  also deletes it from the rig. Built-in plugin voices (Kokoro) stay on this
  computer: a bot speaking one gets the rig's recognition and end-of-turn score, and
  its replies are spoken here. To have the rig speak a bot's replies, give the bot a
  cloned voice (or the stock voice).
- Settings › Voice: *Use the GPU rig when available* with a status line (disabled,
  "Not set up on this computer", until an address and a token exist). The Voice lab
  shows the rig's state, card, engines, GPU lease, limits, deadlines and latencies as
  sentences; `dispatch-voice doctor` prints a GPU rig line. Eight locales.
- Deadlines measured on an RTX 3090 rig over a tailnet: `stt_final_deadline_ms` 350
  (final words 115 ms median, 172 ms worst), `tts_start_deadline_ms` 800 (first audio
  200-275 ms typical, outliers to about 520 ms).
- Harnesses: `tests/latency_ws.py` and `tests/e2e_voice_roundtrip.py` take
  `--rig <ws-url>` (the token from the environment); `latency_ws` adds the `long` and
  `drop` scenarios and records the server's CPU per chat.
- Docs: `docs/voice-chat/rig.md`.

## [1.0.1] — 2026-10-11

A patch release: fixes from the review of 1.0.0. Same extension API (1) and the same
settings; no model or dependency changes.

### Voice chat screen
- The bot's picture no longer jumps to the top at your first word and leaves the
  lower half of the screen empty. The captions area keeps a few lines' room from the
  start and grows with the reply (up to about half the screen, then it scrolls and
  follows the sentence being spoken), so the picture stays where it was and only
  drifts up as a long reply grows. Mute and Hang up stay on screen as before.
- "Low signal — using the phone's voice" is cleared when the chat stops (busy, voice
  switched off, the link gave up) or DisPatch locks; the stopped screen no longer
  shows two lines that contradict each other.
- Mute and Unmute are announced to screen readers on the tap, once, instead of a
  round trip later when the server echoes the state (or never, on a server that
  does not echo it).
- In forced-colours mode (Windows High Contrast) a switch in the quick sheet shows
  whether it is on: a Highlight track with the knob at the end, an outlined track
  with the knob at the start when off. DisPatch's own switches in Settings › Voice
  are DisPatch's to fix.
- The "Show timings" line and the Voice lab's link and MiniMax rows read as short
  sentences ("Heard in 420 ms. Reply started in 1.6 s. 420 ms, 31 kbps, phone
  voice.") instead of middle-dot strings; `drive.timing_link` changed in all eight
  languages.
- The Voice lab names the fallback engine ("Fallback voice: Kokoro.") instead of
  showing its id.
- German: the stock voice is "Mitgelieferte Stimme" (badge "Mitgeliefert"), so it no
  longer shares the word "Standardstimme" with the default voice.

### Spoken replies
- A local phone number ("555-0123") is read digit by digit, not as a range ("555 to
  one hundred twenty-three"). A three-and-four-digit pair still reads as a range when
  it looks like one ("500-1000 people").
- ASCII emoticons (":)", ";-)", ":D", "<3") are dropped instead of being read or
  left in the caption ("run:)").
- With "Skip actions in asterisks", a dropped action no longer leaves its "and" or
  "but" behind ("Then she says let's go and." is now "Then she says let's go.").

### Acknowledgements
- Good news about yourself said with a "!" ("We got the house!", "I passed my
  exam!") gets a new `upbeat` pool of surprise lines ("Oh wow.", "No way!",
  "Whoa.") instead of "Noted." or "Okay."; a negated or bad-news line ("I didn't
  pass!", "Our flight got cancelled!") and "I got it!" do not. `voice-acks.yaml` accepts `upbeat` like the other intents.
- Any "can you believe …?" / "can you imagine …?" is treated as rhetorical and gets
  no thinking line, whatever its length (only short ones did).
- The `trainer` style no longer answers bad news with "Okay.".

### Tests
- The same-chat reconnect test waits for `ready` instead of a fixed pause, and the
  host fixture resets the per-chat registry, learned endpointing and the commit
  cursor between tests (it was order-dependent).
- The doctor tests pass when DisPatch is importable (run from DisPatch's backend
  environment, the way the runbook runs the doctor); the earcon fallback test waits
  for the earcon instead of a 50 ms pause.

## [1.0.0] — 2026-10-11

The first release as DisPatch's voice chat **add-on**: the whole feature (speech
pipeline, session, voices, settings routes and its own frontend) lives here, and
DisPatch 2.2.0 or later hosts it through its extension API 1 (`EXT = {api: [1, 1]}`).
DisPatch runs fully without it. The four milestones of the voice chat v3 plan ship
together in this release.

### Milestone A — the add-on split
- **Voice chat is an add-on.** `setup(ctx)` plugs into DisPatch's extension host:
  routes under `/api/voice` (behind DisPatch's unlocked-only gate), the gated
  `/ws/voice/{thread}` socket, static assets under `/ext/voice/` (content-hashed,
  never cached by DisPatch's service worker), the three voice tools, per-bot field
  validators, a start-up prewarm and a shutdown hook. The profile API moved from
  DisPatch 2.1's `/api/voices` to `/api/voice/profiles` (no alias).
- `dispatch-voice fetch-models` fetches every core model (Silero VAD, Smart Turn
  v3.2, Parakeet TDT 0.6B, Chatterbox-Turbo, the stock voice); `dispatch-voice
  doctor` prints the DisPatch handshake, the models, engines, plugins, the switch,
  the running-vs-checkout version, whether DisPatch's venv has the add-on and
  whether the editable overlay resolves offline (`--backend`), and the latest 1.x
  release (`--check-update`).
- **Drive mode is now called Voice chat** everywhere a person reads it: the UI in
  eight languages, this README, the docs, the standalone Voices page and the
  spoken-reply instructions, which no longer mention driving. Settings keys and
  file names keep their internal names, so existing settings carry over.
- **Spoken-reply instructions** (`text/persona.py`) no longer go in front of the
  person's message (bots used to answer "Hands-free mode, got it…"). Advisor and
  API bots get a "Spoken replies" section in the system prompt; gateway agents a
  short app note at the END of the message. Bots keep their own personality
  ("podcast host" only for a bot without one) and are told never to remark on the
  instructions.
- The power-profile hold: while a voice chat is open the add-on holds the
  `performance` profile through power-profiles-daemon / tuned-ppd (`rt.power_hold`,
  fail-soft).
- MIT licence kept; the Kokoro voices are a separate GPL-3.0-or-later plugin
  (`dispatch-voice-kokoro`); libopus notice added (`THIRD_PARTY_NOTICES.md`).

### Milestone B — conversation
- The session announces every spoken sentence with a `say` frame before its audio
  (and `say_end` after the last one), numbers sentences when they are queued, and
  has a text mode in which the phone reads the `say` and the server never
  synthesises.
- **Acknowledgements** (`text/acks.py`, `text/ack_lines.py`): 70 default lines (62 distinct)
  over seven intents plus per-character pools (`wry`, `bright`, `trainer`, `gruff`),
  chosen by what was said, never repeating any of the last four, only when the reply
  is really going to take a while; spoken status lines for tool calls; wait lines at
  5 s and 15 s. No more "good question" after a statement, and no listening noise
  after a greeting, a thank-you or "I love you". They are rendered once per voice in
  the background (`host/acks.py`, `<data>/voice-cache/`) with a fixed seed;
  operators override pools in `<data>/voice-acks.yaml`. The per-profile fillers are
  no longer played.
- **Speech text** (`text/speech_text.py`): markdown, numbers, money, dates, times and
  units are read the way people say them; emoji and symbols are dropped; role-play
  replies can skip `*actions*`, speak only dialogue, or give actions to a narrator
  voice (`rp_speech`, per-bot `voice_rp`, `narrator_voice`); one-word `*emphasis*`
  stays in the sentence.
- Interrupting a reply persists what was spoken so far and marks the row with how
  much was heard (`voice_heard`), from the client's `heard` frames; the next turn's
  instructions carry the cut-off note once.
- Settings: `rp_speech`, `narrator_voice`, `expressive_tags`, `gateway_trailer` and
  per-bot `bots.<id>.{backchannel, follow_up}` in `voice.yaml` and
  `/api/voice/settings`; `/api/voice/status` reports them and the acknowledgement
  render status; `POST /api/voice/acks/rebuild`. Settings › Voice is one page of six
  groups, a quick sheet on the voice chat screen, and a Voice pill with a popover on
  each Bots-tab card.

### Milestone C — realtime
- **The end of a turn** is decided at every pause from Smart Turn and the streaming
  transcript together (no more flat 1.8 s hold on confident turns; "Got it — go
  on?" via `eot_soft` when it holds); the pause it waits for adapts to how you talk;
  captions while you speak (`partial`).
- **Speculative replies:** the advisor's reply starts while the last words are still
  being transcribed and is kept when the final text matches (≤ 2 per turn).
- **Pipelined Chatterbox:** first audio from the first 25-token chunk, not the whole
  sentence, with a 4 s reference prompt.
- **Talking over a reply** ducks it on the phone within ~100 ms and the words
  decide: our own echo, an "mm-hm" or a cough resume it (rewound when the pause was
  long), "wait" / "stop" cut it at once, anything else cuts it and becomes your next
  message; "oh, and…" a couple of seconds after your question joins it; "say that
  again" repeats the last reply without asking the bot; "Tap to continue" after a
  stop picks the reply up again from the sentence you were in.
- The bot can say "mm-hm" while you talk (per-bot opt-in, Reactions "Often");
  "Still there?" when the client's hang-up-when-quiet timer fires.
- The client plays replies through a player worklet (pause, rewind, sample-accurate
  `heard`) and shows a caption only for a sentence that started playing; the ready
  tone waits until the microphone is really capturing.
- `tests/latency_ws.py` / `tests/latency_harness.py`: the latency budget measured
  through the real session, idle and under load, per power profile.

### Milestone D — low bandwidth
- **Protocol v3:** Opus both ways (ctypes libopus on the server, WebCodecs in the
  browser) with a PCM fallback, sample offsets on the uplink, a persisted commit
  cursor so a reconnect or a restart never submits an utterance twice, and a replay
  ring of the last 30 s of what you said.
- The client watches the link (round trip, playback lead, underruns) and hands
  replies to the phone's own voice — or Wait, or text — with a line on the screen
  saying so, and steps back when the link recovers; "Waiting for signal" for up to
  15 minutes on a visible page.
- A reply cut by a drop resumes after the sentences already heard (also one stored
  before the drop, and one only named by its temporary id); reply audio is paced at
  most 4 s ahead of playback (1.25 s for PCM), so on a slow link the phone voice's
  text and the heartbeat are never stuck behind audio nobody will play.
- `tests/shaper.py` + `tests/e2e_lowbw.py` (seven network profiles), and
  `tests/e2e_browser.py` (headless Chromium in the real DisPatch shell, against a
  scratch DisPatch with the real models and against the v3 reference server).

### Final review (2026-10-11)
- **Voice chat no longer gets locked mid-conversation by DisPatch's idle timer**: the
  client tells the shell it is active (`shell.activity()`, DisPatch 2.2.0) on every
  conversation frame and every 20 s while listening; the server ends a chat nobody
  speaks in after `rt.quiet_max_s` (40 min) with close code 4002 "quiet".
- **The box's default voice is resolved before deciding anything about it**: on a box
  whose default is a Kokoro voice the prompt no longer invites `[laugh]` (Kokoro read
  it aloud), the acknowledgement pool keeps word lines only ("Huh, okay." was heard
  as "Hello K."), and the first-audio estimate is the built-in one. Tag capability
  is decided by the engine that actually speaks a voice.
- One sound per acknowledged turn: the line goes out 0.55 s after the end of the turn,
  inside the client's 800 ms tone hold (both used to play, 50 ms apart).
- "Okay, okay, got it, thanks" over a long reply stops it (up to eight tokens of
  stop phrases and acknowledgements) instead of starting a new turn; "Thanks, that
  helps." gets no reaction line; numeric dates ("12/24/2025") are read as dates.
- The voice chat screen: the captions follow the sentence being spoken and give way
  so Mute and Hang up never leave the screen (also with the phone on its side, in two
  columns); Undo stays visible with Captions off; replies shown as text appear even
  with Captions off; "Pause" is now **Mute** (it only ever muted the microphone);
  hints follow Tones / Interrupt settings; a phone without a voice is no longer
  reported as "Low signal"; selects show the focus ring; the dialog is named "Voice
  chat with {bot}"; Settings controls are described by their hints; forced-colours
  mode keeps the chosen segment visible; the client also rewinds after its own long
  pause.
- Settings › Voice: "Status" is now "Advanced", "This device" is "Screen", the
  server-wide groups say so, the Reactions hint is accurate, middle-dot meta strings
  are gone, and the Voice lab's dead "keep-talking window" is hidden on a streaming
  server.
- Bots: the advisor's voice section names its owner and a persona-less API bot gets
  the conversational default (`BotView.owner` / `has_persona`); the spoken rules no
  longer send a hands-free listener to the chat.
- The tool-status table is generic; install-specific tool names go in
  `voice-acks.yaml: tools`. A new clone inside DisPatch no longer renders the retired
  fillers.
- `dispatch-voice doctor --unit <name>` reads the service's own voice settings;
  `dispatch-voice serve` needs `--data` (never DisPatch's live voices) and answers only
  its own loopback Host (DNS rebinding).
- Notices: libsndfile (LGPL) via soundfile, the optional Moonshine recogniser, and an
  accurate account of the dev scripts that can use Kokoro; CI actions pinned by commit.
- Tests: `e2e_browser.py`'s `livelike` scenario runs a scratch DisPatch set up like a real
  install (entry points, the Kokoro plugin, a PIN, a 60 s idle lock) and proves a silent
  voice chat outlives the idle lock while an untouched page still locks; model-backed
  pytest runs exit with pytest's own status; the round-trip and latency harnesses take
  `--llm-port`.

### Fixed (during development)
- Hanging up (or the link dropping) while the session waited for the phone's "where
  were you" frame after an interruption left the session running, holding the power
  profile and a voice slot, and sent the cut words as a message after the socket was
  gone.
- The note about what was heard before an interruption could carry the previous
  reply's words, lost the sentences before the reply's id switched, and ignored the
  phone's exact answer when it came a moment late.
- A tap-to-stop or a cancel stalled the connection for up to 1.5 s; `say_end` could
  arrive before the last sentence's audio.
- Speech that started while the previous question was being sent held the reply for
  ever or replaced the question instead of joining it; an acknowledgement could play
  over someone still talking; a sentence that began by quoting the bot was thrown away
  as its echo; a malformed frame could end the voice chat; a speculative reply kept
  running after a hang-up; a restart overwrote this device's learned pause.
- A resume after a drop, and "Tap to continue", no longer lose a sentence: a reply's
  sentences depend on its text alone, so every session counts the same ones.
- A reply whose text is rewritten while it streams is no longer spoken as two texts
  spliced mid-word; a reconnect no longer leaves the old session running for 30 s
  beside the new one; inside a burst of replayed audio, where an utterance ended is
  exact; a full uplink queue no longer stalls pings and taps; a big replay is decoded
  off DisPatch's event loop.

## [0.1.0] — 2026-10-09

The cloned-voice engine for voice chat in DisPatch 2.1: Chatterbox-Turbo ONNX (q4) on
the CPU, voice profiles from a 10–20 s clip with a recorded consent, a stock synthetic
voice, the standalone Voices page, and `fetch-models`.
