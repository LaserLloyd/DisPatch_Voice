# Using voice chat

<!-- Owner: track T7 (client app). -->

Voice chat lets you talk with a bot hands-free: you speak, the bot answers in
its own voice, and you can talk over it to interrupt. It works in any chat
whose bot is not a Safe-Mode bot, on a device that is unlocked. The engines
understand and speak **English only for now**; the screen says so the first
time it opens in another UI language, and Settings › Voice says so under the
switch.

## Opening and closing it

- Open a chat and tap the **microphone** in the chat header. The first time,
  the browser asks for the microphone. DisPatch must be opened from its
  `https://` address (the address bar shows a lock), or the browser refuses.
- Voice chat fills the screen. **Hang up** (the wide button) closes it, as do
  Esc and the phone's Back gesture. A toast says "Voice chat ended."
- Voice chat runs **with the screen on**. Locking the phone mutes the
  microphone, and it says so when you come back ("Muted while the screen was
  off").
- A voice chat that is in use never lets DisPatch lock itself for being idle
  (talking makes no touches); a chat nobody speaks in for 40 minutes is ended by
  the server ("Voice chat ended.").

## The screen

One picture in a ring tells you what is happening; the word under it names
the state, and the line under that says what a tap on the picture does.

| Ring | Word | A tap on the picture |
|---|---|---|
| a breathing dot on top | Listening | mutes the microphone |
| green | Hearing you (also while a reply is held because you started talking) | mutes the microphone |
| amber arcs | Thinking (a counter after 4 s) | cancels the turn |
| accent, moving with the voice | Speaking | interrupts the reply |
| dashed, picture greyed | Muted | unmutes |
| dashed, red | No connection / Waiting for signal | — (it retries by itself) |

- **Mute mutes only the microphone.** A reply that is already playing
  finishes; the screen says **"Muted — still speaking"** and a tap on the
  picture interrupts it. **Unmute** (or a tap when nothing plays) listens again.
  (This button was called Pause before 1.0.0, with a pause glyph that promised a
  media pause it never did.)
- **One sound per turn end.** When you stop talking you hear either the
  short "got it" tone or the bot's own quick reaction ("Let me see"), never
  both: the tone waits 800 ms and is dropped when a reaction or the reply
  starts.
- **"Got it — go on?"** appears when DisPatch is holding your turn open
  because you might not be finished (only for holds of 0.6 s or more).
- **Captions** show your words while you speak — settled words in full
  colour, the part that may still change dimmed — then what was heard
  ("…") and the reply sentence by sentence, the one being spoken
  highlighted. A sentence appears when it starts to play (or when the
  phone's voice starts it), so after an interruption the captions never show
  sentences you did not hear. With role-play text set to "Dialogue, plus a
  narrator voice", the narrator's sentences are in italics. The captions
  **follow the sentence being spoken**: it stays in the upper part of the
  captions area, which takes the room the picture does not need once there are
  captions; scroll back by hand and following pauses for a few seconds (the top
  edge fades while the area is scrolled).
- **The buttons never leave the screen.** The picture shrinks and the captions
  scroll instead; with the phone on its side (a landscape mount) the picture is
  on one side, the captions on the other and Mute / Hang up across the bottom.
- **Talking over a reply.** The reply drops to a murmur and pauses the moment
  you start talking (the ring says "Hearing you"). If it was a cough or a
  quick "mm-hm", the reply picks up again — from the start of the sentence
  when the pause was long. If you keep talking, the reply stops and what you
  say is your next turn.
- **The ready tone** plays only once the microphone is really capturing, so
  nothing you say after it is lost, even when a slow connection makes the
  first start take a few seconds.
- **Undo** appears under the state word for 8 seconds after each message you
  said, until the reply starts — also with Captions off. Keyboard: **U**.
- The first three times voice chat opens on a device it shows the tip
  "Talk over me to interrupt, or say 'stop'." ("Tap the picture to stop a
  reply." when Interrupt by talking is off). With Tones off the connecting hint
  says "Start talking when it says Listening" instead of "after the tone".
- Saying **"okay, okay, got it, thanks"** (or "no thanks", "that's enough"…)
  over a long reply stops it without starting a new turn.
- Keyboard: **Space** mutes/unmutes the microphone, **Esc** hangs up, Tab
  stays inside the voice chat screen and focus returns to the microphone
  button afterwards.

### When the connection is slow

The voice chat watches the link. When it can no longer keep up (gaps in the
reply, a sentence that does not arrive in time, a very slow round trip), it
switches as **When the connection is slow** says, and a line on the screen
says which:

- **Phone voice** (the default): "Low signal — using the phone's voice". The
  rest of the reply, and the next ones, are read by the phone's own speech
  engine; nothing is read twice. A slow-but-steady link switches only between
  replies, never in the middle of one.
- **Wait**: "Low signal — waiting for the voice". The bot's own voice stays;
  each sentence waits until it has fully arrived, then plays without gaps.
- **Text only**: "Low signal — replies are shown as text". The reply is shown
  on this screen, not spoken — whatever the Captions setting says.

When the link has been healthy for a while it steps back to the bot's voice
by itself. A device with no phone voice installed falls back to text; with
"Always use the phone's voice" on such a device the line says "This device has
no phone voice — replies are shown as text" instead of blaming the signal.

### When the connection drops

It reconnects by itself (0.5 s, doubling to 8 s). After a minute without
signal the screen says **Waiting for signal** and keeps trying every 15
seconds, and at once when the phone comes back online or the screen comes
back on; after 15 minutes it stops and offers **Try again**. What you say
while there is no signal is kept ("No signal. I'll send that when we're
back.") and sent when the connection returns; anything older than 20 seconds
by then is dropped ("That got cut off. Say it again."). A reply that was
interrupted by the drop continues where it stopped, without repeating the
sentences you already heard. Errors are written on the screen, never only as
a toast: "Voice is busy on this server" (too many voice chats are open),
"Voice isn't set up on this server", "This device is locked", and so on.

### Hang up when quiet

After the chosen time with nothing said (5, 15 or 30 minutes; 15 by
default; or Never) the bot asks "Still there?". If you don't answer in the
next 20 seconds, voice chat hangs up. Only you can answer it: talking, a tap,
the Mute button or a steering-wheel play/pause press. The bot saying
"Still there?" does not count. While the microphone is muted nobody can
answer by voice, so after the same time with nothing said voice chat hangs up
without asking.

## The quick sheet (gear on the voice chat screen)

What people change mid-conversation, in large rows, applied at once:

- **{Bot}'s voice** — sets this bot's voice (everywhere, saved immediately).
  The server-wide default voice stays in Settings › Voice.
- **Pause before replying** — Quick / Normal / Patient.
- **Interrupt by talking**.
- **Reply length** — Short / Normal / Longer.
- **When the connection is slow** — Phone voice (only when this device has
  one) / Wait / Text only. Without a phone voice it highlights what will really
  happen (Text only) and says why.
- **Microphone**.
- **All voice chat settings** opens Settings over the voice chat with the
  microphone muted. Nothing hangs up; tap the picture to carry on.

## Settings › Voice

Six groups, nineteen rows at most, plus a collapsed **Advanced** group.
"Server" rows are shared by everyone on this DisPatch (their groups, and the two
server rows under Advanced, say "Shared by everyone on this server"); "device"
rows apply to this browser only. Every control is described by its hint for a
screen reader, and every select shows the app's focus ring.

| Group | Row | Scope |
|---|---|---|
| Voice chat | Use voice chat (on/off, with the English-only note) | server |
| Voices | Default voice (▶ plays a sample) | server |
| | Speaking speed (built-in voices only) | server |
| | Role-play text: Speak everything / Skip actions in asterisks / Dialogue only / Dialogue, plus a narrator voice | server |
| | Narrator voice (only for the narrator choice; "(not installed)" when missing) | server |
| | Manage voices (opens the Voice cloning tool, when it is on) | — |
| Listening | Pause before replying (Quick / Normal / Patient — it adjusts itself to how you pause) | device |
| | Interrupt by talking | device |
| | Keep listening after a reply (off = push to talk) | device |
| | Microphone | device |
| Replies | Reply length | device |
| | Ask a follow-up question: Bot's default / On / Off ("Bot's default" uses each bot's own setting from the Bots tab) | device |
| | Reactions: Off / Sometimes / Often — short words in the bot's own voice ("let me see" while it works; at Often also "mm-hm" while you talk, for bots that opted in) | device |
| | Tones (the short sounds) | device |
| Connection | When the connection is slow: Phone voice / Wait / Text only / Always use the phone's voice | device |
| | Phone voice (only for the two phone choices; on Android the phone's own Text-to-speech setting decides, so this row says where it is; elsewhere a list of offline voices with ▶) | device |
| Screen | Keep the screen on | device |
| | Captions | device |
| | Hang up when quiet: 5 min / 15 min / 30 min / Never | device |
| Advanced (collapsed) | Engines (in a sentence each) — Speaker (desktop browsers that can choose one) — Thinking during voice turns (server) — Show timings on the voice chat screen — Expressive tags (server, cloned voices) — Voice chats open and the power hold | mixed |

While voice is off, the page shows only the switch (and, when something is
missing on the server, what and the commands that fix it). If the page
cannot reach the server it says so and offers **Try again**.

**Privacy of the phone voice:** with "Phone voice", replies are read by the
phone's own system speech engine. Google's engine speaks offline when its
language pack is installed; the page cannot verify which engine is used.

## Bots tab

Each bot card (except Safe-Mode bots) has a **Voice** pill. It opens a
small panel:

- **Voice** and **Role-play text** (Server default or one of the four) —
  saved with the Bots tab's **Save**, like the other pills.
- **Reactions while I talk** (this bot may say "mm-hm" while you talk, when
  Reactions is set to Often) and **Asks a follow-up** (this bot's default) —
  applied at once, for everyone on this server.

## The voice tools

Three opt-in tools (switched on in Settings › Tools): **Voice cloning** (add
a voice from a 10–20 s clip, set the default, rename, delete), **Built-in
voices** (an optional separately installed plugin) and **Voice lab**:
engines (speech recognition, the cloned voice's sample length), turn-taking
(end of turn, thinking, the pause learned for this device with Reset; the old
"keep-talking window" shows only on a server without streaming recognition,
where it still applies),
acknowledgements per voice with Rebuild, the last voice chat's numbers on
this device (early replies started/used/dropped, round trip, bitrate, which
voice spoke, the power profile), Show timings, and a 3-second
hear-yourself-back test.

## For developers

### Files (`src/dispatch_voice/static/`)

| File | What |
|---|---|
| `index.js` | the entry DisPatch imports: `EXT`, `mount(shell, {assetsVersion})`; wires every shell slot and the voice status |
| `shell.js` | the add-on's handle on DisPatch's frozen `shell` (C-§E.1): `t`, `el`, `railIcon`, toasts… with plain fallbacks |
| `bridge.js` | the one seam to the audio & link modules (C-§E.2): real modules when they export the contract names, fallbacks otherwise |
| `drive.js` | the voice chat screen, the socket's life and the pure helpers behind it |
| `drive-audio.js` | the audio & link engine (no DOM): wire v2/v3, player worklet, mic gate, Opus up/down, link monitor → phone voice / Wait / text, uplink ring + cursor replay, `after_partial`, claims, `heard`, the duck |
| `drive-settings.js` | the device settings and their migration, the `start` frame, Settings › Voice, the quick sheet, the Bots pill |
| `voice-tools.js`, `voices.js` | the three tools and the Voices manager |
| `voice.css` | the screen and the ring; everything else uses DisPatch's own classes and theme tokens |
| `locales/<lang>.json` | the add-on's strings in 8 languages (`drive.*`, `voices.*`, `vtools.*`) |

Internal names keep the old `drive` prefix (file names, CSS classes, the
i18n namespace, the `dispatch-drive` storage key) on purpose: renaming them
would drop every device's settings for nothing a user can see.

**One module instance per file.** The entry imports every sibling with the
server's assets hash (`?v=…`), and every module imports its siblings through
its own `import.meta.url` query, so a cold open is one round of immutable
fetches and no file is ever loaded twice under two URLs.

### Device settings (`localStorage['dispatch-drive']`, C-§F.3)

`v: 3` marks the current layout. Older blobs migrate on first read: `acks`
becomes `reactions` (and `backchannels: true` with acks on becomes "often"),
`partials` folds into `captions`, and a pre-v3 `followUp: true` (the old
default) becomes `null`, meaning "this bot's own default". `start` carries
`acks` = reactions, `rt.backchannels` = (reactions is "often"),
`rt.partials` = captions, `slow_link`, and `follow_up` only when the device
chose. If the device goes back to "Bot's default" while a voice chat is open,
the next `start` carries `follow_up: null` once, so the session drops the
value it was told (Amendment T7a-7).

### Connection details

- **Heartbeat:** a `ping` every 2 s from the moment the socket opens. Before
  `ready` the server is loading models and answers nothing, so the socket
  only counts as dead after 60 s. From `ready` on it is dead after 8 s of
  silence, or 4 × the round trip once that is above 1.5 s.
- **Captions** are keyed by `(sid, seq)`: a new session numbers its
  sentences from 1 again and they are new captions. A caption is added when
  its sentence first reports played audio (`played.ms > 0`), when the phone
  voice starts it, or at once in text mode — never at `say`.
- **The ready tone and Listening** wait for both `ready` and a capturing mic
  worklet (on EDGE the worklet fetch alone took ~3 s after `ready`; words in
  that window were lost). `index.js` also prefetches both worklet files into
  the HTTP cache once voice is on (they are immutable under `?v=`).

### The audio path (`drive-audio.js`)

`drive.js` owns the screen and the socket's life; `createDriveAudio()` owns
what moves through it, with every browser object and clock injected
(`tests/js/drive-audio.test.js` runs it under node on the real T8 modules).

- **Negotiation (C.1).** Every socket asks for `v=3` with the voice chat's
  stream id and the codec lists from `opusSupport()`. A server without
  `ready.proto` gets today's v2 wire: raw PCM16 up, `audio` header + PCM
  down, `playback_done` on every drain (also after a reconnect to a server
  that came back as v2). If its `ready.say` is true (a server that announces
  sentences but still sends v2 audio), the client also routes `say`, sends
  `heard`, and honours "Always use the phone's voice"; link monitoring,
  `link` and `probe` need `ready.proto >= 3` (Amendment T7b-1).
- **Uplink.** Mic frames carry their 16 kHz offsets; before the first
  `ready` they are buffered (10 s), then go through `createUplink` (Opus
  from `ready.codec_in`, else PCM16) and out like a ring replay at 4×
  realtime: 10 s of PCM16 in one burst would overflow the server's
  64,000 B audio queue. `ready.gate` turns the worklet's gate
  on; `{hold}` follows `state` hearing/transcribing. The encoder starts at
  `ready.opus_up_kbps`; the link monitor's uplink rung (16/12 kbps)
  reconfigures it at once.
- **Reconnect (C.5 #10, T6-1/T6-2).** Every uplink message is kept in the
  ring (30 s, capped at 320 KB: 10 s of PCM16). The new socket's `start` carries `cursor {stream, from =
  max(rx_off, transcript upto, eot upto)}` and, mid-reply,
  `after_partial {message_id, sentences}` = the leading sentences whose audio
  arrived complete (`audio_end` on v3, where `say_end` comes before the last
  sentence's audio; the next `say`), that the phone voice holds (it keeps
  speaking across the drop), or that were captioned — counted on
  from earlier resumes and never twice for one session. A reply is one
  reply across its provisional `run:…` id and its row id (T3s2-1; the frame
  names the newest, and a reply cut under its provisional id goes on under
  the row id the next session speaks). It is not resumed once it is
  settled: cut (`barge_in` — in a later session whatever its gen — or a
  tap-stop), failed (`turn_error` other than `tts_failed`), followed by a new
  transcript, or said in full (`say_end`) with every sentence complete.
  The session speaks only replies stored AFTER `start.after`, so while such a
  resume is pending `after` is the user row the reply answers (the last
  `transcript.message_id`; else the newest reply heard in full) and the cut
  reply's ids are left out of `after`/`after_ids`; once the resumed reply is
  said to its end, `after` is the newest reply again. Live audio waits
  until the ring has been replayed from the server's `cursor {from}` (or our
  own after 3 s), at 4× realtime; stale bursts show "That got cut off".
- **Downlink and the player.** wire3 frames go through `createDownlink`
  (WebCodecs Opus, PCM16 fallback) into the player worklet; `say` precedes
  audio, the next `say`/`say_end`/`audio_end` sends the player `end`.
  `playback_done {gen, seq}` goes on `idle {ended: true}`. After a
  reconnect the player keeps playing what the old session delivered
  complete, and the new session's audio waits behind it (at most 15 s) — a
  `player.reset()` at `ready` would cut what `after_partial` skipped. The
  phone voice does the same with what it holds (at most 60 s). Meanwhile
  nothing about the old session's sentences reaches the new one (`heard`,
  `playback_done`), and a tap-stop silences the old audio at once. Every
  `ready` starts gen at 0 again (C.5 #9), so the old session's gen never
  gates `resume` or the floor a tap-stop flushes to.
- **Undo.** The server answers `undo` with `barge_in {gen + 1}` and then
  `undone`; the client does not flush again on `undone` (that would drop the
  next reply, which comes at the same gen).
- **"Tap to continue"** after a stop is offered only when `ready.replay_last`
  (or `ready.rt.replay_last`) is true — `replay_last` is optional (C.3).
- **Claims and the slow link.** A sentence belongs to the first of server
  audio, phone voice or caption (`(sid, seq)` claims). When the link monitor
  goes to text, the reply in flight's trailing sentences still waiting for
  audio go to the phone voice (or the caption) — never an older sentence
  whose speech failed and `link {mode: "text"}` is sent (predictive causes only
  after `say_end`). "Wait" never asks for text: it asks for the lowest
  downlink rung (always, also when the event carried no rate) and
  prebuffers whole sentences. If the phone voice gives up while it holds the
  reply's last sentence, that sentence is captioned and `playback_done` still
  goes out. "Always use the phone's voice" pins text
  from `ready`. A decoder failure sends `link {why: "decoder", down_kbps:
  null}` (the server switches to PCM16 from the next sentence) and the next
  socket asks for `down=pcm16`; no forced reconnect.
- **Talking over a reply (§4.6).** A mic speech edge during a reply ducks
  the gain to 0.15 and pauses the player (the optimistic duck, not in the
  first 1.5 s), with `duck_local`; the server's `duck`, `resume {rewind}`
  and `barge_in` do the rest. When the phone paused first, the rewind rule
  (> 800 ms → restart the sentence) is judged by the phone's own pause as well
  as the server's `resume {rewind}` (the server's clock starts later). While the
  phone voice speaks a server `duck` is informational. After `barge_in` one `heard {gen: new, seq,
  played_ms, source}` goes at once (read before the flush); a tap-stop puts
  the same object on `stop_playback`/`cancel`/`undo`. `heard` also goes
  alone once a second while a reply plays.
- **Bots tab pill:** DisPatch renders the roster when Settings opens, not on
  a tab switch. The add-on therefore returns a slot for every non-Safe bot
  and fills or hides it whenever the status changes. Turning voice on in
  Settings › Voice shows the pills on the Bots tab straight away.
- **Bots popover** is shifted sideways, and narrowed if necessary, to stay
  inside the scrolling Bots pane with an 8 px gutter; its ▶ keeps the button
  chrome and a voice row wraps so the select gets the full width.
- **The idle lock.** `shell.activity()` (DisPatch ≥ 2.2.0) on every conversation
  frame and every `ACTIVITY_EVERY_MS` (20 s) while the chat is open, listening and
  visible; close code 4002 ("quiet", the server's own bound) hangs up without a
  reconnect.
- **Layout.** `.drive-col` is the viewport's height and the picture takes the
  slack. The captions area holds 3.5 lines' room from the start (so the picture does
  not move at the first word), grows with its text up to 55 % of the viewport and
  then scrolls, following the `now` sentence (`follow()`, paused for `FOLLOW_PAUSE_MS` after a wheel/touch/pointer on
  the area). `(orientation: landscape) and (max-height: 500px)` switches to two
  columns with the bar across the bottom.

### Tests

`npm run test:ci` (Node ≥ 20, jsdom): `drive.test.js` (the screen under
jsdom with fake audio, socket and timers; the player node runs the real
`PlayerCore`), `drive-audio.test.js` (the engine on the real T8 modules with
T8's fake clock, codecs and speechSynthesis), `settings.test.js`,
`voice-tab.test.js`, `index.test.js`, `absent-interplay.test.js`.
`python3 scripts/check-locales.py --strict` checks the 8 bundles.

**Browser e2e** (`tests/e2e_browser.py`, headless Chromium through
Playwright, a controllable fake microphone and a fake `speechSynthesis`):

```
# the v3 wire: the real DisPatch shell, the voice socket through tests/shaper.py
# to the reference v3 server (no speech models)
uv run --offline --with playwright python tests/e2e_browser.py --target ref
# the server as it is: a scratch DisPatch with the add-on and real speech models
flock "$VOICE_LOCK" uv run --offline --with playwright \
    python tests/e2e_browser.py --target dispatch
```

Ports `--port-base` (19075) to +3; results in `--out`, screenshots in
`--shots`. v3-only checks against a v2 server are reported as SKIP with the
reason. The dispatch target's `livelike` scenario starts its own scratch
DisPatch set up the way a real install is: the add-on found by its entry point,
voice switched on by `voice.yaml`, the Kokoro plugin on the command line when
its checkout and models are there (`$DISPATCH_VOICE_KOKORO_SRC`, default the
sibling `dispatch-voice-kokoro`), a PIN and a 60 s idle lock. A page nobody
touches must lock (the control); a voice chat left listening for 95 s must
still be open and unlocked; the next questions must be answered aloud, and with
a built-in default voice every acknowledgement must be a word line.

## Phone checklist (manual, on a Pixel in the car)

Run before each release that changes the client (SPEC §12):

1. `speechSynthesis.getVoices()` dump on Chrome Android (expect one entry per
   locale, `localService` true); the Android hint shows instead of a list.
2. `AudioContext.sampleRate` and `outputLatency` with the car kit connected
   (Bluetooth SCO can force 8/16 kHz).
3. Phone voice over car Bluetooth: no cancel-and-repeat loop; a word-confirmed
   "stop" cancels and nothing is re-spoken.
4. Echo level with the phone voice (sets the starting echo bias).
5. Screen-on rule: locking the phone mutes; unlocking says so and listens.
6. The steering-wheel play/pause buttons after a reply; `rate` honoured.
7. Waiting for signal through a tunnel: one message, not two, when back.
8. The relay indicator in the Voice lab after a voice chat on cellular.
