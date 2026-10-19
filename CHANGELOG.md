# Changelog

All notable changes to DisPatch Voice.

## [Unreleased]

### Fixed (ops check 2026-10-19)
- **DisPatch 3.0 no longer copies `voice.yaml` into its settings store at start.** The host calls
  `migrate_voice_yaml(writer, dry_run=False)` once its store exists, but this release reads and
  writes `voice.yaml` only: the store kept a frozen copy of the first start (a later Settings change
  went to `voice.yaml` alone, so `GET /api/settings` and the backups' `settings.json` showed stale
  voice values), and the `voice.yaml.migrated` marker would have kept the release that reads the
  store from ever taking the current file. A real run now answers `deferred` and writes nothing
  until the add-on reads the store (`migration.STORE_IS_READ`); a dry run and `force=True` are
  unchanged. A rollback to 1.5.0 was never affected (`voice.yaml` is left as it was).
- **A rig on the tailnet is no longer "degraded".** A plain `ws://` rig address on the tailnet
  (the 100.64/10 range or a MagicDNS name) made the status page show the rig degraded and the
  journal warn "not encrypted" at every voice chat, although WireGuard encrypts that hop and the
  warning's own advice was "keep it on a VPN or tailnet". Other plain `ws://` addresses off the
  machine still warn.
- **The test suites can no longer touch the real home folder.** `DELETE /api/voice/rig/token`
  (`rig.clear_token()`) removes every token file it knows, including the pre-1.6 one under the
  home folder; a test isolated only the new path, so running the suite on a machine with DisPatch
  installed deleted the live rig token (`~/.config/local-chat/speechforge.env`) and the running
  DisPatch fell back to the CPU. Both suites now give every test its own empty `HOME` (a `models`
  test keeps the real one for the downloaded models, with the token file redirected).

### Changed
- **The host-independent engines are their own package, `forge-voice-core`** (`packages/forge-voice-core/`, import
  `forge_voice_core`): the CPU engines (Silero VAD, Smart Turn, Parakeet, Moonshine, Chatterbox-Turbo, the OpenAI-compatible
  clients), the provider interfaces, the settings-schema primitives and the engine-level settings rows, the model registry
  and download job, the system-requirement checks, the network address policy and the GPU-rig (sf.1) client (extra
  `speechforge`). `dispatch-voice` now depends on it; DisPatch behaves as before (logger names, environment variables,
  settings, wire and defaults are unchanged). Every moved import path (`dispatch_voice.host.tts`, `dispatch_voice.rt.*`,
  `dispatch_voice.engine`, `dispatch_voice.audio`, ...) stays as a deprecated alias of the core module (the same module object).
  The host contract another application implements is `packages/forge-voice-core/docs/HOST_API.md`.

### Fixed
- **A plain `pip install` or `uv pip install` of the add-on works again** (git URL, archive, sdist, wheel). 1.6.0 reached
  `forge-voice-core` only through `[tool.uv.sources]`, which pip never reads and uv reads only for a directory or a git
  checkout: an archive URL failed with "No solution found" and a directory install left an editable `.pth` pointing at the
  checkout (the Docker image then failed with `No module named 'forge_voice_core'`). `dispatch-voice` now declares
  `forge-voice-core[speechforge]` as an archive-URL direct reference to `packages/forge-voice-core` on `main` (no git needed; the commit that is tagged names its own tag); the uv source stays for
  checkouts. CI builds the wheel and the sdist and installs each into a clean venv with plain pip
  (`scripts/check_installable.py`). The sdist now contains only the sources, not every untracked folder.

### Added
- `forge_voice_core.hostlink`: the one seam between the engines and an application; `dispatch_voice/_core_bridge.py` is
  DisPatch's implementation.
- `dispatch-voice --version` prints `dispatch-voice <version>` (it was rejected as a missing subcommand).

### Fixed
- **Security:** a `module:setup` target in the `extensions` setting (writable by any operator
  through Settings / `PATCH /api/voice/settings`) was imported and called at the next start, so
  `sys:exit` stopped DisPatch at every boot. Module targets are honoured only from
  `DISPATCH_VOICE_EXTENSIONS`; the setting names installed plugins.
- **Security:** the profile-deletion sync and the doctor's probe sent the rig token to `rig.url`
  without the network policy; an address the policy refuses never receives it now.
- A public rig on the `rig.allow` list was refused at the voice-chat gate (the chat stayed on
  the CPU while the status page said the rig was ready).
- Windows: a token file or `voice.yaml` whose ACL granted a second account (or any group other
  than the well-known "everybody" ones) counted as private; now anybody but you, SYSTEM and
  Administrators makes it not private.
- A Chatterbox model move that stopped half way (a file in use on Windows) left the engine
  pointed at the half-filled new folder and the rest of the move marked done; it now keeps
  loading from the complete set and a retry finishes the move.
- A plugin's STT/TTS provider (or `stt_provider: rig`) could be selected but voice chats never
  used it; the status health and the provider row now say what runs instead.
- The README screenshots moved to `docs/screenshots/` with an `ALLOWED.txt`, so the privacy
  check (CI and the git hooks) passes again.
- `dispatch-voice doctor` inside the DisPatch container read `~/.local/share/local-chat` and so
  reported the models missing and the switch off: it now honours `LOCAL_CHAT_DATA_DIR` (the legacy
  name DisPatch still answers to) after `DISPATCH_DATA_DIR`, the same order DisPatch uses. The rig
  token file path follows the same rule.
- `dispatch-voice doctor` reported "no SPEECHFORGE_TOKEN" for a token saved in the host's file
  credential store (`<data>/ext/voice/secrets.json`); it now finds it there, and when it finds no
  token the line names every place it looked.

## [1.6.0] — 2026-10-18

One Settings → Voice page, one definition per setting, providers behind one interface, and the
first Windows-supported release. No change to the voice session itself.

### Added
- **Settings → Voice in four sections**, in order: 1 Speech to text, 2 Text to speech, 3 Input
  and devices (input mode, patience, barge-in, microphone and speaker, and a **microphone test**
  with a level bar and a latency readout), 4 **Voices per character** (voice, role-play speech,
  locked-device voice, back-channel and follow-up for each character, saved at once). The
  Bots-tab Voice pill is now a link to that character's card (it opened a popover whose voice
  choices waited for the Bots tab's Save).
- **Fields that had no UI**: the rig address and a **write-only rig token**
  (`PUT/DELETE /api/voice/rig/token`), the TTS provider and engine, an OpenAI-compatible speech
  server and transcription server, and **provider plugin** switches. A setting the server's
  environment decides is shown "Set by environment (NAME)" and disabled.
- **In-app model download** with progress and Cancel (`POST /api/voice/models/fetch`,
  `GET .../status`, `POST .../cancel`), the same code as `dispatch-voice fetch-models`; and
  **one models folder** with `speech/` and `chatterbox/` subfolders (`dispatch-voice
  migrate-models`, `POST /api/voice/models/migrate`; dry run first, never overwrites; an
  install from before keeps working where its files are).
- **Provider interface** (`host/providers.py`): `STTProvider.transcribe`, `TTSProvider.synthesize`,
  `VADProvider`, each with `health()`, `capabilities()` and `settings_schema()`. Adapters for
  the CPU pipeline, the SpeechForge rig (one `RigLink` feeding an STT and a TTS provider) and
  OpenAI-compatible `/v1/audio/speech` and **`/v1/audio/transcriptions`** (new; also selectable
  as `stt_provider: openai`). Plugins register with `api.register_provider` (`ExtensionAPI` 4)
  through the `dispatch_voice.providers` entry-point group. `GET /api/voice/providers`.
- **`GET /api/voice/settings/schema`**: every setting with type, default, range, scope, level,
  secret flag and environment names. `GET /settings` reports `sources` (env / file / default)
  and `env` per key.
- **`GET /api/voice/status` health**: a `health` list with a reason and a fix for each part
  that is not fine, `system` (ffmpeg, libopus with the install command for this OS), the
  provider catalog and the download state.
- `migrate_voice_yaml(store_writer, dry_run=True)` (not wired yet): the one-shot move of
  `voice.yaml` into the host's settings store.
- A **Windows job in CI**, and `fsperm.py` (owner-only files on POSIX and by ACL on Windows).

### Changed
- **The environment now beats `voice.yaml`** (it was the other way round for everything except
  `DISPATCH_VOICE`). A PATCH over a variable-held key is saved but reported `overridden`.
- **One definition per setting** (`host/settings_schema.py`): `clean_prefs`, the PATCH
  validation and the `rt` / `rig` subtrees are generated from it. `stt_engine`, `hard_silence_s`,
  `power_hold` and `bots` are no longer defined twice; the old spellings still read. The
  README's environment table is generated from the schema and a test keeps it current.
- `websockets` is a runtime dependency (the rig client imported it but only dev installs had it).
  ffmpeg and libopus are declared as system requirements (`pyproject.toml`, README,
  `dispatch-voice doctor`).
- `DISPATCH_VOICE_EXTENSIONS` is deprecated (it still loads, with a warning): install the
  plugin and switch it on in Settings.
- `ExtensionAPI.api_version` is 4.

### Security
- **The rig address is checked** (it receives the token and the live microphone): loopback,
  private, tailnet and LAN names are allowed; link-local and cloud-metadata addresses never;
  public addresses only on the administrator's allow-list (`rig.allow`); plain `ws://` off the
  machine warns; a name that resolves somewhere forbidden is not dialled. The same policy
  applies to the OpenAI-compatible speech and transcription servers.
- The bearer token no longer appears in DEBUG logs of the websockets client; a test pins that no
  log line carries it.

### Fixed (host wiring, W6b)
- **Speech works right after the in-app download, with no restart** (review F6). The Chatterbox
  engine and the voice resolver were registered once, at setup, so a download finished
  afterwards left `status.ready` true while the first voice socket failed ("voice models failed
  to load"). `tts.build_engine` / `shared_engine` and `resolve_profile` now register them as
  soon as the models are on disk. `ready` is also exact now: it is true only when the whole
  Chatterbox set (all four graphs and their weights) and the whole Parakeet set are present, not
  while the download has written the tokenizer or the encoder but not the rest. Only the
  cloned-voice management routes (`/api/voice/profiles`) still mount at the next start, and the
  page says so.
- **The rig token lives in DisPatch's data folder** (review F7c): the host's credential store
  (`ctx.secrets`) first, else `<data>/ext/voice/speechforge.env`, so it survives a container
  image update and `PUT /api/voice/rig/token` no longer answers 500 under a read-only root. The
  1.6.0 file `~/.config/local-chat/speechforge.env` and `SPEECHFORGE_TOKEN` are still read.

### Changed (host wiring, W6b)
- **One models folder named `models`** (review F8): the default is `<data>/models` with
  `speech/` and `chatterbox/` (it was `<data>/speech-models`, which held Chatterbox too). An
  install that still has `<data>/speech-models` keeps working; `migrate-models` moves it.

### Fixed (Windows)
- The rig token file was rejected on every Windows file (POSIX mode bits are always 0666 there):
  the ACL is checked instead. Text files are read and written as UTF-8 (a non-ASCII bot id or
  profile name misread on cp1252). The power hold reports "needs Linux" instead of a missing
  `/usr/bin/python3`. A cursor store whose file is corrupt is moved aside (the failed sqlite
  handle is closed first). The `voice.yaml` cache no longer serves a stale file after two writes
  in one 15 ms tick. The rig ping carries `box_load` on Windows too.
- The test suite is green on Windows (it was 27 failed, 1 error): the session tests' 5 ms
  settle loops returned at once there (asyncio fires timers shorter than the 15.6 ms clock tick
  immediately), and file-mode checks are ACL checks on Windows.


## [1.5.0] — 2026-10-17

Background noise and people talking in the room kept pausing and cutting replies.

### Added
- **Interrupt sensitivity** (Settings › Voice › Listening, and the call screen's
  quick settings): Low, Normal or High — how sure the voice chat must be that it is
  you before it stops a reply. A device setting, sent with `start`
  (`interrupt_sensitivity`). **Normal is the new default** and is less eager than
  before; High keeps the 1.4.0 rules exactly.

### Changed
- **An adaptive level gate.** The session follows the room's ambient level and your
  own voice level (learned from the turns you make while nothing plays). At Normal,
  speech over a reply counts only when it is 10 dB over the room and within 10 dB of
  your usual level (Low: 14 dB over the room, within 6 dB of you), so someone across
  the room, a TV or a clatter neither pauses nor cuts the reply. Unknown levels never
  block.
- **The reply pauses after sustained speech.** At Normal the server pauses a reply
  after 0.45 s of speech that passes the gate (Low 0.7 s; High 0.25 s at any level),
  and the phone's instant pause on any sound over its floor (the optimistic duck)
  now runs only at High.
- **Stop words.** At Normal and Low only a clear stop word stops a reply mid-speech
  or at any level: "stop", "wait", "pause", "enough", "quiet", "hold on", "hang on",
  "that's enough", "never mind". "What?", "sorry", "no", "thank you", "okay okay"
  stop a reply only as a whole utterance in your own voice, never from across the
  room. A cut while you are still talking needs four words of your own (Low five;
  High three).
- The journal now says why speech over a reply was ignored ("too faint to be the
  listener", with the levels) and when it cut the reply.

## [1.4.0] — 2026-10-17

A round of fixes from a live call where the voice chat moved to the phone's
voice and stayed there.

### Fixed
- **The microphone no longer hears the phone's voice.** `speechSynthesis` has no
  echo-cancellation reference, so while the phone read a reply its own speaker
  reached the microphone: bogus turns, and 0.6 s of "speech" cut the reply. The
  device now mutes its microphone on the wire while the phone's voice speaks and
  for 0.4 s after (the frames are zeroed, so the uplink's offsets stay intact);
  tap the picture to interrupt a reply the phone is reading.
- **The link recovers after an interrupted reply.** The link monitor cleared its
  "reply in progress" only on `say_end`, which the server never sends for a cut
  reply (a barge-in, a tap, an undo, a failed turn). One interrupted reply while
  on the phone's voice stopped every probe, so the chat never went back to the
  server's voice. The monitor is now told when a reply is cut.
- **A refused probe is answered.** The server answers a probe inside its window
  with `error {code: "rate_limited"}` instead of silence (the client waited 30 s),
  and accepts one per 18 s so a client pacing at 20 s is never refused by jitter.
- **Live captions no longer start with "null".** The caption of what you are
  saying showed the word "null" before the first committed words.

### Added
- **Keyboard shortcuts on the call screen**, and a tooltip with the key when you
  hover or focus a button: Space mutes/unmutes from anywhere on the screen
  (except in the text box), T opens the text box, O the Output menu, U Undo,
  Esc hangs up.

### Changed
- Mute mutes only the microphone, as before; typing a message (the Type button)
  mutes the microphone the same way and the reply is still spoken.

## [1.3.5] — 2026-10-15

### Fixed
- **A reply no longer changes voice half-way.** With role-play speech on and a
  built-in narrator voice, the first narrator piece used to move the whole reply
  from the GPU rig to this computer, so the cloned dialogue voice changed
  mid-reply and the rest came slower. Now only that piece is spoken locally and
  the reply stays on the rig. A reply that opens with narration can start on the
  rig too.
- **The rig forgetting a voice is recovered, not fallen back from.** When the rig
  answers `unknown_voice` (evicted or restarted) while a re-upload is already in
  flight, the host waits up to 1.5 s for it and retries that sentence on the rig
  once per reply before using this computer.

### Changed
- CI installs ffmpeg and libopus; on a machine without ffmpeg the tests that need
  it skip with a reason instead of failing.

## [1.3.4] — 2026-10-14

### Added
- **Voice on locked devices.** Each bot can have its own voice for locked
  (Safe Mode) devices, set in the Bots-tab Voice popover (or `locked_voice` under
  `bots.<id>` in `voice.yaml`). Empty, the default, means the same as the bot's
  standard voice. A voice chat from a locked device speaks with it; every other
  device keeps the standard voice. The host decides from the socket's locked
  flag, never from anything the client sends. The setting is unlocked-only to
  read or change, and is not part of the `locked_status` a locked device sees.

## [1.3.3] — 2026-10-14

### Changed
- **Voice chat on locked devices.** With DisPatch 2.3.4 or later, a locked
  (Safe Mode) device gets the voice chat button on the bots it can chat with
  (the `safe` ones), as long as voice is switched on. DisPatch runs every turn
  from it, spoken or typed on the call screen, with the same locked rules as a
  typed Safe-Mode message. On a locked device the quick sheet keeps this
  device's own settings only; the bot's voice, Settings › Voice, the Bots-tab
  pill and the voice tools stay unlocked-only, as does every `/api/voice` route.
  A locked device learns whether voice is on from the add-on's new
  `locked_status` (`{enabled, ready}`). On an older DisPatch nothing changes.

## [1.3.2] — 2026-10-14

A review pass over 1.2.0–1.3.1.

### Fixed
- **Voice chat screen, Type:** a message longer than the wire's 4,000-character
  bound is refused on the device with a notice and kept in the box; it used to
  be sent, answered with `bad_frame`, and already cleared from the box.
- **Type instead of talking:** Esc in the text box moves the focus to the picture
  instead of dropping it on the page, so Esc (hang up) and Tab keep working.
- **Output:** a remembered output device is named on the button from the start
  of a chat, not only after the menu was opened once.
- Removed the unused `drive.err_safe_bot` string from every locale (no code has
  read it since 1.3.1).

## [1.3.1] — 2026-10-14

### Changed
- **Voice for Safe-Mode bots.** A bot marked `safe` (one that locked family
  devices may also talk to) now gets the voice chat button and the Bots-tab
  voice pill like any other bot, and its per-bot voice settings are accepted.
  Voice chat itself is still unlocked-only: the device must hold the PIN, and a
  Safe-Mode device never reaches it. Needs DisPatch 2.3.1, whose gate stopped
  refusing a safe bot's thread; on an older DisPatch the button shows and the
  socket is refused.

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
