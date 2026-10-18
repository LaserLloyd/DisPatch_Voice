# Voice chat (DisPatch Voice add-on)

Voice chat lets you talk to a DisPatch bot hands-free: you speak, it answers in its own
voice, and you can interrupt it at any time. It is an add-on: DisPatch runs fine without it,
and with it installed DisPatch gains a microphone button in each chat and a Settings → Voice
page.

This page is the entry point. One file per area; each has one owner:

| Page | What it covers |
|---|---|
| [index.md](index.md) | this overview, install and setup |
| [host.md](host.md) | how the add-on plugs into DisPatch: setup, routes, the session hub, settings files, power hold |
| [turn-taking.md](turn-taking.md) | streaming speech recognition, end-of-turn, speculative replies, interruptions, latency |
| [persona-acks-rp.md](persona-acks-rp.md) | spoken-reply instructions, acknowledgements, speech text and role-play |
| [low-bandwidth.md](low-bandwidth.md) | Opus, the v3 framing, slow links, replay after a drop |
| [client.md](client.md) | the voice chat screen and settings, user-facing behaviour |
| [audio-modules.md](audio-modules.md) | the browser audio worklets, codecs, link monitor and phone voice |
| [protocol.md](protocol.md) | the WebSocket protocol v3 (normative) |
| [rig.md](rig.md) | the optional GPU rig (SpeechForge): setup, how it falls back, voices |

## Install and set up

DisPatch 2.2.0 or later is the host. The add-on is a Python package that DisPatch
discovers through an entry point; nothing in DisPatch names it.

1. **Put the add-on on DisPatch's environment.** Either an editable overlay on the
   `uv run` line that starts DisPatch (`--offline` keeps a network-less reboot
   booting):

   ```sh
   cd <dispatch-chat>/backend
   uv run --with-editable <dispatch-voice> python -c "import dispatch_voice"      # once, online
   uv run --offline --with-editable <dispatch-voice> uvicorn app.main:app …
   ```

   On a server, make `<dispatch-voice>` a release checkout that nobody develops in
   (`git worktree add --detach <dir> v1.1.0`): what is in the overlay's folder goes
   live at the next restart. DisPatch's `scripts/local-chat-launch` is the
   recommended service command: it checks the overlay in a throwaway process and
   starts DisPatch without the add-on (Settings → Voice then says why) rather than
   in a restart loop. Details in [host.md](host.md#running-dispatch-with-the-add-on).

   or a bare-metal install into DisPatch's venv:

   ```sh
   uv pip install --python .venv/bin/python "dispatch-voice @ git+https://github.com/LaserLloyd/DisPatch_Voice@v1"
   ```

   (`v1` moves with each 1.x release. A later `uv sync` removes a pip-installed
   add-on; `dispatch-voice doctor` tells you when that happened.)
2. **Restart DisPatch once**, so it discovers the add-on. Everything after this is
   Settings → Voice.
3. **In Settings → Voice**: turn the switch on and press **Download** under *Models*
   (~1.4 GB: Silero VAD, Smart Turn, Parakeet, Chatterbox-Turbo, the stock voice; a
   progress bar and Cancel). The page lists anything still missing with the fix. The
   same download from a shell is `dispatch-voice fetch-models`; since 1.6.0 the files go
   into ONE folder, DisPatch's `models_dir`, as `speech/` and `chatterbox/`
   (`DISPATCH_SPEECH_MODELS` / `DISPATCH_VOICE_MODELS` still point a family elsewhere,
   and an older install keeps working where its files are until `dispatch-voice
   migrate-models --apply` or the Move button). Speech recognition works as soon as the
   download finishes; speaking (the stock voice and cloned voices both run on the
   Chatterbox engine, which the add-on registers at start-up) needs one more DisPatch
   restart, and the page says so. `dispatch-voice doctor`
   prints the same from the shell, plus the handshake and the running version
   (`--unit <name>` reads the voice settings and backend directory of a systemd
   `--user` unit, so it checks what the service really uses).
4. Optional: the Kokoro built-in voices (`dispatch-voice-kokoro`, a separate,
   GPL-3.0-or-later plugin package): install it and switch it on under Settings →
   Voice → *Provider plugins* (the `extensions` setting; the old
   `DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup` still works and is deprecated).
   Or a cloned voice from a clip in the Voice cloning tool, or the stock synthetic voice as it is.
5. Optional: give your bots a character in their acknowledgements (`<data>/voice-acks.yaml`:
   `bots: {<bot id>: {flavour: wry|bright|trainer|gruff}}`) and name your own tools'
   status lines (`tools: [{match: <pattern>, say: <line>}]`); see
   [persona-acks-rp.md](persona-acks-rp.md).

Voice chat understands and speaks English only for now. It is available to an
unlocked device and, since 1.3.3 with DisPatch 2.3.4, to a locked (Safe Mode)
device on the bots it can chat with; DisPatch runs those turns with its locked
rules. The voice settings stay unlocked-only. Since 1.3.4 a bot can speak with a
different voice on locked devices (`bots.<id>.locked_voice` in `voice.yaml`, the
"Voice on locked devices" picker in the Bots-tab Voice popover; empty = the standard
voice). The server picks it from the socket's locked flag; a locked device cannot read it.

## Reading the code

Comments cite the design documents the add-on was built from by section (`SPEC §4.6`,
`contracts §C.4`, `Amendment T6-1`, review findings such as `hardening M8` or
`review-T3s4 #4`). Those documents are not part of this repository; the behaviour each
comment cites is described in these pages, and [protocol.md](protocol.md) is the
normative wire contract.

<!-- Owner: track T3. -->
