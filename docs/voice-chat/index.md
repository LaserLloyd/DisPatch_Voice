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
   (`git worktree add --detach <dir> v1.0.1`): what is in the overlay's folder goes
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
2. **Fetch the models** (~1.4 GB: Silero VAD, Smart Turn, Parakeet, Chatterbox-Turbo, the stock voice):

   ```sh
   dispatch-voice fetch-models
   ```

   The speech models go to `DISPATCH_SPEECH_MODELS` (else `<data>/speech-models`),
   Chatterbox to `DISPATCH_VOICE_MODELS` (else `~/.local/share/dispatch-voice/models`).
3. **Restart DisPatch**, then in Settings → Voice turn the switch on. The page lists
   anything still missing with the command that fixes it. `dispatch-voice doctor`
   prints the same from the shell, plus the handshake and the running version
   (`--unit <name>` reads the voice settings and backend directory of a systemd
   `--user` unit, so it checks what the service really uses).
4. Optional: the Kokoro built-in voices (`dispatch-voice-kokoro`, a separate,
   GPL-3.0-or-later plugin package: `DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup`),
   a cloned voice from a clip in the Voice cloning tool, or the stock synthetic voice as it is.
5. Optional: give your bots a character in their acknowledgements (`<data>/voice-acks.yaml`:
   `bots: {<bot id>: {flavour: wry|bright|trainer|gruff}}`) and name your own tools'
   status lines (`tools: [{match: <pattern>, say: <line>}]`); see
   [persona-acks-rp.md](persona-acks-rp.md).

Voice chat understands and speaks English only for now. It is available to an
unlocked device only, never in Safe Mode.

## Reading the code

Comments cite the design documents the add-on was built from by section (`SPEC §4.6`,
`contracts §C.4`, `Amendment T6-1`, review findings such as `hardening M8` or
`review-T3s4 #4`). Those documents are not part of this repository; the behaviour each
comment cites is described in these pages, and [protocol.md](protocol.md) is the
normative wire contract.

<!-- Owner: track T3. -->
