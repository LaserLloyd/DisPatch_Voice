# dispatch-voice

**DisPatch Voice**: voice chat for [DisPatch Chat](https://github.com/LaserLloyd/DisPatch_Chat), as an add-on. You talk to a DisPatch bot hands-free; it listens, answers in its own voice and can be interrupted. Everything runs on the CPU of the machine DisPatch runs on; by default nothing is sent anywhere (the optional MiniMax and HQ engines, `DISPATCH_VOICE_TTS=minimax` / `DISPATCH_VOICE_HQ_URL`, send the reply text to that service to be spoken).

DisPatch runs fully without it. With the add-on installed, DisPatch gains a microphone button in each chat and a Settings → Voice page. The add-on is the whole feature: the speech pipeline (Silero VAD + Smart Turn v3.2 for turn taking, Parakeet TDT 0.6B for recognition), the voice session, the voices (cloned voices and a stock synthetic voice on Chatterbox-Turbo ONNX; optional built-in voices through a plugin), the settings routes and its own frontend assets. DisPatch keeps only a small, stable extension host and the page that explains how to install this when it is absent.

On the call screen there are three round buttons: **Mute**, **Type** (type your turn instead of speaking it; the reply is still spoken) and **Hang up**, plus an **Output** button on desktop browsers that choose where replies play (phone browsers cannot, so it is not shown there).

Docs: [docs/voice-chat/](docs/voice-chat/index.md) (install and setup, the host side, turn taking, the protocol).

## Install, for DisPatch

DisPatch 2.2.0 or later (newer DisPatch releases unlock more, see [Which DisPatch version](#which-dispatch-version)). Three steps; details in [docs/voice-chat/index.md](docs/voice-chat/index.md).

```sh
# 1. the add-on on DisPatch's environment: an editable overlay on the line that starts DisPatch
cd <dispatch-chat>/backend
uv run --with-editable <dispatch-voice> python -c "import dispatch_voice"       # once, online
uv run --offline --with-editable <dispatch-voice> uvicorn app.main:app …        # the unit line

#    or a bare-metal install into DisPatch's venv (`v1` follows the latest 1.x release)
uv pip install --python .venv/bin/python "dispatch-voice @ git+https://github.com/LaserLloyd/DisPatch_Voice@v1"

# 2. the models (~1.4 GB: Silero VAD, Smart Turn, Parakeet, Chatterbox-Turbo, the stock voice)
dispatch-voice fetch-models

# 3. restart DisPatch, then Settings → Voice → switch it on
```

**On a server you keep running**, point the overlay at a release checkout that nobody develops in, and start DisPatch through its launcher, which falls back to DisPatch without voice (instead of a restart loop) when the overlay cannot be resolved or crashes on import:

```sh
git -C <dispatch-voice> worktree add --detach <release-dir> v1.4.0     # later releases: git -C <release-dir> checkout --detach <tag>
cd <dispatch-chat>/backend
uv run --with-editable <release-dir> python -c "import dispatch_voice"  # warm once, online, after every release
<dispatch-chat>/scripts/local-chat-launch --uv "$(command -v uv)" --with-editable <release-dir> \
    -- uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-ping-timeout 60 --ws-max-size 262144
```

`--offline` (added by the launcher), `--ws-max-size 262144` and `--ws-ping-timeout 60` are load-bearing; [docs/voice-chat/host.md](docs/voice-chat/host.md#running-dispatch-with-the-add-on) says why.

`dispatch-voice doctor` prints the DisPatch handshake, which models are present, the engines and plugins, the switch, and whether the add-on the running server loaded is the one in this checkout (`--unit <name>` reads the voice settings and backend directory of a systemd `--user` unit; `--backend DIR` also checks DisPatch's venv and that the overlay resolves offline; `--check-update` compares with the latest release). `dispatch-voice fetch-models --verify` checks every model file against its checksum.

Optional: the Kokoro built-in voices are a separate, GPL-3.0-or-later plugin package, `dispatch-voice-kokoro` (`DISPATCH_VOICE_EXTENSIONS=dispatch_voice_kokoro:setup`). Optional: per-bot character in the acknowledgements and spoken status lines for your own tools in `<data>/voice-acks.yaml` ([docs/voice-chat/persona-acks-rp.md](docs/voice-chat/persona-acks-rp.md)).

Voice chat understands and speaks English only for now. It works on an unlocked device and, with DisPatch 2.3.4 or later, on a locked (Safe Mode) one too, for the bots that device can chat with (the `safe` ones); every turn from a locked device, spoken or typed, runs with DisPatch's locked rules, and the voice settings stay unlocked-only. Since 1.3.4 a bot can use a different voice for locked devices: `bots.<id>.locked_voice` in `voice.yaml`, or the "Voice on locked devices" picker in the Bots-tab Voice popover. Empty (the default) means the bot's standard voice. The server picks it from the socket's locked flag, and a locked device cannot read the setting.

### Which DisPatch version

| Add-on feature | Add-on | DisPatch |
|---|---|---|
| Voice chat | 1.0 | 2.2.0 |
| GPU rig engine (SpeechForge) | 1.1.0 | 2.2.0 |
| Type button (`typed` frame), Output button | 1.2.0, 1.3.0 | 2.2.0 |
| Voice on a `safe` bot, for an unlocked device | 1.3.1 | 2.3.1 |
| Voice on locked devices | 1.3.3 | 2.3.4 |
| Per-bot `locked_voice` | 1.3.4 | 2.3.4 |

On an older DisPatch the newer buttons are simply not available or the socket is refused, as the [changelog](CHANGELOG.md) notes per release.

## Engine

[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) by Resemble AI, MIT licence, ungated, run through ONNX Runtime at q4 precision (~720 MB download). It was chosen on 2026-10-09 as the best MIT-licensed, ungated cloning model that runs on a CPU:

| Candidate | Licence | Result on the test machine (Ryzen AI 9 365, CPU) |
|---|---|---|
| **Chatterbox-Turbo ONNX q4** | MIT, ungated | profile build 0.7 s; RTF 0.7–0.9 on sentences; ~1.5–2 s to speak a short line; native `[laugh]`/`[chuckle]` tags; built-in Perth watermark |
| Chatterbox-Nano (PyTorch) | MIT, ungated | RTF 1.0–1.6, slower than real time |
| Chatterbox-Turbo (PyTorch) | MIT, ungated | RTF 2.2–3.1 |
| Chatterbox-Flash | MIT, ungated | needs CUDA (FlashInfer) |
| Kyutai Pocket TTS | code MIT, **cloning weights gated** | fastest (0.1 s first audio), but gated |
| NeuTTS Air | Apache-2.0 (not MIT) | not evaluated |

Quality check (2026-10-09): 20 renders across 10 lines, transcribed back with Parakeet. 19 were word-perfect. The miss was "Mm." rendered as "Mm-hmm", so the filler now *is* "Mm-hmm.". The speaker-match score separates voices clearly: clone vs reference 0.91, an unrelated male voice 0.54, a female voice 0.43.

**Sampling matters.** Resemble's ONNX example decodes greedily, and on a cloned voice that looped the reference clip's words for ten seconds. The engine samples the way Resemble's own PyTorch code does (temperature 0.8, top-p 0.95, top-k 1000, repetition penalty 1.2), and caps output length by word count. `tests/test_models.py::test_no_runaway_or_reference_replay` guards this.

Models are pinned to a Hugging Face commit (`MODEL_REVISION` in `engine.py`) and to revisions with sha256 checksums for the speech models (`host/download.py`), so a re-download can't change what you tested. They live in plain directories, not the HF cache: onnxruntime refuses external-data files reached through the cache's symlinks.

## Optional: GPU rig

Since 1.1.0, a machine with an NVIDIA GPU running SpeechForge (a speech server speaking the `sf.1` protocol) can do the speech recognition, the end-of-turn score and the reply voice, so replies start sooner. It is off until you set it up:

```yaml
# <data>/voice.yaml
rig:
  enabled: true
  url: ws://<rig>:1236/v1/realtime
```

```sh
# the token, readable by you only (or SPEECHFORGE_TOKEN in DisPatch's environment)
mkdir -p ~/.config/local-chat && (umask 077 && printf 'SPEECHFORGE_TOKEN=%s\n' '<token>' > ~/.config/local-chat/speechforge.env)
```

Then Settings → Voice → *Use the GPU rig when available*. This computer stays in charge of the conversation and takes over the speech work whenever the rig is slow, busy or away: a turn only starts on a rig that answered recently, a dead rig is noticed in under a second, and the words in flight replay into the local recogniser. Only voice conditionals go to the rig, never the reference clip. Details: [docs/voice-chat/rig.md](docs/voice-chat/rig.md).

## Standalone use

```sh
git clone https://github.com/LaserLloyd/DisPatch_Voice.git dispatch-voice && cd dispatch-voice
uv sync                                  # core: onnxruntime, no torch
uv sync --extra watermark                # + Perth watermark (CPU torch only, from the pytorch-cpu index)
uv run dispatch-voice fetch-models
uv run dispatch-voice serve --data /tmp/voices-test --port 8769      # the Voices panel: loopback only, no auth, its own folder
uv run dispatch-voice say <profile-id> "Okay, here's the thing [chuckle]." --data /tmp/voices-test -o line.wav
```

As a library, the pieces DisPatch uses are importable on their own:

```python
from dispatch_voice.engine import ChatterboxOnnx
from dispatch_voice.profiles import ProfileStore
from dispatch_voice import speak

store = ProfileStore(DATA_DIR / "voices", ChatterboxOnnx())       # load once (~3 s)
for frame in speak.speak(store, llm_token_stream, profile_id=None, stop=barge_in_event):
    ws.send_bytes(frame)            # 16-bit mono PCM, 24 kHz, 40 ms frames
ack = speak.filler(store)           # a pre-rendered "Mm-hmm." / "Right." in the same voice (standalone use)
```

`dispatch-voice serve` refuses to start without `--data` and refuses DisPatch's own voice store: the page has no login. It answers only a request whose `Host` is its own loopback address (a DNS-rebinding page gets 421).

`router.build_router` refuses to build without an auth dependency (`allow_open=True` only for tests, the standalone page and the DisPatch add-on, whose gate is DisPatch's). Write routes also refuse a browser `Origin` that doesn't match `Host`.

## Stock voice

Chatterbox has no built-in voice, so dispatch-voice ships one: `stock-host`, "Stock voice (synthetic)". It is built from `src/dispatch_voice/stock/synthetic-host.wav`, 12.6 s of fully synthetic speech (Kokoro-82M output, Apache-2.0 weights), so it is not a real person's voice. `fetch-models` builds its tensors once into `<models>/stock/stock-host.npz` (`--no-stock` skips this). `dispatch_voice.stock.ensure_stock(store)` then turns them into an ordinary profile; with `build=True` it builds the tensors itself when the cache is missing. `render_extras` adds the audition line and the fillers. DisPatch uses the stock profile when it has no other voice (inside DisPatch only its audition line is rendered: acknowledgements come from DisPatch's render queue, not the fillers). It never becomes the default on its own.

## Voice profiles

`<data>/voices/<id>/`, directory mode 0700:

| File | What |
|---|---|
| `original.<ext>` | the upload as received (so the profile can be rebuilt with a newer model) |
| `reference.wav` | trimmed, best 20 s window, loudness-normalised to −20 LUFS. This is what was cloned. |
| `voice.npz` | **the generation parameters**: the four speech-encoder tensors (`cond_emb`, `prompt_token`, `speaker_embeddings`, `speaker_features`). Loading takes milliseconds. |
| `profile.json` | name, quality report, match score, consent record, engine, timings |
| `audition.wav` | a fixed test line in the clone, worded unlike the recording script so a clone that parrots its reference is audible |
| `fillers/0-5.wav` | reaction sounds in the clone ("Mm-hmm.", "Right.", "Oh, nice."…), rendered in the background after a build on the standalone page, with fixed seeds (not inside DisPatch, whose voice chat renders its own acknowledgement lines) |

## Best practices built in

- **Clip guidance and checks.** Ask for 10–20 s of one person talking naturally, in a quiet room, with no music. Shorter than 5 s of speech is refused. Background noise, clipping and suspiciously flat dynamics (music, heavy compression) are flagged before anything is built. Long clips are cut to the 20 s window with the most speech.
- **Record mode** turns off the browser's echo cancellation, noise suppression and auto-gain: those artifacts would be cloned too. The on-screen script is conversational, with a question and an aside, because an expressive reference gives an expressive voice.
- **Consent is mandatory and recorded per profile:** "my own voice", "I have the speaker's explicit permission", or "synthetic / stock voice". A synthetic host voice is the recommended choice.
- **Watermarking.** With the `watermark` extra, every output carries Resemble's inaudible Perth watermark. Verified: detected as 1.0 in output, 0.0 in clean audio.
- **Private by construction.** Profiles stay on the box. `.gitignore` excludes `voices/`, `models/` and `*.npz`, and the profile store's file route serves only an allowlist of wavs (never `voice.npz`). IDs are validated against path traversal.
- **Upload hardening.** 20 MB cap, ffmpeg decoding with a fixed argv over pipes (no shell, no temp files), atomic profile writes (a failed build leaves nothing behind).

## Tests

```sh
uv run pytest -q                 # Python; model-backed tests skip when the models are absent
npm install && npm test          # client modules (node --test; jsdom for the DOM suites)
python3 scripts/check-locales.py # locale bundles
```

See [tests/README.md](tests/README.md) for the layout, the DisPatch stand-in (`tests/fakes.py`), the fixtures, and how to run the model-backed and browser harnesses on a shared machine. The cross-repo contract test (`tests/contract/`) runs against a DisPatch checkout (`DISPATCH_BACKEND`); `tests/e2e_voice_roundtrip.py` walks a whole voice chat on a scratch server, with or without DisPatch.

Before committing, install the privacy hooks once (`sh scripts/install-hooks.sh`). They run `scripts/scrub_check.py` on every commit, commit message and push; personal identifiers go in the git-ignored `scripts/scrub-rules.local.txt`.

Third-party licences: `THIRD_PARTY_NOTICES.md`.

## Limits

- First audio for a short line is ~1.5–2 s on this CPU, against ~0.1 s for Pocket TTS. Acknowledgements cover the gap in voice chat.
- English only (Turbo is English-only).
- One synthesis at a time per engine instance (CPU-bound).
- The `watermark` extra pulls CPU torch (~200 MB) plus librosa. Perth imports `pkg_resources`, hence `setuptools<81`.
