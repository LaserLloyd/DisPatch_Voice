# dispatch-voice

Local voice cloning and speech for DisPatch Chat's Drive mode. Upload or record a 10–20 second clip, and the panel builds a **voice profile**: the generation parameters the speech model needs to talk in that voice. Spoken replies then use that voice. Everything runs on the CPU of the machine DisPatch runs on. Nothing is sent anywhere.

## Engine

[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) by Resemble AI, MIT licence, ungated, run through ONNX Runtime at q4 precision (~690 MB download). It was chosen on 2026-10-09 as the best MIT-licensed, ungated cloning model that runs on a CPU:

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

## Install

```sh
git clone https://github.com/LaserLloyd/DisPatch_Voice.git dispatch-voice && cd dispatch-voice
uv sync                                  # core: onnxruntime, no torch
uv sync --extra watermark                # + Perth watermark (CPU torch only, from the pytorch-cpu index)
uv run dispatch-voice fetch-models       # ~690 MB into ~/.local/share/dispatch-voice/models, then builds the stock voice
```

Models are pinned to a Hugging Face commit (`MODEL_REVISION` in `engine.py`), so a re-download can't change the voice. They live in a plain directory, not the HF cache: onnxruntime refuses external-data files reached through the cache's symlinks.

For DisPatch Chat, install it into DisPatch's environment instead (see DisPatch's `docs/voice-drive-mode.md`):

```sh
uv run --group voice --with "dispatch-voice @ git+https://github.com/LaserLloyd/DisPatch_Voice" dispatch-voice fetch-models
```

## Use

Standalone, for testing (loopback only, no auth):

```sh
uv run dispatch-voice serve --data /tmp/voices-test --port 8769
# open http://127.0.0.1:8769/voices/ui
uv run dispatch-voice say <profile-id> "Okay, here's the thing [chuckle]." --data /tmp/voices-test -o line.wav
```

Mounted in DisPatch (behind the unlocked tier, never Safe Mode):

```python
from dispatch_voice.engine import ChatterboxOnnx
from dispatch_voice.profiles import ProfileStore
from dispatch_voice.router import build_router
from dispatch_voice import speak

voice_store = ProfileStore(DATA_DIR / "voices", ChatterboxOnnx())       # load once at startup (~3 s)
app.include_router(build_router(voice_store, [Depends(_require_full_access)]), prefix="/api/voices")
# panel: /api/voices/ui

# Drive mode:
for frame in speak.speak(voice_store, llm_token_stream, profile_id=None, stop=barge_in_event):
    ws.send_bytes(frame)            # 16-bit mono PCM, 24 kHz, 40 ms frames
ack = speak.filler(voice_store)     # pre-rendered "Mm-hmm." / "Right." in the same voice, play at turn end
```

`build_router` refuses to build without an auth dependency. Write routes also refuse a browser `Origin` that doesn't match `Host`.

## Stock voice

Chatterbox has no built-in voice, so dispatch-voice ships one: `stock-host`, "Stock voice (synthetic)". It is built from `src/dispatch_voice/stock/synthetic-host.wav`, 12.6 s of fully synthetic speech (Kokoro-82M output, Apache-2.0 weights), so it is not a real person's voice. `fetch-models` builds its tensors once into `<models>/stock/stock-host.npz` (`--no-stock` skips this). `dispatch_voice.stock.ensure_stock(store)` then turns them into an ordinary profile; with `build=True` it builds the tensors itself when the cache is missing. `render_extras` adds the audition line and the fillers. A host app uses the stock profile when it has no other voice. It never becomes the default on its own.

## Voice profiles

`<data>/voices/<id>/`, directory mode 0700:

| File | What |
|---|---|
| `original.<ext>` | the upload as received (so the profile can be rebuilt with a newer model) |
| `reference.wav` | trimmed, best 20 s window, loudness-normalised to −20 LUFS. This is what was cloned. |
| `voice.npz` | **the generation parameters**: the four speech-encoder tensors (`cond_emb`, `prompt_token`, `speaker_embeddings`, `speaker_features`). Loading takes milliseconds. |
| `profile.json` | name, quality report, match score, consent record, engine, timings |
| `audition.wav` | a fixed test line in the clone, worded unlike the recording script so a clone that parrots its reference is audible |
| `fillers/0-5.wav` | reaction sounds in the clone ("Mm-hmm.", "Right.", "Oh, nice."…), rendered in the background after a build, with fixed seeds |

## Best practices built in

- **Clip guidance and checks.** Ask for 10–20 s of one person talking naturally, in a quiet room, with no music. Shorter than 5 s of speech is refused. Background noise, clipping and suspiciously flat dynamics (music, heavy compression) are flagged before anything is built. Long clips are cut to the 20 s window with the most speech.
- **Record mode** turns off the browser's echo cancellation, noise suppression and auto-gain: those artifacts would be cloned too. The on-screen script is conversational, with a question and an aside, because an expressive reference gives an expressive voice.
- **Consent is mandatory and recorded per profile:** "my own voice", "I have the speaker's explicit permission", or "synthetic / stock voice". A synthetic host voice is the recommended choice.
- **Watermarking.** With the `watermark` extra, every output carries Resemble's inaudible Perth watermark. Verified: detected as 1.0 in output, 0.0 in clean audio.
- **Private by construction.** Profiles stay on the box. `.gitignore` excludes `voices/`, `models/` and `*.npz`, and the profile store's file route serves only an allowlist of wavs (never `voice.npz`). IDs are validated against path traversal.
- **Upload hardening.** 20 MB cap, ffmpeg decoding with a fixed argv over pipes (no shell, no temp files), atomic profile writes (a failed build leaves nothing behind).

## Tests

```sh
uv run pytest -q          # 55 tests; the 3 in test_models.py use the real models and skip without them
```

Before committing, install the privacy hooks once (`sh scripts/install-hooks.sh`). They run `scripts/scrub_check.py` on every commit, commit message and push; personal identifiers go in the git-ignored `scripts/scrub-rules.local.txt`.

Third-party licences: `THIRD_PARTY_NOTICES.md`.

## Limits

- First audio for a short line is ~1.5–2 s on this CPU, against ~0.1 s for Pocket TTS. Fillers cover the gap in Drive mode.
- English only (Turbo is English-only).
- One synthesis at a time per engine instance (CPU-bound).
- The `watermark` extra pulls CPU torch (~200 MB) plus librosa. Perth imports `pkg_resources`, hence `setuptools<81`.
