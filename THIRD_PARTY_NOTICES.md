# Third-party notices

dispatch-voice itself is MIT-licensed (see `LICENSE`). It depends on, downloads,
or ships the following third-party material. Nothing below is bundled in this
repository except the stock-voice reference clip noted under "Stock voice clip".

## Speech model (downloaded at runtime, not redistributed)

**Chatterbox-Turbo (ONNX export)** by Resemble AI
- Source: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
  (pinned revision `d21799bd0354adb85e348b8a0442a8405110a2cf`, see
  `MODEL_REVISION` in `src/dispatch_voice/engine.py`)
- Upstream project: https://github.com/resemble-ai/chatterbox
- Licence: MIT, per the model card. Copyright (c) Resemble AI.
- `dispatch-voice fetch-models` downloads the weights, tokenizer and config
  from Hugging Face into the user's data directory. They are never committed
  here (`.gitignore` excludes `models/`, `*.onnx`, `*.onnx_data`).

## Speech-recognition and turn-taking models (downloaded at runtime, not redistributed)

Fetched by `dispatch-voice fetch-models` into the speech-model directory, each
pinned to a revision and sha256-checked (`src/dispatch_voice/host/download.py`):

- **Silero VAD** (`silero_vad.onnx`), snakers4/silero-vad — MIT.
- **Smart Turn v3.2** (`smart-turn-v3.2-cpu.onnx`), pipecat-ai/smart-turn-v3 — BSD-2-Clause.
- **Parakeet TDT 0.6B v2** (ONNX int8 export by istupakov, `istupakov/parakeet-tdt-0.6b-v2-onnx`),
  NVIDIA — CC-BY-4.0. Attribution: NVIDIA Parakeet TDT 0.6B v2, licensed under CC-BY-4.0.
  Run through `onnx-asr` (MIT).

## Test fixture audio (shipped in this repository)

`tests/fixtures/clips/`, `clips-8k/` and `addendum.wav` are synthetic speech:
Kokoro-82M output (Apache-2.0 weights, hexgrad) and this package's own stock
voice. No real person's voice.

## libopus (system library, loaded at run time, not bundled)

**libopus** (Xiph.Org Foundation, Skype Limited, Octasic, Jean-Marc Valin,
Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding, Erik de Castro
Lopo, Mozilla, Amazon; Copyright 2001-2023), BSD-3-Clause. Not bundled:
`dispatch_voice.transport.opus` loads the system library at run time through
ctypes (`libopus.so.0`; opus 1.6 on Fedora 44). Without it, voice chat uses PCM.

## Optional streaming recogniser (Moonshine, not installed by default)

**Moonshine** (Useful Sensors, `moonshine-voice`) — MIT. `rt.stt_engine: moonshine`
uses it when the operator has installed it separately; otherwise the recogniser
falls back to Parakeet (`src/dispatch_voice/rt/stt_stream.py` probes for it with
`importlib.util.find_spec` and never requires it). Its models are downloaded by
that package, not by this one.

## Optional watermark (`watermark` extra)

**resemble-perth** (Perth implicit audio watermarker) by Resemble AI
- Source: https://github.com/resemble-ai/Perth (PyPI `resemble-perth`)
- Licence: MIT
- Only installed with `uv sync --extra watermark`; it in turn pulls PyTorch
  (BSD-3-Clause), torchaudio (BSD-2-Clause) and librosa (ISC).

## Stock voice clip (shipped in this repository)

**`src/dispatch_voice/stock/synthetic-host.wav`** — 12.6 s of fully synthetic
speech generated with **Kokoro-82M** (voice `am_michael`) by hexgrad. It is
the reference clip of the stock voice (`dispatch_voice.stock`) and the tests'
fixture.
- Source: https://huggingface.co/hexgrad/Kokoro-82M
- Licence of the weights: Apache-2.0
- It is not a recording of a real person. The audio is model output, not
  Kokoro's code, so the GPL-3.0 phonemizer used to make it does not reach it.
- The package (`src/`, what a wheel ships) never imports Kokoro or its
  phonemizer. Two developer scripts can use Kokoro when it happens to be
  installed separately: `tests/fixtures/make_fixtures.py` (to regenerate the
  fixture clips) and `tests/latency_harness.py` (a fallback engine for its
  timings). Both import it only at run time, through the separately
  installed, GPL-3.0-or-later `dispatch-voice-kokoro` plugin or `kokoro-onnx`
  (MIT runtime, whose phonemizer chain bundles GPL-3.0 espeak-ng); no Kokoro or
  phonemizer code is copied into this repository, and neither script is part
  of the package. Four end-to-end harnesses (`tests/e2e_voice_roundtrip.py`,
  `tests/latency_ws.py`, `tests/e2e_lowbw.py` with `--kokoro`, and the
  `livelike` scenario of `tests/e2e_browser.py`) never import it: they start a
  separate scratch DisPatch process with the separately installed plugin on its
  command line, the way the live service runs it.

## Python dependencies (installed by the package manager, not redistributed)

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause (plus bundled 0BSD/MIT/Zlib/CC0 parts) |
| onnxruntime | MIT |
| onnx-asr | MIT |
| pyyaml | MIT |
| httpx | BSD-3-Clause |
| tokenizers | Apache-2.0 |
| huggingface_hub | Apache-2.0 |
| soundfile | BSD-3-Clause; its wheels bundle **libsndfile**, LGPL-2.1-or-later (dynamically loaded; replaceable by a system libsndfile, which satisfies the LGPL's relinking terms) |
| pyloudnorm | MIT |
| fastapi | MIT |
| python-multipart | Apache-2.0 |
