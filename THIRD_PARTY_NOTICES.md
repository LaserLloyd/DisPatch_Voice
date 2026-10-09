# Third-party notices

dispatch-voice itself is MIT-licensed (see `LICENSE`). It depends on, downloads,
or ships the following third-party material. Nothing below is bundled in this
repository except the test fixture noted under "Test fixture".

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

## Optional watermark (`watermark` extra)

**resemble-perth** (Perth implicit audio watermarker) by Resemble AI
- Source: https://github.com/resemble-ai/Perth (PyPI `resemble-perth`)
- Licence: MIT
- Only installed with `uv sync --extra watermark`; it in turn pulls PyTorch
  (BSD-3-Clause), torchaudio (BSD-2-Clause) and librosa (ISC).

## Test fixture (shipped in this repository)

**`tests/fixtures/synthetic-host.wav`** — 12.6 s of fully synthetic speech
generated with **Kokoro-82M** (voice `am_michael`) by hexgrad.
- Source: https://huggingface.co/hexgrad/Kokoro-82M
- Licence: Apache-2.0
- It is not a recording of a real person.

## Python dependencies (installed by the package manager, not redistributed)

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause (plus bundled 0BSD/MIT/Zlib/CC0 parts) |
| onnxruntime | MIT |
| tokenizers | Apache-2.0 |
| huggingface_hub | Apache-2.0 |
| soundfile | BSD-3-Clause |
| soxr (python-soxr) | LGPL-2.1-or-later (dynamically linked; unmodified) |
| pyloudnorm | MIT |
| fastapi | MIT |
| python-multipart | Apache-2.0 |
