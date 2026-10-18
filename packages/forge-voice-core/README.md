# forge-voice-core

The host-independent voice engines behind DisPatch Voice, as a package any application can embed.
No web framework, no application: everything application-shaped goes through one small seam
([`docs/HOST_API.md`](docs/HOST_API.md)).

| Area | Modules |
|---|---|
| Voice activity + turn detection (CPU) | `vad_turn` (Silero VAD, Smart Turn v3.2, endpointer) |
| Speech to text (CPU) | `rt.stt_stream` (Parakeet TDT 0.6B, Moonshine), `stt` |
| Text to speech (CPU) | `engine` (Chatterbox-Turbo ONNX q4), `rt.tts_stream`, `engines.chatterbox`, `tts`, `speak`, `profiles`, `stock` |
| OpenAI-compatible clients | `providers.OpenAISTTProvider`, `tts.HQTTS` |
| Provider interfaces | `providers` (`STTProvider`, `TTSProvider`, `VADProvider`; `health()`, `capabilities()`, `settings_schema()`), `registry`, `health` |
| Settings primitives | `settings_schema` (`Setting`, `Schema`, the engine-level rows), `engine_settings` |
| Models | `models` (one models folder), `download` + `fetch_job` (progress and cancel), `sysreq` (ffmpeg, libopus) |
| Network policy | `netpolicy` (which addresses a remote speech server may be) |
| Realtime policy | `rt.*` (endpoint, interrupt, speculate, backchannel, heard) |
| Optional GPU rig (SpeechForge, protocol sf.1) | `rig`, `rt.stt_rig`, `engines.rig_tts` — install the `speechforge` extra |

## Install

```text
pip install forge-voice-core                   # CPU engines
pip install "forge-voice-core[speechforge]"    # + the GPU-rig client (adds websockets)
```

From a checkout of the DisPatch_Voice repository (this package lives in `packages/forge-voice-core/`):

```text
uv pip install ./packages/forge-voice-core      # then, for DisPatch:  uv pip install .
```

Importing `forge_voice_core` imports nothing heavy; each engine imports `numpy`/`onnxruntime` when it
is first used. The package imports without `dispatch_voice`, `fastapi` or `starlette` (a test enforces it).
Models are downloaded on request (`python -m forge_voice_core.download --dest DIR`, or the `fetch_job`
API), never at import or install.

System programs pip cannot install: `ffmpeg` (decodes an uploaded voice clip, needed to clone a voice) and
`libopus` (optional, the low-bandwidth link). `forge_voice_core.sysreq.check()` reports them with the
install command for the operating system.

## Embedding it

```python
from forge_voice_core import hostlink, models
from forge_voice_core.hostlink import HostLink

class MyHost(HostLink):
    def models_root(self): ...          # where the model families live
    def load_settings(self): ...        # an EngineSettings-shaped object

hostlink.install(MyHost())
```

The full contract, the audio/turn callbacks a conversational host wires around the engines, and a
checklist are in [`docs/HOST_API.md`](docs/HOST_API.md).

## Licence

MIT. The engines download third-party models on request: Silero VAD (MIT), Smart Turn (BSD-2-Clause),
Parakeet TDT (CC-BY-4.0, attribution required), Chatterbox-Turbo (MIT). See `THIRD_PARTY_NOTICES.md`
at the repository root.
