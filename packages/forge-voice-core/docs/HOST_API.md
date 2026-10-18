# Host contract: what an application provides to forge-voice-core

`forge-voice-core` is a set of engines and policies. It has no microphone, no speaker, no chat and no
window: an application (the *host*) wires them to its own audio, its own conversation and its own
settings. DisPatch Voice is the first host (`dispatch_voice/_core_bridge.py` plus the FastAPI session code);
ChatForge is the second. This document is the whole contract. The code behind it is
`forge_voice_core/hostlink.py` (the seam) and the engine modules named below.

Everything in section 1 is **optional to override**: the defaults run the package standalone, from the
environment and the platform's data folder. Sections 2 to 6 are what a host has to **build around** the
engines, because the package deliberately does not contain a conversation loop.

## 0. Rules the package keeps (so a host can rely on them)

* `import forge_voice_core` imports nothing heavy and never `dispatch_voice`, `fastapi` or `starlette`
  (a test enforces it with an import-blocking fixture). Engines import `numpy`/`onnxruntime` when first used.
* Nothing loads a model, opens a socket, starts a thread or writes a file at import or install time.
* Every blocking engine call runs in the caller's thread. A host runs them off its UI thread and off its
  event loop (`asyncio.to_thread`, a worker). Only the GUI thread is forbidden.
* Audio formats are fixed: **microphone = PCM16 little-endian mono 16 kHz**; **speech out = int16 mono 24 kHz**
  (Chatterbox; a remote engine declares its own `sample_rate`).
* The package logs through one logger family (`forge_voice`; `forge_voice_core._log.set_name("my.app")` before
  the first engine import renames it). The rig client redacts the bearer token from its own logs and from the
  `websockets` library's debug output.

## 1. The host link (`forge_voice_core.hostlink`)

Subclass `HostLink`, override what you need, call `hostlink.install(MyHost())` **once, before the first engine
call**. `hostlink.get()` is what the package calls; `hostlink.reset()` restores the standalone default (tests).

| Method | Default (standalone) | A host overrides it to | Read by |
|---|---|---|---|
| `models_root()` | `$FORGE_VOICE_MODELS_DIR`, else `<data>/models` | return its models folder; the families live in `speech/` and `chatterbox/` under it | `models`, `download`, `engine` |
| `voices_dir()` | `$FORGE_VOICE_VOICES_DIR`, else `<data>/voices` | where cloned voices are kept (personal data: never in the app bundle or a zip) | `profile_bridge` |
| `state_dir()` / `has_state_dir()` | `$FORGE_VOICE_STATE_DIR`, else `<data>/state` / `True` | a private folder for small secrets files | `rig` (token file) |
| `legacy_token_path()` | `~/.config/forge-voice/speechforge.env` | the old file to still read as a fallback | `rig` |
| `data_dir()` | `$FORGE_VOICE_DATA_DIR`, else the platform data folder + `forge-voice` | its data folder | the defaults above |
| `load_settings()` | `EngineSettings()` | return **its** settings object (see section 2) | `tts`, `registry`, `rig`, `engines.chatterbox` |
| `settings_schema_json()` | the engine-level rows | return the rows of the whole Settings page | `Provider.settings_schema()` |
| `bot_voice(bot_id)` | `""` | the voice profile a character was saved with | `registry.resolve_profile` |
| `profiles()` | `forge_voice_core.profile_bridge` (store under `voices_dir()`, the stock voice) | an object with `store()`, `stock_id()`, `stock_profile(build)`, `release_engine()` | `engines.chatterbox`, `rig` |
| `ensure_engine()` | registers `chatterbox` once its models are on disk | also register its profile resolver | `tts` (before every engine lookup) |
| `secret_store()` | `None` | an object with `get(name)`, `set(name, value)`, `delete(name)` (OS keyring / Credential Manager) | `rig` (the rig token goes here first) |
| `power()` | `None` | an object with `acquire(settings)` / `release()` to hold a performance power profile | `rig` |
| `router_mounted()` | `False` | whether its cloned-voices HTTP router is mounted | `fetch_job` (the "restart required" flag) |
| `client_name`, `client_version` | `forge-voice-core`, the package version | its name and version, sent to the speech rig | `rig` |

The token for the optional GPU rig is **never** a setting value. It comes from `SPEECHFORGE_TOKEN`, then
`secret_store()`, then the token file; `rig.read_token()` says which and why not. A ChatForge build should
implement `secret_store()` on top of Windows Credential Manager so no token file exists at all.

## 2. Settings

The engines take a duck-typed `settings` argument and read these attributes with `getattr`. A host either
passes `forge_voice_core.engine_settings.EngineSettings` (every field has the shipped default) or its own
object with the same names.

| Group | Attributes |
|---|---|
| paths | `models_dir`, `silero_path`, `smart_turn_path`, `parakeet_dir` |
| STT | `stt_engine` (`parakeet`/`moonshine`), `stt_provider` (`cpu`/`openai`/plugin id), `stt_url`, `stt_model`, `stt_timeout_s`, `threads` |
| TTS | `tts_engine` (`auto`/`chatterbox`/`minimax`/plugin), `tts_provider`, `builtin_voice`, `speed`, `hq_url`, `hq_model`, `hq_timeout_s`, `minimax_*`, `tts_threads` |
| turn taking | `vad_threshold`, `min_speech_s`, `turn_probe_silence_s`, `turn_hard_silence_s`, `no_smart_turn_silence_s`, `max_utterance_s`, `smart_turn_threshold` |
| realtime | `rt` (`rt.settings.RtSettings`: timings, `decoder_sessions`, `first_chunk_tokens`, ...) |
| rig | `rig` (`rig.RigSettings`: `enabled`, `url`, deadlines), `rig_allow` (hosts a PUBLIC address may be) |
| plugins | `extensions` (entry-point names), `sources` (where a value came from; only used for a deprecation warning) |

Secrets (`stt_api_key`, `hq_api_key`, `MINIMAX_API_KEY`) are read from the **environment only**, by name,
never from the settings object.

`forge_voice_core.settings_schema` is the declarative part: `Setting` (a row), `ENGINE_SETTINGS` (the rows the
engines own: STT/TTS, `rt.*`, `rig.*`), and `Schema(rows, env_prefix=...)`, which generates `clean_prefs`,
`validate_patch`, `resolve` (environment beats file beats default), `schema_json` and `env_table` for any table.
A host builds `Schema(ENGINE_SETTINGS + its_own_rows)`; the shipped environment names are `DISPATCH_VOICE_*`, and
`env_prefix="CHATFORGE_VOICE_"` renames them.

## 3. Models and the download UI

Models are never fetched at import or install. The host drives them:

* **Where:** `models.speech_dir()` (Silero VAD, Smart Turn, Parakeet) and `models.chatterbox_dir()` (Chatterbox-Turbo
  ONNX), both under `hostlink.models_root()`. `models.migrate(dry_run=...)` moves a pre-1.6 two-folder layout.
* **What is missing:** `sysreq.check()` returns one row per OS program (`ffmpeg`, `libopus`) with the install
  command for the platform; `health.report(settings)` returns `{id, state, reason, fix, code}` rows for a doctor page;
  each provider has `health()`. `engines.chatterbox.available()` and `models.parakeet_complete(dir)` answer
  "are the models there".
* **Download job (progress + cancel):** `fetch_job.JOB` is one background job, one at a time.
  `JOB.start(only=None|"speech"|"chatterbox", force=False, stock=True)` -> `False` if one is running;
  `JOB.status()` -> `{state: idle|running|done|failed|cancelled, phase, file, files_done, files_total, percent,
  error, cancellable, log, restart_required}` (poll it, ~2 Hz); `JOB.cancel()`. Progress is weighted by bytes, so
  the bar moves with the data. A cancelled or failed job removes its partial file; a retry resumes at the next
  missing file. Total ~1.4 GB; the host shows that number and asks first.
* **Blocking variant (a CLI, a first-run wizard):** `download.fetch_all(only=, force=, verify=, stock=, echo=, progress=,
  cancel=)` where `progress(event)` receives `{phase, file, index, total, bytes, bytes_total, weight_mb, done}` and `cancel`
  is a `threading.Event`. Every Silero/Smart-Turn/Parakeet file is sha256-checked; Chatterbox is revision-pinned.
* **Disk and licences:** the host shows the Parakeet attribution (CC-BY-4.0) somewhere users can read it.

## 4. Audio in: capture, then turn detection, then speech to text

The host owns the microphone (a native stream or a browser `getUserMedia` worklet). It hands the package
**PCM16 LE mono 16 kHz bytes in any chunk size**:

```python
from forge_voice_core import vad_turn, rt
from forge_voice_core.rt.stt_stream import build_streaming_stt

settings = host_settings()                       # section 2
detector = vad_turn.build_detector(settings)     # Silero VAD + Smart Turn (if its model is present)
stt = build_streaming_stt(settings)              # Parakeet tail-mode streaming recogniser

def on_mic_chunk(pcm16: bytes) -> None:          # host's capture callback (audio thread -> a worker queue)
    for ev in detector.feed(pcm16):              # 32 ms frames; returns a list of events
        if isinstance(ev, vad_turn.SpeechStart):    # the user started talking: barge-in point
            ...
        elif isinstance(ev, vad_turn.TurnEnd):      # ev.audio: float32 mono 16 kHz, the whole utterance
            text = transcribe(stt, ev.audio)        # see below
            host.on_turn(text)                      # section 5
```

* `detector.feed(pcm16, end=None)` returns `SpeechStart`, `TurnEnd(audio, reason, turn_prob)`, `Discarded`, and (in
  `probe_events` mode) `Pause`/`Resumed`. `detector.reset()` between conversations, `discard_utterance()` to drop an echo.
* Streaming STT: `stt.push(pcm16_array, vad_prob=detector.last_prob)` while speaking, `stt.at_pause()` / `stt.tick()`
  for partial text, `stt.final()` for the final transcript, `stt.reset()` after. For a one-shot "transcribe this
  recorded utterance" use `stt.shared_engine(settings).transcribe(float32_audio)` (a batch Parakeet) instead.
* `rt.endpoint.TurnPolicy` / `rt.speculate` / `rt.backchannel` / `rt.interrupt` / `rt.heard` are pure policy objects
  for hands-free mode (merge window, speculative replies, barge-in, "what the listener heard of a cut reply"). A
  push-to-talk host needs none of them, and not even the detector: buffer the microphone while the key is held, then
  `stt.shared_engine(settings).transcribe(float32_audio)` on release (`detector.end_turn("manual")` ends an utterance the
  detector already opened, and returns `None` when none is open).
* Remote or alternative recognisers implement `providers.STTProvider`; `providers.REGISTRY` lists the built-ins
  (`cpu`, `openai`, `rig`) and plugins from the `dispatch_voice.providers` entry-point group.

## 5. The turn callback and the reply

The package has **no** conversation. The host supplies:

| Host callback | Signature | Notes |
|---|---|---|
| turn | `submit(text: str) -> message_id` | called with the final transcript; the host puts it through its normal send path so a spoken message is an ordinary message |
| reply stream | `reply_frames` -> an iterator of text deltas | the host adapts its chat engine's token stream |
| abort | `cancel(message_id)` | barge-in / the stop button |
| retract | `drop_last_turn()` | optional: the turn-merge window re-submits a merged utterance |
| heard | `note_heard(message_id, text)` | optional: what the listener actually heard of a cut reply |

The reply instruction ("short sentences, no markdown, numbers as words") is the host's system-prompt concern.

## 6. Audio out: text to speech and playback

```python
from forge_voice_core import tts

engine = tts.shared_engine(settings)          # registry picks chatterbox / a plugin / MiniMax / the HQ server (+ fallback)
async for pcm in tts.speak(reply_text_chunks, profile_id=None, engine=engine, stop=stop_event):
    play(pcm, rate=engine.sample_rate)        # bytes, int16 mono; the host's playback callback
```

* `tts.speak` splits the incoming text on sentence boundaries (`text.speech_text.SentenceSplitter`, which keeps source
  spans so a cut reply can be stored up to the last heard word), drops what cannot be spoken (code fences become "The
  code is in the chat."; links, markdown and emoji removed; numbers and dates read as words), and yields frames.
* Playback is the host's: it must accept `engine.sample_rate` (24 kHz for Chatterbox; resample if the device cannot),
  must be able to **stop immediately** (set `stop`; drop queued frames) and should report how much audio has actually
  played if it wants `rt.heard`.
* The first spoken frame is ~1.5-2 s after the first sentence on a laptop CPU (measured upstream; measure on your
  hardware). Chatterbox runs at a real-time factor of 0.7-0.9; budget ~3 GB of RAM while loaded and unload on idle
  (`profile_bridge.release_engine()`, `registry.run_release_hooks()`).
* Voices: `profiles.ProfileStore` (clone from a clip: needs `ffmpeg`), `stock` (a bundled synthetic voice that speaks when
  nobody has cloned one). A cloned voice must be consented to by its speaker; the store records the statement.

## 7. Network policy

Anything that sends audio or reply text to an address (an OpenAI-compatible server, the rig) goes through
`netpolicy.check_url_policy(url, allow, schemes=...)`: loopback, private, tailnet and LAN names are accepted; link-local
and cloud-metadata addresses are never; a public host only when it is on the operator's allow-list
(`settings.rig_allow`). `netpolicy.check_resolved(url)` re-checks after DNS resolution (rebinding). A host's settings UI
should call the first on save and show `verdict.reason` / `verdict.fix`.

## 8. Checklist for a new host

1. Subclass `HostLink`: `models_root`, `voices_dir`, `state_dir`, `load_settings`, `secret_store`; install it first thing.
2. Settings card: render `Schema(ENGINE_SETTINGS + yours).schema_json()`; persist; pass an object with the section 2 names.
3. Models card: `sysreq.check()`, `health.report()`, `fetch_job.JOB` (start / status / cancel); show size and licence first.
4. Capture: 16 kHz mono PCM16 -> `TurnDetector.feed` on a worker; STT final on `TurnEnd`; `submit(text)`.
5. Playback: `tts.speak` on a worker, 24 kHz int16 frames to the output, with an immediate stop.
6. Quit: stop the worker, `profile_bridge.release_engine()`, `fetch_job.JOB.cancel()`.
7. Test with a fake `HostLink` (the package's own tests do) and never against a live service.
