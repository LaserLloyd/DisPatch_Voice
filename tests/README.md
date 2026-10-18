# Tests

```
uv run pytest -q                 # Python (model-backed tests skip when the models are absent)
uv run pytest -q packages/forge-voice-core/tests   # the engines package: rt/, rig/, vad_turn, tts, models, settings schema,
                                                   # and the guard that it imports without dispatch_voice / fastapi / starlette
npm install && npm test          # client modules (node --test; jsdom for the DOM suites)
npm run test:ci                  # same, but a missing jsdom FAILS (CI)
python3 scripts/check-locales.py # locale bundles
```

Use `uv sync --all-extras` (not a bare `uv sync`) on a machine that has the optional
watermark extra installed: a bare sync removes it.

Model-backed runs (`-m models`, the latency harness, the e2e harnesses with real engines)
load 1–2 GB of models and keep several cores busy. On a machine where other jobs do the
same, serialise them with one lock file shared by every such job, named here `$VOICE_LOCK`
(any path; `flock` creates it): `flock "$VOICE_LOCK" uv run pytest -m models`. A run that
loaded models exits through `os._exit` with pytest's own status once the report is printed
(ONNX Runtime's thread pools could abort the interpreter's teardown with exit 134 after
every test passed); `DISPATCH_VOICE_TEST_HARD_EXIT=0` turns that off.

**Windows.** The suite runs on Windows (CI has a job for it). Three rules keep it that way:
file-mode assertions go through `fakes.assert_private` / `make_private` (a POSIX mode on Linux, the
ACL on Windows: `packages/forge-voice-core/src/forge_voice_core/fsperm.py`); every text file is read and written with
`encoding="utf-8"`; and tests never wait with `asyncio.sleep(<15 ms)` - the event loop fires a
timer shorter than the clock tick (15.6 ms on Windows) immediately, so such a wait returns at
once. Use `settling.pause` / `settle` for real time, and `settling.wait_until(cond)` wherever the
test can say WHAT it waits for. Settings tests set `DISPATCH_VOICE_*` through `monkeypatch`
only: the environment now beats `voice.yaml`.

The browser harnesses use Playwright's own Chromium unless `--chrome PATH` or
`DISPATCH_E2E_CHROME` names another; results and screenshots go to `.e2e-out/`
(git-ignored) unless `--out` / `--shots` say otherwise. `e2e_browser.py --target
dispatch --scenarios livelike` runs a scratch DisPatch set up like a real install
(entry points, the Kokoro plugin when present, a PIN, a 60 s idle lock).
`e2e_voice_roundtrip.py` and `latency_ws.py` take `--llm-port` to put the stub LLM
on a fixed port instead of any free one.

## Layout and ownership

Each area has one owner and its own `conftest.py`; nobody edits another area, the root
`conftest.py`, `fakes.py`, `stub_llm.py`, `load.py` or `fixtures/` without a note to the
owner (T0 for the shared files).

| Path | What | Owner |
|---|---|---|
| `test_*.py` (here) | the speech package: engine, audio, profiles, router, speak, stock | existing |
| `test_scaffold.py` | the layout, the fixtures and the doubles below | T0 |
| `session/`, `host/`, `contract/` | the session hub, `host/` modules, the DisPatch contract test | T3 |
| `rt/`, `test_models_rt.py`, `latency_harness.py` | realtime engines | T4 |
| `text/`, `manual/` | spoken-reply text, acknowledgements, role-play | T5 |
| `transport/`, `shaper.py`, `e2e_lowbw.py` | Opus, framing, offsets, slow links | T6 |
| `js/{drive,voice-tab,index,settings,absent-interplay}.test.js`, `e2e_browser.py` | the client app | T7 |
| `js/{worklet,player,opus,link-monitor,local-voice,uplink-ring,seq-claims,wire3}.test.js` | client audio modules | T8 |
| `e2e_voice_roundtrip.py` | wire round trip on a scratch instance | T3 |

`<area>/legacy/` holds the pre-split suites copied byte for byte from DisPatch (they still
import `app.voice.*`). They are never collected (root `conftest.py`, `collect_ignore_glob`)
and `npm test` does not reach `js/legacy/`. The owner ports a file onto the doubles below,
then deletes the legacy copy.

## Shared doubles (`fakes.py`, fixtures in the root `conftest.py`)

- `fake_ctx`: a `FakeContext` standing in for DisPatch's `ExtensionContext` (contracts §B.1):
  in-memory database, four bots (`sage` advisor, `swift` api, `nova` gateway, `alpha` safe),
  thread `t-voice-1` on `sage`. Every call is recorded (`ctx.calls_to("submit_turn")`).
  Knobs: `reply_for` (scripted advisor-style reply frames after `submit_turn`),
  `speculation_reply`, `partial_on_abort`, `operator`, `turn_running`. `subscribe_frames`
  delivers through a bounded queue (256, drop-oldest) and accepts `replay=` (e.g. the recorded
  snapshot); `await ctx.drain()` waits for deliveries. `ctx.connect("voice", thread_id, ws)`
  runs the gate's last steps and the registered handler; `ctx.app` is a FastAPI app with every
  `include_router` mounted behind the operator gate.
- `fake_engines`: `FakeSTT`, `FakeTTS`, `OneTurnDetector` instances plus the `FakeWS` and
  `ScriptedDetector` classes and `utterance()`.
- `b1a_frames` / `fakes.frames_for(name, thread_id)`: the recorded frame sequences in
  `fixtures/frames_b1a.json` (advisor, gateway, simulated-stream, short-reply and error turns,
  the mid-stream snapshot and what follows it, the interrupted partial row, the `note_heard`
  update), with the shapes DisPatch emits.
- Contract mirrors: `VoiceInstructions`, `BotView`, `ExtPaths`, `ToolSpec`, `ThreadSocket`.

## Fixtures (`fixtures/`)

- `clips/`: 16 kHz mono 16-bit speech with known text (`clips/meta.json`): `short`, `medium`,
  `long`, `pause_a`, `pause_b`, `pause` (0.7 s mid-sentence pause), `backchannel` (0.87 s
  "Uh-huh."), `interrupt`, `driver1b`, `driver2`. All synthetic voices (Kokoro `am_michael`;
  the add-on's stock voice for `driver*`), no real person.
- `addendum.wav` (+ `addendum.json`, exact sample offsets): "What's on my calendar this
  afternoon" + exactly 2.0 s of digital silence + "Oh, and tomorrow."
- `clips-8k/`: every clip and the addendum through an 8 kHz capture and the mic worklet's
  linear upsampling (a Bluetooth car kit); still 16 kHz files.
- `transcripts.json`: what Parakeet hears on each clip (baseline). Note the backchannel:
  Parakeet never hears "uh-huh".
- `wire3.json`: shared binary-frame vectors for the Python and JS wire tests (T6 fills it).
- `make_fixtures.py`: rebuilds `clips-8k/` (`derive`, pure numpy) and the addendum
  (`addendum`, needs the optional Kokoro plugin).

## Harness helpers

- `stub_llm.py`: OpenAI-compatible stub (`/v1/chat/completions` streamed or not, 1.2 s to the
  first token by default; `/v1/embeddings` with a settable delay, e.g. 0 / 1.7 / 2.5 s).
  Every reply is different within its first 20 characters, because DisPatch drops a reply that
  shares a 60-character opening with a recent one. `StubLLM(...)` in process, or
  `python tests/stub_llm.py --port N`; runtime knobs on `POST /_stub/config`; a client that
  hangs up mid-stream is logged as `aborted`.
- `load.py`: `cpu_load(8)` keeps 8 processes busy (the "under load" column).

Model-backed runs (Parakeet, Smart Turn, Chatterbox, Kokoro, the latency harness) load
gigabytes; on a shared machine serialise them behind one lock (`flock <lockfile> <cmd>`).

**Where the tests live.** The engines and their tests moved to `packages/forge-voice-core/` (its own `conftest.py`, `settling.py`, `helpers.py` and sf.1 fixtures). `tests/host/test_rig_session.py` and `tests/test_models_rt.py` still import that suite's fake rig and model probes by path. `tests/host/test_core_bridge.py` pins the deprecated import aliases and DisPatch's host link.
