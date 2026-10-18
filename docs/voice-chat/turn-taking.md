# Voice chat: turn-taking and latency

<!-- Owner: track T4. -->

This page covers the realtime engines in `src/dispatch_voice/rt/`: how the add-on hears
you while you are still talking, decides that you have finished, starts the reply before
the last word is confirmed, gets the first sound out of a cloned voice in about a second
and a half, tells a cough from "wait, stop", and remembers how much of an answer you
actually heard. The session hub (`host/session.py`, see [host.md](host.md)) wires these
pieces together; nothing here opens a socket or touches DisPatch.

Every number on this page was measured on the development box (Ryzen AI 9 365, CPU only)
in the state the car sees it in: `power-saver`, `cpuinfo_max_freq` 2.0 GHz. The
`performance` hold ([host.md](host.md)) makes the CPU-bound stages 1.5–1.9× faster. The
latency harness prints the profile with every table so the two are never confused.

## The pipeline in one picture

```
mic frames ─► Silero VAD ─► at ≈ 0.15 s of silence: ParakeetTailSTT.prime() (the tail call starts early)
                        ─► at every pause ≥ 0.25 s:
   ├─► Smart Turn on the last 8 s          (≈ 120–190 ms, thread A)   ─┐
   └─► ParakeetTailSTT.at_pause()          (reuses the primed call;    ┤ TurnPolicy.decide
                                            else ≈ 250–700 ms, thread B) ▼
   end / hold  ─► Speculator.start(partial) ─► the advisor's LLM call starts (≤ 2 per turn)
   final()     ─► similar ≥ 0.95 → adopt the running reply, else submit normally
   first sentence ─► PipelinedChatterbox: LM thread (voice prefix cached) → 25-token chunk → decoder → 40 ms frames
   while the reply plays: a pause partial ─► classify_interruption → echo | backchannel | stop | real
   on a real interruption ─► HeardMap.locate(seq, played_ms) → voice_heard {upto, heard, unheard}
```

## Streaming speech recognition (`rt/stt_stream.py`)

Parakeet TDT 0.6B v2 is a whole-utterance model, but its partial transcripts are stable:
across 44 measured steps the leading words always matched the final transcript except the
last one or two. `ParakeetTailSTT` turns that into a streaming recogniser without a new
model:

* The utterance is a list of closed segments (text committed, never changes) plus one open
  tail. `push()` takes every 32 ms frame with Silero's probability.
* `at_pause()` transcribes only the open tail. Once the tail is longer than 5 s it is first
  closed at the last silence of ≥ 300 ms that ended ≥ 1 s ago, so a monologue never makes a
  call grow without bound (the 15 s fixture clip commits once and every call stays under
  1.2 s). When no such silence exists — road noise or a radio holding Silero up, a speaker
  chaining clauses with < 0.3 s pauses — a tail past `tail_max_s` (10 s) is closed anyway at
  its quietest frame ≥ 1 s old, and a closed segment is never longer than that either, so
  no recogniser call sees more than ≈ 10 s of audio (measured on the 15 s clip with the VAD
  forced to "speech": calls of 10 + 5 s, WER unchanged within the threshold).
* `tick()` gives live captions at most once a second and never while a pause call is in
  flight; it closes a long tail too.
* `reset()` is safe from any thread at any moment (a cancel, undo, hang-up or new `start`
  while the STT thread is inside a call): every call takes its audio and stores its result
  under the buffer lock with a generation check, so the cancelled turn's words are dropped
  and the next turn is always re-transcribed.
* `final()` returns committed + the last partial when less than 0.3 s of audio arrived since
  that partial, or when everything since it was silence — the usual case, since the turn
  ends at the pause that was just transcribed — and measures in the microseconds; otherwise
  it makes one more tail call. If a pause call is in flight it waits for it first (≤ 300 ms
  after it returns on the 7.3 s clip) — `final(timeout_s=…)` bounds that wait and answers
  with the best known text past it, which is how the session honours `hold_max_s`.
* `prime()` lets the session start the tail call **before** the probe, at the first ≈ 0.15 s
  of silence, in a worker thread; the probe's `at_pause()` then reuses it. This is exact:
  Parakeet's transcript of every fixture clip is identical with 0.10 and 0.25 s of trailing
  silence (`run/T4/probe_early_tail.json`; outside that band one clip moves a full stop, the
  words never change). The decision lands ≈ 0.1 s earlier on every turn; a false pause (the
  speaker goes on) wastes the call — 0–1 per clip at 0.15 s, 0–4 at 0.10 s — and the next
  `at_pause()` waits for it, so the worst case is one call later than today, never more.
* `Partial.stable_words` is the agreement with the previous partial minus two, so the
  client can draw the settled words in full colour and the tail dimmed.
* The STT and Smart Turn sessions are created with ORT intra-op spinning **off**
  (`ort_options`): measured by the lead, −9…−23 % under load and the 46–59 % of a core
  burnt after each call drops to zero. The TTS decoder keeps spinning.

Measured at 2 GHz, idle: `at_pause()` 253 ms on a 2.2 s tail, 337 ms at 3.5 s, 523 ms on the
whole 7.3 s clip; transcripts of every fixture clip match the baseline (WER 0, the 8 kHz
band-limited copies too). The 0.87 s "Uh-huh" comes out as "Odd High." from every model
tested — the interruption classifier keys on duration and token count, never on the word.

`MoonshineSTT` (optional, MIT, `rt.stt_engine: moonshine`) gives continuous words 0.4–0.6 s
behind the audio for under one core in single-thread mode, which the engine forces
(`MOONSHINE_ORT_SINGLE_THREAD=1`; the default thread mode pins all 20). It needs the
`moonshine-voice` package and its `medium-streaming-en` model under
`<speech models>/moonshine/` or in the package's cache; when either is missing the add-on
logs once and uses Parakeet tail mode. It is recommended only after a road test on the Pixel:
every number above is from clean synthetic speech.

## End of turn (`rt/endpoint.py`)

The flat 1.8 s merge hold is gone. At every pause the policy sees Smart Turn's P(done) on
the last 8 s and the pause partial, and decides once:

| P(done) | the partial ends with | decision |
|---|---|---|
| ≥ 0.90 | anything | **end** now |
| 0.50–0.90 | `. ? !` and not a continuation word | **end** now |
| 0.50–0.90 | no terminal punctuation, or a continuation word (and, so, could, if, the …) | **hold** for the learned hold × patience; the reply is prepared but not sent; speech inside the hold merges |
| < 0.50 | `. ? !` | **reprobe** at +0.4 s; end after 1.0 s of silence |
| < 0.50 | not terminal | **listen**; hard silence (1.4 s × patience) ends it; the bot may backchannel |

Why both signals: Smart Turn scores complete clauses 0.76–0.99 and mid-clause audio
0.01–0.2, and its one known misfire ("I was thinking that maybe we could" at 0.58) has no
terminal punctuation. Parakeet closes almost every utterance with a full stop, so the
continuation-word rule is what actually catches that case.

When only Smart Turn is in (the tail transcript still running) the session may ask with
`partial=None`: P ≥ 0.90 ends the turn at once; anything else waits. Speculation never
starts on a reprobe row, and never without text.

**Learned endpointing.** Per device the policy keeps an EMA of the driver's resume gaps
(the hold is that EMA clamped to 0.3–1.2 s) and a false-end count: each false end raises
the hold floor by 0.15 s and the probe silence by 50 ms (max 0.5 s); every third clean turn
takes one step back. The values travel in `ready.rt.learned` and the device blob; the
server's win. The Settings hint says only "Adjusts itself to how you pause."; Reset lives in
the Voice lab.

Smart Turn's probabilities depend on the onnxruntime version: the design table was measured
on 1.29; on 1.31 two rows move ("…call the dentist" 0.76 → 0.52, the "Uh-huh" clip
0.55 → 0.36) and the rest stay within 0.03. The decisions are the same; the model-backed test
carries both columns.

## Speculative replies (`rt/speculate.py`)

The advisor's LLM call starts on the first end/hold decision — about 0.55–0.8 s after speech
ends, as soon as the tail transcript exists — and the final transcript either adopts it
(`similar ≥ 0.95`, end-weighted so a changed last word on a short question is a different
question) or discards it. Caps: two starts per turn, three seconds apart, nothing on fewer
than four words, a run older than 20 s is dropped unread, and a run whose thread head moved
(a typed message landed meanwhile) is counted as stale and never adopted. `stats` feeds the
Voice lab line so the cost is visible before `rt.speculate: all` widens it to metered bots.

Gateway agents never speculate (tools and memory writes make a shadow run unsafe); they
still gain the STT overlap and the hold removal.

## First audio (`rt/tts_stream.py`, `engine.py`)

Chatterbox-Turbo's language model is fast once it is running (13 ms per speech token;
25 tokens = 1 s of audio), but its prefill used to cost 0.45–0.55 s per sentence, because it
re-read the voice's conditioning prefix (`cond_emb`, 151 positions, the same for every
sentence) each time. The conditional decoder is the other cost, and it re-decodes the whole
reference prompt on every call: 1.43 s for a 25-token chunk with the 10 s prompt a profile
keeps, 0.67 s with a 4 s slice. So:

* the LM's KV cache for the prefix is computed **once per voice** (`engine.prefix_cache`,
  ≈ 0.45 s, 30 MB; `PipelinedChatterbox.prepare(pid)` does it ahead of the first sentence,
  `warm()` for the default voice) and every sentence prefills only its own text tokens:
  531 → 98 ms for a 24-token sentence, 486 → 55 ms for a short one. The result is identical
  — the last-position logits match the full prefill bit for bit and 40 sampled tokens matched
  under the same seed (`run/T4/probe_prefix_kv.json`);
* the profile store keeps the full tensors; the engine slices `prompt_token` and
  `speaker_features` to `rt.decoder_prompt_s` (default 4 s; 2–10) at load;
* an LM thread streams tokens; the first chunk (25 tokens) is decoded as soon as its tokens
  exist, then 25-token chunks with 8 already-spoken tokens re-decoded as left context;
* every chunk also decodes **three tokens of lookahead** and drops their audio. Measured:
  with only silence tokens after it, a chunk's last token comes out 10–25 dB quiet; with
  three real tokens following it stays within ±12 % of the in-context level;
* the seam is a 20 ms equal-power crossfade of two decodes of the same tokens (the decoder
  draws fresh noise on every call; butted chunks click);
* the output is 40 ms int16 frames at 24 kHz, exactly what the whole-sentence engine
  produced, so the session's protocol does not change; `stop` is honoured per token and
  per chunk;
* `rt.decoder_sessions: 2` lets two chunks decode at once and the first chunk shrink to 15
  tokens (≈ 1.0 s to first audio measured in the prototype); the default stays **1** until
  the 20-thread contention run below passes on the box.

Measured (2 GHz, idle, `test_models_rt.py`): first audio **1.51 s** after the sentence text
on the 25-word sentence with the voice prefix cached (1.81 s before it), chunks every
≈ 0.81 s against 1.0 s of audio each — no underrun; the prefill of that sentence 526 → 110 ms.
Whole-sentence synthesis (`render_line`, used for acknowledgements and notices) keeps the
full 10 s prompt and a seed: the token sequence is reproducible, the vocoder's noise is not.

## Interruptions and backchannels (`rt/backchannel.py`)

The phone ducks its own playback the instant its mic hears speech (the client's job); the
server confirms with words at the first pause partial, 0.4–0.9 s after onset, classified as:

| verdict | rule | the session does |
|---|---|---|
| echo | ≥ 80 % of the words were just said by the bot (digits ignored; a lone command word is never an echo) | `duck off`, `resume` |
| stop | at most eight tokens made only of command words, stop phrases and acknowledgement words, with a command word or stop phrase among them or one ack word repeated — stop, wait, no, hold on, hang on, pause, skip, what, sorry, okay okay, got it, thanks, thank you, enough, that's enough, never mind, quiet, shut up, "okay, okay, got it, thanks", "perfect, thanks", "yeah no, that's enough" — or a second backchannel within 3 s. Different ack words alone ("okay right sure") are not a stop | `barge_in`; nothing becomes a turn |
| backchannel | under 1 s, at most two tokens, none a command word ("okay", "yeah", "Odd High.") | `resume` |
| real | everything else ("Wait, stop, that's not what I meant") | `barge_in` + abort + the words become the next turn |

A bot may say "Mm-hm." while you talk (`BackchannelGate`): only on a listen row, ≥ 4 s into
the turn, after a pause ≥ 0.5 s, ≥ 6 s since the last one, at most three per turn — and
only when the bot opted in (`voice.yaml: bots.<id>.backchannel`, default off) and the device's
Reactions setting is Often.

## What the bot knows after an interruption (`rt/heard.py`)

Each spoken sentence is a `Span`: the speakable text and its `[start, end)` in the stored
markdown. The client reports `heard {seq, played_ms}`; `HeardMap.locate` maps it linearly
over the sentence's text to a word boundary in the source that is never inside a link,
emphasis marker or tag (`**Tuesday at 3pm** — see [the map](…)` is spoken as different
characters), and produces the `heard`/`unheard` texts the next turn's instructions carry.
Sentence-level accuracy is exact; within a sentence the linear map is within ≈ 15 %.

## Latency budget (speech end → first reply audio, advisor bot, cloned voice)

| stage | 2 GHz power-saver | performance hold | mechanism |
|---|---|---|---|
| speech end → tail transcript | the primed call starts at 0.15 s and costs 0.2–0.7 s (2.2–7.3 s tails), parallel to Smart Turn 0.06–0.19 s; without priming it starts at the 0.25 s probe | ≈ 0.6× | tail mode, `prime()` |
| end-of-turn decision | ≈ 0.4–0.9 s after speech end (see the harness `at` column) | ≈ 0.3–0.6 s | policy |
| LLM first sentence | starts with the decision; ≈ 1.2 s to the first token on the stub (MiniMax-M3 median 1.0 s) | same | speculation |
| TTS first chunk | ≈ 1.5 s pipelined, one session, 4 s prompt, voice prefix cached | ≈ 0.8–0.9 s | prefix cache, lookahead chunks |
| **first reply audio** | the harness table in the T4 report (`impl/T4.md`) carries the measured medians per column | | |

The harness (`tests/latency_harness.py`) re-measures this on every change — never quote it
from memory. `--both` runs it idle and under eight busy processes; `--hold` takes the
`performance` hold through the add-on's own `PowerHold` for the run (the performance
column); `--early-tail 0` (default) is the probe-time call the session makes as
specified, `--early-tail 0.15` primes the tail call (opt-in, ≈ 0.03–0.06 s on short turns);
`--contention` is the 20-thread gate for `decoder_sessions: 2`; `--tts kokoro` gives the
plugin column; `ws` mode drives a scratch DisPatch instance once the
session hub is in. Every row prints `platform_profile` and `max_mhz`; a `reprobe` row asks
Smart Turn again at +0.4 s on the real silence, like the session.

## Tests

* `packages/forge-voice-core/tests/rt/` — 101 unit tests with fakes: the policy table incl. `partial=None`, the learned
  EMA, the speculator caps, fail-soft start and `similar()` on the churn cases, the classifier
  incl. stop phrases, the two-backchannels rule and the contiguous-run echo rule, the heard
  map on the markdown golden rows (snake_case, `a < b`, code spans, tags), the tail STT
  commit logic and cap, `reset()` during an in-flight call, the `final()` fast paths and
  timeout, `prime()` (reuse after silence, a fresh call after speech, reset safety), the
  Moonshine push buffer, the pipelined chunk plan, seams, frames, stop, two sessions, the
  swap-safe session change, the voice prefix cache and `prepare()`, cache tags, settings
  validation, and the harness's per-profile verdicts.
* `tests/test_models_rt.py` — 31 model-backed tests (`-m models`, under the box-wide lock):
  partial stability on the clips, the primed tail call answering the probe, the Smart Turn
  table ± 0.1, first audio ≤ 2.0 s with no underrun, the seam test, the 8 kHz WER, LM
  determinism with and without the voice prefix, and the decoder's samples-per-token.
* `tests/latency_harness.py` — the budget above, idle and under load, in both profiles, plus
  the contention gate; run it under `flock "$VOICE_LOCK"` on a shared box.
