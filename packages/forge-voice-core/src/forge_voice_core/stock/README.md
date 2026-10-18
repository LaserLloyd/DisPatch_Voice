`synthetic-host.wav`: 12.6 s of fully synthetic speech (Kokoro-82M TTS, voice
`am_michael`, Apache-2.0 weights). It is not a recording of a real person.

It is the reference clip of the **stock voice** (`dispatch_voice.stock`): the
voice that speaks when no cloned profile is chosen and no built-in voice plugin
is installed. The voice tensors are built from it by `dispatch-voice
fetch-models` (cached under the models directory, never committed) or on first
use. The tests use the same file.
