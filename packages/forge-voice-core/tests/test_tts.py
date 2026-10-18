"""Voice chat: what gets spoken, and when a sentence is ready to speak."""
from __future__ import annotations

import asyncio

import numpy as np

from forge_voice_core.tts import SentenceSplitter, speak, speakable, to_pcm16


def stream(text: str, step: int = 3) -> list[str]:
    """Feed `text` a few characters at a time, like a token stream."""
    sp = SentenceSplitter(min_words=1)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += sp.push(text[i:i + step])
    return out + sp.flush()


# ---- speakable (T5's normalise behind the host's name) -------------------- #

def test_plain_text_passes_through():
    assert speakable("Turn left in two hundred metres.") == "Turn left in two hundred metres."


def test_code_block_becomes_a_pointer_to_the_chat():
    s = speakable("Here you go:\n```python\nprint('x')\n```\nDone.")
    assert "print" not in s and "The code is in the chat." in s and s.endswith("Done.")


def test_all_emoji_reply_says_nothing():
    assert speakable("👍🎉") == ""
    assert speakable("") == ""


def test_splitter_returns_source_spans():
    sp = SentenceSplitter(min_words=1)
    spans = sp.push("Sure. I can") + sp.push(" do that! Next") + sp.flush()
    assert [x.text for x in spans] == ["Sure.", "I can do that!", "Next"]
    assert all(sp.source[x.start:x.end] == x.text for x in spans)


# ---- engine plumbing ----------------------------------------------------- #

class FakeEngine:
    sample_rate = 24000
    sample_format = "pcm_s16le"
    channels = 1
    streaming = True

    def __init__(self):
        self.said = []

    def stream(self, text, profile_id=None, stop=None):
        self.said.append((text, profile_id))
        yield np.zeros(120, "<i2")
        yield np.zeros(120, "<i2")


def test_speak_seam_streams_frames_in_the_profiles_voice():
    eng = FakeEngine()

    async def go():
        async def chunks():
            for c in ["Hello there. How", " are you? 👍", "```x```"]:
                yield c
        return [f async for f in speak(chunks(), "host-voice", engine=eng)]

    frames = asyncio.run(go())
    assert [t for t, _ in eng.said] == ["Hello there. How are you?", "The code is in the chat."]
    assert {p for _, p in eng.said} == {"host-voice"}
    assert len(frames) == 4 and all(isinstance(f, bytes) and len(f) == 240 for f in frames)


def test_to_pcm16_clips():
    a = to_pcm16(np.array([2.0, -2.0, 0.0], np.float32))
    assert a.tolist() == [32767, -32767, 0]


def test_a_long_first_sentence_starts_with_its_first_clause():
    """Latency: the driver waits for the FIRST piece, and synthesis time grows
    with its length. A whole reply's long first sentence is cut at a clause."""
    sp = SentenceSplitter()
    out = [x.text for x in sp.push("You have two things this afternoon: a dentist appointment at three, and a "
                                   "call with the school at four thirty. Want me to remind you? ") + sp.flush()]
    assert out[0] == "You have two things this afternoon:"
    assert out[1].startswith("a dentist appointment") and out[1].endswith("thirty.")
    # Streamed, the same cut happens as soon as the clause end is known.
    sp = SentenceSplitter()
    got = []
    for w in ["You", "have", "two", "things", "this", "afternoon:", "a", "dentist", "appointment", "at", "three"]:
        got += [x.text for x in sp.push(w + " ")]
    assert got and got[0] == "You have two things this afternoon:"
    # Short first sentences are untouched; later sentences are never cut early.
    sp = SentenceSplitter()
    assert [x.text for x in sp.push("Sure, it is sunny. ")] == ["Sure, it is sunny."]
