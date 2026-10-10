"""Speculator caps (2 per turn, 3 s gap, < 4 words), commit/discard/merge/timeout/stale, stats, similar()."""
import pytest

from dispatch_voice.rt.speculate import Speculator, similar


class Run:
    def __init__(self, text, head_id="m-1"):
        self.text_in = text
        self.head_id = head_id
        self.prompt_chars = 1000 + len(text)
        self.discarded = 0

    async def discard(self):
        self.discarded += 1


class Host:
    def __init__(self, refuse=False):
        self.started: list[str] = []
        self.runs: list[Run] = []
        self.refuse = refuse

    async def start(self, text):
        self.started.append(text)
        if self.refuse:
            return None
        r = Run(text)
        self.runs.append(r)
        return r


@pytest.fixture
def host():
    return Host()


async def test_start_commit_adopts_identical_text(host, clock):
    sp = Speculator(host.start, clock=clock)
    assert await sp.start("what's on my calendar this afternoon")
    run = await sp.commit("What's on my calendar this afternoon?")
    assert run is host.runs[0] and run.discarded == 0
    assert sp.stats["started"] == 1 and sp.stats["adopted"] == 1 and sp.stats["discarded"] == 0
    assert sp.stats["prompt_chars"] == run.prompt_chars
    assert not sp.live


async def test_commit_below_threshold_discards(host, clock):
    sp = Speculator(host.start, clock=clock)
    await sp.start("what's on my calendar this afternoon")
    assert await sp.commit("what's on my calendar for tomorrow") is None
    assert host.runs[0].discarded == 1 and sp.stats["discarded"] == 1 and sp.stats["adopted"] == 0


async def test_caps_two_per_turn_and_three_second_gap(host, clock):
    sp = Speculator(host.start, clock=clock, max_per_turn=2, min_gap_s=3.0)
    assert await sp.start("one two three four five")
    clock.advance(1.0)
    assert not await sp.start("one two three four five six")     # < 3 s since the last start
    clock.advance(2.5)
    assert await sp.start("one two three four five six")          # second run: the first is discarded
    assert host.runs[0].discarded == 1
    clock.advance(5.0)
    assert not await sp.start("one two three four five six seven")   # cap reached
    assert len(host.runs) == 2 and sp.stats["started"] == 2
    sp.new_turn()
    clock.advance(0.1)
    assert await sp.start("a brand new turn with words")             # counters reset
    assert host.runs[1].discarded == 1                              # the orphan was reaped


async def test_fewer_than_four_words_never_starts(host, clock):
    sp = Speculator(host.start, clock=clock)
    assert not await sp.start("stop that now")
    assert host.started == [] and sp.stats["started"] == 0


async def test_host_refusal_is_not_a_start(clock):
    h = Host(refuse=True)
    sp = Speculator(h.start, clock=clock)
    assert not await sp.start("tell me about the garage plan")
    assert sp.stats["started"] == 0 and sp.stats["refused"] == 1 and not sp.live
    clock.advance(3.0)
    assert not await sp.start("tell me about the garage plan again")
    assert len(h.started) == 2


async def test_discard_is_idempotent_and_merge_safe(host, clock):
    sp = Speculator(host.start, clock=clock)
    await sp.start("oh and what about tomorrow morning")
    await sp.discard()
    await sp.discard()
    assert host.runs[0].discarded == 1 and sp.stats["discarded"] == 1
    assert await sp.commit("oh and what about tomorrow morning") is None


async def test_timeout_discards_unread(host, clock):
    sp = Speculator(host.start, clock=clock, timeout_s=20.0)
    await sp.start("what's on my calendar this afternoon")
    clock.advance(20.5)
    assert await sp.commit("what's on my calendar this afternoon") is None
    assert host.runs[0].discarded == 1 and sp.stats["discarded"] == 1


async def test_stale_head_is_counted_separately(host, clock):
    sp = Speculator(host.start, clock=clock)
    await sp.start("what's on my calendar this afternoon")
    assert await sp.commit("what's on my calendar this afternoon", head_id="m-2") is None
    assert sp.stats == {"started": 1, "adopted": 0, "discarded": 0, "stale": 1,
                        "prompt_chars": host.runs[0].prompt_chars, "refused": 0}
    clock.advance(3.0)
    assert await sp.start("what's on my calendar this afternoon")
    assert await sp.commit("what's on my calendar this afternoon", head_id="m-1") is not None


async def test_note_stale_after_dispatch_refuses(host, clock):
    sp = Speculator(host.start, clock=clock)
    await sp.start("what's on my calendar this afternoon")
    await sp.commit("what's on my calendar this afternoon")
    sp.note_stale()
    assert sp.stats["adopted"] == 0 and sp.stats["stale"] == 1


# ---- similar() ---------------------------------------------------------------- #

def test_similar_measured_churn_cases():
    assert similar("what's on my calendar this afternoon", "What's on my calendar this afternoon?") == 1.0
    assert similar("", "") == 1.0 and similar("hello", "") == 0.0
    # The last word changed on a short sentence: below the threshold (a different question).
    assert similar("what's on my calendar this afternoon", "what's on my calendar this morning") < 0.95
    # A mid-sentence article changed on a long one: above it.
    long_a = ("so i was looking at the plan for the garage conversion again last night and i think the "
              "budget we talked about is a little tight once you add the electrical work")
    long_b = long_a.replace("the plan", "a plan")
    assert similar(long_a, long_b) >= 0.95
    # A trailing word appended (the tail still churning) on a long sentence: adopt.
    assert similar(long_a, long_a + " and") >= 0.95
    # Two words appended on a short one: reject.
    assert similar("can you remind me to call", "can you remind me to call the dentist") < 0.95
    assert 0.0 < similar("one two three", "four five six") < 0.1 or similar("one two three", "four five six") == 0.0


# ---- review T4 #8: start() is fail-soft and bounded -------------------------------- #

async def test_start_is_fail_soft_when_the_host_raises_or_hangs(clock):
    import asyncio

    async def boom(text):
        raise RuntimeError("advisor down")

    sp = Speculator(boom, clock=clock)
    assert await sp.start("what's on my calendar this afternoon") is False
    assert sp.stats["refused"] == 1 and sp.stats["started"] == 0 and not sp.live

    async def hang(text):
        await asyncio.sleep(5)

    sp2 = Speculator(hang, clock=clock, start_timeout_s=0.05)
    assert await sp2.start("what's on my calendar this afternoon") is False
    assert sp2.stats["refused"] == 1 and not sp2.live
    assert await sp2.commit("what's on my calendar this afternoon") is None
