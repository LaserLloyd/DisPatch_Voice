"""Acknowledgements: the pools, intent, the selector's invariants, the timing gate,
tool status lines, voice-acks.yaml and the background render queue."""
from __future__ import annotations

import itertools
import os
import random
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dispatch_voice.text import ack_lines
from dispatch_voice.text.acks import (
    INTENTS,
    POOL_KEYS,
    RENDER_ORDER,
    AckSelector,
    EtaEstimator,
    PoolsFile,
    RenderQueue,
    TurnAcks,
    cache_key,
    default_pools,
    estimate_clip_s,
    intent_of,
    is_word_line,
    line_seed,
    load_pools,
    tool_status_line,
    validate_line,
)

ALL_POOLS = [("default", ack_lines.DEFAULT)] + list(ack_lines.FLAVOUR.items())


def every_line():
    for name, pool in ALL_POOLS:
        for key, lines in pool.items():
            for line in lines:
                yield name, key, line


# ---- the shipped pools ---------------------------------------------------------- #

def test_default_pool_has_at_least_fifty_distinct_lines_over_seven_intents():
    distinct = {x for v in ack_lines.DEFAULT.values() for x in v}
    assert len(distinct) >= 50
    assert set(ack_lines.DEFAULT) == set(POOL_KEYS)
    assert all(len(v) <= 20 for v in ack_lines.DEFAULT.values())


def test_flavour_pools_are_keyed_by_character_and_add_their_own_lines():
    distinct = {x for v in ack_lines.DEFAULT.values() for x in v}
    for name in ("wry", "bright", "trainer", "gruff"):
        pool = ack_lines.FLAVOUR[name]
        assert set(pool) <= set(POOL_KEYS)
        own = {x for v in pool.values() for x in v} - distinct
        assert len(own) >= 4, name


@pytest.mark.parametrize("name,key,line", list(every_line()))
def test_every_shipped_line_is_valid_and_has_no_praise_promise_or_invitation(name, key, line):
    assert validate_line(line) == line
    low = line.lower()
    assert not any(bad in low for bad in ack_lines.FORBIDDEN), (name, key, line)
    words = len(line.split())
    assert words <= (9 if key.startswith("wait_") else 7)


def test_statement_and_emotional_pools_have_no_yeah_or_got_it():
    for _, pool in ALL_POOLS:
        for key in ("statement", "emotional"):
            for line in pool.get(key, ()):
                assert line not in ("Yeah.", "Got it.") and not line.startswith("Yeah")


def test_word_only_voices_get_word_lines_only():
    p = default_pools()
    _, base = p.lines("statement", bot_id="x", voice="builtin:af_heart")
    assert base and all(is_word_line(x) for x in base)
    assert "Mm-hm." not in base and "Hmm." not in p.lines("question", bot_id="x", voice="builtin:af")[1]
    assert "Mm-hm." in p.lines("statement", bot_id="x", voice="profile-1")[1]
    for key in POOL_KEYS:                      # every intent still has words to say
        assert replace(p, bot_flavour={"b": "gruff"}).lines(key, bot_id="b", voice="builtin:af_heart") != ((), ())


# ---- intent ---------------------------------------------------------------------- #

INTENT_TABLE = [
    # 1.0.1 review: bad news and "understood" said with a "!" are not good news
    ("I got it!", "statement"),
    ("Our flight got cancelled!", "statement"),
    ("I got in an accident!", "statement"),
    ("My dad got arrested!", "statement"),
    ("What's on my calendar?", "question"),
    ("How does this work?", "question"),
    ("Why does Rae sound like Nova?", "question"),
    ("Is it going to rain later?", "question"),
    ("Hey Sage, what's the weather like?", "question"),
    ("Okay so what do you think about jazz", "question"),
    ("Remind me to call the school.", "request"),
    ("Tell me about yourself.", "request"),
    ("Can you book the dentist for Tuesday?", "request"),
    ("Navigate to the nearest gas station.", "request"),
    ("Play some jazz.", "request"),
    ("Skip this song.", "request"),
    ("Set a timer for ten minutes.", "request"),
    ("I need you to text Mom that I'm late.", "request"),
    ("Open the garage notes please.", "request"),
    ("Honestly, rough day.", "emotional"),
    ("I'm so tired of this commute.", "emotional"),
    ("Ugh, the traffic is awful.", "emotional"),
    ("I think we're using it now.", "statement"),
    ("My sister is visiting next week.", "statement"),
    ("So, um, I had a great day today!", "statement"),
    ("I wonder if it'll rain.", "statement"),
    ("ok", "none"),
    ("yes please", "none"),
    ("Thanks!", "none"),
    ("", "none"),
    # review-T5 #2: greetings, thanks, goodbyes, affection and small talk get no
    # listening noise ("I see." after "I love you." is the misfit to avoid)
    ("Thanks a lot.", "none"),
    ("Thanks Mom.", "none"),
    ("That's really helpful, thanks.", "none"),
    ("I love you.", "none"),
    ("Good morning Sage", "none"),
    ("See you later.", "none"),
    ("Sounds good to me.", "none"),
    ("That works for me.", "none"),
    ("How are you?", "none"),
    ("How's it going?", "none"),
    ("How was your day?", "none"),
    ("Hello there, how are you doing today?", "none"),
    ("Have a nice day.", "none"),
    ("Good morning, what's on my calendar today?", "question"),
    ("Thanks, can you also remind me to call Mom?", "request"),
    ("Morning Sage, remind me to call the school.", "request"),
    # a punctuated transcript: an auxiliary start is a question only with "?"
    ("Had a rough day.", "emotional"),
    ("Has been a long week.", "emotional"),
    ("Is it going to rain later", "question"),
    ("Have you seen my keys?", "question"),
    ("What a day!", "statement"),
    # "can you …" asking an opinion or a fact is a question, not a request
    ("Can you believe it?", "none"),                     # rhetorical: no line fits it
    # 1.0.1: a long "can you believe …?" is just as rhetorical as a short one
    ("Can you believe what my boss said in the meeting today?", "none"),
    ("Can you believe they cancelled the show?", "none"),
    ("Could you imagine living there?", "none"),
    ("Do you think it'll snow?", "question"),
    ("Will you be here tomorrow?", "question"),
    ("Would you say that's normal?", "question"),
    ("Would you mind grabbing the milk?", "request"),
    ("Can you remember to buy milk?", "request"),
    ("Would you be able to book a table?", "request"),
    # review-T5 #3: a negated or outweighed trigger is not a hard feeling
    ("No pain today, the knee is fine!", "statement"),
    ("I'm not tired at all, I slept great.", "statement"),
    ("I miss you too, but I'm happy.", "statement"),
    ("I finally beat my anxiety, feeling great.", "statement"),
    ("I hate to say it but that was a great game.", "statement"),
    ("Great news, my dog is not sick anymore.", "statement"),
    ("I'm so stressed, but it's fine.", "emotional"),
    ("I miss my dad so much.", "emotional"),
    # 1.0.1: good news about yourself with a "!" is upbeat, never "Noted."
    ("We got the house!", "upbeat"),
    ("I passed my exam!", "upbeat"),
    ("My daughter got into college!", "upbeat"),
    ("I finally finished the report!", "upbeat"),
    ("We got the house.", "statement"),                  # no "!": a plain statement
    ("I didn't pass!", "statement"),                     # negated
    ("I made a mistake!", "statement"),                  # bad news
    ("I got fired!", "emotional"),
    # review-T5 #15: a lower-case name after a greeting
    ("hey nova tell me a joke", "request"),
    ("hey tell me a joke", "request"),
]


@pytest.mark.parametrize("text,intent", INTENT_TABLE)
def test_intent_table(text, intent):
    assert intent_of(text, after_barge_in=False) == intent


def test_after_a_barge_in_the_intent_is_followup_unless_tiny():
    assert intent_of("No wait, what about tomorrow?", after_barge_in=True) == "followup"
    assert intent_of("Stop.", after_barge_in=True) == "none"


def test_a_question_never_gets_a_request_or_statement_line():
    p = default_pools()
    q = set(p.lines("question", bot_id="x", voice="v")[1])
    assert not q & set(ack_lines.DEFAULT["request"]) - {"Okay."}
    assert not any("check" in x.lower() or "look that up" in x.lower() for x in q)


# ---- selector invariants (property tests over many seeds) ------------------------ #

def simulate(seed: int, *, bot="advisor", voice="v", mode="sometimes", turns=1500):
    rng = random.Random(seed)
    sel = AckSelector(default_pools(), mode=mode, rng=random.Random(seed + 1))
    t, last_at, last_acked = 0.0, None, False
    said, rows = [], []
    for _ in range(turns):
        t += rng.choice([1.5, 2.0, 3.5, 6.0, 12.0, 30.0])
        intent = rng.choice(list(INTENTS) + ["none"])
        first = rng.choice([None, "Yeah", "Okay", "Let", "Sure"])
        line = sel.pick(intent, bot_id=bot, voice=voice, now=t, last_turn_at=last_at,
                        last_acked=last_acked, reply_first_word=first)
        rows.append((t, last_at, last_acked, intent, first, line))
        if line:
            said.append(line)
        last_at, last_acked = t, bool(line)
    return sel, said, rows


@pytest.mark.parametrize("seed", range(8))
def test_no_line_repeats_within_the_last_four(seed):
    _, said, _ = simulate(seed)
    assert len(said) > 300
    for i in range(4, len(said)):
        assert said[i] not in said[i - 4:i]


@pytest.mark.parametrize("seed", range(8))
def test_never_two_acks_on_consecutive_fast_turns(seed):
    _, _, rows = simulate(seed)
    for t, last_at, last_acked, _, _, line in rows:
        if line and last_at is not None and t - last_at < 4.0:
            assert not last_acked


@pytest.mark.parametrize("seed", range(4))
def test_reply_first_word_is_never_the_ack_first_word(seed):
    _, _, rows = simulate(seed)
    for *_, first, line in rows:
        if line and first:
            assert line.split()[0].strip(",.!?").lower() != first.lower()


def test_skip_rates_by_mode():
    def rate(mode):
        _, _, rows = simulate(3, mode=mode, turns=3000)
        eligible = [r for r in rows if r[3] != "none" and not (r[1] is not None and r[0] - r[1] < 4 and r[2])]
        return sum(1 for r in eligible if r[5]) / len(eligible)
    assert rate("off") == 0
    assert 0.55 < rate("sometimes") < 0.75          # 35 % skipped
    assert rate("often") > 0.8                      # 10 % skipped


def test_bag_exhaustion_every_line_before_any_repeat():
    sel = AckSelector(default_pools(), mode="often", rng=random.Random(5))
    sel.rng.random = lambda: 0.99                       # never skip
    own, base = sel.pools.lines("request", bot_id="nobody", voice="v")
    n = len(own) + len(base)
    got = [sel.pick("request", bot_id="nobody", voice="v", now=100.0 * i, last_turn_at=None,
                    last_acked=False) for i in range(n)]
    assert sorted(got) == sorted(own + base)            # one full cycle, no repeats


def test_flavour_lines_come_first_in_every_cycle():
    pools = replace(default_pools(), bot_flavour={"nova": "bright"})
    sel = AckSelector(pools, mode="often", rng=random.Random(2))
    sel.rng.random = lambda: 0.99
    own, base = pools.lines("question", bot_id="nova", voice="v")
    assert own == ack_lines.FLAVOUR["bright"]["question"]
    first = [sel.pick("question", bot_id="nova", voice="v", now=100.0 * i, last_turn_at=None,
                      last_acked=False) for i in range(len(own))]
    assert set(first) == set(own)
    nxt = sel.pick("question", bot_id="nova", voice="v", now=1e6, last_turn_at=None, last_acked=False)
    assert nxt in base


def test_flavour_by_alias_and_operator_override_by_bot_id(tmp_path):
    f = tmp_path / "voice-acks.yaml"
    f.write_text("bots:\n  advisor:\n    question: [\"Let me consult the oracle.\"]\n", encoding="utf-8")
    pools = load_pools(f)
    own, _ = pools.lines("question", bot_id="advisor", voice="v")
    assert own == ("Let me consult the oracle.",)
    own, _ = pools.lines("statement", bot_id="advisor", voice="v")   # falls back to its style
    assert own == ack_lines.FLAVOUR["wry"]["statement"]


def test_yaml_flavour_picks_a_style_per_bot_and_names_can_match(tmp_path, caplog):
    f = tmp_path / "voice-acks.yaml"
    f.write_text("bots:\n  nova: {flavour: Bright}\n  rae: {flavour: trainer, wait_5s: [\"Hold on, boss.\"]}\n"
                 "  odd: {flavour: pirate}\n", encoding="utf-8")
    p = load_pools(f)
    assert p.lines("request", bot_id="nova", voice="v")[0] == ack_lines.FLAVOUR["bright"]["request"]
    assert p.lines("wait_5s", bot_id="rae", voice="v")[0] == ("Hold on, boss.",)
    assert p.lines("request", bot_id="rae", voice="v")[0] == ack_lines.FLAVOUR["trainer"]["request"]
    assert p.lines("request", bot_id="odd", voice="v")[0] == ()
    assert "pirate" in caplog.text
    named = default_pools().bind([("b7", "Gruff")])          # a bot named after a style
    assert named.lines("statement", bot_id="b7", voice="v")[0] == ack_lines.FLAVOUR["gruff"]["statement"]


def test_tiers_ignore_the_skip_roll_but_respect_off_and_recent():
    sel = AckSelector(default_pools(), mode="sometimes", rng=random.Random(1))
    sel.rng.random = lambda: 0.0                        # would always skip a normal ack
    a = sel.tier("wait_5s", bot_id="x", voice="v")
    b = sel.tier("wait_5s", bot_id="x", voice="v")
    assert a and b and a != b and a in ack_lines.DEFAULT["wait_5s"]
    sel.set_mode("off")
    assert sel.tier("wait_15s", bot_id="x", voice="v") is None


def test_pick_with_fits_only_returns_fitting_lines():
    sel = AckSelector(default_pools(), mode="often", rng=random.Random(4))
    sel.rng.random = lambda: 0.99
    for i in range(30):
        line = sel.pick("question", bot_id="x", voice="v", now=10.0 * i, last_turn_at=None,
                        last_acked=False, fits=lambda t: len(t.split()) <= 2)
        assert line is None or len(line.split()) <= 2


@pytest.mark.parametrize("pool", [["Right.", "Okay.", "I see."], ["Right.", "Okay."]])
@pytest.mark.parametrize("seed", range(4))
def test_a_small_pool_never_starves_and_never_repeats_back_to_back(tmp_path, pool, seed):
    """review-T5 #6: the no-repeat window shrinks for a small pool instead of
    silencing it for good once every line has been said."""
    f = tmp_path / "voice-acks.yaml"
    f.write_text("default:\n  statement: [" + ", ".join(f'"{x}"' for x in pool) + "]\n", encoding="utf-8")
    sel = AckSelector(load_pools(f), mode="often", rng=random.Random(seed))
    sel.rng.random = lambda: 0.99                       # never skip
    said = [sel.pick("statement", bot_id="x", voice="v", now=100.0 * i, last_turn_at=None,
                     last_acked=False) for i in range(30)]
    assert None not in said
    assert all(a != b for a, b in itertools.pairwise(said))
    assert set(said) == set(pool)


def test_only_three_rendered_lines_still_ack_every_turn():
    sel = AckSelector(default_pools(), mode="often", rng=random.Random(9))
    sel.rng.random = lambda: 0.99
    _, base = sel.pools.lines("statement", bot_id="x", voice="v")
    rendered = set(base[:3])
    said = [sel.pick("statement", bot_id="x", voice="v", now=100.0 * i, last_turn_at=None,
                     last_acked=False, fits=rendered.__contains__) for i in range(20)]
    assert None not in said and set(said) <= rendered
    assert all(a != b for a, b in itertools.pairwise(said))


def test_unknown_intent_or_none_says_nothing():
    sel = AckSelector(default_pools(), mode="often")
    for intent in ("none", "bogus", ""):
        assert sel.pick(intent, bot_id="x", voice="v", now=0, last_turn_at=None, last_acked=False) is None


# ---- tool status lines --------------------------------------------------------- #

@pytest.mark.parametrize("name,line", [
    ("flashcards_list_decks", "Checking your cards"),
    ("find_cards", "Checking your cards"),
    ("brave_web_search", "Searching"),
    ("web_search", "Searching"),
    ("google_calendar_list", "Checking your calendar"),
    ("calendar", "Checking your calendar"),
    ("mail_search_messages", "Checking your email"),
    ("memory_search", "Checking my notes"),
    ("image_generate", "Working on the picture"),
    ("sessions_spawn", "Checking with the team"),
    ("exec", "Working on it"),
    ("vendor__plan_load", "Looking that up"),
    ("something_else", "Looking that up"),
    ("", None), (None, None),
])
def test_tool_status_lines(name, line):
    assert tool_status_line(name) == line


def test_operator_tool_rules_come_first_and_stay_out_of_the_shipped_table(tmp_path):
    """An install's own tool prefixes are voice-acks.yaml rules, never shipped code
    (FR-hardening M1): the rule wins over the generic table, a bad rule is skipped."""
    f = tmp_path / "voice-acks.yaml"
    f.write_text(
        "tools:\n"
        "  - {match: '^decksrv__', say: Checking your cards}\n"
        "  - {match: '^imagesrv__', say: Working on the picture}\n"
        "  - {match: '(', say: Searching}\n"          # not a pattern: skipped
        "  - {match: '^x', say: 'Say 2 things'}\n"     # digits: not a valid line
        "  - just a string\n", encoding="utf-8")
    pools = load_pools(f)
    assert [line for _rx, line in pools.tool_rules] == ["Checking your cards", "Working on the picture"]
    assert tool_status_line("decksrv__calendar_sync", pools.tool_rules) == "Checking your cards"
    assert tool_status_line("decksrv__calendar_sync") == "Checking your calendar"       # no rules: generic
    assert tool_status_line("imagesrv__list", pools.tool_rules) == "Working on the picture"
    assert pools.tool_texts() == ["Checking your cards", "Working on the picture"]
    assert pools.bind([("b", "Bot")]).tool_rules == pools.tool_rules          # bind keeps them


def test_custom_tool_line_is_rendered_and_said(tmp_path):
    f = tmp_path / "voice-acks.yaml"
    f.write_text("tools:\n  - {match: '^decksrv__', say: Looking through your decks}\n", encoding="utf-8")
    pools = load_pools(f)
    assert "Looking through your decks" in pools.all_texts()
    assert tool_status_line("decksrv__find", pools.tool_rules) == "Looking through your decks"


# ---- ETA and the timing gate (fake clock) ---------------------------------------- #

def test_eta_uses_priors_until_two_samples():
    eta = EtaEstimator(first_chunk_s=1.5, ttft_s=1.2)
    assert eta.eta_s == pytest.approx(2.7)
    eta.observe(ttft_s=0.4, first_chunk_s=0.3)
    assert eta.eta_s == pytest.approx(2.7)              # one sample: still the prior
    eta.observe(ttft_s=0.4, first_chunk_s=0.3)
    assert eta.eta_s == pytest.approx(0.7)


def never_skip(seed=0, mode="often"):
    sel = AckSelector(default_pools(), mode=mode, rng=random.Random(seed))
    sel.rng.random = lambda: 0.99
    return sel


def gate(eta=2.5, **kw):
    args = {"bot_id": "x", "voice": "v", "intent": "question", "t0": 100.0, "eta_s": eta}
    args.update(kw)
    return TurnAcks(never_skip(), **args)


def run(g: TurnAcks, until: float, step: float = 0.05, events=None):
    """Poll from t0 to t0+until; ``events`` = {time_offset: callable(g)}."""
    out = []
    events = dict(events or {})
    t = g.t0
    while t <= g.t0 + until + 1e-9:
        for at in [k for k in events if g.t0 + k <= t + 1e-9]:
            events.pop(at)(g)
        r = g.poll(t)
        if r:
            out.append((round(t - g.t0, 2), *r))
        t += step
    return out


def test_reply_queued_at_half_a_second_means_no_ack():
    g = gate()
    assert run(g, 3, events={0.5: lambda g: g.on_reply_queued()}) == []


def test_reply_at_two_seconds_still_gets_the_ack_at_point_six():
    got = run(gate(eta=2.5), 3, events={2.0: lambda g: g.on_reply_audio()})
    assert len(got) == 1 and got[0][0] == pytest.approx(0.6, abs=0.06) and got[0][1] == "ack"


def test_eta_under_the_gate_means_no_ack():
    assert run(gate(eta=1.1), 3) == []


def test_clip_must_end_before_the_eta():
    # eta 1.7: the clip must end by 1.4 s, i.e. last ≤ 0.8 s — one-word lines only
    for seed in range(10):
        g = TurnAcks(never_skip(seed), bot_id="x", voice="v", intent="statement", t0=100.0,
                     eta_s=1.7)
        got = run(g, 1.0)
        assert got and all(len(x[2].split()) == 1 for x in got)
    # eta 1.6 leaves 0.7 s: even a one-word line (≈ 0.75 s) would run into the reply
    assert run(gate(eta=1.6), 1.0) == []
    # measured clip lengths win over the estimate
    assert run(gate(eta=1.6, clip_s=lambda t: 0.5), 1.0)
    assert run(gate(eta=3.0, clip_s=lambda t: 2.5), 1.0) == []


def test_only_rendered_lines_are_picked_and_real_lengths_count():
    rendered = {"Let me see.": 0.9, "Hang on.": 0.8}
    for seed in range(10):
        g = TurnAcks(never_skip(seed), bot_id="x", voice="v", intent="question", t0=0.0, eta_s=2.5,
                     clip_s=rendered.get)
        r = g.poll(0.6)
        assert r is None or r[1] in rendered
    g = TurnAcks(never_skip(), bot_id="x", voice="v", intent="question", t0=0.0, eta_s=2.5,
                 clip_s=lambda t: None)                 # nothing rendered yet: earcon instead
    assert g.poll(0.6) is None


def test_render_queue_reports_clip_durations(tmp_path):
    rq, _ = queue(tmp_path, FakeEngine())
    rq.ensure("v", ["Okay."])
    assert rq.wait_idle(5)
    assert rq.duration_s("v", "Okay.") == pytest.approx(estimate_clip_s("Okay."), abs=0.01)
    assert rq.duration_s("v", "Never rendered.") is None
    rq.stop()


def test_tier_two_at_five_seconds_once_then_tier_three_at_fifteen():
    got = run(gate(), 20)
    kinds = [(round(t), k) for t, k, _ in got]
    assert kinds == [(1, "ack"), (5, "tier"), (15, "tier")]
    assert got[1][2] in ack_lines.DEFAULT["wait_5s"] and got[2][2] in ack_lines.DEFAULT["wait_15s"]


def test_a_tool_line_replaces_the_five_second_tier():
    got = run(gate(), 12, events={3.0: lambda g: g.on_tool("flashcards_find_cards"),
                                  3.5: lambda g: g.on_tool("flashcards_find_cards"),
                                  4.0: lambda g: g.on_tool("brave_web_search"),
                                  4.5: lambda g: g.on_tool("calendar_read")})
    tools = [x for x in got if x[1] == "tool"]
    assert [x[2] for x in tools] == ["Checking your cards", "Searching"]   # distinct, ≤ 2
    assert not any(x[1] == "tier" for x in got)
    assert tools[1][0] - tools[0][0] >= TurnAcks.MIN_GAP_S - 0.06


def test_nothing_while_speaking_and_nothing_after_the_reply_starts():
    g = gate()
    g.on_speaking(True)
    assert run(g, 8) == []                             # ack window passed while speaking
    g2 = gate()
    got = run(g2, 20, events={1.0: lambda g: g.on_reply_audio()})
    assert [k for _, k, _ in got] == ["ack"]


def test_nothing_after_an_error():
    assert run(gate(), 20, events={0.3: lambda g: g.on_error()}) == []


def test_the_ack_avoids_the_speculative_replys_first_word():
    for seed in range(40):
        sel = never_skip(seed)
        g = TurnAcks(sel, bot_id="x", voice="v", intent="followup", t0=0.0, eta_s=3.0)
        g.on_reply_text("Okay, so tomorrow you have three things.")
        r = g.poll(0.6)
        assert r and not r[1].startswith("Okay")


def test_no_ack_on_a_fast_second_turn():
    g = gate(last_turn_at=97.0, last_acked=True)        # previous turn 3 s ago, acked
    assert [k for _, k, _ in run(g, 3)] == []


def test_next_due_schedules_the_ack_and_tiers():
    g = gate()
    assert g.next_due(100.0) == pytest.approx(100.6)
    g.poll(100.6)
    assert g.next_due(100.7) == pytest.approx(105.0)
    g.on_reply_audio()
    assert g.next_due(101.0) is None


# ---- voice-acks.yaml ------------------------------------------------------------- #

def test_load_pools_validates_and_skips_bad_lines(tmp_path, caplog):
    f = tmp_path / "voice-acks.yaml"
    f.write_text(
        "default:\n"
        "  statement: [\"Mm-hm.\", \"Call me at 5\", \"**bold**\", \"🎉 yay\", "
        "\"one two three four five six seven eight nine ten\", \"Right.\", 7]\n"
        "  bogus: [\"Hi.\"]\n"
        "bots:\n"
        "  trainer-bot:\n"
        "    request: [\"On it, boss.\", \"Done, boss.\"]\n"
        "  broken: \"not a mapping\"\n", encoding="utf-8")
    p = load_pools(f)
    assert p.default["statement"] == ("Mm-hm.", "Right.")
    assert p.default["question"] == ack_lines.DEFAULT["question"]      # untouched intent
    assert p.bots["trainer-bot"]["request"] == ("On it, boss.", "Done, boss.")
    assert "broken" not in p.bots
    assert "bogus" in caplog.text and "skipped" in caplog.text


def test_load_pools_caps_twenty_lines_and_survives_garbage(tmp_path):
    f = tmp_path / "voice-acks.yaml"
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
             "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
             "Sierra", "Tango", "Uniform", "Victor"]
    f.write_text("default:\n  question: [" + ", ".join(f'"{w}."' for w in words) + "]\n", encoding="utf-8")
    assert len(load_pools(f).default["question"]) == 20
    f.write_text(": : not yaml : [", encoding="utf-8")
    assert load_pools(f).default == default_pools().default
    assert load_pools(tmp_path / "missing.yaml").default == default_pools().default


def test_pools_file_hot_reloads_on_mtime(tmp_path):
    f = tmp_path / "voice-acks.yaml"
    pf = PoolsFile(f, check_every_s=0)
    assert pf.get().default["wait_5s"] == ack_lines.DEFAULT["wait_5s"]
    f.write_text("default:\n  wait_5s: [\"Hold tight.\"]\n", encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert pf.get().default["wait_5s"] == ("Hold tight.",)


def test_render_lines_are_in_priority_order_flavour_first():
    pools = replace(default_pools(), bot_flavour={"rae": "trainer"})
    lines = pools.render_lines(bot_id="rae", voice="profile-1")
    assert lines[0] in ack_lines.FLAVOUR["trainer"]["statement"]
    order = [next(k for k in RENDER_ORDER if x in pools.default.get(k, ()) +
                  ack_lines.FLAVOUR["trainer"].get(k, ())) for x in lines]
    assert order == sorted(order, key=RENDER_ORDER.index)
    assert len(lines) == len(set(lines))


# ---- render queue ---------------------------------------------------------------- #

class FakeEngine:
    """Renders 24 kHz silence of the expected length; records each call."""

    def __init__(self, *, bad: set[str] = frozenset(), raise_on: set[str] = frozenset(),
                 delay: float = 0.0) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.bad, self.raise_on, self.delay = set(bad), set(raise_on), delay
        self.lock = threading.Lock()

    def __call__(self, voice: str, text: str, seed: int):
        with self.lock:
            self.calls.append((voice, text, seed))
        if self.delay:
            time.sleep(self.delay)
        if text in self.raise_on:
            raise RuntimeError("engine fell over")
        dur = estimate_clip_s(text) * (5.0 if text in self.bad else 1.0)
        return np.zeros(int(24000 * dur), "<i2"), 24000


def queue(tmp_path, engine, **kw):
    root = tmp_path / "cache"
    return RenderQueue(engine, cache_dir=lambda v: root / (v or "default"), **kw), root


def test_render_queue_renders_with_a_fixed_seed_and_caches_to_disk(tmp_path):
    eng = FakeEngine()
    rq, root = queue(tmp_path, eng)
    rq.ensure("cb-abc", ["Mm-hm.", "On it."])
    assert rq.wait_idle(5)
    assert [(t, s) for _, t, s in eng.calls] == [("Mm-hm.", line_seed("Mm-hm.")),
                                                 ("On it.", line_seed("On it."))]
    assert (root / "cb-abc" / cache_key("On it.")).is_file()
    assert rq.status() == {"cb-abc": {"ready": 2, "total": 2, "failed": 0}}
    assert rq.get("cb-abc", "On it.").dtype == np.dtype("<i2")
    rq.stop()
    # a fresh queue (a restart) reads the disk cache and renders nothing
    eng2 = FakeEngine()
    rq2, _ = queue(tmp_path, eng2)
    rq2.ensure("cb-abc", ["Mm-hm.", "On it."])
    assert rq2.wait_idle(5) and eng2.calls == []
    assert rq2.get("cb-abc", "Mm-hm.") is not None
    rq2.stop()


def test_render_queue_waits_while_a_reply_or_session_is_busy(tmp_path):
    eng = FakeEngine()
    busy = threading.Event()
    busy.set()
    rq, _ = queue(tmp_path, eng, busy=busy.is_set)
    rq.POLL_S = 0.01
    rq.ensure("v", ["Let me see."])
    time.sleep(0.2)
    assert eng.calls == [] and rq.get("v", "Let me see.") is None
    busy.clear()
    assert rq.wait_idle(5) and len(eng.calls) == 1
    rq.stop()


def test_one_engine_call_at_a_time(tmp_path):
    active, peak = [0], [0]
    lock = threading.Lock()

    def render(voice, text, seed):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return np.zeros(int(24000 * estimate_clip_s(text)), "<i2"), 24000

    rq, _ = queue(tmp_path, render)
    for v in ("a", "b", "c"):
        rq.ensure(v, ack_lines.DEFAULT["request"][:5])
    assert rq.wait_idle(10) and peak[0] == 1
    rq.stop()


def test_a_runaway_clip_is_re_rendered_once_then_dropped(tmp_path):
    eng = FakeEngine(bad={"Noted."})
    rq, root = queue(tmp_path, eng)
    rq.ensure("v", ["Noted.", "Okay."])
    assert rq.wait_idle(5)
    seeds = [s for _, t, s in eng.calls if t == "Noted."]
    assert seeds == [line_seed("Noted."), line_seed("Noted.") + 1]
    assert rq.get("v", "Noted.") is None and not (root / "v" / cache_key("Noted.")).exists()
    assert rq.status()["v"] == {"ready": 1, "total": 2, "failed": 1}   # "1 of 2 (1 skipped)"
    rq.ensure("v", ["Noted."])                          # a dropped line is not retried
    assert rq.wait_idle(5) and len([c for c in eng.calls if c[1] == "Noted."]) == 2
    rq.stop()


def test_an_engine_error_is_logged_and_the_queue_carries_on(tmp_path, caplog):
    eng = FakeEngine(raise_on={"Okay."})
    rq, _ = queue(tmp_path, eng)
    rq.ensure("v", ["Okay.", "Sure."])
    assert rq.wait_idle(5)
    assert rq.get("v", "Sure.") is not None and rq.get("v", "Okay.") is None
    assert "ack render failed" in caplog.text
    rq.stop()


def test_rebuild_rerenders_the_voice(tmp_path):
    eng = FakeEngine()
    rq, _ = queue(tmp_path, eng)
    rq.ensure("v", ["Okay.", "Sure."])
    assert rq.wait_idle(5) and len(eng.calls) == 2
    rq.rebuild("v")
    assert rq.wait_idle(5) and len(eng.calls) == 4
    assert rq.status()["v"] == {"ready": 2, "total": 2, "failed": 0}
    rq.stop()


def test_prune_deletes_unnamed_files_keeps_named_and_caps_the_tag(tmp_path):
    eng = FakeEngine()
    rq, root = queue(tmp_path, eng, keep=lambda: ["The rest is in the chat."])
    rq.ensure("v", ["Okay."])
    assert rq.wait_idle(5)
    d = root / "v"
    np.save(d / cache_key("An old pool line."), np.zeros(10, "<i2"))
    np.save(d / cache_key("The rest is in the chat."), np.zeros(10, "<i2"))
    assert rq.prune("v") == 1
    assert sorted(p.name for p in d.iterdir()) == sorted(
        [cache_key("Okay."), cache_key("The rest is in the chat.")])
    rq.MAX_FILES_PER_TAG = 1
    keep_all = [f"Line number {chr(65 + i)}." for i in range(3)]
    for i, t in enumerate(keep_all):
        np.save(d / cache_key(t), np.zeros(10, "<i2"))
        os.utime(d / cache_key(t), (1000 + i, 1000 + i))
    rq.prune("v", keep=keep_all + ["Okay.", "The rest is in the chat."])
    assert len(list(d.iterdir())) == 1
    rq.stop()


def test_prune_before_ensure_is_a_no_op_and_spares_fresh_tmp_files(tmp_path, caplog):
    """review-T5 #9: pruning a voice nobody has named yet would wipe its tag;
    another writer's in-flight ``*.tmp.npy`` is left alone, a stale one goes."""
    eng = FakeEngine()
    rq, root = queue(tmp_path, eng)
    d = root / "v"
    d.mkdir(parents=True)
    np.save(d / cache_key("Okay."), np.zeros(10, "<i2"))
    assert rq.prune("v") == 0 and (d / cache_key("Okay.")).is_file()
    assert "ensure() first" in caplog.text
    rq.ensure("v", ["Okay."])
    assert rq.wait_idle(5)
    fresh = d / "abc.4242.tmp.npy"
    stale = d / "def.4242.tmp.npy"
    np.save(fresh, np.zeros(10, "<i2"))
    np.save(stale, np.zeros(10, "<i2"))
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    assert rq.prune("v") == 1
    assert fresh.is_file() and not stale.exists() and (d / cache_key("Okay.")).is_file()
    rq.stop()


def test_a_raising_busy_check_never_kills_the_render_thread(tmp_path, caplog):
    """review-T5 #7: busy() closes over session state; one exception there must
    not leave the queue dead and full."""
    eng = FakeEngine()
    state = {"n": 0}

    def busy():
        state["n"] += 1
        if state["n"] <= 3:
            raise RuntimeError("session went away mid-check")
        return False

    rq, _ = queue(tmp_path, eng, busy=busy)
    rq.POLL_S = 0.01
    rq.ensure("v", ["Okay.", "Sure."])
    assert rq.wait_idle(5)
    assert rq._thread is not None and rq._thread.is_alive()
    assert rq.status()["v"]["ready"] == 2
    assert caplog.text.count("busy() check failed") == 1     # logged once per streak
    rq.stop()


def test_a_bad_cache_directory_marks_the_voice_failed_instead_of_raising(tmp_path, caplog):
    """review-T5 #7: ensure() runs in the session's warm-up; a bad profile tag
    must not raise into it."""
    eng = FakeEngine()

    def cache_dir(voice):
        if voice == "broken":
            raise ValueError("no such profile")
        return tmp_path / "cache" / voice

    rq = RenderQueue(eng, cache_dir=cache_dir)
    rq.ensure("broken", ["Okay.", "Sure."])            # must not raise
    rq.ensure("fine", ["Okay."])
    assert rq.wait_idle(5)
    st = rq.status()
    assert st["broken"] == {"ready": 0, "total": 2, "failed": 2}
    assert st["fine"] == {"ready": 1, "total": 1, "failed": 0}
    assert rq.get("broken", "Okay.") is None and rq.duration_s("broken", "Okay.") is None
    assert rq.prune("broken") == 0
    assert caplog.text.count("cache directory for broken failed") == 1
    assert [c for c in eng.calls if c[0] == "broken"] == []
    rq.stop()


def test_memory_is_an_lru_of_voices_and_clear_drops_it(tmp_path):
    eng = FakeEngine()
    rq = RenderQueue(eng)                               # no disk cache at all
    rq.LRU_VOICES = 2
    for v in ("a", "b", "c"):
        rq.ensure(v, ["Okay."])
    assert rq.wait_idle(5)
    assert rq.get("a", "Okay.") is None and rq.get("c", "Okay.") is not None
    rq.clear()
    assert rq.get("c", "Okay.") is None
    rq.stop()


def test_full_pool_render_through_the_queue(tmp_path):
    """The Voice lab row: every line of a bot's pools for one voice."""
    pools = default_pools()
    lines = pools.render_lines(bot_id="advisor", voice="cb-1234")
    eng = FakeEngine()
    rq, _ = queue(tmp_path, eng)
    rq.ensure("cb-1234", lines)
    assert rq.wait_idle(10)
    st = rq.status()["cb-1234"]
    assert st["ready"] == st["total"] == len(lines) >= 60


def test_no_host_imports_anywhere_in_text():
    pkg = Path(__file__).resolve().parents[2] / "src" / "dispatch_voice" / "text"
    for f in pkg.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert "from ..host" not in src and "dispatch_voice.host" not in src, f
        assert "from app" not in src and "import app" not in src, f


@pytest.mark.parametrize("text", ["Thanks, that helps.", "That helps, thanks.", "That helped a lot.",
                                  "Thanks, that really helps.", "That is helpful."])
def test_thanks_that_helps_is_phatic(text):
    """FR-works 4: "Thanks, that helps." left "that helps" as a statement and got
    "Huh, okay." back; it is social, so no reaction line."""
    assert intent_of(text) == "none"


def test_that_helps_with_content_is_still_a_statement():
    assert intent_of("That helps me decide what to do tomorrow.") == "statement"


# ---- 1.0.1: upbeat pool, trainer's emotional lines ------------------------------- #

def test_the_upbeat_pool_is_surprise_not_a_verdict_and_never_flat():
    lines = ack_lines.DEFAULT["upbeat"]
    assert len(lines) >= 5
    flat = {"Noted.", "Okay.", "Right.", "Mm.", "Mm-hm.", "I see."}
    assert not flat & set(lines)
    for line in lines:
        assert not any(bad in line.lower() for bad in ack_lines.FORBIDDEN)
    # a word-only voice (Kokoro) still has upbeat lines (all of its lines are words)
    _, base = default_pools().lines("upbeat", bot_id="x", voice="builtin:af_heart")
    assert base


def test_upbeat_is_picked_after_good_news_and_does_not_repeat():
    sel = AckSelector(default_pools(), mode="often", rng=random.Random(3))
    got = []
    for i in range(12):
        line = sel.pick(intent_of("We got the house!"), bot_id="x", voice="profile-1",
                        now=100.0 * i, last_turn_at=None, last_acked=False)
        if line:
            assert line in ack_lines.DEFAULT["upbeat"]
            got.append(line)
    assert got
    assert all(a != b for a, b in zip(got, got[1:]))


def test_the_trainer_never_answers_bad_news_with_okay():
    assert "Okay." not in ack_lines.FLAVOUR["trainer"]["emotional"]
    p = replace(default_pools(), bot_flavour={"t": "trainer"})
    own, base = p.lines("emotional", bot_id="t", voice="profile-1")
    assert "Okay." not in own + base
