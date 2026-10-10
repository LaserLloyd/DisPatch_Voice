"""BackchannelGate timing with a fake clock; classify_interruption incl. stop phrases and the two-backchannels rule."""
from dispatch_voice.rt.backchannel import BackchannelGate, classify_interruption

SAID = ["You have a dentist visit at three.", "The package shipped on the 14th and arrives Thursday."]


def test_gate_timing():
    g = BackchannelGate()
    assert not g.ok(turn_s=3.9, pause_s=0.6, now=10.0)
    assert not g.ok(turn_s=4.0, pause_s=0.49, now=10.0)
    assert g.ok(turn_s=4.0, pause_s=0.5, now=10.0)
    g.sent(10.0)
    assert not g.ok(turn_s=9.0, pause_s=0.8, now=15.9)
    assert g.ok(turn_s=9.0, pause_s=0.8, now=16.0)
    g.sent(16.0)
    g.sent(22.0)
    assert g.count == 3
    assert not g.ok(turn_s=30.0, pause_s=1.0, now=40.0)      # ≤ 3 per turn
    g.new_turn()
    assert g.ok(turn_s=30.0, pause_s=1.0, now=40.0)


def cls(words, dur=1.5, prev=None, now=100.0, said=SAID):
    return classify_interruption(words, dur_s=dur, recent_said=said, prev_verdict=prev, now=now)


def test_echo_needs_80_percent_overlap_digits_ignored():
    assert cls("dentist visit at three") == "echo"
    assert cls("the package shipped on the 14th") == "echo"
    assert cls("the package shipped on the 99th") == "echo"       # digits ignored
    assert cls("the package I ordered yesterday is late") == "real"


def test_stop_words_and_phrases():
    for s in ("stop", "Wait.", "No!", "hold on", "hang on", "pause", "skip", "what", "sorry",
              "okay okay", "got it", "thanks", "thank you", "enough", "that's enough", "never mind",
              "quiet", "shut up", "got it, thanks", "no no", "uh, stop", "okay okay okay"):
        assert cls(s, dur=0.6) == "stop", s


def test_a_single_command_word_is_never_an_echo():
    assert cls("stop", said=["Stop by the store at three."]) == "stop"
    assert cls("no", dur=0.4, said=["No, the dentist is at three."]) == "stop"


def test_plain_okay_once_is_a_backchannel_twice_is_stop():
    assert cls("okay", dur=0.5) == "backchannel"
    assert cls("yeah", dur=0.5) == "backchannel"
    assert cls("Odd High.", dur=0.87) == "backchannel"     # Parakeet's "uh-huh"
    assert cls("mm hm", dur=0.6) == "backchannel"
    assert cls("okay", dur=0.5, prev=("backchannel", 98.0), now=100.0) == "stop"
    assert cls("okay", dur=0.5, prev=("backchannel", 96.5), now=100.0) == "backchannel"   # > 3 s apart
    assert cls("okay", dur=0.5, prev=("echo", 99.0), now=100.0) == "backchannel"


def test_backchannel_limits():
    assert cls("okay", dur=1.0) == "real"                 # not under 1.0 s
    assert cls("okay right sure", dur=0.6) == "real"       # > 2 tokens
    assert cls("", dur=0.5) == "backchannel"              # nothing recognisable, short
    assert cls("", dur=1.5) == "real"


def test_content_after_a_command_is_a_real_interruption():
    assert cls("Wait, stop, that's not what I meant.", dur=2.28) == "real"
    assert cls("Actually, wait, stop. Just tell me the first one.", dur=3.72) == "real"
    assert cls("no, I meant the other one", dur=1.8) == "real"


# ---- review T4 #5 / Amendment T4-3: echo = a contiguous run of ONE recent sentence ---- #

REPLY = [("So I was looking at the plan for the garage conversion again last night and I think the budget we "
          "talked about is a little tight once you add the work."),
         "Pushing the start back a month gives you room to get a second quote and keep the kitchen on track."]


def test_echo_is_a_contiguous_run_of_one_sentence():
    assert cls("get a second quote", dur=0.9, said=REPLY) == "echo"            # bleed: one run, one sentence
    assert cls("pushing the start back a month gives you room", dur=2.0, said=REPLY) == "echo"
    assert cls("uh, pushing the start back a month gives you room", dur=2.1, said=REPLY) == "echo"   # filler
    assert cls("pushing the start back a week gives you room", dur=2.0, said=REPLY) == "echo"    # one substitution
    assert cls("the kitchen is a little tight", dur=1.1, said=REPLY) == "real"  # words of two sentences
    assert cls("I was looking at the budget", dur=1.2, said=REPLY) == "real"    # a new last word
    assert cls("I think we should push the start back", dur=1.4, said=REPLY) == "real"
    assert cls("pushing the start back a month gives you time", dur=2.0, said=REPLY) == "real"
    assert cls("so pushing the start back", dur=1.2, said=REPLY) == "real"      # a new first word
    assert cls("at three", dur=0.5) == "backchannel"                            # ≤ 2 tokens: never an echo
    assert cls("at three", dur=1.5) == "real"
    assert cls("the package shipped on the 14th", said=SAID) == "echo"
    assert cls("the package shipped on the 99th", said=SAID) == "echo"           # ordinals ignored too
    assert cls("you have a dentist visit at 3", said=SAID) == "echo"


def test_ack_runs_with_stop_phrases_up_to_eight_tokens_are_a_stop():
    """FR-works 7 / experience M4: "okay okay, got it, thanks" over a long answer
    means "you can stop" — it used to start a new turn (five tokens > the old four)."""
    for words in ("Okay, okay, got it, thanks.", "Okay, okay, got it.", "okay got it thanks", "great great",
                  "got it thanks so much", "perfect, thanks", "yeah no, that's enough",
                  "alright that's enough, thank you", "no no no, that's enough"):
        assert cls(words, dur=1.6) == "stop", words
    # One acknowledgement word is still a backchannel, and content still makes a turn.
    assert cls("okay", dur=0.5) == "backchannel"
    assert cls("great", dur=0.5) == "backchannel"
    for words in ("okay thanks, and what about tomorrow", "got it, now tell me about the weather",
                  "thanks, can you also check my email"):
        assert cls(words, dur=1.6) == "real", words
    # Never longer than eight tokens.
    assert cls("okay okay okay okay okay okay okay okay okay", dur=2.0) == "real"
