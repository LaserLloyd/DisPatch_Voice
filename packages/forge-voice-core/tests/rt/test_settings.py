"""RtSettings: C-§F.1 defaults, ranges, validation."""
from forge_voice_core.rt import RtSettings


def test_defaults_match_the_contract():
    s = RtSettings()
    assert s.stt_engine == "parakeet" and s.speculate == "advisor"
    assert s.speculate_max_per_turn == 2 and s.speculate_min_gap_s == 3.0
    assert s.preemptive_tts is False and s.decoder_prompt_s == 4.0
    assert s.first_chunk_tokens == 25 and s.decoder_sessions == 1
    assert s.stt_spinning is False and s.power_hold == "performance"
    assert (s.probe_silence_s, s.hold_min_s, s.hold_max_s, s.addendum_s) == (0.25, 0.3, 1.2, 3.0)
    assert (s.echo_guard_s, s.false_interrupt_timeout_s, s.stt_threads_while_speaking) == (1.5, 2.0, 4)


def test_from_mapping_clamps_and_drops_unknown_keys():
    s = RtSettings.from_mapping({"decoder_prompt_s": 99, "first_chunk_tokens": 3, "decoder_sessions": 7,
                                 "probe_silence_s": "soon", "speculate": "sometimes", "stt_engine": "whisper",
                                 "bogus": 1, "hold_min_s": 0.9, "hold_max_s": 0.6, "stt_spinning": "yes"})
    assert s.decoder_prompt_s == 10.0 and s.first_chunk_tokens == 10 and s.decoder_sessions == 2
    assert s.probe_silence_s == 0.25 and s.speculate == "advisor" and s.stt_engine == "parakeet"
    assert s.hold_max_s == 0.6 and s.hold_min_s == 0.6        # min never above max
    assert s.stt_spinning is False
    assert "bogus" not in s.to_dict()


def test_two_sessions_default_to_the_15_token_first_chunk():
    assert RtSettings.from_mapping({"decoder_sessions": 2}).first_chunk_tokens == 15
    assert RtSettings.from_mapping({"decoder_sessions": 2, "first_chunk_tokens": 20}).first_chunk_tokens == 20
    assert RtSettings.from_mapping({"decoder_sessions": 1}).first_chunk_tokens == 25


def test_bots_block_and_per_bot_defaults():
    s = RtSettings.from_mapping({"bots": {"doxy": {"backchannel": True, "follow_up": "no"}, 7: {}, "x": "bad"}})
    assert s.bots == {"doxy": {"backchannel": True}}
    assert s.bot("doxy") == {"backchannel": True, "follow_up": True}
    assert s.bot("sage", backend="advisor") == {"backchannel": False, "follow_up": False}
    assert s.bot("nova", backend="gateway") == {"backchannel": False, "follow_up": True}
    assert s.bot(None) == {"backchannel": False, "follow_up": True}


def test_hard_silence_is_carried_in():
    assert RtSettings.from_mapping({}, hard_silence_s=2.0).hard_silence_s == 2.0
    assert RtSettings.from_mapping(None, hard_silence_s=9.0).hard_silence_s == 3.0
    assert RtSettings.from_mapping(None).hard_silence_s == 1.4
