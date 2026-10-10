"""host/persona.py: the host's inputs to T5's builder — backend placement from
the context, the per-bot follow-up default, tag capability of the target
voice, the extension persona seam, the interruption note."""
from __future__ import annotations

from dispatch_voice.host import persona, registry
from dispatch_voice.host.config import VoiceSettings


def test_placement_follows_the_bots_backend_from_the_context(host_env):
    sage = persona.instructions_for("sage", None, tags_ok=False)          # advisor
    nova = persona.instructions_for("nova", None, tags_ok=False)          # gateway
    swift = persona.instructions_for("swift", None, tags_ok=False)        # api
    assert sage.system.startswith("## Spoken replies") and sage.trailer == ""
    assert swift.system.startswith("## Spoken replies") and swift.trailer == ""
    assert nova.system == "" and nova.trailer.startswith("[voice chat]") and len(nova.trailer) <= 320
    unknown = persona.instructions_for("ghost", None, tags_ok=False)      # unknown bot: both placements
    assert unknown.system and unknown.trailer
    for instr in (sage, nova, swift):
        text = instr.system + instr.trailer
        assert "podcast host" not in text and "driving" not in text and "[Voice mode]" not in text


def test_follow_up_default_is_per_bot_from_voice_yaml_then_by_backend(host_env):
    st = VoiceSettings(bots={"nova": {"follow_up": False}, "sage": {"follow_up": True}})
    assert persona.follow_up_default("sage", VoiceSettings()) is False       # advisor: off
    assert persona.follow_up_default("nova", VoiceSettings()) is True        # gateway: on
    assert persona.follow_up_default("nova", st) is False                    # the file wins
    assert persona.follow_up_default("sage", st) is True
    nova = persona.instructions_for("nova", None, settings=st, tags_ok=False)
    assert "No closing question" in nova.trailer
    sage = persona.instructions_for("sage", None, settings=st, tags_ok=False)
    assert "follow-up question" in sage.system
    # The device's choice overrides the bot's default.
    assert "follow-up question" in persona.instructions_for("nova", None, settings=st, follow_up=True,
                                                            tags_ok=False).trailer.lower() or \
        "No closing question" not in persona.instructions_for("nova", None, settings=st, follow_up=True,
                                                              tags_ok=False).trailer


def test_tags_bullet_only_for_a_tag_capable_voice(host_env, monkeypatch):
    from dispatch_voice.host.engines import chatterbox
    monkeypatch.setattr(chatterbox, "available", lambda: True)
    on = VoiceSettings(expressive_tags=True)
    off = VoiceSettings(expressive_tags=False)
    assert persona.tags_capable("host-voice", on) is True
    assert persona.tags_capable("builtin:af_heart", on) is False            # a built-in voice never
    assert persona.tags_capable("host-voice", off) is False                 # the switch is off
    monkeypatch.setattr(chatterbox, "available", lambda: False)
    assert persona.tags_capable(None, on) is False

    class Tagless:
        supports_tags = False

    class Tagged:
        supports_tags = True

    assert persona.tags_capable("x", on, tts_engine=Tagless()) is False
    assert persona.tags_capable("x", on, tts_engine=Tagged()) is True
    with_tags = persona.instructions_for("swift", "host-voice", settings=on, tags_ok=True)
    without = persona.instructions_for("swift", "host-voice", settings=on, tags_ok=False)
    assert "[laugh]" in with_tags.system and "[laugh]" not in without.system


def test_gateway_trailer_tag_mode_and_extension_persona(host_env):
    st = VoiceSettings(gateway_trailer="tag")
    assert persona.instructions_for("nova", None, settings=st, tags_ok=False).trailer == "[voice chat]"
    registry.set_persona_provider(lambda bot, pid: "Be a pirate." if bot == "swift" else None)
    try:
        instr = persona.instructions_for("swift", None, tags_ok=False, style="short")
        assert instr.system.startswith("Be a pirate.") and "two short sentences" in instr.system
    finally:
        registry.reset()


def test_interruption_note_in_both_placements_and_caps(host_env):
    note = {"heard": "You have two things", "unheard": "this afternoon. A dentist appointment at three."}
    swift = persona.instructions_for("swift", None, tags_ok=False, interrupted=note)
    nova = persona.instructions_for("nova", None, tags_ok=False, interrupted=note)
    assert 'cut off after "You have two things"' in swift.system and "dentist" in swift.system
    assert "dentist" in nova.trailer and len(nova.trailer) <= 560 and len(swift.system) <= 2200
    empty = persona.instructions_for("swift", None, tags_ok=False, interrupted={"heard": "", "unheard": ""})
    assert "previous reply was cut off" not in empty.system


# ---- FR-works 3: the voice that actually speaks decides the tags -------------- #

class _Cloned:
    """The cloned-voice engine's shape: tags on."""
    supports_tags = True
    sample_rate = 24000


class _Builtin:
    """A built-in voice plugin (Kokoro): reads [laugh] aloud as a word."""
    supports_tags = False
    sample_rate = 24000


def _chain():
    from dispatch_voice.host.tts import FallbackTTS
    return FallbackTTS(_Cloned(), _Builtin())


def _with_builtin_default(monkeypatch):
    voices = [{"id": "builtin:af_heart", "name": "Heart", "lang": "en-us", "default": True},
              {"id": "builtin:am_michael", "name": "Michael", "lang": "en-us"}]
    registry.set_builtin_voices_provider(lambda settings=None: voices)
    return voices


def test_default_voice_on_a_kokoro_box_gets_no_tags(host_env, monkeypatch):
    """The live box: Kokoro behind Chatterbox, no cloned default. "No voice" is spoken
    by Kokoro af_heart, so the prompt must not invite [laugh] (it was read aloud)."""
    from dispatch_voice.host import routes
    _with_builtin_default(monkeypatch)
    monkeypatch.setattr(routes, "_dv", lambda: None)
    try:
        st = VoiceSettings(expressive_tags=True)
        assert routes.effective_voice(st, None) == "builtin:af_heart"
        eng = _chain()
        assert persona.tags_capable(None, st, tts_engine=eng) is False
        assert persona.tags_capable("builtin:am_michael", st, tts_engine=eng) is False
        assert persona.tags_capable("host-voice", st, tts_engine=eng) is True      # a clone still may
        instr = persona.instructions_for("swift", None, settings=st, tts_engine=eng)
        assert "[laugh]" not in instr.system
    finally:
        registry.reset()


def test_a_declared_tagless_fallback_does_not_switch_tags_off_for_clones(host_env, monkeypatch):
    """Kokoro now declares supports_tags = False; the chain walk used to pop the
    fallback first, so that flag would have turned tags off for cloned voices too."""
    st = VoiceSettings(expressive_tags=True)
    eng = _chain()
    assert persona.tags_capable("a-cloned-profile", st, tts_engine=eng) is True
    from dispatch_voice.host.tts import speaker
    assert isinstance(speaker(eng, "a-cloned-profile"), _Cloned)
    assert isinstance(speaker(eng, "builtin:af_heart"), _Builtin)


def test_owner_and_persona_flags_reach_the_builder(host_env, monkeypatch):
    """FR-works 12: an advisor's voice section names its owner; an API bot with no
    persona of its own gets the warm conversational manner, one with a persona keeps it."""
    from types import SimpleNamespace

    from dispatch_voice.host import context
    views = {
        "sage": SimpleNamespace(id="sage", name="Sage", backend="advisor", owner="Sam", has_persona=True),
        "plain": SimpleNamespace(id="plain", name="Plain", backend="api", owner="", has_persona=False),
        "swift": SimpleNamespace(id="swift", name="Swift", backend="api", owner="", has_persona=True),
    }
    monkeypatch.setattr(context, "bot", lambda bot_id: views.get(bot_id))
    monkeypatch.setattr(context, "bot_backend", lambda bot_id: views[bot_id].backend if bot_id in views else None)
    sage = persona.instructions_for("sage", None, tags_ok=False)
    assert "Sam is talking by voice" in sage.system and "The user is talking" not in sage.system
    plain = persona.instructions_for("plain", None, tags_ok=False)
    assert "podcast host" in plain.system and "Stay in character" not in plain.system
    swift = persona.instructions_for("swift", None, tags_ok=False)
    assert "Stay in character" in swift.system and "podcast host" not in swift.system
    # An older host (no fields): the bot keeps its own manner, "the user".
    monkeypatch.setattr(context, "bot", lambda bot_id: SimpleNamespace(id="x", name="X", backend="api"))
    old = persona.instructions_for("x", None, tags_ok=False, backend="api")
    assert "The user is talking by voice" in old.system and "Stay in character" in old.system


def test_spoken_rules_never_send_the_listener_to_the_chat(host_env):
    """FR-works 11: "nothing you write is seen on a screen" and "say the details are
    in the chat" contradicted each other; a hands-free listener is never sent to a screen."""
    instr = persona.instructions_for("swift", None, tags_ok=False)
    assert "nothing you write is seen on a screen" in instr.system
    assert "in the chat" not in instr.system
