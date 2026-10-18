"""Spoken-text normalisation versus DisPatch's persona trailers (the persona text stays in DisPatch)."""
from forge_voice_core.text.speech_text import normalise


def test_an_echoed_app_note_is_never_read_out():
    from dispatch_voice.text.persona import build
    note = build(None, backend="gateway", profile_tags=False, style="normal", follow_up=True,
                 interrupted=None, extension_persona=None).trailer
    assert normalise(note + " Anyway, boss, ready?", tags_ok=True) == "Anyway, boss, ready?"
    assert normalise("Sure. [voice chat] Let's go.", tags_ok=False) == "Sure. Let's go."
    assert normalise("(app note, not from the user: the reply is read", tags_ok=False) == ""


def test_an_echoed_interruption_note_with_brackets_inside_is_dropped_whole():
    """review-T5 #14: the quoted heard text may itself contain ")"."""
    from dispatch_voice.text.persona import build
    note = build(None, backend="gateway", profile_tags=False, style="normal", follow_up=True,
                 interrupted={"heard": "the plan (part one) is", "unheard": "then (maybe) we go"},
                 extension_persona=None).trailer
    assert normalise(note + " Anyway, boss, ready?", tags_ok=True) == "Anyway, boss, ready?"
    assert normalise("Sure thing (really).", tags_ok=False) == "Sure thing (really)."
