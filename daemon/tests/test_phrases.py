from macd import phrases as P


def test_rename_and_voice_reset_are_model_tools_not_regexes() -> None:
    """AG15/AG16: the model decides a rename or voice reset (mac.rename_assistant / mac.reset_voice);
    no phrase list intercepts speech any more ("I'll call you back later" renamed Mac to "Back Later")."""
    for gone in ("rename_request", "_RENAME_RE", "is_voice_reset_request", "VOICE_RESET_PHRASES", "correction_text"):
        assert not hasattr(P, gone)
