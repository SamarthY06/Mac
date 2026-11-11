import pytest

from macd import phrases as P


def test_rename_and_voice_reset_are_model_tools_not_regexes() -> None:
    """AG15/AG16: the model decides a rename or voice reset (mac.rename_assistant / mac.reset_voice);
    no phrase list intercepts speech any more ("I'll call you back later" renamed Mac to "Back Later")."""
    for gone in ("rename_request", "_RENAME_RE", "is_voice_reset_request", "VOICE_RESET_PHRASES", "correction_text"):
        assert not hasattr(P, gone)


@pytest.mark.parametrize("t,expected", [("stop everything", True), ("Hey Mac, stop everything please", True),
                                        ("stop all tasks", True), ("okay stop everything now", True),
                                        ("stop", False), ("stop it", False), ("stop everything you said", False)])
def test_is_stop_all(t: str, expected: bool) -> None:
    assert P.is_stop_all(t) is expected


@pytest.mark.parametrize("t,expected", [
    ("That's a good idea.", True), ("It's really pretty.", True), ("I love it", True), ("sounds good to me", True),
    ("oh that's clever", True), ("yeah okay got it", True),
    ("no, make it shorter", False), ("what about Friday", False), ("actually just open the first one", False),
    ("can you repeat that", False), ("that's wrong", False), ("stop", False), ("wait", False),
    ("that's a good idea but use the second one", False),
])
def test_backchannel_appraisals_do_not_interrupt(t: str, expected: bool) -> None:
    assert P.is_backchannel(t) is expected
