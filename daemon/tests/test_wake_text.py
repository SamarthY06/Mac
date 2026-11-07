import pytest

from macd.speech.wake import WakePhrases, strip_wake, verify_wake


@pytest.mark.parametrize(
    "text",
    ["Hey Mac", "hey mac, what time is it?", "Hey, Mac.", "OK Mac open Safari", "Okay, Mack.", "hi mac", "Hey Mak",
     "Heymac what's up", "um hey mac", "Hey Max, what's the weather?"],
)
def test_verify_positive(text):
    assert verify_wake(text)


@pytest.mark.parametrize(
    "text",
    ["big mac", "I ate a big mac yesterday", "macbook pro", "my mac is slow", "my macbook pro is slow", "", "hey",
     "the weather is nice today", "a max of five", "mac and cheese", "hey you", "make it so",
     # person names are not the agent (N6)
     "Hey Mark, can you order pizza?", "hey matt", "Hey Mike, how are you?", "okay mat, let's go", "Hi Mary",
     "hey macbook", "hey machine learning is cool", "hey macro"],
)
def test_verify_negative(text):
    assert not verify_wake(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hey Mac, what is two plus two?", "What is two plus two?"),
        ("hey mac what time is it", "What time is it"),
        ("OK Mac.", ""),
        ("so uh hey mac, open mail", "Open mail"),
        ("What is the time?", "What is the time?"),
        # STT fused/punctuated forms
        ("Hey Macstop!", "Stop!"),
        ("hey mac,what time is it", "What time is it"),
        ("Hey Mac.Stop", "Stop"),
        ("Hey Mac!Stop.", "Stop."),
    ],
)
def test_strip_wake(text, expected):
    assert strip_wake(text) == expected


def test_mentions():
    wp = WakePhrases(["hey mac"])
    assert wp.mentions("I'm Mac, your assistant.")
    assert not wp.mentions("Opening your MacBook settings")


def test_configured_phrases_drive_verification():
    wp = WakePhrases(["hey jarvis", "computer"])
    assert wp.verify("hey jarvis, lights on") and wp.verify("Hey Jervis.")  # fuzzy on the configured name
    assert not wp.verify("hey mac")
    assert not wp.verify("computer, lights on")  # a bare name never verifies
    assert wp.remove("hey jarvis, lights on") == "Lights on"
    assert wp.mentions("Jarvis is here")
    with pytest.raises(ValueError):
        WakePhrases([])
