

def test_rename_back_with_current_name_as_wake() -> None:
    from macd.phrases import rename_request
    assert rename_request("Hey Jarvis, change your name to Mac.") == "Mac"
    assert rename_request("Change your name back to Mac.") == "Mac"
