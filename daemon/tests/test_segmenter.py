from macd.speech.segmenter import (
    CODE_PLACEHOLDER,
    SentenceSegmenter,
    speakable,
    split_sentences,
)


def run(text: str, step: int | None = None, **kw) -> list[str]:
    seg = SentenceSegmenter(**kw)
    out: list[str] = []
    if step is None:
        out += seg.feed(text)
    else:
        for i in range(0, len(text), step):
            out += seg.feed(text[i : i + step])
    return out + seg.flush()


def test_basic_split_streaming_matches_batch():
    text = "The capital of France is Paris. It has about two million people! Do you want to know more?"
    expected = ["The capital of France is Paris.", "It has about two million people!", "Do you want to know more?"]
    assert run(text) == expected
    for step in (1, 3, 7):
        assert run(text, step) == expected


def test_sentence_emitted_before_flush():
    seg = SentenceSegmenter()
    assert seg.feed("I found three results for you. The ") == ["I found three results for you."]
    assert seg.feed("first is good") == []
    assert seg.flush() == ["The first is good."]


def test_abbreviations_and_numbers():
    s = split_sentences("Dr. Smith met Mr. Jones at 3.5 p.m. on Monday. Version 1.2.3 is out. The U.S. economy grew. See e.g. this one.")
    assert s[0] == "Dr. Smith met Mr. Jones at 3.5 p.m. on Monday."
    assert "Version 1.2.3 is out." in s
    assert "The U.S. economy grew." in s


def test_abbrev_expansion():
    assert speakable("Use a fruit, e.g. an apple.") == "Use a fruit, for example an apple."


def test_merge_short_fragments():
    out = run("Okay. Sure. Here is a much longer sentence that follows.")
    assert out == ["Okay. Sure. Here is a much longer sentence that follows."]
    assert all(out)


def test_markdown_headers_bullets_bold():
    md = "## Results\n\nHere's what I found:\n- **LinkedIn** has 3 profiles\n- *Twitter* has none\n1. First item here\n"
    out = run(md, min_chars=0)
    assert out == ["Results.", "Here's what I found:", "LinkedIn has 3 profiles.", "Twitter has none.", "First item here."]


def test_tables_skipped():
    md = "Summary below.\n| a | b |\n|---|---|\n| 1 | 2 |\nThat is all."
    assert run(md, min_chars=0) == ["Summary below.", "That is all."]


def test_code_fence_placeholder_once():
    md = "Here you go:\n```python\nprint('hi. there')\n```\nAnd another:\n```\nx = 1\n```\nDone now."
    out = run(md, min_chars=0)
    assert out.count(CODE_PLACEHOLDER) == 1
    assert "print" not in " ".join(out)
    assert out[-1] == "Done now."


def test_code_fence_streamed_in_pieces():
    md = "Look:\n``"
    seg = SentenceSegmenter(min_chars=0)
    out = seg.feed(md)
    out += seg.feed("`bash\nrm -rf /tmp/x. Then\n```\nAll set.")
    out += seg.flush()
    assert out == ["Look:", CODE_PLACEHOLDER, "All set."]


def test_inline_code_urls_emoji():
    out = speakable("Run `ls -la` then open https://www.linkedin.com/in/someone?x=1. Great 🎉!")
    assert out == "Run ls -la then open linkedin.com. Great!"
    assert speakable("See [the docs](https://docs.python.org/3/library) now.") == "See the docs now."
    assert speakable("Visit <https://example.com/a/b>") == "Visit example.com."


def test_no_empty_strings():
    assert run("") == []
    assert run("   \n\n---\n🎉\n") == []
    assert run("**") == []


def test_snake_case_preserved():
    assert speakable("Set my_var_name to 3.") == "Set my_var_name to 3."


def test_dashes_become_comma_pauses():
    assert speakable("Paris — the capital – is big - really.") == "Paris, the capital, is big, really."
    assert speakable("Use well-known APIs.") == "Use well-known APIs."  # hyphenated words untouched
