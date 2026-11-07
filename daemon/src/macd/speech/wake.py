"""Wake word: sherpa-onnx open-vocabulary KWS (stage 1) + transcript checks (stage 2).

Both stages are driven by ``WakeConfig.phrases`` (e.g. "hey mac", "ok mac"): stage 1 tokenises
them with the KWS model's SentencePiece model, stage 2 (``WakePhrases``) matches them fuzzily
in the verification transcript and strips them from the final transcript.
"""

from __future__ import annotations

import difflib
import logging
import re
import tempfile
from pathlib import Path

import numpy as np

from macd.contracts import SAMPLE_RATE, WakeHit

log = logging.getLogger(__name__)

DEFAULT_PHRASES = ("hey mac", "ok mac")

# Known acoustic confusions. Person names (mark, matt, mike...) are deliberately NOT here:
# "Hey Mark, can you order pizza?" must not wake Mac.
_PREFIX_ALIASES: dict[str, set[str]] = {
    "hey": {"hey", "hi", "hay", "hei", "hej", "he", "a", "eh"},
    "hi": {"hi", "hey", "hay"},
    "ok": {"ok", "okay", "o.k", "kay"},
    "okay": {"ok", "okay", "o.k", "kay"},
}
# TODO(later): weak prefixes still accept the exact name anywhere in a sentence ("I bought a Mac" verifies);
# consider requiring the phrase at/near the start of the verification transcript.
_WEAK_PREFIXES = {"a", "he", "eh"}  # only accept the exact name after these
_NAME_ALIASES: dict[str, set[str]] = {
    "mac": {"mac", "mack", "mak", "mach", "mag", "mc", "macs", "mac's", "max"},
}
_NAME_WEAK = {"max"}  # accepted only after a strong prefix ("a max of five" is not a wake)
# STT sometimes fuses the name with the next word ("Hey Macstop!", "Mac.Stop"); these are real
# compounds that must not be read as "mac" + remainder.
_NAME_COMPOUNDS = {"macbook", "macbooks", "macos", "macintosh", "machine", "machines", "macaroni", "macro", "macros", "macdonald", "macdonalds"}
_WORD = re.compile(r"[a-z]+(?:[.'][a-z]+)*")
_LEADING_PUNCT = re.compile(r"^[\s,.;:!?\-–—]+")


def validate_wake_phrases(phrases: list[str]) -> list[str]:
    """Normalise user-entered wake phrases; ValueError with a human message if unusable."""
    out = []
    for raw in phrases:
        words = raw.lower().split()
        if not words:
            continue
        if not all(w.isascii() and w.isalpha() for w in words):
            raise ValueError(f"'{raw}' must contain only letters (a-z) and spaces")
        if not 2 <= len(words) <= 4:
            raise ValueError(f"'{raw}' must be 2 to 4 words (e.g. \"hey jarvis\")")
        out.append(" ".join(words))
    if not 1 <= len(out) <= 3:
        raise ValueError("choose between 1 and 3 wake phrases")
    return out


def _words(phrase: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(phrase.lower()))


def _fuzzy(word: str, target: str) -> bool:
    if word == target:
        return True
    return len(target) >= 4 and difflib.SequenceMatcher(None, word, target).ratio() >= 0.8


class WakePhrases:
    """Stage-2 matcher for the configured wake phrases (fuzzy, name-safe)."""

    def __init__(self, phrases: list[str] | tuple[str, ...] = DEFAULT_PHRASES) -> None:
        self.phrases = [w for w in (_words(p) for p in phrases) if w]
        if not self.phrases:
            raise ValueError("no wake phrases configured")
        self.names = {p[-1] for p in self.phrases}
        aliases = {a for n in self.names for a in _NAME_ALIASES.get(n, {n})}
        self._mention = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted(aliases)) + r")\b")

    def _match_prefix(self, word: str, target: str) -> bool:
        return word in _PREFIX_ALIASES.get(target, {target}) or _fuzzy(word, target)

    def _match_name(self, word: str, target: str, strong_prefix: bool) -> int:
        """Length of the matched name at the start of ``word`` (0 = no match).

        Shorter than the word means the name was fused with the next word ("macstop").
        """
        if word in _NAME_WEAK and target not in _NAME_WEAK:
            return len(word) if strong_prefix and word in _NAME_ALIASES.get(target, set()) else 0
        if word in _NAME_ALIASES.get(target, {target}) or _fuzzy(word, target):
            return len(word)
        head = re.split(r"[.']", word, 1)[0]
        if head != word and head in _NAME_ALIASES.get(target, {target}):
            return len(head)  # "mac.stop"
        if word.startswith(target) and len(word) >= len(target) + 2 and word not in _NAME_COMPOUNDS:
            return len(target)  # "macstop"
        return 0

    def find(self, text: str) -> tuple[int, int] | None:
        """(start, end) char span of the first wake phrase in ``text``, or None."""
        toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text.lower())]
        for i, (w, s, e) in enumerate(toks):
            for phrase in self.phrases:
                *prefix, name = phrase
                if not prefix:
                    continue  # a bare name is never a wake word in stage 2
                if len(prefix) == 1 and w in {p + name for p in _PREFIX_ALIASES.get(prefix[0], {prefix[0]})}:
                    return s, e  # glued: "heymac"
                if i + len(phrase) > len(toks):
                    continue
                if not all(self._match_prefix(toks[i + k][0], prefix[k]) for k in range(len(prefix))):
                    continue
                strong = toks[i + len(prefix) - 1][0] not in _WEAK_PREFIXES
                nw, ns, _ = toks[i + len(prefix)]
                n = self._match_name(nw, name, strong)
                if n:
                    return s, ns + n
        return None

    def verify(self, text: str) -> bool:
        return self.find(text) is not None

    def remove(self, text: str) -> str:
        """Remove everything up to and including the wake phrase; tidy punctuation/case."""
        span = self.find(text)
        rest = text[span[1] :] if span else text
        rest = _LEADING_PUNCT.sub("", rest).strip()
        if rest and rest[0].islower():
            rest = rest[0].upper() + rest[1:]
        return rest

    def mentions(self, text: str) -> bool:
        """True if spoken text contains the agent's name (could self-trigger the KWS)."""
        return self._mention.search(text.lower()) is not None


_DEFAULT = WakePhrases()


def verify_wake(text: str, phrases: WakePhrases = _DEFAULT) -> bool:
    return phrases.verify(text)


def strip_wake(text: str, phrases: WakePhrases = _DEFAULT) -> str:
    return phrases.remove(text)


# ----------------------------------------------------------------------------- KWS


def keyword_lines(phrases: list[str] | tuple[str, ...], kws_dir: Path, score: float, threshold: float) -> list[str]:
    """sherpa keywords-file lines, e.g. '▁HE Y ▁MA C :1.5 #0.25 @hey_mac'."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(kws_dir / "bpe.model"))
    vocab = {ln.split()[0] for ln in (kws_dir / "tokens.txt").read_text().splitlines() if ln.strip()}
    lines = []
    for phrase in phrases:
        upper = " ".join(phrase.upper().split())
        if not upper:
            continue
        toks = sp.encode(upper, out_type=str)
        bad = [t for t in toks if t not in vocab]
        if bad:
            raise ValueError(f"wake phrase {phrase!r} has tokens not in the KWS vocabulary: {bad}")
        lines.append(" ".join(toks) + f" :{score} #{threshold} @{upper.lower().replace(' ', '_')}")
    if not lines:
        raise ValueError("no wake phrases configured")
    return lines


def _pick(dir_: Path, stem: str) -> str:
    """Prefer the int8 variant of an exported onnx file (smaller, faster)."""
    cands = sorted(dir_.glob(f"{stem}*.int8.onnx")) or sorted(dir_.glob(f"{stem}*.onnx"))
    if not cands:
        raise FileNotFoundError(f"no {stem}*.onnx in {dir_}")
    return str(cands[0])


class KeywordSpotterModel:
    """Shared KeywordSpotter (thread-safe in sherpa); create one stream per detector."""

    def __init__(self, kws_dir: Path, phrases: list[str], score: float, threshold: float, num_threads: int = 1) -> None:
        import sherpa_onnx

        self._tmp = tempfile.TemporaryDirectory(prefix="macd-kws-")
        kw_file = Path(self._tmp.name) / "keywords.txt"
        kw_file.write_text("\n".join(keyword_lines(phrases, kws_dir, score, threshold)) + "\n")
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(kws_dir / "tokens.txt"),
            encoder=_pick(kws_dir, "encoder"),
            decoder=_pick(kws_dir, "decoder"),
            joiner=_pick(kws_dir, "joiner"),
            keywords_file=str(kw_file),
            num_threads=num_threads,
            keywords_score=score,
            keywords_threshold=threshold,
            max_active_paths=4,
            num_trailing_blanks=1,
            provider="cpu",
        )
        # Warm-up: the first decode of a fresh session is ~100 ms (ORT lazy init).
        stream = self.spotter.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)


class SherpaWakeDetector:
    """WakeDetector over one KeywordSpotter stream (one instance per audio source)."""

    def __init__(self, model: KeywordSpotterModel) -> None:
        self._spotter = model.spotter
        self._stream = self._spotter.create_stream()

    def accept(self, frame: np.ndarray, sample_index: int) -> WakeHit | None:
        """Feed one frame starting at absolute ``sample_index``. Returns an unverified hit."""
        x = np.asarray(frame, dtype=np.float32)
        self._stream.accept_waveform(SAMPLE_RATE, x)
        hit: WakeHit | None = None
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            kw = self._spotter.get_result(self._stream)
            if kw:
                hit = WakeHit(phrase=kw.replace("_", " ").strip().lower(), sample_index=sample_index + x.shape[0])
                self._spotter.reset_stream(self._stream)  # required after a hit
        return hit

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()
