"""
Light text front-end for speech synthesis and document translation.

* :func:`split_sentences` — regex sentence splitter (no NLTK downloads), aware
  of common abbreviations, and capping very long sentences.
* :func:`expand_numbers_en` — digits → English words (simple, offline).
* :func:`filter_to_vocab` — keep only characters a TTS vocabulary knows,
  decomposing unknown precomposed letters (``é`` → ``e`` + U+0301) so tone
  marks survive whichever Unicode form the model was trained on.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "no", "fig",
        "gen", "gov", "hon", "rev", "sen", "rep", "col", "capt", "lt", "sgt", "inc", "ltd", "co", "corp",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept",
    }
)
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|[。！？]\s*|\n{2,}")
_WORD_BEFORE = re.compile(r"([\w.]+)[.!?…]+[\"'”’)\]]*\s*$")


def _is_abbreviation(prefix: str) -> bool:
    m = _WORD_BEFORE.search(prefix)
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    if word in _ABBREVIATIONS:
        return True
    # single initials ("J. R. R. Tolkien") and dotted acronyms ("U.S.")
    return len(word) == 1 and word.isalpha() or bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]", word))


def split_sentences(text: str, *, max_chars: int = 300) -> list[str]:
    """Split ``text`` into sentences; overly long ones are split at , ; : boundaries."""
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        candidate = text[start:end]
        nxt = text[end : end + 1]
        if m.group(0).strip() and m.group(0)[0] in ".!?…" and (_is_abbreviation(candidate) or (nxt and nxt.islower())):
            continue
        if candidate.strip():
            sentences.append(candidate.strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    out: list[str] = []
    for s in sentences:
        out.extend(_cap_length(s, max_chars))
    return out


def _cap_length(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind("; "), window.rfind(": "), window.rfind(", "))
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars - 1
        parts.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        parts.append(rest)
    return parts


# ── numbers (English) ───────────────────────────────────────────────────────

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"), (100, "hundred")]


def number_to_words_en(n: int) -> str:
    """Spell a non-negative integer in English (``1204`` → ``one thousand two hundred four``)."""
    if n < 0:
        return "minus " + number_to_words_en(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    for value, word in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return f"{number_to_words_en(head)} {word}" + (f" {number_to_words_en(rest)}" if rest else "")
    return str(n)  # pragma: no cover


def expand_numbers_en(text: str) -> str:
    """Replace digit groups with English words; decimals read digit-by-digit."""

    def _decimal(m: re.Match[str]) -> str:
        whole, frac = m.group(1).replace(",", ""), m.group(2)
        return f"{number_to_words_en(int(whole))} point " + " ".join(_ONES[int(d)] for d in frac)

    text = re.sub(r"(\d[\d,]*)\.(\d+)", _decimal, text)
    text = re.sub(r"(\d+)%", lambda m: f"{m.group(1)} percent", text)
    return re.sub(
        r"\d{1,3}(?:,\d{3})+|\d+",
        lambda m: number_to_words_en(int(m.group(0).replace(",", ""))) if len(m.group(0)) <= 13 else m.group(0),
        text,
    )


# ── vocabulary filtering ────────────────────────────────────────────────────

def filter_to_vocab(text: str, vocab: Iterable[str]) -> str:
    """Drop characters unknown to a character-level TTS vocabulary.

    Unknown precomposed characters are decomposed (NFD) and their known parts
    kept, so ``ẹ́`` survives as ``ẹ`` + U+0301 when only that form is known.
    Runs of whitespace collapse to a single space.
    """
    known = set(vocab)
    out: list[str] = []
    for ch in text:
        if ch in known:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.extend(c for c in unicodedata.normalize("NFD", ch) if c in known)
    return " ".join("".join(out).split())


def clean_tts_text(text: str, lang: str) -> str:
    """NFC-normalize, expand numbers for English and normalise quotes/dashes."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[–—]", ", ", text)
    if lang == "english":
        text = expand_numbers_en(text.replace("&", " and "))
    return " ".join(text.split())
