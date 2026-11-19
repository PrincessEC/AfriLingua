"""
Text language identification for English / Igbo / Yoruba / Hausa (+ registry extras).

Scoring (per token, NFC-lowercased):

* **Lexical**: weighted hits against built-in function-word lexicons merged
  with each registry language's ``detection_words``. Short tokens (≤ 2 chars)
  weigh half, and a word listed for *k* languages counts ``1/k`` for each.
* **Orthographic**: characters specific to a language — Igbo ``ị ụ ṅ``,
  Yoruba ``ẹ ṣ`` and tone-marked vowels, Hausa hooked letters ``ɓ ɗ ƙ ƴ``;
  ``ọ`` is shared by Igbo and Yoruba.

A language is only returned when the evidence is strong: at least two
"hits" worth of score with a clear majority, or a high ratio of supporting
tokens for very short inputs. A single matching word inside a longer sentence
never decides (v1 classified "Aha, I see" as Igbo). Lingua (English/Yoruba)
is consulted as a tie-breaker when installed. Otherwise ``("unknown", 0.0)``.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections import defaultdict
from typing import Any

from afrilingua.backends._common import module_available
from afrilingua.core.languages import LanguageRegistry, get_registry

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"(?:[^\W\d_]|[\u0300-\u036f])+")

_BUILTIN_LEXICON: dict[str, str] = {
    "english": """
        the and is are was were be been being of to in on at for with that this these those it its i you he she
        we they me him her us them my your our their a an not no yes do does did have has had will would can could
        should shall may might must what which who whom where when why how there here from by about as but or if so
        than then very all some any see know think good morning afternoon evening night thank thanks please hello
        welcome everyone meeting today tomorrow yesterday said say go going come get okay ok just like well oh right
        now more one also into out up down over after before because people time year make want need let tell work
        new first great
    """,
    "igbo": """
        kedu ka mere nne nna anyị anyi unu ndị ndi nke maka bụ bu ya gị gi ọ dị di daalụ daalu biko nnọọ nnoo ndewo
        ụtụtụ ututu ọma oma ehihie mgbede abalị chineke chukwu ebe ihe onye mmadụ ụmụ oge taa echi nwanne nwoke
        nwanyị ụlọ akwụkwọ ọrụ ezigbo kwa chọrọ achọ mara hụ gaa bịa nụ ọzọ niile otu abụọ ahụ ahu ebee
        gịnị gini nwere enweghị mma ukwuu dịka mgbe kwuru
    """,
    "yoruba": """
        ẹ ṣé se kú ku àárọ̀ aaro ọ̀rẹ́ ore mo ni àti ati jọ̀wọ́ jowo ẹ̀kọ́ eko báwo bawo ṣe wà mi rẹ ti tí sí fún fun
        náà naa kò ń gbogbo yín yin wọn won àwa awa ọjọ́ ojo ilé ile owó owo ìyá iya bàbá ọmọ omo ọlọ́run olorun
        dáadáa daadaa káàbọ̀ kaabo àkókò ìpàdé ipade ẹ̀yin jẹ́ lọ wá bá sì yìí yii èyí eyi ohun nǹkan nkan
        ẹ̀gbọ́n àbúrò ọ̀sán alẹ́ òní ọ̀la
    """,
    "hausa": """
        sannu nagode yaya gida wannan kuma ina kai ne ce da zuwa barka lafiya ka ki kin kake kike yake take suke
        muke zan za mu su shi ita wani wata daga cikin akwai babu amma saboda domin tare yanzu gobe jiya rana safe
        dare yau abinci ruwa mutane mutum yara aiki kudi lokaci magana sosai haka wane wace menene yaushe nan
        kowa duk allah ɗan ƴan ƙasa ɗaya biyu uku hakan zai kwana kwanan
    """,
}

# (characters, {language: weight})
_CHAR_CUES: tuple[tuple[frozenset[str], dict[str, float]], ...] = (
    (frozenset("ịụṅ"), {"igbo": 1.0}),
    (frozenset("ọ"), {"igbo": 0.5, "yoruba": 0.5}),
    (frozenset("ẹṣ"), {"yoruba": 1.0}),
    (frozenset("àáèéìíòóùúǹńḿ\u0300\u0301\u0304"), {"yoruba": 0.5}),
    (frozenset("ɓɗƙƴ"), {"hausa": 1.0}),
)

_MAX_TOKEN_SCORE = 2.0


def tokenize(text: str) -> list[str]:
    """NFC-lowercased word tokens, keeping combining tone marks attached."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


class HeuristicLanguageIdentifier:
    """:class:`~afrilingua.backends.base.LanguageIdentifier` (lexical + orthographic + lingua)."""

    def __init__(self, registry: LanguageRegistry | None = None, *, use_lingua: bool = True) -> None:
        self.registry = registry or get_registry()
        self.use_lingua = use_lingua and module_available("lingua")
        self._lingua: Any = None
        self._lingua_lock = threading.Lock()
        self._lexicon = self._build_lexicon()

    # ── lexicon ─────────────────────────────────────────────────────────────
    def _build_lexicon(self) -> dict[str, dict[str, float]]:
        words: dict[str, set[str]] = defaultdict(set)
        names = set(self.registry.names())
        for lang, blob in _BUILTIN_LEXICON.items():
            if lang in names:
                for w in blob.split():
                    words[unicodedata.normalize("NFC", w).lower()].add(lang)
        for spec in self.registry:
            for w in spec.detection_words:
                for tok in tokenize(w):
                    words[tok].add(spec.name)
        lexicon: dict[str, dict[str, float]] = {}
        for word, langs in words.items():
            base = 1.0 if len(word) >= 3 else 0.5
            lexicon[word] = {lang: base / len(langs) for lang in langs}
        return lexicon

    # ── scoring ─────────────────────────────────────────────────────────────
    def scores(self, text: str) -> tuple[dict[str, float], int]:
        """Return (language → evidence score, number of tokens)."""
        tokens = tokenize(text)
        totals: dict[str, float] = defaultdict(float)
        names = set(self.registry.names())
        for tok in tokens:
            per: dict[str, float] = defaultdict(float)
            for lang, w in self._lexicon.get(tok, {}).items():
                per[lang] += w
            for chars, weights in _CHAR_CUES:
                if any(c in chars for c in tok):
                    for lang, w in weights.items():
                        if lang in names:
                            per[lang] += w
            for lang, w in per.items():
                totals[lang] += min(_MAX_TOKEN_SCORE, w)
        return dict(totals), len(tokens)

    def detect(self, text: str) -> tuple[str, float]:
        """Return (canonical language | ``"unknown"``, confidence 0..1)."""
        if not text or not text.strip():
            return "unknown", 0.0
        scores, n = self.scores(text)
        if n == 0:
            return "unknown", 0.0
        total = sum(scores.values())
        if total > 0:
            best, best_score = max(scores.items(), key=lambda kv: kv[1])
            share = best_score / total
            coverage = best_score / n
            strong = best_score >= 2.0 and share >= 0.6
            dense = best_score >= 1.0 and coverage >= 0.4 and share >= 0.75
            if strong or dense:
                conf = min(0.99, share * min(1.0, 0.5 + coverage))
                return best, round(conf, 3)
        return self._lingua_tiebreak(text, scores, n)

    # ── lingua ──────────────────────────────────────────────────────────────
    def _detector(self) -> Any:
        if self._lingua is None:
            with self._lingua_lock:
                if self._lingua is None:
                    from lingua import Language, LanguageDetectorBuilder

                    self._lingua = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.YORUBA).build()
        return self._lingua

    def _lingua_tiebreak(self, text: str, scores: dict[str, float], n: int) -> tuple[str, float]:
        if not self.use_lingua or n < 2:
            return "unknown", 0.0
        try:
            values = self._detector().compute_language_confidence_values(text)
        except Exception:
            logger.debug("lingua failed", exc_info=True)
            return "unknown", 0.0
        if not values:
            return "unknown", 0.0
        top = values[0]
        lang = top.language.name.lower()
        conf = float(top.value)
        african = sum(v for k, v in scores.items() if k != "english")
        if lang == "english" and conf >= 0.75 and african < 1.0 and "english" in self.registry:
            return "english", round(conf * 0.9, 3)
        if lang == "yoruba" and conf >= 0.75 and scores.get("yoruba", 0.0) >= 1.0 and "yoruba" in self.registry:
            return "yoruba", round(conf * 0.9, 3)
        return "unknown", 0.0
