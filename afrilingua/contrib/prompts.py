"""Bundled prompt corpora for the contribution hub.

* ``data/prompts_en.txt``: ~280 short English sentences (``domain<TAB>sentence``)
  across greetings, conference, health, government, education, faith, market and
  numbers/dates/time. Used as sources for ``translate`` prompts.
* ``data/prompts_<iso639-1>.txt``: a *small* list of native seed sentences
  (common greetings with proper diacritics, ``domain=seed``) used for ``record``
  prompts until enough approved community sentences exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["DATA_DIR", "Prompt", "load_prompt_file", "seed_prompts"]

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    lang: str
    domain: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "lang": self.lang, "domain": self.domain}


@lru_cache(maxsize=16)
def load_prompt_file(name: str, lang: str, prefix: str) -> tuple[Prompt, ...]:
    """Parse ``data/<name>``; ids are ``<prefix>-<n>`` (stable line ordinals)."""
    path = DATA_DIR / name
    if not path.exists():
        return ()
    out: list[Prompt] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        domain, _, text = line.partition("\t")
        if not text:
            domain, text = "general", domain
        out.append(Prompt(f"{prefix}-{len(out) + 1:04d}", text.strip(), lang, domain.strip() or "general"))
    return tuple(out)


def seed_prompts(lang: str, iso639_1: str | None) -> tuple[Prompt, ...]:
    """Bundled prompts written *in* ``lang`` (English corpus for English)."""
    if lang == "english":
        return load_prompt_file("prompts_en.txt", "english", "en")
    if not iso639_1:
        return ()
    return load_prompt_file(f"prompts_{iso639_1}.txt", lang, f"seed-{iso639_1}")
