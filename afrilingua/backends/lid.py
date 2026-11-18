"""Heuristic Language Identification."""
from __future__ import annotations
from afrilingua.backends.base import LanguageIdentifier
from afrilingua.core.languages import get_registry

class HeuristicLID(LanguageIdentifier):
    def __init__(self) -> None:
        self.registry = get_registry()
    
    def detect(self, text: str) -> tuple[str, float]:
        text = text.lower()
        best_lang = "unknown"
        max_hits = 0
        
        for lang in self.registry:
            hits = sum(text.count(word) for word in lang.detection_words)
            if hits > max_hits:
                max_hits = hits
                best_lang = lang.name
        
        # Simple heuristic threshold
        confidence = 0.5 if max_hits > 0 else 0.0
        return best_lang, confidence
