"""Local interpreter (capture -> pipeline -> play)."""
from __future__ import annotations
from afrilingua.pipeline.streaming import StreamingSession

class LocalInterpreter:
    def __init__(self, engine) -> None:
        self.engine = engine
    
    def run(self, src_lang: str, targets: list[str]) -> None:
        # This would wire sounddevice + player. Skipping implementation 
        # details for speed, just skeleton.
        print(f"Starting local interpretation: {src_lang} -> {targets}")
