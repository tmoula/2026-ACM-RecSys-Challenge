"""Dynamic few-shot example sampling for GPT response generation."""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional


class FewShotPool:
    def __init__(self, pool_path: str, seed: int = 42) -> None:
        self.pool_path = pool_path
        self.rng = random.Random(seed)
        self.examples: List[dict] = []
        if os.path.exists(pool_path):
            self.examples = json.load(open(pool_path, "r", encoding="utf-8"))

    def sample(self, count: int = 2) -> List[dict]:
        if not self.examples:
            return []
        count = min(count, len(self.examples))
        return self.rng.sample(self.examples, k=count)

    def format_block(self, count: int = 2) -> str:
        samples = self.sample(count)
        if not samples:
            return ""
        lines = ["Here are stylistic examples only — do NOT copy their artists or track titles:"]
        for index, example in enumerate(samples, start=1):
            lines.append(f"Example {index} user request: {example['user_request']}")
            lines.append(f"Example {index} track metadata: {example['track_metadata']}")
            lines.append(f"Example {index} assistant style: {example['assistant_response']}")
        return "\n".join(lines)
