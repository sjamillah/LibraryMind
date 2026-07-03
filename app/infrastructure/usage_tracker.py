import time
from dataclasses import dataclass
from datetime import datetime, timezone

import tiktoken

PRICING_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
}
DEFAULT_PRICING = {"prompt": 0.001, "completion": 0.002}


@dataclass
class UsageRecord:
    timestamp: float
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class UsageTracker:
    def __init__(self):
        self._records: list[UsageRecord] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def record(self, provider: str, model: str, prompt_text: str, completion_text: str) -> UsageRecord:
        prompt_tokens = self.count_tokens(prompt_text)
        completion_tokens = self.count_tokens(completion_text)
        pricing = PRICING_PER_1K_TOKENS.get(model, DEFAULT_PRICING)
        cost = (prompt_tokens / 1000) * pricing["prompt"] + (completion_tokens / 1000) * pricing["completion"]

        entry = UsageRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
        )
        self._records.append(entry)
        return entry

    def total_cost_today(self) -> float:
        today = datetime.now(timezone.utc).date()
        return round(sum(
            r.cost_usd for r in self._records
            if datetime.fromtimestamp(r.timestamp, tz=timezone.utc).date() == today
        ), 6)

    def total_requests_today(self) -> int:
        today = datetime.now(timezone.utc).date()
        return sum(
            1 for r in self._records
            if datetime.fromtimestamp(r.timestamp, tz=timezone.utc).date() == today
        )


usage_tracker = UsageTracker()
