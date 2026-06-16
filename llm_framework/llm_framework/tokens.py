"""Dependency-light heuristic token estimator.

We deliberately AVOID tiktoken: it downloads encoder files on first use and
pulls a non-trivial dependency, which fights the "lean for phi's small
context window" constraint (d10) and the offline-first build order (d7/d8).

This estimator is intentionally cheap and approximate — its job is context
budgeting / compaction triggers, NOT exact billing. It blends a char-based
and a word/punctuation-based estimate, which tracks BPE token counts closely
enough for English prose and code to drive compaction decisions.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

# Words and standalone punctuation runs each tend to map to >=1 BPE token.
_TOKENISH = re.compile(r"\w+|[^\w\s]+")


def estimate_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in ``text``.

    Heuristic: take the max of a ~4-chars-per-token estimate and a
    word/punctuation-piece count, both of which under-shoot on their own for
    different inputs (long words vs. dense punctuation). The max of the two is
    a safe-ish upper-leaning estimate for budgeting.
    """
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4  # ceil(len/4)
    piece_estimate = len(_TOKENISH.findall(text))
    return max(char_estimate, piece_estimate)


def estimate_message_tokens(
    messages: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    per_message_overhead: int = 4,
) -> int:
    """Estimate tokens for a list of chat messages.

    Adds a small fixed per-message overhead to approximate the role/delimiter
    tokens chat templates insert around each turn.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, Mapping) else ""
        total += estimate_tokens(str(content)) + per_message_overhead
    return total
