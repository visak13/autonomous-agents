"""llm_framework — standalone LLM transport layer for the reactive agent.

RUNTIME = pure phi4-mini via Ollama (http://localhost:11435). ZERO Claude
anywhere in the runtime (d1). Built first, tested live later (d7/d8): the
deterministic FakeTransport lets the whole chain run fully offline with zero
GPU use.

Public surface
--------------
- ``Transport``       : the protocol every transport satisfies
- ``ChatResult``      : a (role, content, raw) reply
- ``OllamaTransport`` : real phi4-mini transport (OpenAI-compat + native API)
- ``FakeTransport``   : scripted, deterministic, offline transport
- ``estimate_tokens`` : dependency-light heuristic token estimator
- ``Context`` / ``Chain`` : the lambda-chainable pipeline core
- built-in stages : ``prompt_assembly``, ``call_stage``, ``structured_output``,
  the ``tool_hook`` / ``memory_injection`` seams, and ``build_default_chain``
"""

from __future__ import annotations

from .transport import (
    ChatResult,
    FakeTransport,
    Message,
    OllamaTransport,
    Transport,
    TransportError,
)
from .tokens import estimate_tokens, estimate_message_tokens
from .chain import Chain, Context, Stage
from .stages import (
    build_default_chain,
    call_stage,
    memory_injection,
    prompt_assembly,
    structured_output,
    tool_hook,
)
from .context import (
    CompactionEvent,
    Conversation,
    HeuristicTokenCounter,
    SUMMARY_HEADER,
    TokenCounter,
    TransportSummarizer,
    deterministic_summary,
)

__all__ = [
    # transport
    "Transport",
    "ChatResult",
    "Message",
    "OllamaTransport",
    "FakeTransport",
    "TransportError",
    # tokens
    "estimate_tokens",
    "estimate_message_tokens",
    # chain core
    "Chain",
    "Context",
    "Stage",
    # stages
    "prompt_assembly",
    "call_stage",
    "structured_output",
    "tool_hook",
    "memory_injection",
    "build_default_chain",
    # context management (a3)
    "Conversation",
    "CompactionEvent",
    "TokenCounter",
    "HeuristicTokenCounter",
    "TransportSummarizer",
    "deterministic_summary",
    "SUMMARY_HEADER",
]
