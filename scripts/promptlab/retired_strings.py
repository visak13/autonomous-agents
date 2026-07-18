"""The RETIRED STEERING STRINGS registry — the single source of truth for the
CoT-autonomy refactor's self-policing.

Two lists:

* ``ENFORCED`` — strings that must NEVER appear in any package source (and, via
  ``trace_assert.py``, in any live trace's prompts/observations). A phase moves its
  retired strings here the moment the deletion lands.
* ``PENDING`` — strings inventoried as spoon-feeding but not yet removed; tracked so
  the registry is the complete refactor worklist.

Both ``agent_runtime/tests/test_no_steering_strings.py`` (source grep-gate) and
``scripts/promptlab/trace_assert.py`` (live-trace gate) import THIS file, so there is
exactly one registry.

Entries are plain substrings (case-sensitive) — no regex, so a hit is unambiguous.
"""

# Strings whose presence in package source is a test FAILURE.
ENFORCED: list[str] = [
    # P1 — researcher role framing (scripted loop) deleted
    "Work the canonical loop",
    # P1 — per-turn reply-format nudges collapsed to the neutral unusable-turn note
    "(to search or fetch), OR your",
    "final output as plain prose — nothing else",
    # P1 — reviewer/synthesizer framing how-to bodies retired (mechanics move to
    # bundle doctrine/specs in P4; the role is a drive statement only)
    "Inspect it by BOUNDED REGION",
    "one bounded section per turn via the file_write tool",
    # P2 — get_bundles / note / unloaded-tool ack tails made fact-only
    "Use them now to do your task",
    "Continue: read another source",
    "Search or fetch a source first",
    "to LOAD the one your task needs",
    "Self-select its bundle ",
    # P2 — web_ingest observation tails made fact-only
    "Try again.",
    "Try a different query",
    "Try another query",
    "Try a broader query",
    "Try another source",
    "Choose a different source",
    "Choose a DIFFERENT source",
    "Fetch a DIFFERENT source or write",
    "To read one, reply with ONLY a web_fetch",
    "these are the ONLY URLs you may web_fetch",
    "Note follow-ups or fetch it ",
    "Choose one from the search results",
    # P2 — research bundle per-fetch take-a-note chain retired
    "BEFORE you fetch another or write your findings",
    # P2 — file tool result tails made pure state
    "Continue it (file_write append=true)",
    "resend the SAME call with append=true",
    "re-read the exact on-disk text (whitespace/newlines included) and resend",
    # P3 — first-turn operational scripting deleted
    "To GATHER evidence",
    "then search, read the real sources",
    ") and write it there",
    # P3 — finalize commands became the turn-budget fact
    "Stop searching. Write your FINDINGS",
    "Emit your final output now",
    "reached. Write your FINDINGS",
    # P3 — the bounce-gates (gather-more / note gate / target gate) deleted
    "now and FETCH at least one relevant",
    "Record it now",
    "Load the file tools first",
    "finish only after the write is acknowledged",
    "NOT write the final report or any",
]

# Inventoried spoon-feeding, scheduled for removal in later phases (P4-P6).
PENDING: list[str] = []
