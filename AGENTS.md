## Engineering style

Prefer boring, flat, readable code.

- Apply YAGNI aggressively. Do not introduce factories, generic wrappers,
  plugin systems, configuration frameworks, extension points, or generalized
  abstractions for hypothetical future requirements.
- Implement the minimum functionality required by the current specification.
- Prefer a small concrete implementation over a generalized framework.
- Prefer duplication over a premature or incorrect abstraction. Extract an
  abstraction when repeated code represents the same stable concept, not merely
  because two blocks look similar.
- Prefer standard-library functionality when it is simple and sufficient.
  Add dependencies when they materially simplify or improve the solution.
- Keep control flow shallow. Prefer early returns and straightforward sequential
  logic over deeply nested branches.
- Keep functions cohesive and reasonably small, but do not split code merely to
  satisfy a line-count rule.
- Use clear names instead of comments that restate the code. Comments should
  explain non-obvious constraints, invariants, tradeoffs, or reasons.
- Do not create architecture for requirements that are explicitly deferred.
- Preserve existing architecture boundaries unless the task requires changing them.
- Before introducing a new abstraction, layer, dependency, or subsystem, consider
  whether a simpler concrete solution solves the current problem more robustly.

## Research-system invariants

Hypertrace is an evidence-preserving research harness. Correctness of the research
record is more important than convenience or model fluency.

- The model is not an authority. Persisted source evidence is the system of record.
- Do not promote inference, similarity, chronology, or model confidence into causal
  or genealogical claims without the required evidence.
- Preserve provenance across retrieval, extraction, interpretation, review, and
  reporting.
- Never treat search snippets, page furniture, retroactive metadata, or inferred
  dates as equivalent to verified contemporary source text.
- Keep page publication dates separate from dates verified for the quoted text.
- Keep observed historical usage separate from claims about origin, lineage, or
  transmission.
- Strong epistemic states must be enforced by code-level rules, not merely requested
  in prompts.
- Provider failures, malformed output, retries, partial writes, and process crashes
  are expected operating conditions. They must not silently corrupt or advance
  research state.
- Durable multi-record state transitions should be atomic when partial completion
  would make resume behavior ambiguous.
- Preserve failed retrievals and unresolved sources as explicit gaps rather than
  silently dropping them.
- Do not weaken provenance or review requirements merely to accommodate a model or
  provider.

## Provider and secret handling

- Never commit API keys, access tokens, credentials, or secret-bearing local config.
- Read secrets from the environment or explicitly designated external secret files.
- Do not log secrets or full request headers.
- Diagnostic logging for model/provider failures should preserve useful response
  metadata without leaking credentials or unnecessarily storing full prompts.
- Treat provider-reported usage as authoritative when available. Clearly distinguish
  reported, locally estimated, and unknown usage.
- Keep application-side limits independent from provider-side spending limits.

## Testing guidance

Prefer a small number of high-value tests for realistic regression risks at stable
boundaries.

Do test:
- durable state transitions and recovery
- protocol parsing and validation
- deduplication and retry behavior
- persistence and crash consistency
- provenance and epistemic promotion boundaries
- authority and permission boundaries
- artifact identity and integrity
- provider failure handling
- externally observable workflow behavior

Do not add tests merely to increase coverage.

Avoid tests that:
- mirror implementation details
- assert constructor signatures or trivial getters/setters
- test every enum value or branch independently
- pin exact log text, UI wording, formatting, or internal call order
- mock large parts of the system only to verify plumbing
- duplicate guarantees already provided by type checking, compilation, or framework
  behavior

Prefer integration tests when the important property crosses process, persistence,
protocol, provider, or filesystem boundaries.

Use compilation, type checking, linting, and manual validation where those provide
better confidence than unit tests.

Before adding a test, ask:
1. What realistic regression would this catch?
2. Is this boundary stable?
3. Would this test still be useful after an internal refactor?

If those answers are weak, do not add the test.

Do not create tests for behavior that is already better validated by an end-to-end
slice acceptance test.

## Scope discipline

Implement the requested slice, not the imagined future product.

When several solutions satisfy the requirements, prefer the one with:
1. fewer concepts,
2. fewer moving parts,
3. fewer dependencies,
4. less hidden behavior,
5. easier failure recovery.

Do not generalize from one implementation unless the current requirements already
contain multiple real cases that require the generalization.

When fixing a live failure, first identify the concrete failure mode and preserve a
regression test for that boundary. Do not use the incident as justification for
unrelated refactoring or architecture expansion.
