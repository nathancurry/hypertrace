# Hypertrace

Hypertrace is an auditable, bounded research harness for historical terminology and provenance questions. The model proposes searches and interpretations; fetched source text, exact excerpts, source dates, relationships, and review notes live in SQLite.

## Setup

Python 3.13+ and `uv` are required.

```sh
uv sync --extra dev
uv run hypertrace init
```

`init` creates `hypertrace.db` and seeds the hyperpop/Hyperballad question with competing hypotheses and independent search avenues. For a different question:

```sh
uv run hypertrace research "Where did this term originate?" \
  --hypothesis "The term arose in publication A." \
  --hypothesis "The term arose independently elsewhere."
```

Configure a Brave Search API key and any OpenAI-compatible chat-completions service:

```sh
export BRAVE_SEARCH_API_KEY=...
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_API_KEY=...
export RESEARCH_MODEL=your-model
export ROUTER_MODEL=your-model
export REVIEW_MODEL=your-model
```

The defaults for `ROUTER_MODEL` and `REVIEW_MODEL` are `RESEARCH_MODEL`. `LLM_JSON_MODE=false` disables the `response_format: json_object` request for providers that only support JSON via prompting. When using a cost limit, set `LLM_INPUT_COST_PER_MILLION` and `LLM_OUTPUT_COST_PER_MILLION` in USD for the models used in that run. Provider attempts are logged separately from logical actions. The run summary distinguishes known estimated cost from requests whose spend is unknown; cost-limited runs stop when usage is unreported. Action and time limits are enforced independently.

```sh
uv run hypertrace run --max-actions 30 --max-minutes 20
uv run hypertrace run --question-id 2 --max-actions 100 --max-cost 1.00 --model stronger-model
uv run hypertrace status
uv run hypertrace evidence
uv run hypertrace hypotheses
uv run hypertrace report --output report.md
```

Use `--db PATH` before the subcommand or set `HYPERTRACE_DB`. A run stops at its configured action, elapsed-time, cost, or yield limit. Pending candidate pages and queries remain in the database for later runs. `HYPERTRACE_MIN_YIELD` or `--min-yield` sets the minimum observed evidence per distinct assessed document after three such documents; pending query avenues are still searched.

Each OpenAI-compatible request records sanitized response metadata in `provider_attempts.diagnostics_json`: HTTP status, finish reason, choice and message shape, content state, provider error shape, and top-level keys. Response text, prompts, and API keys are not stored there. Empty, missing, null, or token-limited final content is a provider output failure; reasoning text is never treated as the final answer. A failed adversarial review is recorded in `review_attempts`, stops the run with `review_failed`, and stays due for the next `hypertrace run` on the same database and question. The CheaperInference `glm-5.3-flash` review request uses low reasoning effort and a larger completion budget because reasoning tokens otherwise exhaust the final-answer budget.

## Evidence rules and limits

Search snippets and excerpts from page furniture are discovery leads only. Evidence excerpts must match fetched article-body text exactly and retain surrounding context and page region. Page publication metadata never dates an excerpt; only independently verified quote dates enter the chronology. Reports cite retrieved URLs and group identical text and revisions, with source independence unresolved. Automated runs cannot establish demonstrated transmission or change accepted hypothesis status; model interpretations are provisional. Earlier uses of the string remain separate from claims about the modern genre term's origin.

The v1 backend reads ordinary public HTML and plain text. It does not resolve paywalls, JavaScript-only pages, historical snapshots, book scans, or archive-specific dating. Inaccessible, unsupported, and overlong pages remain explicit gaps in reports. Reports may correctly remain inconclusive. No live research is performed by `init`.
