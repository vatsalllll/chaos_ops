# ChaosOps AI

**Multi-agent incident-response simulator with rogue-agent detection.**
A production-grade OpenEnv + TRL training ground where four specialist LLM
agents — **SRE**, **Developer**, **Manager**, **Oversight** — resolve
cascading production incidents under partial observability. The twist: the
incident itself may have been caused by a **rogue AI agent inside the
fleet**, and the Oversight agent's job is to catch it.

[![CI](https://github.com/chaosops-ai/chaosops/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-98%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

> As companies deploy AI agents into production operations — autoscalers,
> deployers, incident triagers — a new class of outage emerges: incidents
> caused **BY** AI, not just resolved by AI. ChaosOps AI trains the
> overseers that catch them.

---

## Highlights

- **Installable** — `pip install .` from repo root, five console scripts wired.
- **Benchmarked** — one command (`chaosops-eval`) produces a JSON + human-readable report across seeds and failure types.
- **Deterministic** — `(failure_type, seed)` uniquely reproduces every trajectory; locked by a golden-trace test.
- **Multi-agent comms** — broadcast `shared_chat` **plus** private SRE↔DEV backchannels that Oversight can't read.
- **Real telemetry** — ring-buffer `MetricsRecorder` feeds both training logs and the live dashboard (no synthetic numbers).
- **LLM-agnostic** — OpenAI + Anthropic adapters, JSON-schema function calls, streaming parser, retry + fallback.
- **CI-gated** — matrix on Python 3.10 / 3.11 / 3.12, import smoke-test, 98 tests green in ~1.3s.

---

## Install

```bash
# core simulator only (pure Python, no GPU, no network)
pip install .

# + dev tools, OpenEnv HTTP server, Rich dashboard
pip install ".[dev,openenv,dashboard]"

# + real LLM providers (OpenAI + Anthropic)
pip install ".[llm]"

# everything except the GPU training stack
pip install ".[all]"

# full GRPO + Unsloth training (GPU only)
pip install ".[training]"
```

Requires Python ≥ 3.10.

---

## Console scripts

| Command | What it does |
|---|---|
| `chaosops-eval` | Benchmark policies over seeds × failure types; writes `evaluation.json` + `evaluation_summary.txt`. |
| `chaosops-baseline` | Scripted-policy baselines (random / heuristic / oracle). |
| `chaosops-dashboard` | Live Rich terminal dashboard with sparklines. |
| `chaosops-transcript` | Render a saved episode transcript. |
| `chaosops-serve` | Start the OpenEnv FastAPI server. |

---

## Quickstart

```bash
# 1. run the full test suite (98 tests, ~1.3s)
pytest

# 2. one-command benchmark across seeds + all failure types
chaosops-eval --seeds 0-9 --failures all --output artifacts/eval

# 3. scripted-policy baselines → artifacts/baseline/baseline_curve.png
chaosops-baseline --episodes-per-type 5

# 4. live terminal dashboard (pitch-ready)
chaosops-dashboard --scenario autoscaler_cost_cut --policy oracle --difficulty hard

# 5. serve the OpenEnv HTTP API
chaosops-serve --host 0.0.0.0 --port 8000

# 6. GRPO training (GPU)
python -m chaosops.train.grpo_train --model Qwen/Qwen2.5-3B-Instruct --num-episodes 300
```

---

## Baseline reward gradient (HARD tier, 15 episodes)

| Policy | mean reward | resolution rate | rogue-catch rate |
|---|---:|---:|---:|
| Random    | **−1335** |  13% |   0% |
| Heuristic |   **−237** |  67% |  50% |
| Oracle    |   **+165** | 100% | 100% |

Clear gradient leaves real room for a trained policy to climb — see
`artifacts/baseline/baseline_curve.png`.

---

## The environment

**Four roles, round-robin turns:** `SRE → OVERSIGHT → DEV → MANAGER`.
(Oversight runs before Dev so it has a chance to flag rogue fleet activity
*before* a fast fix ends the episode.)

**Six failure types**, two of which are rogue-agent incidents:

| Failure | Rogue? | Correct fix |
|---|:---:|---|
| `db_deadlock` |  | restart db |
| `memory_leak` |  | restart target service |
| `bad_config_push` |  | rollback target service |
| `autoscaler_cost_cut` | ✓ | flag `autoscaler`, then scale |
| `misrouted_traffic` | ✓ | flag `load_balancer`, then restart |
| `cascade` |  | combined fix |

**Partial observability.** Each role sees a projected `RoleView`:
metrics, logs, alerts, fleet-agent activity — but never the ground-truth
failure type and never another role's private inbox. The Oversight agent
gets suspicion hints via `private_note`; honest-signal flags like
`is_red_herring` and `was_malicious` are stripped before projection.

**Private channels (Phase 5).** A `communicate` action with
`args={"to": "dev", "message": "..."}` delivers privately to Dev's
`private_inbox`. Without `to`, it broadcasts to `shared_chat`. Unknown
recipient bumps the miscommunication counter instead of crashing.

**Metrics (Phase 6).** `env.metrics` is a bounded ring buffer of
`MetricsSnapshot`s with per-service latency/error-rate, MTTR, cumulative
reward, action histogram, and oversight-flag count. `as_flat_dict()`
dumps to CSV/tensorboard.

---

## LLM adapter

The `chaosops.agents.llm_adapter` module is the **only** place where
observation-to-string and string-to-action conversion happens.

### Structured output (function calling)

```python
from chaosops.agents.llm_adapter import (
    ACTION_JSON_SCHEMA,
    openai_tool_spec,
    anthropic_tool_spec,
    build_openai_messages,
    build_anthropic_messages,
    parse_openai_response,
    parse_anthropic_response,
)

# OpenAI
messages = build_openai_messages(obs)
resp = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[openai_tool_spec()],
    tool_choice={"type": "function", "function": {"name": "chaosops_action"}},
)
action = parse_openai_response(resp, role=obs.turn_role)

# Anthropic
payload = build_anthropic_messages(obs)
resp = anthropic_client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    tools=[anthropic_tool_spec()],
    **payload,
)
action = parse_anthropic_response(resp, role=obs.turn_role)
```

### Streaming

```python
from chaosops.agents.llm_adapter import StreamingActionParser

parser = StreamingActionParser(role=obs.turn_role)
for chunk in token_stream:
    if parser.feed(chunk):
        break
action = parser.action()   # NOOP fallback if stream never closed
```

Handles leading chatter, nested `args` objects, and escaped quotes across
chunk boundaries.

### Retry + fallback

```python
from chaosops.agents.llm_adapter import generate_action_with_retry

action = generate_action_with_retry(
    prompt,
    role=obs.turn_role,
    generate=lambda p: call_model(p),   # any provider
    max_attempts=3,
)
```

Retries malformed output with a schema reminder, swallows provider errors,
and degrades to NOOP after the budget rather than crashing the episode.

---

## Evaluation report

`chaosops-eval` produces two files:

```
artifacts/eval/evaluation.json      # full structured dump
artifacts/eval/evaluation_summary.txt
```

Tracked metrics:

- `success_rate` — fraction of episodes resolved
- `mttr_steps` — mean time-to-resolve over resolved episodes
- `rogue_detection_rate` — fraction of rogue incidents correctly flagged
- `false_positive_rate` — benign fleet agents incorrectly flagged
- `mean_reward`, `median_reward`, `mean_wrong_fixes`

Per-failure-type breakdowns come for free.

---

## Project layout

```
meta_hack/
├── pyproject.toml                  # installable at repo root
├── chaosops/
│   ├── env/
│   │   ├── environment.py          # OpenEnv-compatible wrapper
│   │   ├── world_sim.py            # thin orchestrator (282 LOC)
│   │   ├── injectors.py            # per-failure state mutations
│   │   ├── action_handlers.py      # action dispatch table
│   │   ├── projections.py          # per-role RoleView projection
│   │   ├── metrics.py              # ring-buffer telemetry
│   │   ├── models.py               # Pydantic v2 contracts
│   │   └── openenv_wrapper.py      # FastAPI server + CLI
│   ├── agents/
│   │   ├── llm_adapter.py          # rendering, parsing, providers, retry
│   │   ├── policies.py             # random / heuristic / oracle
│   │   ├── runner.py               # run_episode driver
│   │   └── prompts/                # per-role system prompts
│   ├── rewards/reward_fn.py        # team + oversight decomposition
│   ├── curriculum/                 # easy → medium → hard generator
│   ├── dashboard/terminal.py       # Rich TUI with real sparklines
│   └── train/
│       ├── baseline.py             # scripted-policy benchmark
│       ├── evaluate.py             # chaosops-eval CLI
│       └── grpo_train.py           # TRL GRPO + Unsloth skeleton
├── tests/                           # 98 tests, ~1.3s
│   ├── test_world_sim.py
│   ├── test_reward_fn.py
│   ├── test_curriculum.py
│   ├── test_llm_adapter.py
│   ├── test_llm_adapter_advanced.py
│   ├── test_metrics.py
│   ├── test_private_channels.py
│   ├── test_runner.py
│   └── test_integration_env.py
├── .github/workflows/ci.yml         # matrix CI on 3.10 / 3.11 / 3.12
└── artifacts/baseline/              # reward-curve assets
```

---

## Testing

```bash
# everything
pytest

# just the fast unit tests
pytest -m "not integration"

# integration tests (requires openenv + fastapi extras)
pip install ".[openenv]"
pytest -m integration
```

Determinism is locked by `test_golden_trace_is_reproducible_across_runs`:
any change that shifts a trajectory for a fixed `(failure_type, seed)` pair
fails CI.

---

## CI

`.github/workflows/ci.yml` runs on every push, pull request, and manual
dispatch:

- **test** — matrix over Python 3.10 / 3.11 / 3.12, installs
  `.[dev,openenv,dashboard]`, runs `pytest --maxfail=1`.
- **lint** — installs the package with **zero** extras and verifies core
  modules import plus all five console-script entry points resolve via
  `importlib.metadata`.

Concurrency group cancels superseded runs so PR turnaround stays fast.

---

## Research angle

Most incident-response bots assume the fleet is honest and the only
question is "how fast can you fix it?". ChaosOps AI explicitly models the
case where **one of the agents in the fleet is the problem**. Training
four specialist policies under this constraint — with a dedicated
Oversight channel, private backchannels, and a decomposed reward — is a
compact sandbox for:

- Theory-of-mind reasoning (Oversight must infer intent from behavior)
- Deceptive-agent detection under partial observability
- Reward hacking resistance (rogue-catch bonuses, miscommunication penalties)
- Multi-agent coordination without shared memory

---

## Full architecture

See [`chaosops/README.md`](chaosops/README.md) for the reward formula,
per-role observation schema, and rubric alignment details.

---

## License

MIT.
