# ChaosOps AI

**Multi-agent incident-response simulator with rogue-agent detection.** An OpenEnv + TRL training ground where four specialist LLM agents (SRE, Developer, Manager, Oversight) resolve cascading production incidents under partial observability — and where the incident itself may have been caused by a rogue AI agent inside the fleet.

> As companies deploy AI agents into production operations — autoscalers, deployers, incident triagers — a new class of outage emerges: incidents caused **BY** AI, not just resolved by AI. ChaosOps AI trains the overseers that catch them.

---

## Quickstart

```bash
# 1. unit tests (no LLM, no GPU)
python -m pytest tests/

# 2. scripted-policy baselines → writes artifacts/baseline/{baseline.json, baseline_curve.png}
python -m chaosops.train.baseline --episodes-per-type 5

# 3. live Rich dashboard demo (3-minute pitch asset)
python -m chaosops.dashboard.terminal --scenario autoscaler_cost_cut --policy oracle --difficulty hard

# 4. GRPO training (onsite, HF-credit GPU)
python -m chaosops.train.grpo_train --model Qwen/Qwen2.5-3B-Instruct --num-episodes 300
```

## Baseline reward gradient (HARD tier, 15 episodes)

| Policy | mean reward | resolution rate | rogue-catch rate |
|---|---|---|---|
| Random    | **−1335** |  13% |   0% |
| Heuristic |   **−237** |  67% |  50% |
| Oracle    |   **+165** | 100% | 100% |

Clear gradient leaves real room for a trained policy to climb — see `artifacts/baseline/baseline_curve.png`.

## Full documentation

See [`chaosops/README.md`](chaosops/README.md) for architecture, reward formula, rubric alignment, and package layout.

## Layout

```
chaos_ops/
├── chaosops/            # main package
│   ├── env/             # deterministic simulator + OpenEnv wrapper
│   ├── agents/          # role prompts, LLM adapter, scripted policies
│   ├── rewards/         # reward function + team/oversight split
│   ├── curriculum/      # easy → medium → hard generator
│   ├── dashboard/       # Rich terminal demo
│   └── train/           # baseline + GRPO training
├── tests/               # 19 unit tests
└── artifacts/baseline/  # reward-curve PNG + JSON for the pitch
```
