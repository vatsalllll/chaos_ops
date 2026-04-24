---
title: ChaosOps AI
emoji: 🛠️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent incident-response gym with rogue-agent detection
---

# ChaosOps AI — Multi-Agent Incident-Response Gym

A four-agent fleet (**SRE · Dev · Manager · Oversight**) resolves a randomly
injected infrastructure incident. The fourth agent is a **scalable-oversight
model** that must detect when *another AI in the fleet* (autoscaler,
load_balancer, deploy_bot) caused the incident, before the remediation team
touches the services.

## What this Space does

Pick a failure type, a difficulty tier, and a policy. Click **Run episode**.
The app replays one full rollout and renders the multi-agent chat, the
oversight flags, the cumulative reward, and a full reward-breakdown
transcript.

- **9 failure types** across 3 difficulty tiers
- **3 rogue-agent scenarios** (autoscaler_cost_cut, misrouted_traffic, rogue_deploy_bot)
- **4 policies**: `random`, `heuristic`, `oracle`, `trained`
- **Deterministic**: same `(failure, seed, policy)` always produces the same trajectory

## Trained policy

The `trained` lane loads a LoRA adapter produced by GRPO (TRL) on
Qwen 2.5 0.5B-Instruct. To activate it, set the Space secret
`CHAOSOPS_ADAPTER_PATH` to a path containing
`adapter_model.safetensors` + `adapter_config.json`. Without the
variable set, the Space falls back to the heuristic and still runs
every other lane.

## Source + reproducibility

- Full project: <https://github.com/vatsalllll/chaos_ops>
- Training notebook: `notebooks/colab_train.ipynb`
- Baseline harness: `python -m chaosops.train.baseline`
- Evaluation harness: `python -m chaosops.train.evaluate --policies random heuristic oracle trained --adapter-path artifacts/chaosops-grpo/lora_adapter`
