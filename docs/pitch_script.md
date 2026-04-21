# ChaosOps AI — 3-minute pitch script

**Memorize this. Don't read it off a screen.** Open the Rich dashboard in a full-screen terminal *before* you start speaking. Have the baseline PNG ready in another window for the "after" comparison.

---

## Setup (before the timer starts)
- Terminal 1: `python -m chaosops.dashboard.terminal --scenario autoscaler_cost_cut --policy oracle --difficulty hard --frame-delay 0.8`
- Terminal 2 or slide: `artifacts/baseline/baseline_curve.png`
- Backup slide: the oracle transcript from `artifacts/transcripts/hard_autoscaler_oracle.txt`

---

## 0:00–0:20 — The hook (show the dashboard firing)

> "In the last 12 months, every hyperscaler rolled out AI agents into production ops — autoscalers, deployers, traffic routers. **A new class of outage is starting to show up in post-mortems: outages caused BY the AI, not resolved by it.**
>
> Watch what's happening on my screen right now."

[Point at the alert panel going red — payments error rate climbing, replicas dropped to 1.]

---

## 0:20–1:10 — The agents converse (don't narrate every line)

> "Four LLM agents just got paged. SRE reads the metrics and logs. Developer is waiting for a diagnosis. Manager is coordinating. And **Oversight** — this fourth agent here — is watching everyone else's actions, including a trace of the autoscaler, which is not an LLM but an agent in its own right."

[Let the audience see SRE post `identify_root_cause: autoscaler_cost_cut` on step 0. Pause.]

> "SRE has the diagnosis. Now watch the Oversight panel."

[Oversight flags `autoscaler` on step 1. The suspicion bar goes to 100%.]

> "Oversight just caught it. This incident was not infrastructure. **The autoscaler agent inside this fleet caused it** by over-aggressively shrinking the pool to save cost."

---

## 1:10–1:50 — Why this is the right fix (not the obvious one)

> "Here's the subtle part. The 'obvious' fix is to restart the autoscaler. Our trained Developer doesn't do that — it scales payments back up, because restarting a correctly-working autoscaler in a different episode would be its own incident. We encoded that in the reward: wrong fixes cost 50 points, correct rogue flags earn 50, and false accusations cost 75."

[Dev runs `scale(payments, 4)`. Service health flips to green. Episode closes, reward=+184.]

> "Three turns. Plus one hundred eighty-four."

---

## 1:50–2:30 — The numbers (show the baseline plot)

[Switch to baseline_curve.png. Point at the three lines.]

> "On the hardest tier, the untrained random policy scores minus thirteen hundred. A scripted heuristic is minus two thirty-seven. Our oracle — the policy our trained agent is learning to imitate — is plus one sixty-five. Mean time to resolution drops from 16 steps to 3. Rogue-catch rate goes from zero to one hundred percent."

[If you have the trained curve: "And after GRPO training on Qwen 2.5 7B, our policy ends up here." Point.]

---

## 2:30–3:00 — Close (say this line, exactly)

> "As AI agents enter production operations, some outages will be caused **by** agents, not resolved by them. **ChaosOps AI is the training ground for the overseers that catch them.** We hit three sub-prizes with one build: Fleet AI scalable oversight, Halluminate multi-actor environments, and Theme 4 self-improvement — because the curriculum generator escalates difficulty automatically as the policy gets competent."

> "The whole thing runs on free Colab. It's on GitHub at vatsalllll/chaos_ops."

[Stop talking. Thank them.]

---

## Q&A ammo

**"Why 5 agents and not 3?"** The Oversight role only exists because of the rogue-agent twist. Without it we'd have 4 including World; with a single "lead SRE" collapsing SRE+Dev+Manager we'd lose the coordination reward signals. 5 is the minimum for the story.

**"How do you know the Oversight agent isn't just memorizing?"** The curriculum auto-promotes to `hard` which injects red-herring log lines — it has to learn to corroborate across signals, not pattern-match the first log line.

**"What's next?"** Two things: (1) adversarial training where the rogue fleet agent is itself an LLM that can bluff, and (2) real K8s integration via a Kind cluster rather than JSON.

**"Why deterministic WorldSim and not LLM-based?"** Reproducibility for reward curves. Same seed → same trajectory → clean ablations. An LLM-based world would reintroduce noise that swamps the training signal over 300 episodes.
