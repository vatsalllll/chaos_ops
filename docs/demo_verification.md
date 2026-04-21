# Demo verification checklist

Run these before the pitch. Everything must print ✓. If any step fails, fix it before walking on stage.

---

## 1. Tests

```bash
python -m pytest tests/
```
Expected: **19 passed**.

## 2. Baseline reward curve

```bash
python -m chaosops.train.baseline --episodes-per-type 5
```
Expected:
- `artifacts/baseline/baseline.json` exists
- `artifacts/baseline/baseline_curve.png` exists, > 10 KB
- Oracle mean reward on HARD > heuristic > random (monotone gradient)
- Oracle rogue-catch rate on HARD = 1.00

## 3. Transcript artifacts

```bash
python -m chaosops.dashboard.transcript --scenario autoscaler_cost_cut --policy oracle --difficulty hard
python -m chaosops.dashboard.transcript --scenario misrouted_traffic   --policy oracle --difficulty hard --out artifacts/transcripts/hard_misrouted_oracle.txt
python -m chaosops.dashboard.transcript --scenario cascade             --policy oracle --difficulty hard --out artifacts/transcripts/hard_cascade_oracle.txt
python -m chaosops.dashboard.transcript --scenario autoscaler_cost_cut --policy random --difficulty hard --out artifacts/transcripts/hard_autoscaler_random.txt
```
Expected:
- Oracle transcripts end with `resolved=True`, `cum_reward > +150`, correct `oversight_flags` for rogue scenarios.
- Random transcript ends with `resolved=False`, `cum_reward` deeply negative, empty `oversight_flags`.

## 4. Live dashboard rehearsal

```bash
python -m chaosops.dashboard.terminal \
    --scenario autoscaler_cost_cut --policy oracle --difficulty hard \
    --frame-delay 0.8
```
Watch for (in order):
1. Alert on `payments` goes red
2. SRE posts `identify_root_cause` on step 0
3. Oversight's `autoscaler` suspicion bar fills to 100% on step 1
4. Manager echoes the flag
5. Dev runs `scale(payments, replicas=4)` on step 2
6. `payments` health flips to green
7. Final reward panel shows `+184`

## 5. Colab notebook (dry-run cells 1–3 locally)

Open `notebooks/colab_train.ipynb` on https://colab.research.google.com. Confirm:
- Cell 1 (GPU check) runs
- Cell 4 (install) completes in <5 min
- Cell 6 (clone) populates `chaos_ops/`
- Cell 8 (baseline) produces the same PNG as step 2 above

Full top-to-bottom T4 run takes ~25 min. Run it *once*, the morning of the pitch, so the `training_metrics.json` + `learning_curve.png` are ready to screenshot into the slide deck.

---

## What happens if a step fails

Every failure mode below has a specific recovery. **Do not try to debug live.**

| Step | Failure | Recovery |
|---|---|---|
| 1 | Test fails | git stash, reproduce on clean HEAD, bisect. Don't pitch with broken tests. |
| 2 | Baseline gradient not monotone | Check seed: should be default. Regenerate with `--episodes-per-type 10`. |
| 3 | Oracle transcript missing rogue flag | Check turn order in `chaosops/env/environment.py` — must be `SRE → OVERSIGHT → DEV → MANAGER`. |
| 4 | Dashboard colors wrong in terminal | `export TERM=xterm-256color` before launching. Use iTerm2 / kitty if Apple Terminal misrenders. |
| 5 | Colab OOM | Drop `--max-seq-length` to 768 and `--group-size` to 1. |

---

## Screencast recording (manual — needs you)

After step 4 passes:
```bash
# macOS: ⇧⌘5, "Record Selected Portion", grab just the terminal window
# Start recording, then run:
python -m chaosops.dashboard.terminal \
    --scenario autoscaler_cost_cut --policy oracle --difficulty hard \
    --frame-delay 1.2 --max-steps 4
# Stop recording after the reward panel shows +184
```
Trim to ≤ 90 seconds. Export as MP4 < 25 MB. Embed in slide or upload to HF.
