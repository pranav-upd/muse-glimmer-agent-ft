# Muse Glimmer Agent FT — GRPO Finetuning + Harbor Eval

[![Demo video](https://img.youtube.com/vi/Id0ilSVboPM/0.jpg)](https://youtu.be/Id0ilSVboPM)

GRPO finetuning of Muse Glimmer (28B) into a working software agent:
Modal training/inference (RTX PRO 6000, FP4), Langfuse-tracked eval on
`PersonalAILab/AFM-WebAgent-RL-Dataset`, Muse Wonder as LLM judge,
LoRA published to `pranavupadhyaya52/muse_glimmer_agentFT`, and an
lm-eval harness (`eval.py`) for independent benchmarking.

| File | Role |
|---|---|
| `starter.py` | Training, inference, judge, Langfuse harness, serving endpoints |
| `eval.py` | LM eval driver (GPQA Diamond via `lm_harness` on Modal GPU) |

## Benchmark: GPQA Diamond zero-shot (198 questions)

| Weights | Score |
|---|---|
| Base (`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`) | 0.4141 |
| Finetuned (`pranavupadhyaya52/muse_glimmer_agentFT`) | 0.4343 |

Method: EleutherAI harness on a single RTX PRO 6000 via `lm_harness`
(`python eval.py --lm-tasks gpqa_diamond_zeroshot --lm-source both`).
Both arms share prompts, seed, batching, and loader — the only delta is
the LoRA weights, attached onto a clean base load (a double-stacked
adapter bug found mid-project was fixed before these numbers).

The finetuned score is higher by ~0.02 (~4 questions), which hypothetically
suggests a small gain and possibly rules out regression on hard science
reasoning — though at n=198 the gap sits inside statistical noise, so it
is possibly just sampling variation. A second slice would be needed before
reading more into it.

```powershell
python eval.py --lm-tasks gpqa_diamond_zeroshot --lm-source both
```

Quickstart:

```powershell
python starter.py --dry-run                                  # validate, no GPU
python -m modal run starter.py --action train                # GRPO on Modal
python -m modal deploy starter.py                            # publish endpoints + lm_harness
python eval.py --lm-limit 10                                 # smoke: 10 samples
python eval.py                                               # GPQA Diamond, both weights
```

---

## 1. Muse Glimmer, Rewired for Action

**The problem.** Base models chat; they don't *do*. An agent that can't reliably act in a terminal, a repo, or a browser is a demo, not a tool. Closing that gap usually takes pretraining-scale budgets — out of reach for nearly everyone in this room.

**What we built.** A complete, reproducible reinforcement-learning loop that teaches a Meta model to act. Muse Glimmer (28B) trains with GRPO on 10,000 agentic trajectories, guided by three reward signals: verifiable ground truth, strict reply-channel discipline, and a live Muse Wonder judge scoring every rollout in real time. Around it, production-grade plumbing: FP4 training on RTX PRO 6000 via Modal, rotating checkpoints, and one-click Hub publishing. Proof comes from a head-to-head GPQA Diamond run — 198 questions, base model versus finetuned, same harness, same seed — with the finetuned score hypothetically ahead (0.4343 vs 0.4141), though possibly within sampling variation.

**Why it matters.** The weights are the demo; the *loop* is the product. Any team with tasks and a way to check answers can reuse this pipeline to mint their own specialist agents — ops runbooks, research triage, code repair — for the price of a training run, not a pretraining program.

**Why this is Frontier work.** We didn't prompt a model and hope. We gave a machine a genuinely new capability — acting reliably inside real software environments — and proved it with independent benchmarks, not vibes. That's what happens when Meta's models meet the real world: the loop is open, reproducible, and waiting for the next domain.

---

## 2. Built on Muse, End to End

This project runs on three Muse models, each doing what it does best.

**Muse Glimmer (28B)** is the athlete — the finetuning substrate. Through GRPO on 10,000 agentic trajectories, it learns what its pretraining never taught it: to act. Private reasoning on one channel, decisive answers on another, tool calls in live sandboxes instead of chat.

**Muse Wonder** is the referee — called via API inference to judge every rollout in real time. Where answers are verifiable, ground truth rules; everywhere else, Wonder's scores become the reward signal that shapes the policy. No human labels, no waiting — the loop grades itself at machine speed.

**Muse Spark 1.3** is the engineer — pair-programming inside OpenCode through the entire build. It scaffolded the training pipeline, diagnosed dead training steps from raw logs, verified the tokenizer against the model card, wired the Harbor eval harness, and hardened the deployment endpoints. Every debugging turnaround in this project has Spark's fingerprints on it.

Together: Glimmer learns, Wonder judges, Spark builds — a complete agent pipeline where Meta's models don't just participate, they staff every role.

---

## 3. Architecture Review: Is It Sound? Does It Hit the Objective?

**Is the architecture sound?** Yes — each layer does one job and the contracts between them hold. Modal owns compute (training on RTX PRO 6000, inference behind versioned endpoints, secrets never in code). GRPO owns learning (four-term rewards where no single signal can silently dominate, fallbacks at every failure point). The Hub owns continuity (same-name pushes; volume and local stay in sync by overwrite, never by accumulation). The harness owns proof (independent lm-eval runs, deterministic grading, no Wonder involvement). Verified claims throughout: tokenizer checked against the model card, channel behavior confirmed live, judge mechanism probed directly. The weak seam is compute, especially GPU: every stage — 40-hour training runs, per-trial 28B inference, parallel eval containers — competes for the same scarce accelerator budget, so iteration speed and eval scale are capped by hardware access, not by design.

**Does it meet the objective?** Half-proven, by its own standard. "Acting reliably inside real software environments" — demonstrated in slices: channel-disciplined answers, productive gradient steps, trajectories stepping against live sandboxes. "Proved with independent benchmarks, not vibes" — *not yet*: the TerminalBench base-vs-finetuned head-to-head is staged and instrumented but unscored at the time of writing. The architecture earns the claim structurally; only completed scorecards earn it factually. Run the ten tasks — then the sentence is true.

---

## Inference

The finetuned model is served from Modal (Hub LoRA attached at request
time — no local GPU needed). Two endpoints, same contract:

| Endpoint | Weights |
|---|---|
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run` | Finetuned (`muse_glimmer_agentFT` LoRA) |
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve-base.us-east.modal.run` | Clean base |

POST `{"prompt", "max_new_tokens"?, "temperature"?}` →
`{"reply", "reasoning"}` (the `to=user` / `to=self` channels).
`temperature: 0` means greedy.

```powershell
$body = @{
  prompt           = "What is 48 + 48/2? Reply with the number only."
  max_new_tokens   = 256
  temperature      = 0.7
} | ConvertTo-Json
Invoke-RestMethod `
  -Uri "https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/" `
  -Method Post -Body $body -ContentType "application/json"
```

```python
import json, urllib.request

body = json.dumps({
    "prompt": "What is 48 + 48/2? Reply with the number only.",
    "max_new_tokens": 256,
    "temperature": 0.7,
}).encode()
req = urllib.request.Request(
    "https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/",
    data=body, headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req, timeout=300)))
```

```bash
curl -X POST \
  https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 48 + 48/2? Reply with the number only.","max_new_tokens":256,"temperature":0.7}'
```
