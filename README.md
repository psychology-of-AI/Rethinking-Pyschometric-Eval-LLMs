# Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior

🚩 **News**: Accepted as **Oral Presentation** at the [**Combining Theory and Benchmarks (CTB) Workshop**](https://sites.google.com/view/icml-ctb/home), **ICML 2026** — Seoul, South Korea, July 10–11, 2026.

This work builds on our earlier work, [*The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs*](https://github.com/psychology-of-AI/Personality-Illusion) (**ICML 2026** + **Best Paper Honorable Mention** @ NeurIPS 2025 LAW Workshop), which first documented systematic self-report–behavior dissociation in LLMs. This follow-up identifies *when* and *why* coherence emerges.

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://psychology-of-ai.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.12730-red?logo=arxiv)](https://arxiv.org/abs/2606.12730)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](./LICENSE)

![Workflow](./assets/workflow.jpg)

## Overview

Anticipating LLM behavioral tendencies from low-cost psychometric probes is critical for safe deployment — but only if self-reports (SR) reliably predict behavior. Prior work (including our own *Personality Illusion*) documented substantial SR–behavior dissociation, but did not pin down *why*. Two methodological assumptions in the literature explain the gap:

- 📏 **Coarse instruments.** Big Five traits are designed to be cross-situational and weakly predict specific behaviors even in humans (*r* ≈ 0.20).
- 🪟 **Weak context matching.** SR and behavior have typically been measured in independent sessions with only loose parameter matching, hiding any coherence that depends on shared context.

We address both with a **2 × 2 × 2 factorial design** varying:

- 🧭 **Framework**: Theory of Planned Behavior (TPB, fine-grained) vs. Big Five (coarse)
- 💬 **Session context**: same-session (shared message thread) vs. separate-sessions (independent API calls)
- 🎭 **Identity induction**: parameter grid (temperature × seed × system prompt) vs. persona prompting (30 PersonaHub characters)

Applied across **4 behavioral tasks** (risk-taking, sycophancy, honesty, implicit bias) and **11 frontier LLMs**.

### 🔑 Key Findings

- 🎯 **Granularity matters.** Same-session TPB reaches the human meta-analytic intention–behavior baseline (*r* ≈ 0.40); Big Five does not predict at all (best |*r*| < 0.07).
- 🧩 **Cross-session coherence is task-dependent.** It survives for behaviors anchored outside the prompt (implicit bias, honesty) but collapses for context-primed behaviors (sycophancy).
- 🎭 **Personas stabilize self-reports but not behavior.** Persona prompting makes SR more consistent across sessions yet does not rescue behavioral coupling — a safety-relevant pattern for persona-customized deployments.

## 📦 Repository Structure

```
.
├── configs/      # Experiment configs (tasks, TPB/Big5 sweeps, personas, model registry)
├── src/          # Core library (LLM client, runners, task envs, surveys, perturbations)
├── scripts/      # Sweep, merge, and RQ analysis scripts
└── results/      # Pre-collected experiment data
    ├── between/  # Separate-sessions (grid + persona inductions)
    ├── within/   # Same-session (SR + behavior in shared context)
    └── merged/   # Joined SR × behavior CSVs (ready for analysis)
```

## 🧪 Experimental Design

| Task                       | Abbrev.    | Construct                         | Primary TPB anchor       |
| -------------------------- | ---------- | --------------------------------- | ------------------------ |
| Columbia Card Task         | CCT        | Risk-taking                       | Perceived Behav. Control |
| Sycophancy probe (Asch)    | Sycophancy | Social conformity                 | Subjective Norm          |
| Honesty calibration        | Honesty    | Confidence calibration & updating | Attitude                 |
| Implicit Association Test  | IAT        | Implicit bias (6 domains)         | Intention                |

- **Self-report instruments:** TPB (TACT-anchored per task) and Big Five (BFI-44).
- **Induction:** Parameter grid (3 system prompts × 3 temperatures × 3 seeds = 27 conditions) or 30 PersonaHub personas.
- **Sessions:** Same-session (within) vs. separate-sessions (between).

## ⚙️ Setup

Requires Python 3.10+ and an [OpenRouter](https://openrouter.ai) API key.

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
set -a; source .env; set +a
```

## ▶️ Reproducing the Results

Pre-collected results are included under `results/` — you can re-run the analysis directly. Full sweep reproduction requires substantial API time and cost.

| Step | What it does | Entry point |
| ---- | ------------ | ----------- |
| 1 | TPB & Big Five self-report sweeps | `scripts/config_based_sweeps/sweep_tpb_variants.py` |
| 2 | Behavioral task sweeps (CCT / Sycophancy / Honesty / IAT) | `scripts/config_based_sweeps/sweep_{task}_variants.py` |
| 3 | Merge SR with behavior | `scripts/merging/merge_*.py` |
| 4 | RQ1–RQ4 analyses and figures | `scripts/analysis/rq{1,2,3,4}_*.py` |

Each script reads a config from `configs/` and writes to the matching `results/` subdirectory. See the README under each scripts subfolder for exact invocations.

## 🤖 Models

All 11 models are accessed via [OpenRouter](https://openrouter.ai) (registry: `configs/openrouter_models.json`):

- **Proprietary:** Claude 3.7 Sonnet, Claude Haiku 4.5, GPT-4o mini, Gemini 2.5 Flash
- **Open-weight:** LLaMA-3.3 70B, LLaMA-4 Maverick, Qwen2.5 72B, Qwen3 235B-A22B, DeepSeek V3.1, Phi-4, Mistral Large

## 🤝 Contributions

We **welcome contributions** — new self-reports, behavioral tasks, or LLMs. Please open a PR with a brief description and any relevant setup details. For larger changes, start a Discussion first to align efforts.

## 💬 Getting in Touch

- General questions → GitHub Discussions
- Bugs → open an Issue with reproduction steps and logs
- Feature requests → Discussions

## 📑 Citation

If you find this work useful, please also consider citing our prior paper this work builds upon:

```bibtex
@misc{han2025personalityillusionrevealingdissociation,
  title={The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs},
  author={Pengrui Han and Rafal Kocielnik and Peiyang Song and Ramit Debnath and Dean Mobbs and Anima Anandkumar and R. Michael Alvarez},
  year={2025},
  eprint={2509.03730},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2509.03730},
}
```

## 📜 License

MIT — see [LICENSE](./LICENSE).
