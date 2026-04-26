# MLOps Incident Response — OpenEnv Hackathon mini-blog

**Publish this post on your Hugging Face *Space* (not the model repo):**  
[Create a new Discussion → `jason9150/mlops-incident-env`](https://huggingface.co/spaces/jason9150/mlops-incident-env/discussions/new)

Official materials ask judges to open **your environment URL** first; the story below matches what they score (innovation, storytelling, reward improvement, pipeline).

---

## 1. Problem (why this exists)

Production ML fails in ways that are expensive to rehearse: schema drift, bad deploys, silent feature skew, and multi-service cascades. On-call engineers **read evidence** (logs, metrics, configs, drift) and **name the right component** before fixing anything. We built an OpenEnv environment so an LLM can **practice that loop** with **verifiable scores**—not trivia, not a toy grid world.

This aligns with **Theme #3 — World modeling / professional tasks** in the hackathon brief: tools, partial observability, and multi-step workflows instead of shortcuts.

---

## 2. Environment (what the agent does)

**MLOps Incident Response** is OpenEnv-compliant (`openenv.yaml` in the [GitHub repo](https://github.com/jas0nt0dd/mlops-incident-env)):

- **Four tasks:** `easy`, `medium`, `hard`, `cascade` — from a single bad pipeline to a three-way cascade after a bad release.
- **Actions:** inspect components, query logs, check metrics, compare configs, check feature drift, submit a structured diagnosis.
- **Rewards:** Rubric-style scoring (component ID, root cause depth, investigation quality, synthesis, efficiency)—**dense signal during the episode**, not only a terminal 0/1.

**Live demo (submit this URL to judges):**

- **Space:** https://jason9150-mlops-incident-env.hf.space  
- **Health:** https://jason9150-mlops-incident-env.hf.space/health  
- **Training artifacts (metrics + plot links):** https://jason9150-mlops-incident-env.hf.space/training-artifacts  
- **API:** `POST /reset` with `task_id`, then `POST /step` with `action_type`, `target`, `parameters`.

**Training & compute (what judges can open next):**

- **Google Colab** — [training / repro notebook](https://colab.research.google.com/drive/1BAduA4fXaV6QGvjUWi1VZPdyW1-PdTRr): Unsloth + TRL GRPO against the **same live Space** as production; good for stepping through installs, SFT warm-up, reward curves, and ablations.
- **Hugging Face GPU Job** — [hosted job `69edb45e…`](https://huggingface.co/jobs/jason9150/69edb45ed70108f37acdfbfa): one **HF Jobs** execution of `hf_train.py` with full logs and hardware metadata on Hugging Face (pairs with Colab as the “cloud GPU” story).

---

## 3. Training pipeline (Unsloth + TRL, against the live Space)

Hackathon **minimum requirements** call for a **training script using Unsloth or HF TRL** (ideally Colab). We use **both**:

- **Base model:** Meta **Llama 3.2 3B Instruct** (gated on Hub; license + token required).
- **Method:** **GRPO** (`trl.GRPOTrainer`) + **Unsloth** 4-bit LoRA for efficiency.
- **Reward:** The **environment** scores `submit_diagnosis` (same Space URL as production). Small **capped** format shaping keeps JSON + thinking tags stable without drowning the env signal.
- **Curriculum:** `RUN_MODE` / `TRAIN_COUNTS_JSON` bias easy/medium so a 3B model gets non-zero reward early (per organizer guidance: *success probability must be > 0 for RL to work*).

**Optional:** Oracle traces → `scripts/oracle_traces_to_sft_jsonl.py` → **SFT warm-up** via `SFT_ORACLE_JSONL`, then GRPO.

**Training code & artifacts (separate Hub model repo — adapters + metrics):**  
https://huggingface.co/jason9150/mlops-incident-agent-grpo-hf  
After a run: `metrics.json`, `grpo_reward_curves.png`, and LoRA weights when `HF_TOKEN` is set.

**Source:** https://github.com/jas0nt0dd/mlops-incident-env  

---

## 4. Evidence of improvement (what judges look for — 20%)

Organizers ask for **observable** training progress: *reward curves, metrics, before/after behavior*.

Our runner (`hf_train.py`) logs **per-task baseline vs post-train** and saves a **before/after bar chart + mid-training eval curve**. **Replace the line below** with your latest numbers or attach the Hub `metrics.json` / PNG in this Discussion.

> **Latest run (edit me):** average baseline → post-train: *(paste from Colab / HF Job `RESULTS` block)*  
> Example pattern from internal runs: easy/medium often move first; hard/cascade stay harder for a 3B one-shot policy—honest reporting beats cherry-picking one tier.

**Plots:** label axes (step vs score), keep baseline and trained on **comparable** eval settings (`EVAL_GREEDY=1` by default).

---

## 5. How this maps to judging (official weights)

From **Themes & Judging Criteria** (OpenEnv India 2026):

| Weight | Criterion | How we address it |
|--------|-----------|-------------------|
| **40%** | Environment innovation | MLOps incident domain; multi-tier scenarios; rubric + process-aware rewards; cascade synthesis. |
| **30%** | Storytelling & presentation | This post + README + live Space; problem → env → train → evidence → why it matters. |
| **20%** | Improvement in rewards | Hub / local `metrics.json` + `grpo_reward_curves.png`; before vs after table. |
| **10%** | Reward & training pipeline | Env score is teacher; shaping capped; GRPO connects to **your** Space, not a static offline-only dataset. |

**Minimum checklist (non‑negotiable from the same doc):**

- [x] OpenEnv (latest) — spec + server in repo.  
- [x] Training script — Unsloth + TRL (`hf_train.py`).  
- [x] Evidence of a real train — plots + metrics (link files in this Discussion or in README).  
- [x] **This mini-blog on Hugging Face** — *you are reading it; keep the Discussion URL in your README.*  
- [x] Environment on **HF Space** — URL above.  
- [x] README on GitHub links Space + materials.

---

## 6. Resources we actually used

From **OpenEnv_Hackathon_Resources.md**:

- OpenEnv: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/)  
- Hub collections: [huggingface.co/openenv](https://huggingface.co/openenv)  
- Tutorials: [OpenEnv/tutorial](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)  

Participant guide themes: RL loop, verifier-first rewards, GRPO/Unsloth stack, deploy Space early—we followed that order.

---

## 7. TL;DR (organizer quote)

> Build an environment an LLM can **train on** to get measurably better at something interesting. **Then show that training. Then tell the story.**

We built the MLOps simulator, wired **GRPO to live rewards**, and publish **metrics + plots**. Judges: start at the **Space URL**, then open the **model repo** for training artifacts if needed.

---

*Tags: `openenv`, `trl`, `unsloth`, `grpo`, `mlops`, `hackathon`*
