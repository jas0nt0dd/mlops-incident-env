# Teaching a 3B model to triage MLOps incidents (OpenEnv × HF)

**META × Hugging Face OpenEnv Hackathon — mini blog**

---

## The problem in one sentence

Production ML breaks in boring, expensive ways—schema drift, bad configs, silent feature skew, deploy cascades—and on-call engineers pattern-match across logs, metrics, and component graphs. We asked: **can we turn that into an OpenEnv task and actually move a small LLM with RL-style training?**

---

## What we built (environment)

**MLOps Incident Response** is an [OpenEnv](https://github.com/huggingface/openenv)-style environment: the agent issues structured actions (`query_logs`, `check_metrics`, `compare_configs`, `check_feature_drift`, …) against named components, then submits a diagnosis. Four task tiers—**easy**, **medium**, **hard**, **cascade**—ramp from single-pipeline clues to multi-root post-deploy chaos.

Scoring is **process-aware** (investigation quality, efficiency, synthesis), not a single sparse terminal reward, so learning signal exists before the final submit.

- **Spec & tasks:** [`openenv.yaml`](https://github.com/jas0nt0dd/mlops-incident-env/blob/main/openenv.yaml) in the repo root  
- **Server implementation:** FastAPI app under `server/`  
- **Public demo:** [HF Space — live environment](https://jason9150-mlops-incident-env.hf.space) · [health](https://jason9150-mlops-incident-env.hf.space/health)

---

## How we train (pipeline)

We fine-tune **Meta Llama 3.2 3B Instruct** with **Unsloth** (4-bit LoRA) and **Hugging Face TRL GRPO** (`GRPOTrainer`). Training is **not** offline JSON-only: `hf_train.py` talks to the **same Space** over HTTP—`reset` / `step` build the prompt context, and **environment reward** (diagnosis score) is the teacher, with small **capped** format shaping so gradients do not collapse on broken JSON.

Optional path: collect oracle traces from `inference.py`, convert with `scripts/oracle_traces_to_sft_jsonl.py`, warm up with **SFT** via `SFT_ORACLE_JSONL`, then GRPO.

- **Training script & Hub artifacts:** [model repo `jason9150/mlops-incident-agent-grpo-hf`](https://huggingface.co/jason9150/mlops-incident-agent-grpo-hf) (`hf_train.py`; after a run, `metrics.json` and `grpo_reward_curves.png` when pushed)  
- **Source repo:** [github.com/jas0nt0dd/mlops-incident-env](https://github.com/jas0nt0dd/mlops-incident-env)

---

## Did it learn? (evidence)

We care about **observable** movement for judges and for us: same eval harness before and after training, per-task scores plus average, and mid-training eval checkpoints saved when `SAVE_BEST_BY_EVAL` is on.

**→ After your next successful run, paste one line here** (and attach the Hub plot if you like), for example:

- *Baseline avg … → post-train avg … (see `metrics.json` on the model repo)*

That artifact is the honest story—3B one-shot on the hardest tier is still a frontier, but easy/medium and holistic averages are where a hackathon demo shines.

---

## Try it yourself

1. Open the [Space](https://jason9150-mlops-incident-env.hf.space), hit `/health`, then `POST /reset` with a `task_id`.  
2. Clone the [GitHub repo](https://github.com/jas0nt0dd/mlops-incident-env), read `README.md` for `openenv validate` and Docker smoke-test.  
3. For GRPO: set `HF_TOKEN`, `HF_SPACE_URL`, `HF_REPO_ID`, run `hf_train.py` on a GPU (Colab or HF Jobs)—see repo `README.md` for Colab / dependency notes.

---

## Why this fits OpenEnv

We wanted an environment that **MLOps and RL people** immediately recognize—not a toy grid world—and a pipeline where **the deployment is the simulator**. If this post helped you navigate the submission: the Space URL above is the one judges should open first.

---

*Tags: `openenv`, `trl`, `unsloth`, `grpo`, `mlops`, `hackathon`*
