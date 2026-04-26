"""Upload hf_train.py to the Hub model repo and submit an HF Job.

Scheduling (stuck on "Scheduling") is Hugging Face GPU queue / capacity — not a bug in
hf_train.py (that file only runs after a machine is assigned). a10g-small can sit in
queue when demand is high. Use a cheaper tier to get a slot sooner:

  Add to .env:  HF_JOB_FLAVOR=t4-medium
  or:          HF_JOB_FLAVOR=t4-small
  or:          HF_JOB_FLAVOR=l4x1

Omit HF_JOB_FLAVOR to default to t4-medium (usually schedules faster than a10g-small).

Push only, then submit (no double upload):
  python scripts/_submit_hf_job_once.py --upload-only
  $env:HF_JOB_FLAVOR = "a100-large"; python scripts/_submit_hf_job_once.py --job-only
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# Cheapest GPU tiers typically leave "Scheduling" sooner than a10g-small.
_DEFAULT_FLAVOR = "t4-medium"


def _read_dotenv_key(key: str) -> str | None:
    path = PROJECT / ".env"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(f"{key}="):
            return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    upload_only = "--upload-only" in sys.argv
    job_only = "--job-only" in sys.argv
    if upload_only and job_only:
        print("Use either --upload-only or --job-only, not both.", file=sys.stderr)
        return 2

    token = _read_dotenv_key("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in .env", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo

    repo_id = "jason9150/mlops-incident-agent-grpo-hf"
    if not job_only:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        HfApi(token=token).upload_file(
            path_or_fileobj=str(PROJECT / "hf_train.py"),
            path_in_repo="hf_train.py",
            repo_id=repo_id,
            repo_type="model",
        )
        print("upload_ok", repo_id)

    if upload_only:
        print("Done (--upload-only); no job submitted.")
        return 0

    flavor = (
        os.environ.get("HF_JOB_FLAVOR", "").strip()
        or _read_dotenv_key("HF_JOB_FLAVOR")
        or _DEFAULT_FLAVOR
    )
    print(f"Using HF Job flavor: {flavor} (set HF_JOB_FLAVOR in .env or env to override)")
    if not str(flavor).lower().startswith("cpu"):
        print(
            "NOTE: GPU Jobs need prepaid Hugging Face credits (see https://huggingface.co/pricing ). "
            "Without credits, jobs often stay in SCHEDULING and never start. "
            "CPU jobs (e.g. cpu-basic) can still complete — use Billing → Compute to verify balance."
        )

    hf = str(PROJECT / ".venv" / "Scripts" / "hf.exe")
    env = {**os.environ, "HUGGINGFACE_HUB_TOKEN": token}
    subprocess.run([hf, "auth", "login", "--token", token], env=env, capture_output=True, text=True)
    args = [
        "jobs",
        "run",
        "--detach",
        "--namespace",
        "jason9150",
        "--flavor",
        flavor,
        "--timeout",
        "6h",
        "--secrets",
        "HF_TOKEN",
        "-e",
        "HF_SPACE_URL=https://jason9150-mlops-incident-env.hf.space",
        "-e",
        "HF_REPO_ID=jason9150/mlops-incident-agent-grpo-hf",
        "-e",
        "MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct",
        "-e",
        "AUTO_PIP_INSTALL=1",
        "-v",
        "hf://jason9150/mlops-incident-agent-grpo-hf:/repo:ro",
        "unsloth/unsloth:latest",
        "python",
        "/repo/hf_train.py",
    ]
    p = subprocess.run([hf, *args], capture_output=True, text=True, env=env)
    out = PROJECT / "hf_job_last_submit.txt"
    out.write_text(
        f"flavor:{flavor}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}\nreturncode:{p.returncode}\n",
        encoding="utf-8",
    )
    print(p.stdout or p.stderr or f"rc={p.returncode}")
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
