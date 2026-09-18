# Artifact of AgentDiet

This is the artifact for the FSE26 paper: Reducing Cost of LLM Agents with Trajectory Reduction  
https://arxiv.org/abs/2509.23586

**Structure:**

- `code/`: AgentDiet on Trae Agent
  - `code/trae_agent/agents/traj_analyzer.py`: trajectory reduction / reflection
  - `code/trae_agent/swebench_main.py`: main runner (SWE-bench / Multi-SWE-bench)
  - `code/trae_agent/main.sh`: batch entrypoint
  - `code/trae_agent/main_args*.txt`: experiment configs (`out_name|benchmark|arg_json`)
  - `code/trae_agent/utils/llm_upstreams.example.py`: LLM API template (copy → `llm_upstreams.py`)
  - `code/trae_agent/utils/sandbox.py`: Docker sandbox (auto-cleanup of containers)
  - `code/subjects/`: subject lists (`approach_100.json`, `eval_200.json`, …)
  - `code/data/`: SWE-bench Verified metadata
- `result/`: notebooks / scripts to render paper tables and figures
  - `result/trajs.7z`: collected trajectories from the paper

## ➡️ Reproduce the full experiment

> **WARNING:**
>
> Fully re-running the paper can cost ~$2k and ~500GB disk.
>
> Alternatives: run a subset, or inspect `result/trajs.7z` (see below).

### Install dependencies

Requirements:

- x86-64 Linux, Python 3.11+ (we use a conda env named `py312`)
- Docker
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) Verified harness (`swebench==4.1.0` is pinned in `code/requirements.txt`)
- [Multi-SWE-bench Flash](https://github.com/multi-swe-bench/multi-swe-bench) (only if you run `multiswebench-flash`)
- Other packages: `cd code && pip install -r requirements.txt`

Data / env paths expected by the code:

- `~/miniconda3/envs/py312/` — copied into each eval container
- For Multi-SWE-bench Flash only: `~/Multi-SWE-bench-flash/multi_swe_bench_flash.jsonl`  
  ([HuggingFace](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench-flash/blob/main/multi_swe_bench_flash.jsonl))

### Configure LLM API keys

API keys live in a **gitignored** file so they are not committed:

```bash
cd code/trae_agent/utils
cp llm_upstreams.example.py llm_upstreams.py
# edit llm_upstreams.py — fill real base_url / api_key for each model you use
```

`llm_polytool.py` imports `UPSTREAMS_PER_MODEL` from `llm_upstreams.py`. Model names in `main_args*.txt` must match keys in that dict.

Example (OpenAI-compatible gateway):

```python
from utils.llm_polytool import send_request_openai

UPSTREAMS_PER_MODEL = {
    'gpt-5.6-terra': send_request_openai('https://your_base_url/v1', 'your_api_key'),
    'gpt-5.6-luna': send_request_openai('https://your_base_url/v1', 'your_api_key'),
    'gpt-5-mini': send_request_openai('https://your_base_url/v1', 'your_api_key'),
    'deepseek-v4-flash': send_request_openai('https://api.deepseek.com/v1', 'your_api_key'),
    # ...
}
```

### Choose which experiments to run

`main.sh` reads **`main_args-paper.txt`** by default (see the `done < …` line at the end of `main.sh`).

Each line is:

```text
out_name|benchmark|arg_json
```

| Field | Meaning |
| --- | --- |
| `out_name` | Output dir under `code/out/<out_name>/` |
| `benchmark` | `swebench-verified-appr100`, `swebench-verified-eval200`, `swebench-verified-quick10`, or `multiswebench-flash` |
| `arg_json` | Passed as env `TRAJ_ANALYSIS` (mode / models / thresholds) |

Blank lines and lines starting with `#` are skipped.

Common `arg_json` fields:

- `mode`: `skip` (no reduction), `ours` (AgentDiet), `lingua` / `random` / `delete`, or `arbiteros*` variants
- `fix_model`: agent coding model
- `model`: trajectory-compression model (when `mode` is `ours`, etc.)

Ready-made configs in `code/trae_agent/`:

| File | Typical use |
| --- | --- |
| `main_args-paper.txt` | Current paper-style run (default for `main.sh`) |
| `main_args.txt` | Local / ad-hoc setting |
| `main_args-skip.txt` | Baseline without reduction |
| `main_args-bak.txt` | Full paper matrix (design space + eval + multi) |
| `main_args-arbiteros*.txt` | ArbiterOS-related modes |

To switch config, either edit `main_args-paper.txt`, or change the filename at the bottom of `main.sh`.

### Run

```bash
conda activate py312   # or your env with the deps above
cd code/trae_agent
bash main.sh
```

Expect output like:

```text
=== name=design_space/paper_gpt_5_6_terra_gpt_5_6_luna benchmark=swebench-verified-appr100 {...}
-- analysis args: {...}
tot tasks: 100
processing:  django__django-16667
processing:  django__django-11477
processing:  django__django-15987
== finished (val = fail / gen = task_done): django__django-16667 @ ...
```

Notes:

- Prefer `tmux` / `screen` — runs are long.
- Default parallelism is **3** worker processes (`num_processes` in `swebench_main.py`).
- Outputs go to `code/out/<out_name>/{log,patch,output}/`.
- Sandbox containers are force-removed after each task; orphans are swept on startup / exit / Ctrl+C.
- Optional monitor: edit `watch.sh`’s `OUT=` path, then `bash watch.sh`.

### Run a single instance

```bash
cd code/trae_agent
INSTANCE_ID=django__django-16667 TRAJ_ANALYSIS='{"mode":"ours","model":"gpt-5.6-luna","fix_model":"gpt-5.6-terra"}' \
  python3 swebench_main.py \
    --benchmark swebench-verified-appr100 \
    --log_path ../out/debug/log \
    --patches_path ../out/debug/patch \
    --output_path ../out/debug/output
```

`INSTANCE_ID` can be a comma-separated list.

## ➡️ Reproduce on a subset

1. Edit the args file used by `main.sh` (default `main_args-paper.txt`): keep only the lines you need, or comment others with `#`.
2. Or point `main.sh` at `main_args-bak.txt` and trim that file (e.g. keep only `multi/...` for Multi-SWE-bench Flash).
3. Or use `INSTANCE_ID=...` as above.

Already-finished instances (valid non-error trajectory JSON under `log/`) are skipped automatically.

## ➡️ Inspect collected trajectories

Collected trajectories are in `result/trajs.7z`:

```bash
cd result && 7z x trajs.7z
```

Open `result/exporter.ipynb` to render paper tables / figures from those trajectories.
