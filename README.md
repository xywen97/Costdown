# Artifact of AgentDiet

This is the artifact for the FSE26 paper: Reducing Cost of LLM Agents with Trajectory Reduction
https://arxiv.org/abs/2509.23586

**Structure:**

- `code/`: The tool implementation of AgentDiet on Trae Agent
  - `code/trae_agent/agents/traj_analyzer.py`: The reflection module introduced in the paper
  - `code/main_args.txt`: Parameters for all experiment settings
  - `code/subjects/`: The list of subjects used for the experiment
- `result/`: Scripts to render tables and figures of experiment results in the paper
  - `result/trajs.7z`: Raw trajectories collected in the experiment

## ➡️ Reproduce the full experiment

> **WARNING:**
> 
> It may cost ~$2k and ~500GB of disk space to fully run all experiment subjects in the paper.
> 
> If this is unacceptable for you, you can:
> 
> - run the experiment only on a small subset, or
> - inspect the collected trajectories instead.
> 
> Instructions for such alternatives are provided in other sections below.

### Install dependencies

The artifact has the below requirements:

- An x86-64 Linux machine with Python 3.11+ (we used Debian 12)
- Docker
- Evaluation harness of SWE-bench Verified (refer to https://github.com/SWE-bench/SWE-bench)
- Evaluation harness of Multi-SWE-bench Flash (refer to https://github.com/multi-swe-bench/multi-swe-bench)
- Other Python packages (`cd code && pip install -r requirements.txt`)

You also need to prepare necessary data files in the below locations:

- `~/miniconda3/envs/py312/` (install with [Miniconda](https://www.anaconda.com/download))
- `~/Multi-SWE-bench-flash/multi_swe_bench_flash.jsonl` (download from [HuggingFace](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench-flash/blob/main/multi_swe_bench_flash.jsonl))

### Apply for LLM API keys

You need API keys to call various LLMs. You can apply a key from [OpenRouter](https://openrouter.ai/).

Then, modify the `UPSTREAMS_PER_MODEL` variable in `code/trae_agent/utils/llm_polytool.py` to fill in your API keys. For example:

```python
UPSTREAMS_PER_MODEL = {
    'gpt-5-mini-2025-08-07': send_request_openai('https://your_base_url', 'your_api_key'),
    # ...
}
```

For the full experiment, we used below LLMs:

- `claude4-sonnet` (Claude 4 Sonnet)
- `claude35-haiku` (Claude 3.5 Haiku)
- `gemini-2.5-pro` (Gemini 2.5 Pro)
- `gemini-2.5-flash` (Gemini 2.5 Flash)
- `gpt-5-mini-2025-08-07` (GPT-5 mini)
- `deepseek-chat` (DeepSeek v3)
- `qwen3-235b-a22b-instruct-2507` (Qwen3)

You can certainly replace them with other models, but make sure to also update `code/main_args.txt` to the new model names.

### Run the experiment

Use `cd code/trae_agent && ./main.sh` to run the full experiment. You will see outputs like this:

```bash
username@machine:/path/to/artifact/code/trae_agent$ ./main.sh 
=== name=design_space/baseline benchmark=swebench-verified-appr100 {"mode": "skip"}
-- analysis args: {'mode': 'skip'}
tot tasks: 100
processing:  django__django-16667
processing:  django__django-11477
processing:  django__django-15987
processing:  django__django-16315
processing:  pylint-dev__pylint-7080
processing:  sphinx-doc__sphinx-11510
processing:  astropy__astropy-14598
processing:  pylint-dev__pylint-6903
processing:  sphinx-doc__sphinx-9698
== finished (val = fail / gen = task_done): django__django-16667 @ Mon Sep  8 15:24:34 2025
processing:  sphinx-doc__sphinx-7910
== finished (val = fail / gen = turn_capped): sphinx-doc__sphinx-11510 @ Mon Sep  8 15:27:20 2025
processing:  django__django-11451
......
```

Since the experiment will run for a long time, it is recommended to run it in a `screen` or `tmux`.

The trajectories, logs, and generated patches will be saved in `code/out/`.

## ➡️ Reproduce on a subset

The file `code/trae_agent/main_args.txt` includes all experiment settings to run, and will be read by `code/trae_agent/main.sh`.

Each line in the file follows the format: `out_name|benchmark|arg_json`. You can delete or comment out (with `#`) lines that you want to skip.

For example, if you only want to reproduce the experiment on Multi-SWE-bench Flash, delete everything except for the last four lines (`multi/...`).

## ➡️ Inspect collected trajectories

We are aware that the experiment may be costly to run, so we collected trajectories for future researchers to interpret the results without re-running the experiment.

Collected trajectories are located in `result/trajs.7z`. You can extract the zipped file with the command `7z x trajs.7z`.

The Jupyter notebook at `result/exporter.ipynb` reads the trajectories and renders the experiment results. Refer to that notebook for how to use the trajectories.