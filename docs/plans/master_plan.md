# robot-rodeo — master plan

## 1. What This Is

robot-rodeo is the operator's public, learning-first robotics project. It starts entirely in
simulation with Hugging Face **LeRobot** (an open-source Python library and command-line
toolkit for robot learning: datasets, simulators, training and evaluating robot "policies")
and ends v1 ready to buy and use a real **SO-101** arm (a low-cost, open-source 6-motor robot
arm kit sold as a "leader" arm you move by hand and a "follower" arm that copies it). The
operator is new to robotics, so every step also ships a short plain-language explainer.

The product is observable progress: each experiment ends with something to watch (a rollout
video) and a log entry saying what happened and what was learned.

Single user (the operator), learning in public. Local only: nothing is uploaded to the Hugging
Face Hub in v1. Public GitHub repository.

Proposal: docs/plan-redline/robot-rodeo-proposal.html

**v1 goals (operator-approved 2026-11-18):**
- **(a) Watch it work** — run pretrained policies in two simulators (PushT, ALOHA) and watch them succeed or fail.
- **(b) Train your own** — train a PushT policy on the laptop overnight and compare it with the pretrained one.
- **(c) Drive and record** — steer a simulated arm with the keyboard to record about 20 demonstrations, train a policy on them, and evaluate it. This is the same workflow a real SO-101 uses.
- **(d) Buying note** — an SO-101 buying note ready by 2026-12-05 for a possible Christmas purchase.

**Out of scope for v1:** real hardware (a v2 phase after a purchase); uploading to the Hugging
Face Hub; Weights & Biases or any online tracking; Isaac Sim / LeIsaac (too heavy for an 8 GB
laptop GPU); LIBERO (Linux-only, per LeRobot's `pyproject.toml`); reinforcement learning
(HIL-SERL, human-in-the-loop sample-efficient reinforcement learning, LeRobot's real-robot RL method); multi-GPU or cloud training.

### Relationship to community-commando

community-commando (the operator's private community hub) watches `huggingface/lerobot`
releases and may hand robot-rodeo **seed files** in `docs/seeds/`. robot-rodeo owns all robot
build work; it never writes into community-commando.

## 2. Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.12 | LeRobot requires `>=3.12` (`pyproject.toml` line 32, v0.7.x) |
| Package manager | uv | Workspace default; LeRobot documents a uv + Windows PowerShell install |
| Robot learning | `lerobot==0.7.0` with extras `pusht`, `aloha`, `training`, `dataset` | Latest release (v0.7.0, 2026-11-05); pinned so experiments are reproducible |
| Keyboard simulator | project extra `hil = ["gym-hil>=0.1.14,<0.2.0"]` (`uv sync --extra hil`) | Declared so `uv sync` keeps it; `gym-hil` 0.1.14 is a pure-Python wheel |
| GPU compute | PyTorch with CUDA on the RTX 4070 Laptop GPU (8 GB) | Driver 596.49 supports CUDA 13.2. LeRobot's docs say Windows uses PyPI's default torch wheels; Step 1 verifies `torch.cuda.is_available()` and, if false, pins a CUDA torch index in `[tool.uv.index]` chosen by this rule: the newest `https://download.pytorch.org/whl/cu13<N>` index that offers a torch version inside lerobot 0.7.0's torch requirement and a Windows wheel; the chosen URL and torch version are recorded in the probe findings |
| Video | ffmpeg 8.0.1 (already installed via winget) | LeRobot's TorchCodec video decoding requires it |
| Simulators | `gym-pusht` (2D push-the-T task), `gym-aloha` (two-arm robot in the MuJoCo physics engine), `gym-hil` (Franka Panda arm in MuJoCo with keyboard control) | The three simulators that cover goals (a)-(c) |
| Own code | `rodeo` CLI (argparse, stdlib + `tomllib`) | Makes each experiment one repeatable command and writes the log |
| Tests / lint / types | pytest, ruff, mypy | Workspace default |
| Fallback runtime | WSL2 Ubuntu (already installed) | Only for a simulator that fails natively (see §4.5) |

## 3. Data Store

Flat files; nothing secret.

```
recipes/<recipe-name>.toml       experiment definitions (committed)
log/experiments.jsonl            append-only experiment records (committed)
docs/rodeo-log/<exp-id>.md       one readable entry per experiment (committed)
docs/media/<exp-id>.mp4          one short clip per experiment, <= 5 MB (committed)
docs/learn/NN-<topic>.md         plain-language explainers (committed)
docs/findings/                   probe, smoke and throughput findings (committed)
outputs/<exp-id>/                full LeRobot output: checkpoints, all videos, eval_info.json (gitignored; see Outputs root)
%USERPROFILE%\.cache\huggingface  downloaded datasets and models (LeRobot/HF default cache)
```

**Outputs root.** `rodeo` writes every run to `<outputs-root>/<exp-id>/`, where
`<outputs-root>` is the environment variable `RODEO_OUTPUTS` when set, else the `outputs/`
folder of the **main checkout**, found as the parent of `git rev-parse --git-common-dir`
(so a `/build-step` worktree and the main checkout share one outputs folder and one set of
checkpoints). Committed artifacts (`log/`, `docs/rodeo-log/`, `docs/media/`) are written in the
current checkout as usual.

### Identifiers

- **Recipe name:** slug `^[a-z0-9][a-z0-9-]{1,48}$`, equals the TOML file stem. Example: `pusht-pretrained-eval`.
- **Experiment id:** `exp-<YYYYMMDD>-<HHMM>-<recipe-name>` (local time at start). Example: `exp-20261120-2130-pusht-diffusion-train`. Generated by `rodeo run`; used for the output folder, the log record, the Markdown entry and the clip name.

### Recipe file (`recipes/<name>.toml`)

| field | type | note |
|---|---|---|
| `name` | str | equals file stem |
| `kind` | str | `eval` \| `train` \| `record` |
| `tool` | str | `lerobot-eval` \| `lerobot-train` \| `gym-manipulator` (`python -m lerobot.rl.gym_manipulator`) |
| `runtime` | str | `native` (default) \| `wsl` |
| `args` | table | LeRobot flags without leading `--`, e.g. `"policy.path" = "lerobot/diffusion_pusht"` |
| `config_file` | str or absent | path to a JSON config (gym-manipulator only) |
| `expect_minutes` | int | rough wall-clock estimate shown before the run |
| `explainer` | str | path of the related `docs/learn/` page |

`rodeo` always adds `--output_dir=<outputs-root>/<exp-id>` (for `lerobot-eval` and
`lerobot-train`) so each run has its own folder.

### Experiment record (`log/experiments.jsonl`, one JSON object per line)

| field | type | note |
|---|---|---|
| `exp_id` | str | see Identifiers |
| `recipe` | str | recipe name |
| `kind` | str | from recipe |
| `started`, `finished` | str | ISO 8601 local timestamps |
| `status` | str | `ok` \| `failed` \| `interrupted` |
| `command` | list[str] | exact argv run |
| `git_commit` | str | `git rev-parse --short HEAD` at start |
| `lerobot_version` | str | `importlib.metadata.version("lerobot")` |
| `output_dir` | str | absolute `<outputs-root>/<exp-id>` |
| `metrics` | object | eval: `pc_success`, `avg_sum_reward`, `n_episodes` (from `eval_info.json`); train: `steps_done`, `final_loss` if parsed, `steps_per_sec`; record: `episodes_recorded` |
| `clip` | str or null | `docs/media/<exp-id>.mp4` when one was copied |

### Findings data files (machine-readable hand-off between steps)

Each findings Markdown file has a JSON twin that later steps read; the Markdown is for the
operator. Both are rewritten whenever the probe or smoke run is repeated.

`docs/findings/probe.json` (written by Step 1, read by Steps 3 and 8):

| field | type | note |
|---|---|---|
| `cuda` | bool | `torch.cuda.is_available()` |
| `gpu` | str | GPU name |
| `torch` | str | installed torch version |
| `torch_index` | str or null | CUDA index URL pinned by the §2 rule, else null |
| `sims` | object | keys `pusht`, `aloha`, `gym_hil`; each `{ "status": "pass" \| "fail", "reason": str or null, "runtime": "native" \| "wsl" }` |
| `gym_manipulator_import` | object | same shape as a `sims` entry |

`docs/findings/throughput.json` (written by Step 4, read by Step 5):

| field | type | note |
|---|---|---|
| `batch_size` | int | largest of 64/32/16 that fit |
| `steps_per_sec` | float | measured at that batch size |
| `peak_gpu_mb` | int | peak GPU memory |
| `peak_ram_mb` | int | peak system RAM |
| `num_workers` | int | data-loader workers used |
| `dataset_download_mb` | object | dataset id -> MB downloaded on first run |

### Eval output contract (read from LeRobot source, `src/lerobot/scripts/lerobot_eval.py`)

`lerobot-eval` writes `<output_dir>/eval_info.json` (line 939). Its `overall` object carries
`pc_success` (percent of episodes that succeeded), `avg_sum_reward` and `pc_success_ci95`
(lines 736-742, 911-914). Rendered episode videos go under `<output_dir>/videos/`.

### Deduplication and corruption protection

`experiments.jsonl` is append-only; a malformed line is reported and skipped, never rewritten.
A run never reuses an experiment id (the timestamp makes ids unique; a clash exits 1). The clip
copy refuses files over 5 MB (then it keeps only the path to the full video in `outputs/`).

## 4. Domain

### 4.1 Simulators and pretrained policies (verified 2026-11-18 on the Hugging Face Hub API)

| Goal | Simulator (`--env.type`, task) | Pretrained policy | Dataset |
|---|---|---|---|
| (a), (b) | `pusht`, task `PushT-v0` (LeRobot default, `envs/configs.py` line 196) | `lerobot/diffusion_pusht` | `lerobot/pusht` |
| (a) | `aloha`, task `AlohaTransferCube-v0` (default is `AlohaInsertion-v0`, line 174) | `lerobot/act_aloha_sim_transfer_cube_human` | `lerobot/aloha_sim_transfer_cube_human` |
| (c) | `gym_hil`, task `PandaPickCubeKeyboard-v0` | none (operator records demos) | recorded locally |

**Policy types used:** a **Diffusion policy** generates the next actions by gradually denoising
random noise, conditioned on what the robot sees; **ACT** (Action Chunking with Transformers)
predicts a short chunk of future actions at once and learns well from few demonstrations.
Explainer 05 covers both in plain language.

### 4.2 LeRobot commands used (verbatim shapes from LeRobot source and docs)

- Evaluate: `lerobot-eval --policy.path=<hub id or checkpoint>/pretrained_model --env.type=<env> [--env.task=<task>] --eval.batch_size=10 --eval.n_episodes=<n> --policy.use_amp=false --policy.device=cuda --output_dir=<dir>` (docstring of `lerobot_eval.py`, lines 20-45).
- Train: `lerobot-train --policy.type=<diffusion|act> --dataset.repo_id=<hf-dataset-id, e.g. lerobot/pusht> --env.type=<env> --steps=<n> --batch_size=<n> --save_freq=<n> --env_eval_freq=<n> --log_freq=200 --output_dir=<dir>`; resume with `--resume=true --config_path=<dir>/checkpoints/last/pretrained_model/train_config.json` (fields from `configs/train.py` lines 133-172: `output_dir`, `resume`, `seed` default 1000, `num_workers` default 4, `batch_size` default 8, `steps` default 100000, `log_freq` 200, `save_freq` default 20000, `env_eval_freq` default 20000).
- Record in gym_hil: `python -m lerobot.rl.gym_manipulator --config_path <env_config.json>` with `"mode": "record"`, `"task": "PandaPickCubeKeyboard-v0"` and `num_episodes_to_record` (LeRobot `docs/source/hilserl_sim.mdx`; example config `https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json`). Step 8 copies that example to `configs/panda-cube-record.json` and changes only: `task` -> `PandaPickCubeKeyboard-v0`, `control_mode` -> `keyboard`, `mode` -> `record`, `num_episodes_to_record` -> 20, the dataset `repo_id` -> a local name (`local/panda-cube-demos`, never pushed) and its root -> `<outputs-root>/datasets/panda-cube-demos`; every other field stays as in the example.

### 4.3 Training budget

Each training run is capped at 8 hours wall clock (operator-approved overnight runs, laptop
plugged in, sleep disabled for the run). Step 4's smoke run measures steps per second (`docs/findings/throughput.json`) on this
GPU; Step 5 sets `steps = floor(0.9 * 8 h * measured steps/sec)` rounded down to a multiple of
`save_freq`, and records the arithmetic in the recipe comment. `save_freq` is set so a
checkpoint lands at least every 45 minutes, so an interrupted run loses under an hour.

### 4.4 Explainers (`docs/learn/`)

Each is 300-900 words, plain language, written for someone new to robotics, ending with
"What to look for" tied to this repo's experiments:

| File | Topic | Step |
|---|---|---|
| `01-lerobot-and-policies.md` | What LeRobot is; what a robot policy is; observations and actions | 1 |
| `02-reading-results.md` | Success rate, reward, confidence intervals, why 50 episodes beat 5 | 2 |
| `03-the-simulators.md` | PushT, ALOHA, MuJoCo; what a simulator does and does not model | 3 |
| `04-how-training-works.md` | Datasets, steps, batches, loss, checkpoints, overfitting | 5 |
| `05-diffusion-vs-act.md` | The two policy types used here and why each fits its task | 5 |
| `06-teleop-and-demos.md` | Teleoperation, demonstrations, imitation learning, demo quality | 8 |
| `07-sim-to-real.md` | Why sim success does not transfer automatically | 9 |
| `08-so101-anatomy.md` | The SO-101: motors, leader vs follower, calibration | 9 |

### 4.5 Native Windows first, WSL2 fallback

Every recipe runs natively by default. Step 1's probe tries each simulator natively. Known risk:
LeRobot's `hilserl` extra depends on `placo`, which publishes no Windows wheels (PyPI 0.12.0:
macOS and manylinux only), while `gym-hil` itself is pure Python (`any` wheel); `gym-aloha`
depends on `dm-control`, whose Windows support is unverified. Any recipe whose simulator fails
the Step 1 probe natively sets `runtime = "wsl"` (goal (a) and (c) alike). `rodeo` (Step 2) then
runs it as `wsl -d Ubuntu -- bash -lc "<command>"`, where `<command>` is built from the argv
with every element passed through `shlex.quote` and Windows paths rewritten to `/mnt/c/...`,
against a separate WSL virtual environment at `~/robot-rodeo-venv` created by
`scripts/wsl-setup.sh` (Step 2), with the repo at `/mnt/c/Users/abero/dev/robot-rodeo`.
Keyboard input in WSL needs WSLg (the graphical display support built into WSL2 on Windows 11) and `evdev` (the Linux input-event library LeRobot uses to read the keyboard) (LeRobot `installation.mdx`
WSL note); `wsl-setup.sh` installs `libevdev-dev` and `evdev`.

## 5. Modules

`src/robot_rodeo/`:

- `cli.py` — `rodeo` entry point: `list`, `show <recipe>`, `run <recipe> [--dry-run] [--resume <exp-id>]`, `log [--last N]`, `compare <exp-id> <exp-id>`.
- `recipes.py` — load and validate recipe TOML (§3); build argv.
- `paths.py` — outputs-root resolution (`RODEO_OUTPUTS`, else the main checkout's `outputs/` via `git rev-parse --git-common-dir`).
- `runner.py` — start the LeRobot command natively or through WSL (§4.5, `shlex.quote` per argument), stream output to `<outputs-root>/<exp-id>/run.log`, record exit status; Ctrl+C or a killed process records `interrupted`.
- `metrics.py` — parse `eval_info.json` (`overall.pc_success`, `overall.avg_sum_reward`) and training logs (steps, steps/sec, last loss).
- `logbook.py` — append the experiment record; write `docs/rodeo-log/<exp-id>.md` (record table + empty "What I noticed" section for the operator); copy the first rendered video as the clip when <= 5 MB.
- `compare.py` — side-by-side table of two experiments' metrics into `docs/results/<a>-vs-<b>.md`.

Scripts: `scripts/probe.py` (environment probe), `scripts/wsl-setup.sh` (WSL environment for any simulator that fails natively, Step 2).

## 6. API Route Contract

Not applicable — no server. CLI surface is listed in §5 `cli.py`.

## 7. Project Structure

```
robot-rodeo/
  AGENTS.md / CLAUDE.md (pointer)
  README.md
  pyproject.toml            # lerobot==0.7.0[pusht,aloha,training,dataset]; console script rodeo
  recipes/                  # one TOML per experiment
  src/robot_rodeo/          # §5
  scripts/probe.py
  tests/                    # pytest; LeRobot faked in unit tests
  log/experiments.jsonl
  docs/plans/master_plan.md
  docs/learn/               # explainers
  docs/rodeo-log/           # experiment entries
  docs/media/               # short clips
  docs/results/             # comparisons
  docs/findings/            # probe, smoke, throughput
  docs/seeds/               # seeds from community-commando
  docs/so101-buying-note.md
  outputs/                  # gitignored
```

## 8. Key Design Decisions

- **Use LeRobot's own tools; wrap, don't re-implement.** `rodeo` only builds commands, captures output and writes the log, so experiments match what the LeRobot community runs and docs stay applicable.
- **Recipes over ad-hoc commands.** A recipe file makes every experiment repeatable and reviewable in a PR.
- **Pin LeRobot 0.7.0.** LeRobot moves fast (main is already 0.7.1); a pin keeps results comparable. Upgrades are a deliberate step, prompted by community-commando's release watch.
- **Small clips in git, everything else out.** The public log can show what happened without committing gigabytes.
- **Explainers are deliverables.** The operator's goal is to learn; each step's explainer is part of its Done-when.
- **Local only.** No Hub uploads or online trackers in v1 (operator decision).
- **Plan location** `docs/plans/master_plan.md` (workspace canonical).

## 9. Open Questions / Risks

| Item | Risk | Mitigation |
|---|---|---|
| PyTorch CUDA on Windows | PyPI's default Windows torch wheel may be CPU-only | Step 1 asserts `torch.cuda.is_available()`; if false, pins a CUDA index (`cu13x` per driver 596.49) in `[tool.uv.index]` |
| gym_hil on Windows | `placo` has no Windows wheel | Probe decides; WSL2 fallback (§4.5) |
| gym-aloha on Windows | depends on `dm-control` (MuJoCo wrapper) whose Windows support is unverified | Probe decides; WSL2 fallback |
| 8 GB VRAM | Default batch sizes may not fit | Smoke run finds a batch size that fits; recorded in findings |
| 16 GB RAM | LeRobot's 4 default data-loader workers plus a simulator may exhaust RAM | Smoke findings record peak RAM; recipes lower `num_workers` if peak exceeds 12 GB |
| Worktree checkpoints | `/build-step` worktrees start without `outputs/` | Shared outputs root (§3) |
| Laptop thermals / sleep | Overnight run throttles or stops | Checkpoint every <= 45 min; `rodeo run --resume`; operator disables sleep for the run |
| Disk | torch + datasets + checkpoints | ~124 GB free; recipes keep `save_freq` coarse; `outputs/` prune note in README |
| Own policy underperforms | Laptop-sized training is shorter than the reference | Expected and fine; the comparison and explainer say why |
| SO-101 prices move | Buying note goes stale | Note is dated; sources linked |

## 10. How to Run

Prerequisites: Windows 11, Python 3.12, `uv`, NVIDIA driver with CUDA (596.49 present),
`ffmpeg` on PATH (`ffmpeg -version`), about 30 GB free disk.

```powershell
cd C:\Users\abero\dev\robot-rodeo
uv sync --extra hil
uv run python scripts/probe.py
uv run rodeo list
uv run rodeo run pusht-pretrained-eval
```

Then open the clip and log entry printed at the end of the run.

## 11. Development Process

Built with `/build-phase` (the workspace skill that walks this plan's `### Step N:` blocks in
order) dispatching `/build-step` (implements one step in its own git worktree, runs the quality
gate, and gets independent reviewer verdicts before merging). Isolation is `worktree`.
Reviewers are `--reviewers code` (standard code-review lenses); there is no web UI. Per-change
gate: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check src tests scripts`,
`uv run mypy src`. Install is `uv sync --extra hil` (bare `uv sync` would drop the `hil`
extra); there is no dev server or build artifact. A fresh worktree needs its own
`uv sync --extra hil` before the first gate; it re-resolves PyTorch (several GB) from uv's
shared cache, so the first sync in a worktree is slow but does not re-download. Every step that
writes an explainer adds it to `tests/test_explainers.py`, which checks 300-900 words and a
final `## What to look for` section. Training runs are `Type: wait` steps:
`/build-phase` halts there so the long run happens outside the agent's context, and the
operator resumes with `--resume <next step>`.

## Build Steps

### Automated Steps
(These run unattended via /build-phase, except the Type: wait step, where it halts.)

### Step 1: Scaffold, install and simulator probe
- **Problem:** Create the uv project pinned to `lerobot==0.7.0` with extras `pusht`, `aloha`, `training`, `dataset`, and the project extra `hil = ["gym-hil>=0.1.14,<0.2.0"]`, plus `scripts/probe.py`, which checks: `torch.cuda.is_available()` and the GPU name; `ffmpeg -version`; creating and stepping each simulator natively for 10 random actions (`pusht` `PushT-v0`, `aloha` `AlohaTransferCube-v0`, `gym_hil` `PandaPickCubeKeyboard-v0`); importing `lerobot.rl.gym_manipulator` (the recording entry point, which may need `placo`); and writes `docs/findings/probe-<YYYY-MM-DD>.md` with pass/fail per item and the runtime each recipe should use (`native` or `wsl`), plus `docs/findings/probe.json` (§3). If CUDA is unavailable, pin a CUDA torch index by the §2 rule, record the URL and torch version, and re-probe. Write explainer `docs/learn/01-lerobot-and-policies.md` and start `tests/test_explainers.py`.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `pyproject.toml`, `scripts/probe.py`, `docs/findings/probe-<YYYY-MM-DD>.md`, `docs/findings/probe.json`, `docs/learn/01-lerobot-and-policies.md`, `tests/test_explainers.py`, `.gitignore`, `README.md`
- **Produces:** the files above, `tests/test_probe.py`
- **Done when:** after `uv sync --extra hil`, `uv run python scripts/probe.py` exits 0 and the findings file shows CUDA available and pass/fail plus a runtime choice for each simulator and for the `gym_manipulator` import (a failing item is a recorded finding, not a step failure); `tests/test_probe.py` proves the report writer marks a raised error as `fail` with its one-line reason and that `probe.json` has every §3 field; `tests/test_explainers.py` passes for explainer 01.
- **Depends on:** none
- **Status:** TODO

### Step 2: `rodeo` runner and experiment log
- **Problem:** Implement `paths.py`, `recipes.py`, `runner.py`, `metrics.py`, `logbook.py` and `cli.py` (`list`, `show`, `run [--dry-run]`, `log`) per §3 and §5, with both runtimes: `native`, and `wsl` per §4.5 (`shlex.quote` per argument, `/mnt/c/...` path rewrite), plus `scripts/wsl-setup.sh` creating `~/robot-rodeo-venv`. Write explainer `docs/learn/02-reading-results.md`.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `src/robot_rodeo/cli.py`, `src/robot_rodeo/paths.py`, `src/robot_rodeo/recipes.py`, `src/robot_rodeo/runner.py`, `src/robot_rodeo/metrics.py`, `src/robot_rodeo/logbook.py`, `scripts/wsl-setup.sh`, `docs/learn/02-reading-results.md`
- **Produces:** the files above, `tests/test_paths.py`, `tests/test_recipes.py`, `tests/test_runner.py`, `tests/test_logbook.py`, `tests/fixtures/eval_info.json`
- **Done when:** tests prove (through `rodeo run` with a fake `lerobot-eval` script on PATH that writes the fixture `eval_info.json`): the record has every §3 field, `pc_success` is read from `overall`, a non-zero exit records `failed`, a malformed log line is skipped and reported, `--dry-run` prints the argv and writes nothing, and a clip over 5 MB is not copied; `test_paths.py` proves a worktree resolves the main checkout's `outputs/` and that `RODEO_OUTPUTS` overrides it; `test_runner.py` proves a `wsl` recipe becomes `wsl -d Ubuntu -- bash -lc <command>` with each argument `shlex`-quoted (an argument containing a space or `;` stays one argument) and `/mnt/c/...` paths; `tests/test_explainers.py` passes for explainer 02.
- **Depends on:** 1
- **Status:** TODO

### Step 3: Pretrained evaluation recipes (goal a)
- **Problem:** Add `recipes/pusht-pretrained-eval.toml` (`lerobot/diffusion_pusht`, `pusht`, 50 episodes) and `recipes/aloha-cube-pretrained-eval.toml` (`lerobot/act_aloha_sim_transfer_cube_human`, `aloha`, task `AlohaTransferCube-v0`, 50 episodes), run both for real, and commit their log entries and clips. Write explainer `docs/learn/03-the-simulators.md`.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `recipes/pusht-pretrained-eval.toml`, `recipes/aloha-cube-pretrained-eval.toml`, `docs/learn/03-the-simulators.md`
- **Produces:** the recipes, two `docs/rodeo-log/exp-*.md` entries, two `docs/media/exp-*.mp4` clips (or recorded paths if over 5 MB), `log/experiments.jsonl` lines
- **Done when:** both runs end `status: ok` with `pc_success` recorded in `log/experiments.jsonl`, each using the runtime (`native` or `wsl`) recorded for its simulator in `docs/findings/probe.json`; `tests/test_explainers.py` passes for explainer 03.
- **Depends on:** 2
- **Status:** TODO

### Step 4: Smoke gate and GPU throughput
- **Problem:** Add `recipes/pusht-diffusion-smoke.toml` (diffusion on `lerobot/pusht`, 100 steps, `save_freq=100`, `env_eval_freq=0`) and run it for real, then evaluate its checkpoint for 2 episodes through `rodeo`. Find the largest batch size from {64, 32, 16} that fits in 8 GB, and record steps/sec for it in `docs/findings/throughput-<YYYY-MM-DD>.md` and `docs/findings/throughput.json` (§3).
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `recipes/pusht-diffusion-smoke.toml`, `docs/findings/throughput-<YYYY-MM-DD>.md`, `docs/findings/throughput.json`
- **Produces:** the recipe, the findings file, two log entries (train + eval)
- **Done when:** the 100-step train and the 2-episode eval both end `status: ok` through `rodeo run` within 15 minutes combined (first-run dataset downloads excluded and timed separately), and the findings file records batch size, steps/sec, peak GPU memory, peak system RAM, the `num_workers` used (lowered from 4 if peak RAM exceeded 12 GB), and the downloaded size of each dataset.
- **Depends on:** 3
- **Status:** TODO

### Step 5: Laptop-sized PushT training recipe and resume
- **Problem:** Add `recipes/pusht-diffusion-train.toml` sized by §4.3 from `docs/findings/throughput.json` (`batch_size`, `steps_per_sec`, `num_workers`), `recipes/pusht-diffusion-own-eval.toml` (50 episodes on the latest checkpoint under the shared outputs root), and `rodeo run --resume <exp-id>`. Write explainers `docs/learn/04-how-training-works.md` and `docs/learn/05-diffusion-vs-act.md`.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `recipes/pusht-diffusion-train.toml`, `recipes/pusht-diffusion-own-eval.toml`, `src/robot_rodeo/cli.py`, `src/robot_rodeo/runner.py`, `docs/learn/04-how-training-works.md`, `docs/learn/05-diffusion-vs-act.md`
- **Produces:** the files above, `tests/test_resume.py`
- **Done when:** tests prove `--resume` builds `--resume=true --config_path=<outputs-root>/<exp-id>/checkpoints/last/pretrained_model/train_config.json` for the same exp-id and appends a second record linked to it; the train recipe's comment shows the §4.3 arithmetic and expected hours <= 8; `tests/test_explainers.py` passes for explainers 04 and 05.
- **Depends on:** 4
- **Status:** TODO

### Step 6: Overnight PushT training run (goal b)
- **Problem:** Run `uv run rodeo run pusht-diffusion-train` overnight with the laptop plugged in and sleep disabled; resume with `rodeo run --resume <exp-id>` if interrupted.
- **Type:** wait
- **Issue:**
- **Produces:** `outputs/<exp-id>/checkpoints/`, a train log entry
- **Done when:** the train record shows `status: ok` with `steps_done` equal to the recipe's `steps`.
- **Depends on:** 5
- **Status:** TODO

### Step 7: Evaluate your own policy and compare
- **Problem:** Add `compare.py` and `rodeo compare <exp-id> <exp-id>`, run `recipes/pusht-diffusion-own-eval.toml` against Step 6's final checkpoint (found under the shared outputs root), and compare it with Step 3's pretrained PushT experiment, writing `docs/results/<own>-vs-<pretrained>.md` with a short plain-language reading of the difference.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `src/robot_rodeo/compare.py`, `src/robot_rodeo/cli.py`, `docs/results/`
- **Produces:** `src/robot_rodeo/compare.py`, `tests/test_compare.py`, an eval log entry and clip, the comparison file
- **Done when:** `tests/test_compare.py` proves `compare` renders both experiments' `pc_success`, `pc_success_ci95` and `avg_sum_reward`; the real eval record shows `status: ok` with 50 episodes; the comparison file shows both success rates with their 95% intervals.
- **Depends on:** 6
- **Status:** TODO

### Step 8: Keyboard recording and ACT training recipes (goal c tooling)
- **Problem:** Add `recipes/panda-cube-record.toml` (gym-manipulator, `PandaPickCubeKeyboard-v0`, `"mode": "record"`, 20 episodes, local dataset under the shared outputs root) built from LeRobot's example `env_config.json`, `recipes/panda-cube-act-train.toml` (ACT on the recorded dataset, sized by §4.3) and `recipes/panda-cube-act-eval.toml`, each using the runtime recorded for `gym_hil` and `gym_manipulator_import` in `docs/findings/probe.json` (reusing Step 2's WSL support). Write explainer `docs/learn/06-teleop-and-demos.md`, including the keyboard controls.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `recipes/panda-cube-record.toml`, `recipes/panda-cube-act-train.toml`, `recipes/panda-cube-act-eval.toml`, `configs/panda-cube-record.json`, `docs/learn/06-teleop-and-demos.md`
- **Produces:** the files above, `tests/test_panda_recipes.py`
- **Done when:** `rodeo run panda-cube-record --dry-run` prints the exact command for the probed runtime; `tests/test_panda_recipes.py` proves all three recipes load and build argv for that runtime; the explainer lists the keyboard keys from the gym_hil docs and `tests/test_explainers.py` passes for explainer 06.
- **Depends on:** 2
- **Status:** TODO

### Step 9: SO-101 buying note and hardware explainers (goal d)
- **Problem:** Write `docs/so101-buying-note.md` (dated, sources linked): at least three SO-101 kit options (assembled vs self-assembly, leader + follower vs follower only) with current prices, the extras needed (power supplies, USB cables, table clamps, a camera), a total budget range, assembly and calibration time, and what changes in this repo for a v2 hardware phase. Write explainers `docs/learn/07-sim-to-real.md` and `docs/learn/08-so101-anatomy.md`.
- **Type:** code
- **Issue:**
- **Flags:** --reviewers code --isolation worktree
- **Files:** `docs/so101-buying-note.md`, `docs/learn/07-sim-to-real.md`, `docs/learn/08-so101-anatomy.md`
- **Produces:** the three documents
- **Done when:** the note lists >= 3 kit options each with a price, vendor link and access date, and a total budget range; motor facts match LeRobot `docs/source/so101.mdx` (6x STS3215 follower motors at 1/345 gearing, leader with mixed gearing); `tests/test_explainers.py` passes for explainers 07 and 08.
- **Depends on:** none
- **Status:** TODO

### Manual Steps
(These run after /build-phase completes. Operator drives.)

### Step M1: Watch the pretrained policies
- **Source step:** Step 3
- **Issue:**
- **Commands:**
  ```powershell
  cd C:\Users\abero\dev\robot-rodeo
  uv run rodeo log --last 2
  ```
  Open each clip in `docs\media\`, then add two or three lines under "What I noticed" in each `docs\rodeo-log\` entry.
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | PushT clip | The pusher moves the T onto the target outline in most episodes |
  | ALOHA clip | One arm grasps the cube and hands it to the other |
  | Success rates | Recorded in each entry; explainer 02 says how to read them |

### Step M2: Record keyboard demonstrations
- **Source step:** Step 8
- **Issue:**
- **Commands:**
  ```powershell
  cd C:\Users\abero\dev\robot-rodeo
  uv run rodeo run panda-cube-record
  ```
  Use the keys listed in `docs\learn\06-teleop-and-demos.md`; aim for 20 clean pick-ups.
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | Simulator window | Panda arm responds to the keys |
  | Log entry | `episodes_recorded` = 20 |

### Step M3: Train and evaluate on your demonstrations
- **Source step:** Step 8
- **Issue:**
- **Commands:**
  ```powershell
  cd C:\Users\abero\dev\robot-rodeo
  uv run rodeo run panda-cube-act-train
  ```
  Run overnight, then:
  ```powershell
  uv run rodeo run panda-cube-act-eval
  ```
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | Train record | `status: ok` |
  | Eval clip | The arm reaches for the cube; any success is a win with 20 demos |

### Step M4: Read the buying note
- **Source step:** Step 9
- **Issue:**
- **Commands:**
  ```powershell
  start C:\Users\abero\dev\robot-rodeo\docs\so101-buying-note.md
  ```
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | Options and budget | Enough to decide by Christmas, or to note what is missing |

After all Automated Steps complete: **Please run M1 next.**

## 12. Appendix

### Passing Notes project id

`robot-rodeo`.

### Decision Inventory

| ID | P/D | Choice (short) | Status |
|---|---|---|---|
| P1 | P | Public repo named robot-rodeo | stands |
| P2 | P | Simulation first with LeRobot; SO-101 maybe at Christmas | stands |
| P3 | P | v1 = all four goals: watch, train own, keyboard demos + train, buying note | stands |
| P4 | P | Local only; no Hugging Face Hub uploads in v1 | stands |
| P5 | P | No gamepad; keyboard teleoperation | stands |
| P6 | P | Plain-language explainer per step | stands |
| P7 | P | Overnight training runs OK | stands |
| P8 | P | Native Windows first, WSL2 fallback | stands |
| D1 | D | Pin lerobot 0.7.0 with pusht, aloha, training, dataset extras | stands |
| D2 | D | `rodeo` CLI with TOML recipes wrapping LeRobot's own commands | stands |
| D3 | D | Experiment log: append-only JSONL + Markdown entry per experiment | stands |
| D4 | D | Clips <= 5 MB in git; outputs/ gitignored | stands |
| D5 | D | 8 h cap per training run; checkpoint every <= 45 min; steps from measured throughput | stands |
| D6 | D | 50 eval episodes per evaluation | stands |
| D7 | D | PushT diffusion for goal (b); ACT on Panda keyboard demos for goal (c); ~20 demos | stands |
| D8 | D | No W&B; no Isaac Sim; no LIBERO; no RL in v1 | stands |
| D9 | D | Buying note due 2026-12-05 | stands |
| D10 | D | Plan at docs/plans/master_plan.md | stands |
| D11 | D | WSL2 runtime available to every simulator from Step 2, chosen by the Step 1 probe | stands |
| D12 | D | gym-hil declared as `hil` extra; probe imports lerobot.rl.gym_manipulator | stands |
| D13 | D | Shared outputs root (RODEO_OUTPUTS, else main checkout outputs/) | stands |
| D14 | D | Explainers tested: 300-900 words, final What to look for section | stands |
| D15 | D | rodeo compare in Step 7; num_workers lowered if peak RAM > 12 GB | stands |
