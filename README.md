# robot-rodeo

Learning robot learning in public, starting in simulation. This project uses Hugging Face
[LeRobot](https://github.com/huggingface/lerobot) to watch pretrained robot policies at work,
train a policy overnight on a laptop GPU, drive a simulated arm with the keyboard to record
demonstrations and train on them, and get ready for a real
[SO-101](https://huggingface.co/docs/lerobot/so101) arm. Every step comes with a short
plain-language explainer, because the author is new to robotics.

Status: planned, not yet built. Plan: [docs/plans/master_plan.md](docs/plans/master_plan.md).

## What you will find here (as it gets built)

| Folder | What it holds |
|---|---|
| `docs/learn/` | Plain-language explainers: policies, simulators, training, ACT vs Diffusion, teleoperation, sim-to-real, the SO-101 |
| `docs/rodeo-log/` | One entry per experiment: what ran, the result, what I noticed |
| `docs/media/` | Short clips of each experiment |
| `docs/results/` | Side-by-side comparisons (my policy vs the pretrained one) |
| `recipes/` | One TOML file per experiment, so every run is repeatable |

## Stack

| Tool | Why |
|---|---|
| Python 3.12 + uv | LeRobot requires Python 3.12+; uv installs it on Windows |
| LeRobot 0.7.0 (pusht, aloha, training, dataset extras) + gym-hil | Datasets, simulators, training and evaluation; pinned so results stay comparable |
| PyTorch + CUDA | Training on an RTX 4070 Laptop GPU (8 GB) |
| ffmpeg | Video decoding for LeRobot datasets and rollout clips |
| `rodeo` (this repo) | Small CLI that runs a recipe, captures results and writes the log |
| WSL2 Ubuntu (fallback) | Only for a simulator that will not run natively on Windows |

## Prerequisites

- Windows 11 with an NVIDIA GPU and a CUDA-capable driver
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on PATH (`winget install Gyan.FFmpeg`)
- About 30 GB free disk

## Setup

1. `uv sync --extra hil`
2. `uv run python scripts/probe.py` — checks the GPU, ffmpeg and each simulator
3. `uv run rodeo list` — list the experiment recipes
4. `uv run rodeo run pusht-pretrained-eval` — evaluate a pretrained policy and write a log entry with a clip

## Key design decisions

- **LeRobot does the robotics.** `rodeo` only builds LeRobot's own commands from recipes and records what happened, so LeRobot's docs apply directly.
- **Everything stays local.** No uploads to the Hugging Face Hub and no online experiment tracking in v1.
- **Small clips in git, big files out.** Checkpoints and full videos live in a gitignored `outputs/` folder.
- **Overnight training, capped at 8 hours,** with checkpoints at least every 45 minutes so an interrupted run can resume.
