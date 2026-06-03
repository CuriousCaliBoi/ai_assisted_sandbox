# CORAL task: CS336 Systems Leaderboard

Autonomous agents optimize the full 8B training-step benchmark from
[CS336 Assignment 2 Systems](https://github.com/stanford-cs336/assignment2-systems-leaderboard).

## Prerequisites

- [CORAL](https://docs.coralxyz.com/getting-started/installation) (`uv tool install git+https://github.com/Human-Agent-Society/CORAL.git`)
- CUDA GPUs (2× B200/B300 class recommended)
- `uv sync` in the parent repo (`..`)

## Validate grader

Run from this directory so the grader can find the parent assignment repo:

```bash
cd cs336-leaderboard
export PATH="$HOME/.local/bin:$PATH"

# Quick smoke eval: set grader.args.quick to true in task.yaml first (~2 min GPU)
coral validate .

# Full official timing uses warmup=10000, rep=30000 (default; ~10+ min GPU)
```

## Launch agents

```bash
cd cs336-leaderboard
export PATH="$HOME/.local/bin:$PATH"
coral start -c task.yaml
coral log      # leaderboard
coral status   # agent health
coral ui       # web dashboard
coral stop
```

## Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `grader.direction` | `minimize` | Lower training-step ms wins |
| `workspace.repo_path` | `..` | Clones `ai_assisted_sandbox` for each run |
| `grader.args.quick` | `false` | Use short `do_bench` warmup/rep when `true` |
| `grader.args.attention` | `cute` | Passed to benchmark script |
| `grader.args.checkpoint_every` | `1` | Gradient checkpoint interval |

Override at CLI, e.g. `coral start -c task.yaml -o grader.args.quick=true` for faster iteration.

## What agents should change

Primarily `cs336_systems/` and `tests/adapters.py`. See `task.yaml` description for constraints.
