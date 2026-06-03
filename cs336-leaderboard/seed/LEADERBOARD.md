# CS336 Systems Leaderboard — Agent Starting Point

This CORAL task optimizes the **full training step** benchmark from
[assignment2-systems-leaderboard](https://github.com/stanford-cs336/assignment2-systems-leaderboard).

## Current baseline in this repo

- `cs336_systems/model.py` — `FlashBasicsTransformerLM` with FA4 CuTe attention
- `cs336_systems/flash_attention.py` — attention backend wired via `tests/adapters.py`
- Benchmark: `scripts/naive_leaderboard_benchmark.py`

Default grader settings use `--attention cute --checkpoint-every 1` (~6 s quick bench on 1× B300).

## Score

**Lower is better** — wall-clock milliseconds for one training step.

## Quick local timing

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/naive_leaderboard_benchmark.py \
  --attention cute --checkpoint-every 1 --quick
```

## Official leaderboard timing

```bash
uv run python scripts/naive_leaderboard_benchmark.py \
  --attention cute --checkpoint-every 1
```

See `docs/leaderboard_progress.md` for experiment history.
