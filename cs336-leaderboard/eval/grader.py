"""Grader for CS336 Assignment 2 Systems leaderboard optimization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    """Score = training-step wall time in ms (lower is better)."""

    def evaluate(self) -> float | ScoreBundle:
        sandbox = self._resolve_sandbox_root()
        if isinstance(sandbox, ScoreBundle):
            return sandbox

        if not self._cuda_available(sandbox):
            return self.fail("CUDA is required for leaderboard timing (check GPUs and drivers).")

        if self.args.get("run_correctness", True):
            correctness = self._run_correctness_tests(sandbox)
            if correctness is not None:
                return correctness

        timing = self._run_benchmark(sandbox)
        if isinstance(timing, ScoreBundle):
            return timing

        ms, details = timing
        baseline_ms = float(self.args.get("baseline_ms", 10_000))
        top_ms = float(self.args.get("top_ms", 3_837))

        explanation = (
            f"Training step: {ms:.1f} ms ({ms / 1000:.3f} s). "
            f"Leaderboard baseline: {baseline_ms:.0f} ms; top entry: {top_ms:.0f} ms."
        )
        if ms < baseline_ms:
            explanation += " Beats verified naive baseline."
        if ms < top_ms:
            explanation += " Beats current #1 (self-reported; needs verification)."

        feedback = details.get("feedback") or ""
        if not self.args.get("correctness_strict", False) and self.args.get("run_correctness", True):
            causal_note = (
                "Causal FA backward not in correctness gate yet (test_flash_backward_triton[True])."
            )
            feedback = f"{feedback} | {causal_note}" if feedback else causal_note

        return self.score(
            ms,
            explanation=explanation,
            feedback=feedback or None,
            metadata=details,
        )

    def _cuda_available(self, sandbox: Path) -> bool:
        result = self._uv_run(
            sandbox,
            "python",
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            timeout=120,
        )
        return result.returncode == 0

    def _resolve_sandbox_root(self) -> Path | ScoreBundle:
        """Agent worktree, or parent repo when validate runs on an empty workspace."""
        root = Path(self.codebase_path).resolve()
        if (root / "pyproject.toml").exists() and (root / "cs336_systems").is_dir():
            return root

        cfg_yaml = Path(self.private_dir).parent / "config.yaml"
        config_dir_file = Path(self.private_dir).parent / "config_dir"
        if cfg_yaml.exists() and config_dir_file.exists():
            from coral.config import CoralConfig

            cfg = CoralConfig.from_yaml(cfg_yaml)
            task_dir = Path(config_dir_file.read_text().strip())
            candidate = (task_dir / cfg.workspace.repo_path).resolve()
            if (candidate / "pyproject.toml").exists():
                return candidate

        # `coral validate .` from cs336-leaderboard/: parent is the assignment repo.
        for candidate in (
            root.parent,
            Path.cwd().resolve().parent,
            Path.cwd().resolve(),
        ):
            if (candidate / "pyproject.toml").exists() and (candidate / "cs336_systems").is_dir():
                return candidate

        return self.fail(
            "Could not find the assignment repo (expected pyproject.toml + cs336_systems/). "
            "Set workspace.repo_path in task.yaml or run validate from cs336-leaderboard/."
        )

    def _uv_run(self, sandbox: Path, *cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return subprocess.run(
            ["uv", "run", "--project", str(sandbox), *cmd],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
            env=env,
        )

    def _run_correctness_tests(self, sandbox: Path) -> ScoreBundle | None:
        pytest_args = self.args.get(
            "correctness_pytest_args",
            ["tests/test_attention.py", "-k", "triton", "-q", "--tb=short"],
        )
        result = self._uv_run(sandbox, "pytest", *pytest_args, timeout=min(self.timeout or 600, 600))
        log_path = self.eval_logs_dir / "pytest_correctness.log"
        log_path.write_text(
            f"exit={result.returncode}\n\n=== stdout ===\n{result.stdout}\n\n=== stderr ===\n{result.stderr}"
        )

        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip()[-2000:]
            return self.fail(
                "Flash-attention correctness tests failed. "
                "Model outputs must match the reference implementation.\n"
                f"Log: {self.eval_logs_worktree_path(log_path)}\n{tail}",
                feedback=tail,
            )
        return None

    def _run_benchmark(self, sandbox: Path) -> tuple[float, dict] | ScoreBundle:
        script = self.args.get("benchmark_script", "scripts/naive_leaderboard_benchmark.py")
        script_path = sandbox / script
        if not script_path.exists():
            return self.fail(f"Benchmark script not found: {script}")

        cmd = [
            "python",
            str(script_path),
            "--device",
            str(self.args.get("device", "cuda:0")),
            "--attention",
            str(self.args.get("attention", "cute")),
            "--checkpoint-every",
            str(int(self.args.get("checkpoint_every", 1))),
        ]
        if self.args.get("quick", False):
            cmd.append("--quick")
        else:
            cmd.extend(
                [
                    "--warmup-ms",
                    str(int(self.args.get("warmup_ms", 10_000))),
                    "--rep-ms",
                    str(int(self.args.get("rep_ms", 30_000))),
                ]
            )
        if ctx_len := self.args.get("ctx_len"):
            cmd.extend(["--ctx-len", str(int(ctx_len))])
        if self.args.get("compile", False):
            cmd.append("--compile")
            if compile_mode := self.args.get("compile_mode"):
                cmd.extend(["--compile-mode", str(compile_mode)])

        json_path = self.eval_logs_dir / "benchmark_result.json"
        cmd.extend(["--output-json", str(json_path)])

        result = self._uv_run(sandbox, *cmd, timeout=self.timeout)
        log_path = self.eval_logs_dir / "benchmark.log"
        log_path.write_text(
            f"cmd={' '.join(cmd)}\nexit={result.returncode}\n\n"
            f"=== stdout ===\n{result.stdout}\n\n=== stderr ===\n{result.stderr}"
        )

        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip()[-2000:]
            oom_hint = ""
            if "out of memory" in tail.lower() or "oom" in tail.lower():
                oom_hint = " OOM at ctx=32768 usually means flash attention / checkpointing / sharding is missing."
            return self.fail(
                f"Benchmark failed (exit {result.returncode}).{oom_hint}\n"
                f"Log: {self.eval_logs_worktree_path(log_path)}\n{tail}",
                feedback=tail,
            )

        if not json_path.exists():
            return self.fail(
                "Benchmark did not write --output-json file.\n"
                f"Log: {self.eval_logs_worktree_path(log_path)}",
                feedback=(result.stdout + result.stderr)[-1500:],
            )

        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            return self.fail(
                f"Could not parse benchmark JSON file: {exc}\n"
                f"Log: {self.eval_logs_worktree_path(log_path)}",
                feedback=result.stdout[-1500:],
            )

        ms = float(payload["training_step_ms"])
        details = {
            "training_step_ms": ms,
            "training_step_s": payload.get("training_step_s", ms / 1000),
            "peak_memory_allocated_gib": payload.get("peak_memory_allocated_gib"),
            "peak_memory_reserved_gib": payload.get("peak_memory_reserved_gib"),
            "gpu_name": payload.get("gpu_name"),
            "attention": payload.get("attention"),
            "checkpoint_every": payload.get("checkpoint_every"),
            "compiled": payload.get("compiled"),
            "config": payload.get("config"),
            "benchmark_log": str(self.eval_logs_worktree_path(log_path)),
            "feedback": _format_timing_feedback(payload),
        }
        return ms, details


def _format_timing_feedback(payload: dict) -> str:
    ms = payload.get("training_step_ms")
    peak = payload.get("peak_memory_allocated_gib")
    gpu = payload.get("gpu_name", "unknown GPU")
    parts = [f"{ms:.1f} ms on {gpu}"]
    if peak is not None:
        parts.append(f"peak alloc {peak:.1f} GiB")
    if payload.get("attention"):
        parts.append(f"attention={payload['attention']}")
    if payload.get("checkpoint_every"):
        parts.append(f"checkpoint_every={payload['checkpoint_every']}")
    return "; ".join(parts)
