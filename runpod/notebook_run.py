from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_ROOT = REPO_ROOT / "telemetry"

TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+)\s+train_loss:(?P<train_loss>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"(?P<name>final_[^ ]+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
SIZE_RE = re.compile(r"Total submission size [^:]+:\s*(?P<size>\d+)\s+bytes")
CODE_SIZE_RE = re.compile(r"Code size:\s*(?P<size>\d+)\s+bytes")


@dataclass(frozen=True)
class RunProfile:
    name: str
    script: str
    run_id: str
    train_shards: int
    env: dict[str, str]
    nproc_per_node: int = 1


PROFILES: dict[str, list[RunProfile]] = {
    "control-smoke": [
        RunProfile(
            name="control-smoke",
            script="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
            run_id="control_smoke_pr1019",
            train_shards=1,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "MAX_WALLCLOCK_SECONDS": "600",
                "VAL_LOSS_EVERY": "2000",
            },
        )
    ],
    "branch-smoke": [
        RunProfile(
            name="branch-smoke",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_smoke_sp1024",
            train_shards=1,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "MAX_WALLCLOCK_SECONDS": "600",
                "VAL_LOSS_EVERY": "2000",
            },
        )
    ],
    "control-screen": [
        RunProfile(
            name="control-screen",
            script="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
            run_id="control_screen_pr1019",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "MAX_WALLCLOCK_SECONDS": "4800",
                "VAL_LOSS_EVERY": "4000",
            },
        )
    ],
    "branch-screen": [
        RunProfile(
            name="branch-screen",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_screen_sp1024",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "PARALLEL_START_LAYER": "7",
                "RECUR_LAYERS": "4,5",
                "RECUR_UNTIE_MLP": "1",
                "QK_GAIN_INIT": "5.0",
                "TRAIN_LOG_EVERY": "100",
                "MAX_WALLCLOCK_SECONDS": "4800",
                "VAL_LOSS_EVERY": "1000",
            },
        )
    ],
    "branch-screen-4k": [
        RunProfile(
            name="branch-screen-4k",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_screen_sp4096",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
                "VOCAB_SIZE": "4096",
                "MATCHED_FINEWEB_REPO_ID": "kevclark/parameter-golf",
                "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX": "datasets",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "PARALLEL_START_LAYER": "7",
                "RECUR_LAYERS": "4,5",
                "RECUR_UNTIE_MLP": "1",
                "QK_GAIN_INIT": "5.0",
                "TRAIN_LOG_EVERY": "100",
                "VAL_LOSS_EVERY": "1000",
                "MAX_WALLCLOCK_SECONDS": "4800",
            },
        )
    ],
    "branch-final-4k-8x": [
        RunProfile(
            name="branch-final-4k-8x",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_final_sp4096_8xh100",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp4096",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
                "VOCAB_SIZE": "4096",
                "MATCHED_FINEWEB_REPO_ID": "kevclark/parameter-golf",
                "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX": "datasets",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "PARALLEL_START_LAYER": "7",
                "RECUR_LAYERS": "4,5",
                "RECUR_UNTIE_MLP": "1",
                "QK_GAIN_INIT": "5.0",
                "TRAIN_LOG_EVERY": "100",
                "VAL_LOSS_EVERY": "2000",
                "MAX_WALLCLOCK_SECONDS": "600",
            },
            nproc_per_node=8,
        )
    ],
    "branch-final-8x": [
        RunProfile(
            name="branch-final-8x",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_final_sp1024_8xh100",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "PARALLEL_START_LAYER": "7",
                "RECUR_LAYERS": "4,5",
                "RECUR_UNTIE_MLP": "1",
                "QK_GAIN_INIT": "5.0",
                "TRAIN_LOG_EVERY": "100",
                "VAL_LOSS_EVERY": "2000",
                "MAX_WALLCLOCK_SECONDS": "600",
            },
            nproc_per_node=8,
        )
    ],
    "first-smoke": [
        RunProfile(
            name="control-smoke",
            script="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
            run_id="control_smoke_pr1019",
            train_shards=1,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "MAX_WALLCLOCK_SECONDS": "600",
                "VAL_LOSS_EVERY": "2000",
            },
        ),
        RunProfile(
            name="branch-smoke",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_smoke_sp1024",
            train_shards=1,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "MAX_WALLCLOCK_SECONDS": "600",
                "VAL_LOSS_EVERY": "2000",
            },
        ),
    ],
    "first-screen": [
        RunProfile(
            name="control-screen",
            script="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
            run_id="control_screen_pr1019",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "MAX_WALLCLOCK_SECONDS": "4800",
                "VAL_LOSS_EVERY": "4000",
            },
        ),
        RunProfile(
            name="branch-screen",
            script="working/current_frontier/train_gpt.py",
            run_id="frontier_branch_screen_sp1024",
            train_shards=80,
            env={
                "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
                "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
                "VOCAB_SIZE": "1024",
                "BIGRAM_VOCAB_SIZE": "0",
                "SMEAR_ENABLED": "0",
                "MAX_WALLCLOCK_SECONDS": "4800",
                "VAL_LOSS_EVERY": "4000",
            },
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command notebook launcher for RunPod")
    parser.add_argument(
        "--profile",
        default="frontier-auto",
        choices=sorted(list(PROFILES.keys()) + ["frontier-auto", "frontier-final-auto"]),
        help="Which run profile or pipeline to execute.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip auto-install of optional telemetry packages.",
    )
    return parser.parse_args()


def ensure_optional_package(module_name: str, pip_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            cwd=REPO_ROOT,
            check=False,
        )
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def get_console():
    try:
        from rich.console import Console
        from rich.theme import Theme

        theme = Theme(
            {
                "info": "cyan",
                "ok": "green",
                "warn": "yellow",
                "err": "bold red",
                "metric": "magenta",
            }
        )
        return Console(theme=theme), True
    except Exception:
        class PlainConsole:
            def print(self, *args: Any, **kwargs: Any) -> None:
                print(*args)

        return PlainConsole(), False


class Telemetry:
    def __init__(self, run_id: str, console: Any, plots_enabled: bool) -> None:
        self.run_id = run_id
        self.console = console
        self.plots_enabled = plots_enabled
        self.root = TELEMETRY_ROOT / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {"run_id": run_id}
        self.csv_path = self.root / "metrics.csv"
        self.jsonl_path = self.root / "metrics.jsonl"
        self.summary_path = self.root / "summary.json"
        self.plot_path = self.root / "metrics.png"

    def add_row(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row["timestamp"] = time.time()
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._write_csv()
        self._write_plots()

    def update_summary(self, **kwargs: Any) -> None:
        self.summary.update(kwargs)
        self.summary_path.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")

    def _write_csv(self) -> None:
        keys: list[str] = []
        for row in self.rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_plots(self) -> None:
        if not self.plots_enabled:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        train_rows = [r for r in self.rows if r.get("kind") == "train"]
        val_rows = [r for r in self.rows if r.get("kind") == "val"]
        final_rows = [r for r in self.rows if r.get("kind") == "final"]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        if train_rows:
            axes[0].plot([r["step"] for r in train_rows], [r["train_loss"] for r in train_rows], marker="o")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        if val_rows:
            axes[1].plot([r["step"] for r in val_rows], [r["val_bpb"] for r in val_rows], marker="o", label="val")
        if final_rows:
            axes[1].scatter(
                [r.get("step", 0) for r in final_rows],
                [r["val_bpb"] for r in final_rows],
                label="final",
            )
        axes[1].set_title("Validation BPB")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("BPB")
        axes[1].grid(True, alpha=0.3)
        if val_rows or final_rows:
            axes[1].legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=140)
        plt.close(fig)


def print_status(console: Any, message: str, style: str = "info") -> None:
    try:
        console.print(f"[{style}]{message}[/{style}]")
    except Exception:
        console.print(message)


def run_checked(command: list[str], *, env: dict[str, str] | None = None, console: Any) -> None:
    print_status(console, "$ " + " ".join(command), "info")
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def ensure_data(profile: RunProfile, console: Any) -> None:
    dataset_dir = REPO_ROOT / Path(profile.env["DATA_PATH"]).relative_to(".")
    train_shards = profile.train_shards
    existing = len(list(dataset_dir.glob("fineweb_train_*.bin"))) if dataset_dir.is_dir() else 0
    if existing >= train_shards:
        print_status(console, f"dataset already has {existing} train shards; need {train_shards}", "ok")
        return
    variant = f"sp{profile.env['VOCAB_SIZE']}"
    data_env = os.environ.copy()
    data_env.update({k: v for k, v in profile.env.items() if k.startswith("MATCHED_FINEWEB_")})
    if profile.env["VOCAB_SIZE"] != "1024":
        repo_id = profile.env.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
        print_status(console, f"downloading {variant} from {repo_id}", "info")
    run_checked(
        [
            sys.executable,
            "data/cached_challenge_fineweb.py",
            "--variant",
            variant,
            "--train-shards",
            str(train_shards),
        ],
        env=data_env,
        console=console,
    )


def _has_local_sp4096() -> bool:
    dataset_dir = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp4096"
    tokenizer = REPO_ROOT / "data" / "tokenizers" / "fineweb_4096_bpe.model"
    return dataset_dir.is_dir() and tokenizer.is_file() and any(dataset_dir.glob("fineweb_train_*.bin"))


def visible_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return len(lines)
    except Exception:
        return 0


def resolve_profiles(profile_name: str, console: Any) -> list[RunProfile]:
    if profile_name == "frontier-auto":
        gpu_count = visible_gpu_count()
        print_status(console, f"frontier-auto: {gpu_count} visible GPU(s); using sp4096 branch-screen-4k", "ok")
        return PROFILES["branch-screen-4k"]
    if profile_name == "frontier-final-auto":
        gpu_count = visible_gpu_count()
        if gpu_count >= 8:
            print_status(console, "frontier-final-auto: 8+ visible GPUs; using sp4096 branch-final-4k-8x", "ok")
            return PROFILES["branch-final-4k-8x"]
        print_status(
            console,
            f"frontier-final-auto: only {gpu_count} visible GPU(s); falling back to sp4096 branch-screen-4k",
            "warn",
        )
        return PROFILES["branch-screen-4k"]
    return PROFILES[profile_name]


def run_preflight(profile: RunProfile, console: Any) -> None:
    run_checked(
        [
            sys.executable,
            "runpod/preflight.py",
            "--script",
            profile.script,
            "--data-path",
            profile.env["DATA_PATH"],
            "--tokenizer-path",
            profile.env["TOKENIZER_PATH"],
            "--vocab-size",
            profile.env["VOCAB_SIZE"],
            "--min-train-shards",
            str(profile.train_shards),
            "--min-val-shards",
            "1",
            "--min-gpus",
            str(profile.nproc_per_node),
        ],
        console=console,
    )


def consume_training_output(profile: RunProfile, console: Any) -> int:
    env = os.environ.copy()
    env.update(profile.env)
    env["RUN_ID"] = profile.run_id
    env["PYTHONUNBUFFERED"] = "1"
    env["FORCE_COLOR"] = "1"

    telemetry = Telemetry(
        run_id=profile.run_id,
        console=console,
        plots_enabled=ensure_optional_package("matplotlib", "matplotlib"),
    )
    telemetry.update_summary(profile=profile.name, script=profile.script)

    command = ["torchrun", "--standalone", f"--nproc_per_node={profile.nproc_per_node}", profile.script]
    print_status(console, "$ " + " ".join(command), "metric")

    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue

        style = "info"
        if "ERROR" in line or "Traceback" in line:
            style = "err"
        elif "final_" in line or "Total submission size" in line:
            style = "ok"
        elif "val_bpb" in line or "train_loss" in line:
            style = "metric"
        print_status(console, line, style)

        if match := TRAIN_RE.search(line):
            row = {
                "kind": "train",
                "step": int(match.group("step")),
                "iters": int(match.group("iters")),
                "train_loss": float(match.group("train_loss")),
                "train_time_ms": float(match.group("train_time_ms")),
            }
            telemetry.add_row(row)
            telemetry.update_summary(last_train=row)
            continue

        if match := VAL_RE.search(line):
            row = {
                "kind": "val",
                "step": int(match.group("step")),
                "iters": int(match.group("iters")),
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
                "train_time_ms": float(match.group("train_time_ms")),
            }
            telemetry.add_row(row)
            telemetry.update_summary(last_val=row)
            continue

        if match := FINAL_RE.search(line):
            row = {
                "kind": "final",
                "name": match.group("name"),
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
            }
            telemetry.add_row(row)
            telemetry.update_summary(**{match.group("name"): row})
            continue

        if match := SIZE_RE.search(line):
            telemetry.update_summary(total_submission_bytes=int(match.group("size")))
            continue

        if match := CODE_SIZE_RE.search(line):
            telemetry.update_summary(code_bytes=int(match.group("size")))
            continue

    proc.wait()
    telemetry.update_summary(return_code=proc.returncode)
    print_status(console, f"telemetry saved to {telemetry.root}", "ok" if proc.returncode == 0 else "warn")
    if telemetry.plot_path.exists():
        print_status(console, f"plot saved to {telemetry.plot_path}", "ok")
    return proc.returncode


def main() -> None:
    args = parse_args()

    if not args.skip_install:
        ensure_optional_package("rich", "rich")

    console, _ = get_console()

    profiles = resolve_profiles(args.profile, console)
    print_status(console, f"profile={args.profile}", "ok")

    for profile in profiles:
        print_status(console, f"starting {profile.name}", "ok")
        ensure_data(profile, console)
        run_preflight(profile, console)
        return_code = consume_training_output(profile, console)
        if return_code != 0:
            raise SystemExit(return_code)

    print_status(console, "all runs completed", "ok")


if __name__ == "__main__":
    main()
