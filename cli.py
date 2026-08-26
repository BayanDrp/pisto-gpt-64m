from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_SCRIPT = ROOT / "ui" / "pisto-server"  # Go web server (built from ui/main.go)
PRETRAIN_SCRIPT = ROOT / "training" / "pretrain.py"
FINE_TUNE_SCRIPT = ROOT / "training" / "finetune_hf.py"       # AraGPT2 finetune (primary)
FINE_TUNE_LEGACY_SCRIPT = ROOT / "training" / "finetune.py"  # legacy 68M model


def prompt_choice(title: str, options: dict[str, str]) -> str:
    print(f"\n{title}")
    for key, label in options.items():
        print(f"  {key}) {label}")

    while True:
        choice = input("Choose an option: ").strip().lower()
        if choice in options:
            return choice
        print("Invalid choice, try again.")


def run_script(script_path: Path) -> int:
    if not script_path.exists():
        print(f"Missing script: {script_path}")
        return 1

    print(f"\nRunning: {script_path.relative_to(ROOT)}\n")
    # Compiled binaries (e.g. the Go server) are executed directly;
    # Python scripts are run with the current interpreter.
    if script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path)]
    else:
        cmd = [str(script_path)]
    result = subprocess.run(cmd)
    return result.returncode


def run_command(command: str | None, mode: str | None) -> int | None:
    if command == "app":
        return run_script(APP_SCRIPT)

    if command == "train":
        if mode == "pretrain":
            return run_script(PRETRAIN_SCRIPT)
        if mode in {"finetune", "fine-tune", "fine_tune"}:
            return run_script(FINE_TUNE_SCRIPT)
        if mode in {"finetune-legacy", "finetune_legacy", "legacy"}:
            return run_script(FINE_TUNE_LEGACY_SCRIPT)
        raise SystemExit("Use: python cli.py train pretrain|finetune|finetune-legacy")

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("command", nargs="?", choices=["app", "train"])
    parser.add_argument("mode", nargs="?")
    args = parser.parse_args(argv)

    direct = run_command(args.command, args.mode)
    if direct is not None:
        return direct

    choice = prompt_choice(
        "Main menu",
        {
            "1": "Launch web app (Go server)",
            "2": "Train",
        },
    )

    if choice == "1":
        return run_script(APP_SCRIPT)

    train_choice = prompt_choice(
        "Training mode",
        {
            "1": "Pretrain (68M from scratch)",
            "2": "Fine-tune AraGPT2 (HuggingFace)",
            "3": "Fine-tune legacy 68M model",
        },
    )

    if train_choice == "1":
        return run_script(PRETRAIN_SCRIPT)
    if train_choice == "2":
        return run_script(FINE_TUNE_SCRIPT)
    return run_script(FINE_TUNE_LEGACY_SCRIPT)


if __name__ == "__main__":
    raise SystemExit(main())
