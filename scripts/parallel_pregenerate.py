#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel semantic patch cache pregeneration script (Python version).

Usage:
    python scripts/parallel_pregenerate.py --num_gpus 8

Notes:
    - Automatically launches pregeneration jobs on multiple GPUs in parallel.
    - Each GPU processes a shard of the dataset.
    - All cache files are written to the same cache directory.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Parallel semantic patch cache pregeneration")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/train_lora/qwen2_5vl_lora_sft.yaml",
        help="Training config file path",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Dataset directory",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default="data/dataset_info.json",
        help="Path to dataset_info.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Parallel semantic patch cache pregeneration")
    print("=" * 60)
    print(f"GPU count: {args.num_gpus}")
    print(f"Config file: {args.config}")
    print(f"Dataset directory: {args.dataset_dir}")
    print("=" * 60)
    print()

    # Create log directory
    log_dir = Path("logs/pregenerate_cache")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Starting parallel cache pregeneration...")
    print()

    # Launch worker processes
    processes = []
    for gpu_id in range(args.num_gpus):
        log_file = log_dir / f"gpu_{gpu_id}.log"
        print(f"[GPU {gpu_id}] Starting pregeneration task, log: {log_file}")

        # Build command
        cmd = [
            "python",
            "scripts/pregenerate_semantic_cache.py",
            "--config", args.config,
            "--dataset_dir", args.dataset_dir,
            "--dataset_info", args.dataset_info,
            "--gpu_id", str(gpu_id),
            "--num_gpus", str(args.num_gpus),
        ]

        # Launch process
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            processes.append((gpu_id, proc, log_file))
            print(f"[GPU {gpu_id}] PID: {proc.pid}")

    print()
    print("All pregeneration tasks have started and are running in the background...")
    print(f"Monitor progress: tail -f {log_dir}/gpu_*.log")
    print("Waiting for all tasks to finish...")
    print()

    # Monitor process status
    try:
        while True:
            all_done = True
            for gpu_id, proc, _ in processes:
                if proc.poll() is None:
                    all_done = False
                    break

            if all_done:
                break

            # Check every 30 seconds
            time.sleep(30)

    except KeyboardInterrupt:
        print("\nInterrupt detected, terminating all tasks...")
        for gpu_id, proc, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        print("All tasks terminated")
        return

    # Wait for all processes to finish
    for gpu_id, proc, _ in processes:
        proc.wait()

    print()
    print("=" * 60)
    print("All pregeneration tasks completed!")
    print("=" * 60)
    print()

    # Show per-GPU summary
    print("Per-GPU summary:")
    for gpu_id, proc, log_file in processes:
        print()
        print(f"=== GPU {gpu_id} ===")
        if log_file.exists():
            # Read the last lines of the log file
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Print lines containing completion or summary keywords
                for line in lines[-20:]:
                    if any(keyword in line for keyword in ["Success", "Failed", "Completed", "images"]):
                        print(line.strip())
        else:
            print(f"Log file not found: {log_file}")

    print()
    print("Cache files saved to: src/semantic_patch_cache/")
    print(f"You can now start training: llamafactory-cli train {args.config}")


if __name__ == "__main__":
    main()
