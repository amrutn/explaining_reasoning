#!/usr/bin/env python3
"""Remove bogus entries (empty response_text, token_count=0) from result files.

These entries are created when API calls fail (e.g. missing module, rate limit
errors) and the error handler saves placeholder results. The resume logic then
treats them as complete, preventing re-processing.

This script strips those entries so the next run of api_accuracy.py will
re-process only the affected (question, prompt_level, replicate) combinations.

Usage:
    python code/clean_results.py [--results-dir results/] [--dry-run]
"""

import argparse
import json
import os
from glob import glob


def is_bogus(entry):
    """An entry is bogus if it has empty response and zero tokens."""
    return entry.get("response_text", "") == "" and entry.get("token_count", 0) == 0


def clean_file(filepath, dry_run=False):
    with open(filepath, "r") as f:
        data = json.load(f)

    total_removed = 0
    empty_prompt_keys_removed = 0

    for question in data:
        results = question.get("results", {})
        for prompt_key in list(results.keys()):
            replicates = results[prompt_key]
            before = len(replicates)
            results[prompt_key] = [r for r in replicates if not is_bogus(r)]
            removed = before - len(results[prompt_key])
            total_removed += removed

            # Remove the prompt key entirely if no replicates remain
            if not results[prompt_key]:
                del results[prompt_key]
                empty_prompt_keys_removed += 1

    basename = os.path.basename(filepath)
    if total_removed == 0:
        print(f"  {basename}: clean (no bogus entries)")
    else:
        print(f"  {basename}: removed {total_removed} bogus entries "
              f"({empty_prompt_keys_removed} prompt keys now empty)")
        if not dry_run:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"    -> saved")

    return total_removed


def main():
    parser = argparse.ArgumentParser(description="Clean bogus entries from result files")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result JSON files (default: results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without modifying files")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.results_dir, "*.json")))
    if not files:
        print(f"No JSON files found in {args.results_dir}/")
        return

    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    total = 0
    for filepath in files:
        total += clean_file(filepath, dry_run=args.dry_run)

    print(f"\nTotal bogus entries {'found' if args.dry_run else 'removed'}: {total}")


if __name__ == "__main__":
    main()
