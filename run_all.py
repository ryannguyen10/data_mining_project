"""
run_all.py
----------
Master script: runs effectiveness, efficiency, and (optionally) scalability
experiments in sequence.

Usage:
    python run_all.py                  # effectiveness + efficiency only
    python run_all.py --scalability    # all three experiments
"""

import sys
import os
import json
import argparse
import time

# Make sure src/ is on the path before any experiment module is imported
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def print_banner(text):
    print("\n" + "█" * 60)
    print(f"  {text}")
    print("█" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="FedAvg project — run all experiments")
    parser.add_argument("--scalability", action="store_true",
                        help="Also run scalability test on com-dblp")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    t_start = time.time()

    # ── Check data files ─────────────────────────────────────────────────
    data_dir = os.path.join(ROOT, "data")
    small_path = os.path.join(data_dir, "CA-GrQc.txt")
    large_path = os.path.join(data_dir, "com-dblp.ungraph.txt")

    if not os.path.exists(small_path):
        print(f"\n[ERROR] Small dataset not found: {small_path}")
        print("  → Place CA-GrQc.txt in the data/ directory.")
        sys.exit(1)

    if args.scalability and not os.path.exists(large_path):
        print(f"\n[ERROR] Large dataset not found: {large_path}")
        print("  → Place com-dblp.ungraph.txt in the data/ directory.")
        sys.exit(1)

    # ── Experiment 1: Effectiveness ───────────────────────────────────────
    print_banner("EXPERIMENT 1: EFFECTIVENESS  (CA-GrQc)")
    from effectiveness import run_effectiveness
    eff_results = run_effectiveness()

    # ── Experiment 2: Efficiency ──────────────────────────────────────────
    print_banner("EXPERIMENT 2: EFFICIENCY  (CA-GrQc)")
    from efficiency import run_efficiency
    timing_results = run_efficiency()

    # ── Experiment 3: Scalability (optional) ─────────────────────────────
    if args.scalability:
        print_banner("EXPERIMENT 3: SCALABILITY  (com-dblp)")
        from scalability import run_scalability
        scale_results = run_scalability()
    else:
        print("\n[skip] Scalability experiment skipped.")
        print("       Re-run with --scalability to include it.")
        scale_results = None

    # ── Combined summary ──────────────────────────────────────────────────
    total_time = time.time() - t_start
    combined = {
        "effectiveness": eff_results,
        "efficiency_summary": {
            k: v for k, v in timing_results.get("vary_rounds", {}).items()
        },
        "scalability": scale_results,
        "total_experiment_time_s": round(total_time, 2),
    }
    summary_path = os.path.join(results_dir, "all_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All experiments complete in {total_time:.1f}s")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Results directory: {results_dir}/")
    print(f"{'='*60}")

    # ── Print final table ─────────────────────────────────────────────────
    print("\n  ┌─────────────────────┬────────────┬────────────┬──────────────┐")
    print("  │ Method              │  Macro-F1  │  Micro-F1  │ Hamming Loss │")
    print("  ├─────────────────────┼────────────┼────────────┼──────────────┤")
    for name, data in eff_results.items():
        macro = f"{data['macro_f1']:.4f}"
        micro = f"{data['micro_f1']:.4f}"
        hamm  = f"{data['hamming_loss']:.4f}"
        print(f"  │ {name:<19} │ {macro:>10} │ {micro:>10} │ {hamm:>12} │")
    print("  └─────────────────────┴────────────┴────────────┴──────────────┘")


if __name__ == "__main__":
    main()
