"""
experiments/scalability.py
---------------------------
Scalability test on com-dblp (large dataset, ~317k nodes).

Reports:
  - Macro-F1, Micro-F1, Hamming Loss on the large graph
  - Total wall-clock time
  - Running time per round

Feature extraction uses a random sample of 50k nodes
for clustering coefficient computation (too expensive on full graph).

Saves results to results/scalability_results.json
        figures  to results/scalability_timing.png
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from preprocess import prepare_dataset
from fedavg import run_fedavg
from evaluate import print_report


# ── Configuration ───────────────────────────────────────────────────────
DATA_PATH   = os.path.join(_HERE, "../data/com-dblp.ungraph.txt")
RESULTS_DIR = os.path.join(_HERE, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CLASSES    = 5
N_CLIENTS    = 20
N_ROUNDS     = 20      # fewer rounds — large graph
LOCAL_EPOCHS = 3
LR           = 0.01
BATCH_SIZE   = 64
HIDDEN_DIM   = 64
SEED         = 42
SAMPLE_SIZE  = 50_000  # for clustering coeff approximation


def run_scalability():
    print("\n" + "="*60)
    print("  SCALABILITY EXPERIMENT  —  com-dblp (large dataset)")
    print("="*60)

    # ── Prepare data ─────────────────────────────────────────────────────
    t_prep = time.time()
    _, nodes, X, y, clients_iid = prepare_dataset(
        DATA_PATH,
        n_classes=N_CLASSES,
        n_clients=N_CLIENTS,
        partition="iid",
        label_method="cores",
        sample_size=SAMPLE_SIZE,
        seed=SEED)
    prep_time = time.time() - t_prep
    print(f"\n  Data preparation time: {prep_time:.2f}s")
    print(f"  Dataset size: {len(nodes):,} nodes")

    # ── IID run ───────────────────────────────────────────────────────────
    print("\n  Running FedAvg IID on com-dblp …")
    res_iid = run_fedavg(
        X, y, clients_iid,
        n_rounds=N_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        fraction_c=0.5,    # only 50% of clients per round for scalability
        lr=LR,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        seed=SEED,
        verbose=True)

    print_report(res_iid["y_true"], res_iid["y_pred"],
                 N_CLASSES, title="Scalability Test (com-dblp, IID)")

    # ── Non-IID run ────────────────────────────────────────────────────────
    print("\n  Running FedAvg non-IID on com-dblp …")
    _, nodes2, X2, y2, clients_noniid = prepare_dataset(
        DATA_PATH,
        n_classes=N_CLASSES,
        n_clients=N_CLIENTS,
        partition="noniid",
        label_method="cores",
        sample_size=SAMPLE_SIZE,
        seed=SEED)

    res_noniid = run_fedavg(
        X2, y2, clients_noniid,
        n_rounds=N_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        fraction_c=0.5,
        lr=LR,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        seed=SEED,
        verbose=True)

    print_report(res_noniid["y_true"], res_noniid["y_pred"],
                 N_CLASSES, title="Scalability Test (com-dblp, non-IID)")

    # ── Save results ──────────────────────────────────────────────────────
    summary = {
        "dataset": "com-dblp",
        "n_nodes": len(nodes),
        "prep_time_s": round(prep_time, 2),
        "FedAvg_IID": {
            "macro_f1":     round(res_iid["macro_f1"], 4),
            "micro_f1":     round(res_iid["micro_f1"], 4),
            "hamming_loss": round(res_iid["hamming_loss"], 4),
            "total_time_s": round(res_iid["total_time_s"], 2),
        },
        "FedAvg_nonIID": {
            "macro_f1":     round(res_noniid["macro_f1"], 4),
            "micro_f1":     round(res_noniid["micro_f1"], 4),
            "hamming_loss": round(res_noniid["hamming_loss"], 4),
            "total_time_s": round(res_noniid["total_time_s"], 2),
        },
    }
    out_path = os.path.join(RESULTS_DIR, "scalability_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")

    # ── Plot per-round timing ─────────────────────────────────────────────
    def get_round_times(res):
        return [log["round"] for log in res["round_logs"]], \
               [log["round_time_s"] for log in res["round_logs"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("FedAvg Scalability — com-dblp (~317k nodes)", fontsize=13)

    for ax, res, title in zip(
            axes,
            [res_iid, res_noniid],
            ["IID Partition", "non-IID Partition"]):
        rounds, times = get_round_times(res)
        macro_rounds = [l["round"] for l in res["round_logs"] if "macro_f1" in l]
        macro_vals   = [l["macro_f1"] for l in res["round_logs"] if "macro_f1" in l]

        ax2 = ax.twinx()
        ax.bar(rounds, times, color="steelblue", alpha=0.6, label="Round time (s)")
        ax2.plot(macro_rounds, macro_vals, color="darkorange",
                 marker="o", ms=4, label="Macro-F1")
        ax.set_xlabel("Round")
        ax.set_ylabel("Round Time (s)", color="steelblue")
        ax2.set_ylabel("Macro-F1", color="darkorange")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "scalability_timing.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")

    return summary


if __name__ == "__main__":
    run_scalability()
