"""
experiments/efficiency.py
--------------------------
Efficiency test on CA-GrQc (small dataset).

Reports:
  - Wall-clock time per communication round
  - Total training time vs. number of rounds
  - Time breakdown: local training vs aggregation

Saves results to results/efficiency_results.json
        figures  to results/efficiency_timing.png
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


# ── Configuration ───────────────────────────────────────────────────────
DATA_PATH   = os.path.join(_HERE, "../data/CA-GrQc.txt")
RESULTS_DIR = os.path.join(_HERE, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CLASSES    = 5
N_CLIENTS    = 10
LR           = 0.01
BATCH_SIZE   = 32
HIDDEN_DIM   = 64
SEED         = 42


def run_efficiency():
    print("\n" + "="*60)
    print("  EFFICIENCY EXPERIMENT  —  CA-GrQc (small dataset)")
    print("="*60)

    # ── Prepare data once ────────────────────────────────────────────────
    _, nodes, X, y, clients_iid = prepare_dataset(
        DATA_PATH, n_classes=N_CLASSES, n_clients=N_CLIENTS,
        partition="iid", label_method="cores", seed=SEED)

    # ── Vary number of rounds ─────────────────────────────────────────────
    round_configs = [10, 20, 30, 50]
    round_timing = {}

    for n_rounds in round_configs:
        print(f"\n  Running {n_rounds} rounds …")
        t0 = time.time()
        res = run_fedavg(X, y, clients_iid,
                         n_rounds=n_rounds,
                         local_epochs=5,
                         fraction_c=1.0,
                         lr=LR,
                         batch_size=BATCH_SIZE,
                         hidden_dim=HIDDEN_DIM,
                         seed=SEED,
                         verbose=False)
        wall = time.time() - t0
        per_round = [log["round_time_s"] for log in res["round_logs"]]
        round_timing[n_rounds] = {
            "total_s": round(wall, 3),
            "avg_per_round_s": round(float(np.mean(per_round)), 4),
            "min_round_s": round(float(np.min(per_round)), 4),
            "max_round_s": round(float(np.max(per_round)), 4),
        }
        print(f"    total={wall:.2f}s  avg/round={np.mean(per_round):.4f}s")

    # ── Vary local epochs ─────────────────────────────────────────────────
    epoch_configs = [1, 2, 5, 10, 20]
    epoch_timing = {}

    for e in epoch_configs:
        print(f"\n  Running E={e} local epochs …")
        t0 = time.time()
        res = run_fedavg(X, y, clients_iid,
                         n_rounds=20,
                         local_epochs=e,
                         fraction_c=1.0,
                         lr=LR,
                         batch_size=BATCH_SIZE,
                         hidden_dim=HIDDEN_DIM,
                         seed=SEED,
                         verbose=False)
        wall = time.time() - t0
        per_round = [log["round_time_s"] for log in res["round_logs"]]
        epoch_timing[e] = {
            "total_s": round(wall, 3),
            "avg_per_round_s": round(float(np.mean(per_round)), 4),
            "macro_f1": round(res["macro_f1"], 4),
            "micro_f1": round(res["micro_f1"], 4),
        }
        print(f"    total={wall:.2f}s  macro-F1={res['macro_f1']:.4f}")

    # ── Vary number of clients ─────────────────────────────────────────────
    client_configs = [2, 5, 10, 20]
    client_timing = {}

    for k in client_configs:
        print(f"\n  Running K={k} clients …")
        _, _, X_k, y_k, clients_k = prepare_dataset(
            DATA_PATH, n_classes=N_CLASSES, n_clients=k,
            partition="iid", label_method="cores", seed=SEED)
        t0 = time.time()
        res = run_fedavg(X_k, y_k, clients_k,
                         n_rounds=20,
                         local_epochs=5,
                         fraction_c=1.0,
                         lr=LR,
                         batch_size=BATCH_SIZE,
                         hidden_dim=HIDDEN_DIM,
                         seed=SEED,
                         verbose=False)
        wall = time.time() - t0
        per_round = [log["round_time_s"] for log in res["round_logs"]]
        client_timing[k] = {
            "total_s": round(wall, 3),
            "avg_per_round_s": round(float(np.mean(per_round)), 4),
            "macro_f1": round(res["macro_f1"], 4),
        }
        print(f"    total={wall:.2f}s  macro-F1={res['macro_f1']:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "vary_rounds": round_timing,
        "vary_local_epochs": epoch_timing,
        "vary_n_clients": client_timing,
    }
    out_path = os.path.join(RESULTS_DIR, "efficiency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FedAvg Efficiency — CA-GrQc", fontsize=13)

    # 1. Total time vs. rounds
    ax = axes[0]
    rk = sorted(round_timing.keys())
    ax.plot(rk, [round_timing[r]["total_s"] for r in rk],
            marker="o", color="steelblue")
    ax.set_xlabel("Number of Rounds")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Total Time vs. Rounds")
    ax.grid(True, alpha=0.3)

    # 2. Avg round time vs. local epochs
    ax = axes[1]
    ek = sorted(epoch_timing.keys())
    ax.plot(ek, [epoch_timing[e]["avg_per_round_s"] for e in ek],
            marker="s", color="darkorange")
    ax.set_xlabel("Local Epochs (E)")
    ax.set_ylabel("Avg Round Time (s)")
    ax.set_title("Round Time vs. Local Epochs")
    ax.grid(True, alpha=0.3)

    # 3. Total time vs. number of clients
    ax = axes[2]
    ck = sorted(client_timing.keys())
    ax.plot(ck, [client_timing[c]["total_s"] for c in ck],
            marker="^", color="seagreen")
    ax.set_xlabel("Number of Clients (K)")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Total Time vs. Clients")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "efficiency_timing.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")

    return results


if __name__ == "__main__":
    run_efficiency()
