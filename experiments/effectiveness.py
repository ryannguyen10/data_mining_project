"""
experiments/effectiveness.py
-----------------------------
Effectiveness test on CA-GrQc (small dataset).

Reports:
  - Macro-F1
  - Micro-F1
  - Hamming Loss

Compares:
  - FedAvg (IID partition)
  - FedAvg (non-IID partition)
  - Centralised baseline (all data, no federation)

Saves results to results/effectiveness_results.json and
            figures to results/effectiveness_convergence.png
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json
import numpy as np
import matplotlib.pyplot as plt

from preprocess import prepare_dataset
from fedavg import run_fedavg
from evaluate import print_report
from model import MLP


# ── Configuration ───────────────────────────────────────────────────────
DATA_PATH   = os.path.join(_HERE, "../data/CA-GrQc.txt")
RESULTS_DIR = os.path.join(_HERE, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CLASSES     = 5
N_CLIENTS     = 10
N_ROUNDS      = 50
LOCAL_EPOCHS  = 5
LR            = 0.01
BATCH_SIZE    = 32
HIDDEN_DIM    = 64
SEED          = 42


def centralised_baseline(X, y, seed=42):
    """Train a single MLP on all training data (no federation)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    stratify_arg = y if np.bincount(y).min() >= 2 else None
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                      stratify=stratify_arg, random_state=seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    model = MLP(input_dim=X.shape[1], n_classes=N_CLASSES,
                hidden_dim=HIDDEN_DIM, seed=seed)
    n_epochs = N_ROUNDS * LOCAL_EPOCHS     # same total gradient steps
    batch_size = BATCH_SIZE
    lr = LR

    rng = np.random.default_rng(seed)
    for _ in range(n_epochs):
        perm = rng.permutation(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            b_idx = perm[start: start + batch_size]
            model.train_step(X_tr[b_idx], y_tr[b_idx], lr)

    y_pred = model.predict(X_te)
    from evaluate import evaluate_classification
    return evaluate_classification(y_te, y_pred, n_classes=N_CLASSES)


def run_effectiveness():
    print("\n" + "="*60)
    print("  EFFECTIVENESS EXPERIMENT  —  CA-GrQc (small dataset)")
    print("="*60)

    # ── IID experiment ───────────────────────────────────────────────────
    print("\n[1/3] FedAvg — IID partition")
    _, nodes_iid, X_iid, y_iid, clients_iid = prepare_dataset(
        DATA_PATH, n_classes=N_CLASSES, n_clients=N_CLIENTS,
        partition="iid", label_method="cores", seed=SEED)

    res_iid = run_fedavg(
        X_iid, y_iid, clients_iid,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        fraction_c=1.0, lr=LR, batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM, seed=SEED, verbose=True)

    print_report(res_iid["y_true"], res_iid["y_pred"],
                 N_CLASSES, title="FedAvg IID")

    # ── Non-IID experiment ───────────────────────────────────────────────
    print("\n[2/3] FedAvg — non-IID partition (Dirichlet α=0.5)")
    _, nodes_noniid, X_noniid, y_noniid, clients_noniid = prepare_dataset(
        DATA_PATH, n_classes=N_CLASSES, n_clients=N_CLIENTS,
        partition="noniid", label_method="cores", seed=SEED)

    res_noniid = run_fedavg(
        X_noniid, y_noniid, clients_noniid,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        fraction_c=1.0, lr=LR, batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM, seed=SEED, verbose=True)

    print_report(res_noniid["y_true"], res_noniid["y_pred"],
                 N_CLASSES, title="FedAvg non-IID")

    # ── Centralised baseline ─────────────────────────────────────────────
    print("\n[3/3] Centralised baseline")
    res_central = centralised_baseline(X_iid, y_iid, seed=SEED)
    print(f"  Macro-F1    : {res_central['macro_f1']:.4f}")
    print(f"  Micro-F1    : {res_central['micro_f1']:.4f}")
    print(f"  Hamming Loss: {res_central['hamming_loss']:.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    summary = {
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
        "Centralised": {
            "macro_f1":     round(res_central["macro_f1"], 4),
            "micro_f1":     round(res_central["micro_f1"], 4),
            "hamming_loss": round(res_central["hamming_loss"], 4),
        },
    }
    out_path = os.path.join(RESULTS_DIR, "effectiveness_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")

    # ── Plot convergence ─────────────────────────────────────────────────
    def extract_metric(logs, key):
        rounds, vals = [], []
        for log in logs:
            if key in log:
                rounds.append(log["round"])
                vals.append(log[key])
        return rounds, vals

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FedAvg Convergence — CA-GrQc", fontsize=13)

    for ax, metric, label in zip(
            axes,
            ["macro_f1", "micro_f1", "hamming_loss"],
            ["Macro-F1", "Micro-F1", "Hamming Loss"]):

        r_iid, v_iid     = extract_metric(res_iid["round_logs"],    metric)
        r_noniid, v_noniid = extract_metric(res_noniid["round_logs"], metric)

        ax.plot(r_iid,    v_iid,    label="FedAvg IID",    marker="o", ms=3)
        ax.plot(r_noniid, v_noniid, label="FedAvg non-IID", marker="s", ms=3)
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "effectiveness_convergence.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")

    return summary


if __name__ == "__main__":
    run_effectiveness()
