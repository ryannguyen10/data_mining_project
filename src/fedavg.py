"""
fedavg.py
---------
Main FedAvg training loop.
Combines server, clients, and evaluation into one callable function.
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split

from model import MLP
from client import FederatedClient
from server import FederatedServer
from evaluate import evaluate_classification


def run_fedavg(X: np.ndarray,
               y: np.ndarray,
               client_indices: list,
               n_rounds: int = 50,
               local_epochs: int = 5,
               fraction_c: float = 1.0,
               lr: float = 0.01,
               batch_size: int = 32,
               hidden_dim: int = 64,
               test_size: float = 0.2,
               seed: int = 42,
               verbose: bool = True) -> dict:
    """Run the complete FedAvg experiment.

    Parameters
    ----------
    X               : feature matrix [N, d]
    y               : label array [N]
    client_indices  : per-client lists of training node indices
    n_rounds        : number of communication rounds (T)
    local_epochs    : local SGD epochs per round (E)
    fraction_c      : fraction of clients per round (C)
    lr              : learning rate for local SGD (η)
    batch_size      : mini-batch size (B)
    hidden_dim      : MLP hidden layer width
    test_size       : held-out test fraction
    seed            : random seed
    verbose         : print round-by-round progress

    Returns
    -------
    results dict with keys:
        macro_f1, micro_f1, hamming_loss  (test-set metrics)
        round_logs                         (per-round dicts)
        total_time_s
        y_pred, y_true
    """
    n_classes = int(y.max()) + 1
    input_dim = X.shape[1]
    rng = np.random.default_rng(seed)

    # ── Global train/test split (held-out for evaluation) ──────────────
    all_train_idx = np.concatenate(client_indices).astype(int)
    all_idx = np.arange(len(y))
    train_set = set(all_train_idx.tolist())
    test_idx = np.array([i for i in all_idx if i not in train_set])

    # If no explicit test set (all nodes in clients), split off 20%
    if len(test_idx) == 0:
        all_idx_arr = np.arange(len(y))
        # Only stratify if every class has at least 2 members
        min_class_count = np.bincount(y).min()
        stratify_arg = y if min_class_count >= 2 else None
        _, test_idx = train_test_split(all_idx_arr, test_size=test_size,
                                       stratify=stratify_arg,
                                       random_state=seed)

    X_test = X[test_idx]
    y_test = y[test_idx]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  FedAvg  |  rounds={n_rounds}  E={local_epochs}  "
              f"C={fraction_c}  lr={lr}")
        print(f"  clients={len(client_indices)}  "
              f"hidden={hidden_dim}  classes={n_classes}")
        print(f"  train nodes={len(all_train_idx)}  "
              f"test nodes={len(test_idx)}")
        print(f"{'='*55}")

    # ── Instantiate server ──────────────────────────────────────────────
    server = FederatedServer(input_dim=input_dim,
                             n_classes=n_classes,
                             n_clients=len(client_indices),
                             fraction_c=fraction_c,
                             hidden_dim=hidden_dim,
                             seed=seed)

    # ── Instantiate clients ─────────────────────────────────────────────
    clients = [
        FederatedClient(client_id=k,
                        X=X, y=y,
                        local_idx=client_indices[k],
                        n_classes=n_classes,
                        lr=lr,
                        batch_size=batch_size)
        for k in range(len(client_indices))
    ]

    # ── Training loop ───────────────────────────────────────────────────
    t_total_start = time.time()
    round_logs = []

    for r in range(1, n_rounds + 1):
        log = server.run_round(clients, local_epochs=local_epochs, round_num=r)

        # Evaluate on test set every round (or every 5 for large datasets)
        eval_interval = 5 if len(X_test) > 50_000 else 1
        if r % eval_interval == 0 or r == n_rounds:
            y_pred = server.predict(X_test)
            metrics = evaluate_classification(y_test, y_pred,
                                              n_classes=n_classes)
            log.update(metrics)
            if verbose:
                print(f"  Round {r:3d}/{n_rounds} | "
                      f"loss={log['avg_train_loss']:.4f} | "
                      f"macro-F1={metrics['macro_f1']:.4f} | "
                      f"micro-F1={metrics['micro_f1']:.4f} | "
                      f"ham={metrics['hamming_loss']:.4f} | "
                      f"t={log['round_time_s']:.2f}s")

        round_logs.append(log)

    total_time = time.time() - t_total_start

    # ── Final evaluation ────────────────────────────────────────────────
    y_pred_final = server.predict(X_test)
    final_metrics = evaluate_classification(y_test, y_pred_final,
                                            n_classes=n_classes)

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  FINAL RESULTS  (total time: {total_time:.2f}s)")
        print(f"  Macro-F1    : {final_metrics['macro_f1']:.4f}")
        print(f"  Micro-F1    : {final_metrics['micro_f1']:.4f}")
        print(f"  Hamming Loss: {final_metrics['hamming_loss']:.4f}")
        print(f"{'─'*55}\n")

    return {
        **final_metrics,
        "round_logs": round_logs,
        "total_time_s": total_time,
        "y_pred": y_pred_final,
        "y_true": y_test,
        "server": server,
    }
