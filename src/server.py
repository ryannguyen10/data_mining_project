"""
server.py
---------
Federated server: orchestrates client selection and weight aggregation
using the FedAvg algorithm (McMahan et al., 2017).
"""

import numpy as np
import time
from model import MLP


class FederatedServer:
    """Central server for FedAvg.

    Parameters
    ----------
    input_dim   : feature dimensionality
    n_classes   : number of output classes
    n_clients   : total number of available clients
    fraction_c  : fraction of clients selected each round (C in paper)
    hidden_dim  : MLP hidden layer size
    seed        : global random seed
    """

    def __init__(self, input_dim: int, n_classes: int,
                 n_clients: int, fraction_c: float = 1.0,
                 hidden_dim: int = 64, seed: int = 42):
        self.n_clients = n_clients
        self.fraction_c = fraction_c
        self.rng = np.random.default_rng(seed)

        # Global model
        self.global_model = MLP(input_dim=input_dim,
                                n_classes=n_classes,
                                hidden_dim=hidden_dim,
                                seed=seed)

        self.round_logs = []   # stores per-round metrics

    # ──────────────────────────────────────────
    # Client Selection
    # ──────────────────────────────────────────

    def select_clients(self, clients: list) -> list:
        """Return a random subset of clients for this round."""
        k = max(1, int(self.fraction_c * self.n_clients))
        selected_idx = self.rng.choice(len(clients), size=k, replace=False)
        return [clients[i] for i in selected_idx]

    # ──────────────────────────────────────────
    # One FedAvg Round
    # ──────────────────────────────────────────

    def run_round(self, clients: list, local_epochs: int,
                  round_num: int) -> dict:
        """Execute one communication round.

        Steps:
            1. Select K clients
            2. Broadcast global weights
            3. Each client trains locally
            4. Aggregate (weighted average)
            5. Update global model

        Returns
        -------
        dict with round statistics
        """
        t_start = time.time()
        selected = self.select_clients(clients)
        global_w = self.global_model.get_weights()

        all_weights = []
        all_counts = []
        round_losses = []

        for client in selected:
            # Broadcast global model
            client.set_weights(global_w)
            # Local training
            result = client.train(epochs=local_epochs,
                                  seed=int(self.rng.integers(0, 1_000_000)))
            all_weights.append(result["weights"])
            all_counts.append(result["n_samples"])
            round_losses.append(result["train_loss"])

        # Federated averaging
        new_weights = MLP.average_weights(all_weights, all_counts)
        self.global_model.set_weights(new_weights)

        elapsed = time.time() - t_start
        log = {
            "round": round_num,
            "n_selected": len(selected),
            "avg_train_loss": float(np.mean(round_losses)),
            "round_time_s": elapsed,
        }
        self.round_logs.append(log)
        return log

    # ──────────────────────────────────────────
    # Prediction on held-out data
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.global_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.global_model.predict_proba(X)

    def get_global_weights(self) -> list:
        return self.global_model.get_weights()
