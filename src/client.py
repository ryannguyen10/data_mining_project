"""
client.py
---------
Federated client: holds local data and performs local SGD training.
"""

import numpy as np
import copy
from model import MLP


class FederatedClient:
    """Simulates one federated client.

    Parameters
    ----------
    client_id   : integer identifier
    X           : full feature matrix [N, d]
    y           : full label array [N]
    local_idx   : indices belonging to this client
    n_classes   : number of output classes
    lr          : local learning rate
    batch_size  : mini-batch size for SGD
    """

    def __init__(self, client_id: int,
                 X: np.ndarray, y: np.ndarray,
                 local_idx: list,
                 n_classes: int,
                 lr: float = 0.01,
                 batch_size: int = 32):
        self.client_id = client_id
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size

        self.X_local = X[local_idx].copy()
        self.y_local = y[local_idx].copy()
        self.n_samples = len(local_idx)

        # Local model — will be overwritten by server weights each round
        self.model = MLP(input_dim=X.shape[1], n_classes=n_classes)

    # ──────────────────────────────────────────
    # Weight management
    # ──────────────────────────────────────────

    def set_weights(self, weights: list):
        """Load global model weights into local model."""
        self.model.set_weights(weights)

    def get_weights(self) -> list:
        """Return a deep copy of local model weights."""
        return self.model.get_weights()

    # ──────────────────────────────────────────
    # Local training
    # ──────────────────────────────────────────

    def train(self, epochs: int = 1, seed: int = None) -> dict:
        """Run local SGD for `epochs` passes over local data.

        Returns
        -------
        dict with keys:
            weights     : updated weight list
            n_samples   : number of local training samples
            train_loss  : average cross-entropy loss over last epoch
        """
        rng = np.random.default_rng(seed)
        n = self.n_samples
        losses = []

        for epoch in range(epochs):
            # Shuffle local data
            perm = rng.permutation(n)
            X_shuf = self.X_local[perm]
            y_shuf = self.y_local[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                X_batch = X_shuf[start: start + self.batch_size]
                y_batch = y_shuf[start: start + self.batch_size]

                loss = self.model.train_step(X_batch, y_batch, self.lr)
                epoch_loss += loss
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        return {
            "weights": self.model.get_weights(),
            "n_samples": self.n_samples,
            "train_loss": float(np.mean(losses)),
        }
