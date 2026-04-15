"""
model.py
--------
Lightweight two-layer MLP implemented in pure NumPy.
No deep-learning framework required.

Architecture:
    Input (d) → Dense(64) → ReLU → Dense(n_classes) → Softmax
"""

import numpy as np
import copy


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)   # numerical stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy over a batch."""
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_p.mean()


class MLP:
    """Two-layer MLP with ReLU hidden activation.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dim  : size of hidden layer  (default 64)
    n_classes   : number of output classes
    seed        : random seed for weight initialisation
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)

        # He initialisation
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = (rng.standard_normal((input_dim, hidden_dim)) * scale1
                   ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, n_classes)) * scale2
                   ).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

    # ──────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────

    def forward(self, X: np.ndarray):
        """Return (probabilities, cache) for a batch X [B, d]."""
        z1 = X @ self.W1 + self.b1          # [B, hidden]
        a1 = relu(z1)                        # [B, hidden]
        z2 = a1 @ self.W2 + self.b2          # [B, n_classes]
        probs = softmax(z2)                  # [B, n_classes]
        cache = (X, z1, a1, z2, probs)
        return probs, cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions [B]."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs

    # ──────────────────────────────────────────
    # Backward pass + SGD update
    # ──────────────────────────────────────────

    def train_step(self, X: np.ndarray, y: np.ndarray,
                   lr: float) -> float:
        """Single mini-batch forward + backward + SGD update.

        Returns mean cross-entropy loss.
        """
        B = len(y)
        probs, (X_in, z1, a1, z2, _) = self.forward(X)
        loss = cross_entropy_loss(probs, y)

        # Gradient of loss w.r.t. z2  (softmax + cross-entropy combined)
        d_z2 = probs.copy()
        d_z2[np.arange(B), y] -= 1.0
        d_z2 /= B                           # [B, n_classes]

        # Layer 2 gradients
        dW2 = a1.T @ d_z2                   # [hidden, n_classes]
        db2 = d_z2.sum(axis=0)              # [n_classes]

        # Back-propagate through ReLU
        d_a1 = d_z2 @ self.W2.T             # [B, hidden]
        d_z1 = d_a1 * relu_grad(z1)         # [B, hidden]

        # Layer 1 gradients
        dW1 = X_in.T @ d_z1                 # [d, hidden]
        db1 = d_z1.sum(axis=0)              # [hidden]

        # SGD update
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return loss

    # ──────────────────────────────────────────
    # Weight serialisation (for FedAvg)
    # ──────────────────────────────────────────

    def get_weights(self) -> list:
        """Return a deep copy of all parameters."""
        return [copy.deepcopy(self.W1), copy.deepcopy(self.b1),
                copy.deepcopy(self.W2), copy.deepcopy(self.b2)]

    def set_weights(self, weights: list):
        """Load parameters from a weight list."""
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in weights]

    @staticmethod
    def average_weights(weight_list: list, sample_counts: list) -> list:
        """Weighted average of multiple weight lists.

        Parameters
        ----------
        weight_list   : list of weight lists (one per client)
        sample_counts : list of integers (n_samples per client)
        """
        total = sum(sample_counts)
        averaged = []
        for param_idx in range(len(weight_list[0])):
            agg = sum(
                (n / total) * weight_list[k][param_idx]
                for k, n in enumerate(sample_counts)
            )
            averaged.append(agg.astype(np.float32))
        return averaged
