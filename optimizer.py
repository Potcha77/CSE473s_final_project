"""
optimizer.py
------------
Gradient-based parameter update rules.

Implemented:
  SGDMomentum : Stochastic Gradient Descent with Momentum

The optimizer is responsible for updating the learnable parameters (W, b)
of Dense layers using the gradients computed during backpropagation.
"""

import numpy as np
from .layers import Dense


class SGDMomentum:
    """
    SGD with Momentum.

    Update rule for each parameter θ:
      v  ←  momentum * v  -  lr * ∂L/∂θ
      θ  ←  θ + v

    The velocity term v accumulates a fraction of past gradients, which:
      • Accelerates learning along consistent gradient directions.
      • Dampens oscillations in noisy or curved loss surfaces.

    Parameters
    ----------
    lr       : learning rate  (e.g. 0.01)
    momentum : momentum coefficient  (e.g. 0.9, typical range 0.8–0.99)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr       = lr
        self.momentum = momentum
        # velocity dictionaries – keyed by layer id, populated lazily
        self._velocity: dict[int, dict[str, np.ndarray]] = {}

    def update(self, layers: list) -> None:
        """
        Iterate over all Dense layers in the network and apply the
        momentum SGD update.

        Non-Dense layers (activations, dropout) are silently skipped
        because they have no learnable parameters.
        """
        for layer in layers:
            if not isinstance(layer, Dense):
                continue

            lid = id(layer)   # use Python object id as dictionary key

            # Initialise velocities to zero on first encounter
            if lid not in self._velocity:
                self._velocity[lid] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b),
                }

            v = self._velocity[lid]

            # Momentum update for weights
            v['W'] = self.momentum * v['W'] - self.lr * layer.dW
            layer.W += v['W']

            # Momentum update for biases
            v['b'] = self.momentum * v['b'] - self.lr * layer.db
            layer.b += v['b']
