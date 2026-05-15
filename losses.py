"""
losses.py
---------
Loss functions used to measure how far predictions are from targets.

Each loss class exposes:
  loss(y_true, y_pred)  -> scalar loss value
  gradient(y_true, y_pred) -> ∂L/∂y_pred  (the seed gradient for backprop)

Implemented:
  MSE                : Mean Squared Error
  BinaryCrossEntropy : Binary Cross-Entropy
"""

import numpy as np


class MSE:
    """
    Mean Squared Error:
      L = (1/N) * Σ (y_true - y_pred)²

    Gradient w.r.t. y_pred:
      ∂L/∂y_pred = (2/N) * (y_pred - y_true)
    """

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        N = y_true.shape[0]
        return (2.0 / N) * (y_pred - y_true)


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy (log loss):
      L = -(1/N) * Σ [ y_true*log(y_pred) + (1-y_true)*log(1-y_pred) ]

    Gradient w.r.t. y_pred:
      ∂L/∂y_pred = (1/N) * [ -y_true/y_pred + (1-y_true)/(1-y_pred) ]

    y_pred is clipped to (ε, 1-ε) to avoid log(0).
    """

    EPS = 1e-12   # small constant for numerical stability

    @classmethod
    def loss(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, cls.EPS, 1.0 - cls.EPS)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    @classmethod
    def gradient(cls, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        N = y_true.shape[0]
        p = np.clip(y_pred, cls.EPS, 1.0 - cls.EPS)
        return (1.0 / N) * (-y_true / p + (1.0 - y_true) / (1.0 - p))
