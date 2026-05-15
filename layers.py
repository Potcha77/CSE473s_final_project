"""
layers.py
---------
Core layer abstractions for the neural network library.

Contains:
  - Layer      : Abstract base class for all layers.
  - Dense      : Fully-connected (linear) layer.
  - Dropout    : Regularisation layer that randomly zeroes activations.
"""

import numpy as np


class Layer:
    """
    Abstract base class that every layer must inherit.

    Every layer exposes two methods:
      forward(input)  -> output
      backward(grad)  -> upstream gradient
    """

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dense (fully-connected) layer
# ---------------------------------------------------------------------------

class Dense(Layer):
    """
    Fully-connected layer: output = input @ W + b

    Parameters
    ----------
    input_size  : number of input features
    output_size : number of neurons (output features)

    Learnable parameters (updated by the optimizer)
    -----------------------------------------------
    W : weight matrix  shape (input_size, output_size)
    b : bias vector    shape (1, output_size)

    Gradients stored after backward()
    ----------------------------------
    dW : ∂L/∂W  same shape as W
    db : ∂L/∂b  same shape as b
    """

    def __init__(self, input_size: int, output_size: int):
        # He / Xavier-style initialisation keeps gradients well-scaled.
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

        # Gradients – populated during backward()
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = X @ W + b
        Caches X so we can compute ∂L/∂W during backward.
        """
        self._input = input                      # cache for backward
        return input @ self.W + self.b           # shape: (batch, output_size)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass (chain rule):
          ∂L/∂W = Xᵀ @ output_gradient
          ∂L/∂b = sum of output_gradient over batch
          ∂L/∂X = output_gradient @ Wᵀ  (passed to previous layer)
        """
        batch_size = self._input.shape[0]
        self.dW = self._input.T @ output_gradient / batch_size
        self.db = np.sum(output_gradient, axis=0, keepdims=True) / batch_size
        return output_gradient @ self.W.T        # upstream gradient


# ---------------------------------------------------------------------------
# Dropout layer
# ---------------------------------------------------------------------------

class Dropout(Layer):
    """
    Dropout regularisation layer.

    During training, each neuron is independently zeroed with probability
    (1 - keep_prob).  The surviving activations are scaled by 1/keep_prob
    (inverted dropout) so that the expected value is unchanged at test time.

    During inference (training=False) the layer is a transparent pass-through.

    Parameters
    ----------
    keep_prob : fraction of neurons to *keep* (e.g. 0.8 means 20% dropout)
    """

    def __init__(self, keep_prob: float = 0.8):
        self.keep_prob = keep_prob
        self.training  = True          # toggled by Sequential.set_training()
        self._mask: np.ndarray | None = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            # Bernoulli mask – 1 where neuron is kept, 0 where dropped
            self._mask = (np.random.rand(*input.shape) < self.keep_prob) / self.keep_prob
            return input * self._mask
        return input                   # inference: identity

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.training:
            return output_gradient * self._mask
        return output_gradient
