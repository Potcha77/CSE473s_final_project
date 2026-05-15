"""
activations.py
--------------
Non-linear activation functions implemented as Layer subclasses.

Each activation layer:
  • Has NO learnable parameters.
  • forward()  applies the element-wise function.
  • backward() multiplies the incoming gradient by the local derivative
               (chain rule).

Implemented:
  ReLU | Sigmoid | Tanh | Softmax
"""

import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit:  f(x) = max(0, x)

    Derivative:
      f'(x) = 1  if x > 0
              0  otherwise
    """

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = input
        return np.maximum(0, input)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Gradient flows only where input was positive
        return output_gradient * (self._input > 0)


class Sigmoid(Layer):
    """
    Logistic (sigmoid) function:  f(x) = 1 / (1 + e^{-x})

    Derivative:
      f'(x) = f(x) * (1 - f(x))
    """

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Clip input to avoid overflow in exp
        self._output = 1.0 / (1.0 + np.exp(-np.clip(input, -500, 500)))
        return self._output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        sig_deriv = self._output * (1.0 - self._output)
        return output_gradient * sig_deriv


class Tanh(Layer):
    """
    Hyperbolic tangent:  f(x) = (e^x - e^{-x}) / (e^x + e^{-x})

    Derivative:
      f'(x) = 1 - tanh²(x)
    """

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._output = np.tanh(input)
        return self._output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (1.0 - self._output ** 2)


class Softmax(Layer):
    """
    Softmax over the last axis:
      f(x)_i = e^{x_i} / Σ_j e^{x_j}

    Numerically stable version subtracts max(x) before exponentiating.

    Backward:
      The full Jacobian of Softmax is a matrix for each sample; however,
      when paired with Cross-Entropy loss the combined gradient simplifies to
      (prediction - target).  For generality we implement the full Jacobian
      version here so the layer is correct in any context.
    """

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Subtract row-wise max for numerical stability
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        self._output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        For each sample i:
          grad_input[i] = s * (grad_output - (grad_output · s))
        where s = softmax output for sample i.
        """
        batch_size, n = self._output.shape
        input_gradient = np.empty_like(output_gradient)
        for i in range(batch_size):
            s  = self._output[i].reshape(-1, 1)          # (n,1)
            J  = np.diagflat(s) - s @ s.T                # Jacobian (n,n)
            input_gradient[i] = J @ output_gradient[i]
        return input_gradient
