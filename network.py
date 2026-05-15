"""
network.py
----------
The Sequential model: a linear stack of layers that wires together
forward passes, backward passes, and optimizer updates.
"""

import numpy as np
from .layers import Dropout


class Sequential:
    """
    A sequential container – layers execute in the order they are added.

    Usage
    -----
    model = Sequential()
    model.add(Dense(2, 4))
    model.add(Tanh())
    model.add(Dense(4, 1))
    model.add(Sigmoid())
    model.compile(loss=MSE(), optimizer=SGDMomentum(lr=0.1))
    history = model.fit(X, y, epochs=5000)
    predictions = model.predict(X)
    """

    def __init__(self):
        self.layers    = []
        self.loss      = None
        self.optimizer = None

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def add(self, layer) -> None:
        """Append a layer to the stack."""
        self.layers.append(layer)

    def compile(self, loss, optimizer) -> None:
        """Attach a loss function and an optimizer."""
        self.loss      = loss
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Train / inference mode toggle (controls Dropout behaviour)
    # ------------------------------------------------------------------

    def set_training(self, training: bool) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Push X through every layer in order."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, loss_gradient: np.ndarray) -> None:
        """
        Propagate the loss gradient backwards through every layer
        in reverse order, accumulating ∂L/∂W and ∂L/∂b in Dense layers.
        """
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Forward → loss → backward → optimizer update. Returns scalar loss."""
        self.set_training(True)
        y_pred = self.forward(X)
        loss_val = self.loss.loss(y, y_pred)
        grad = self.loss.gradient(y, y_pred)
        self.backward(grad)
        self.optimizer.update(self.layers)
        return loss_val

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int | None = None,
        verbose: int = 100,
    ) -> list[float]:
        """
        Train the network.

        Parameters
        ----------
        X          : input array  (N, features)
        y          : target array (N, outputs)
        epochs     : number of full passes over the data
        batch_size : if None, full-batch gradient descent is used;
                     otherwise mini-batch SGD.
        verbose    : print loss every `verbose` epochs (0 = silent)

        Returns
        -------
        history : list of per-epoch loss values
        """
        N       = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            if batch_size is None:
                # Full-batch
                epoch_loss = self.train_step(X, y)
            else:
                # Mini-batch: shuffle then iterate over chunks
                indices = np.random.permutation(N)
                epoch_loss = 0.0
                n_batches  = 0
                for start in range(0, N, batch_size):
                    idx   = indices[start : start + batch_size]
                    epoch_loss += self.train_step(X[idx], y[idx])
                    n_batches  += 1
                epoch_loss /= n_batches

            history.append(epoch_loss)

            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch:>6}/{epochs}  loss = {epoch_loss:.6f}")

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass with dropout disabled."""
        self.set_training(False)
        return self.forward(X)

    # ------------------------------------------------------------------
    # Encoder helper (for autoencoder use-case)
    # ------------------------------------------------------------------

    def encode(self, X: np.ndarray, encoder_layers: int) -> np.ndarray:
        """
        Run input through only the first `encoder_layers` layers.
        Used to extract latent representations from a trained autoencoder.
        """
        self.set_training(False)
        output = X
        for layer in self.layers[:encoder_layers]:
            output = layer.forward(output)
        return output
