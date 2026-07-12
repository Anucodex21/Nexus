import numpy as np
from .activation import ReLU, Sigmoid
from .loss import MSELoss

class NeuralNetwork:
    """Multi-layer neural network with backpropagation."""

    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = ReLU() if activation == 'relu' else Sigmoid()
        self.loss_fn = MSELoss()

        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass."""
        self.activations = [X]
        self.z_values = []

        current = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            current = self.activation.forward(z)
            self.activations.append(current)

        return current

    def backward(self, y_true, learning_rate):
        """Backward pass with gradient computation."""
        m = y_true.shape[0]
        delta = self.loss_fn.derivative(self.activations[-1], y_true) * self.activation.derivative(self.z_values[-1])

        for i in range(self.num_layers - 2, -1, -1):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation.derivative(self.z_values[i-1])

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        """Train the network."""
        history = {'loss': []}
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X[indices[i:i+batch_size]]
                y_batch = y[indices[i:i+batch_size]]

                output = self.forward(X_batch)
                loss = self.loss_fn.forward(output, y_batch)
                epoch_loss += loss
                self.backward(y_batch, learning_rate)

            history['loss'].append(epoch_loss / max(1, n_samples // batch_size))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {history['loss'][-1]:.4f}")

        return history

    def predict(self, X):
        """Make predictions."""
        return self.forward(X)
