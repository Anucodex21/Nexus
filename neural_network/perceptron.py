import numpy as np

class Perceptron:
    """Single-layer perceptron with binary classification."""

    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs
        self.history = {'loss': []}

    def _activation(self, x):
        """Step function for binary classification."""
        return 1 if x >= 0 else 0

    def predict(self, X):
        """Make predictions."""
        linear = np.dot(X, self.weights) + self.bias
        return np.array([self._activation(x) for x in linear])

    def train(self, X, y):
        """Train using perceptron learning rule."""
        for epoch in range(self.epochs):
            total_loss = 0
            for xi, target in zip(X, y):
                prediction = self._activation(np.dot(xi, self.weights) + self.bias)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_loss += error ** 2
            self.history['loss'].append(total_loss / len(y))
        return self.history

    def accuracy(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
