import numpy as np

class ReLU:
    """Rectified Linear Unit activation."""
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class Sigmoid:
    """Sigmoid activation."""
    def forward(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)

class Tanh:
    """Hyperbolic tangent activation."""
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Softmax:
    """Softmax activation for multi-class classification."""
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)
