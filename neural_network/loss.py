import numpy as np

class MSELoss:
    """Mean Squared Error loss."""
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class CrossEntropyLoss:
    """Cross-entropy loss for classification."""
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))

    def derivative(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) / y_pred.shape[0]
