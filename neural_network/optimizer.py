import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, params, grads, layer_idx):
        if layer_idx not in self.velocities:
            self.velocities[layer_idx] = np.zeros_like(params)
        self.velocities[layer_idx] = self.momentum * self.velocities[layer_idx] - self.lr * grads
        return params + self.velocities[layer_idx]

class Adam:
    """Adam optimizer."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads, layer_idx):
        if layer_idx not in self.m:
            self.m[layer_idx] = np.zeros_like(params)
            self.v[layer_idx] = np.zeros_like(params)

        self.t += 1
        self.m[layer_idx] = self.beta1 * self.m[layer_idx] + (1 - self.beta1) * grads
        self.v[layer_idx] = self.beta2 * self.v[layer_idx] + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m[layer_idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_idx] / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
