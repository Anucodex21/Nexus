import numpy as np

class RNN:
    """Vanilla Recurrent Neural Network."""

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
        self.h = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        self.h = np.zeros((batch_size, seq_len + 1, self.hidden_size))
        y = np.zeros((batch_size, seq_len, self.output_size))

        for t in range(seq_len):
            self.h[:, t+1] = np.tanh(np.dot(x[:, t], self.Wxh) + np.dot(self.h[:, t], self.Whh) + self.bh)
            y[:, t] = np.dot(self.h[:, t+1], self.Why) + self.by
        return y

    def predict(self, x):
        y = self.forward(x)
        return y[:, -1, :]
