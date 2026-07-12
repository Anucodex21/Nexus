import numpy as np

class LSTM:
    """Long Short-Term Memory network."""

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        concat_size = input_size + hidden_size
        self.Wf = np.random.randn(concat_size, hidden_size) * 0.01
        self.Wi = np.random.randn(concat_size, hidden_size) * 0.01
        self.Wo = np.random.randn(concat_size, hidden_size) * 0.01
        self.Wc = np.random.randn(concat_size, hidden_size) * 0.01

        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)

        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, seq_len + 1, self.hidden_size))
        c = np.zeros((batch_size, seq_len + 1, self.hidden_size))
        y = np.zeros((batch_size, seq_len, self.output_size))

        for t in range(seq_len):
            concat = np.hstack([x[:, t], h[:, t]])
            f = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
            i = self.sigmoid(np.dot(concat, self.Wi) + self.bi)
            o = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
            c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)

            c[:, t+1] = f * c[:, t] + i * c_tilde
            h[:, t+1] = o * np.tanh(c[:, t+1])
            y[:, t] = np.dot(h[:, t+1], self.Why) + self.by
        return y

    def predict(self, x):
        y = self.forward(x)
        return y[:, -1, :]
