import numpy as np

class TokenEmbedding:
    """Learnable token embeddings."""

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, x):
        return self.embedding[x]

    def __call__(self, x):
        return self.forward(x)

class PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

    def __call__(self, x):
        return self.forward(x)
