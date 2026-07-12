import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SelfAttention:
    """Single-head self-attention mechanism."""

    def __init__(self, d_model):
        self.d_model = d_model
        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        scores = np.dot(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_model)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attn_weights = softmax(scores)
        output = np.dot(attn_weights, V)
        output = np.dot(output, self.Wo)
        return output, attn_weights

class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.split_heads(np.dot(x, self.Wq))
        K = self.split_heads(np.dot(x, self.Wk))
        V = self.split_heads(np.dot(x, self.Wv))

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attn_weights = softmax(scores, axis=-1)
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(output, self.Wo)
        return output
