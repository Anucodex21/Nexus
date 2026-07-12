import numpy as np
from .attention import MultiHeadAttention

class TransformerDecoderLayer:
    """Single transformer decoder layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def feed_forward(self, x):
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        return np.dot(hidden, self.W2) + self.b2

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_out = self.self_attn.forward(x, tgt_mask)
        x = self.layer_norm(x + attn_out, self.gamma1, self.beta1)
        cross_out = self.cross_attn.forward(x, src_mask)
        x = self.layer_norm(x + cross_out, self.gamma2, self.beta2)
        ff_out = self.feed_forward(x)
        x = self.layer_norm(x + ff_out, self.gamma3, self.beta3)
        return x

class TransformerDecoder:
    """Stack of transformer decoder layers."""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        self.layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask)
        return x
