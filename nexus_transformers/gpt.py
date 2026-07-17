import numpy as np
from .embeddings import TokenEmbedding, PositionalEncoding
from .attention import MultiHeadAttention

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class GPTBlock:
    """Single GPT decoder block."""

    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def forward(self, x, mask):
        attn_out = self.attn.forward(x, mask)
        x = self.layer_norm(x + attn_out, self.gamma1, self.beta1)
        ff_out = np.dot(np.maximum(0, np.dot(x, self.W1) + self.b1), self.W2) + self.b2
        x = self.layer_norm(x + ff_out, self.gamma2, self.beta2)
        return x

class GPT:
    """GPT-style decoder-only transformer."""

    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12,
                 d_ff=3072, max_len=1024, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = [GPTBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f_gamma = np.ones(d_model)
        self.ln_f_beta = np.zeros(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def create_causal_mask(self, seq_len):
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask.reshape(1, 1, seq_len, seq_len)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.pos_encoding(self.embedding(x))
        mask = self.create_causal_mask(seq_len)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        logits = np.dot(x, self.lm_head)
        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
        return input_ids
