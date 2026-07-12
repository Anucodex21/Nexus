import numpy as np
from .embeddings import TokenEmbedding, PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class Transformer:
    """Complete Transformer model."""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        self.d_model = d_model
        self.src_embedding = TokenEmbedding(vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.pos_encoding(self.src_embedding(src))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        encoder_output = self.encoder.forward(src_emb, src_mask)
        decoder_output = self.decoder.forward(tgt_emb, encoder_output, src_mask, tgt_mask)
        output = np.dot(decoder_output, self.output_projection) + self.output_bias
        return output

    def generate(self, src, max_len=50, start_token=2, end_token=3):
        batch_size = src.shape[0]
        tgt = np.full((batch_size, 1), start_token)
        for _ in range(max_len):
            output = self.forward(src, tgt)
            next_token = np.argmax(output[:, -1, :], axis=-1, keepdims=True)
            tgt = np.concatenate([tgt, next_token], axis=1)
            if np.all(next_token == end_token):
                break
        return tgt
