import numpy as np
from tqdm import tqdm

class TransformerTrainer:
    """Trainer for transformer models."""

    def __init__(self, model, learning_rate=0.0001, warmup_steps=4000):
        self.model = model
        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def get_lr(self):
        self.step_count += 1
        return self.base_lr * min(self.step_count ** (-0.5), self.step_count * self.warmup_steps ** (-1.5))

    def cross_entropy_loss(self, logits, targets):
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        log_probs = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        return np.mean(log_probs)

    def train_step(self, src, tgt_input, tgt_output):
        logits = self.model.forward(src, tgt_input)
        loss = self.cross_entropy_loss(logits, tgt_output)
        return loss

    def train(self, dataloader, epochs=10):
        history = {'loss': []}
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                src, tgt = batch
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                loss = self.train_step(src, tgt_input, tgt_output)
                epoch_loss += loss
                num_batches += 1
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {self.get_lr():.6f}")
        return history
