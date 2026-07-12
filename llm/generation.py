import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class TextGenerator:
    """Text generation with various sampling strategies."""

    def __init__(self, model_path, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else None
        )
        self.model.eval()

    def generate_greedy(self, prompt, max_new_tokens=100):
        """Greedy decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_sampling(self, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
        """Sampling with temperature and top-k/top-p filtering."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class BeamSearchGenerator:
    """Beam search generation."""

    def __init__(self, model_path, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else None
        )
        self.model.eval()

    def generate(self, prompt, max_new_tokens=100, num_beams=4, num_return_sequences=1):
        """Beam search decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )

        if num_return_sequences == 1:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
