import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInference:
    """Inference engine for LLMs."""

    def __init__(self, model_path, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # float16 on CPU is either unsupported or extremely slow depending
        # on torch build - only use it when a real GPU is available.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "auto" else None
        )

        if device != "auto":
            self.model.to(self.device)

        self.model.eval()

    def generate(self, prompt, max_new_tokens=512, temperature=0.7, 
                 top_p=0.9, top_k=50, repetition_penalty=1.1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def chat(self, messages, system_prompt="You are a helpful assistant."):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        prompt = self._format_chat_prompt(messages, system_prompt)
        return self.generate(prompt)

    def _format_chat_prompt(self, messages, system_prompt):
        prompt = f"<|system|>\n{system_prompt}\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def batch_generate(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]
