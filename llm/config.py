from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LLMConfig:
    """Configuration for LLM training and inference."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096

    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    model_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./checkpoints"
    data_path: str = "./data"

    device: str = "auto"
    use_fp16: bool = True
    use_4bit: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
