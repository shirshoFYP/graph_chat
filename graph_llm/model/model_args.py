from typing import Tuple
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096

    adapter_len: int = 0
    adapter_layer: int = 0
    adapter_dim: int = 512
    adapter_n_heads: int = 4

    num_hops: int = 2
    w_adapter: bool = True
    w_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.05
    rrwp: int = 8

    n_decoder_layers: int = 2
    n_mp_layers: int = 2
    n_encoder_layers: int = 2

    # target_modules: Tuple[str] = ('q_proj', 'v_proj')     # Option
    fans_out: Tuple[int] = (50, 50, 50)

    # target_modules: Tuple[str] = ('q_proj', 'v_proj', 'k_proj')     # Option
    # target_modules: Tuple[str] = ('o_proj')     # Option
    target_modules: Tuple[str] = ("down_proj", "up_proj", "gate_proj")  # Option
    task_level: str = "node"
