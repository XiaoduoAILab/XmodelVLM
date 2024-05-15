# Copyright (c) 2023 XiaoDuo AI. All rights reserved.

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing_extensions import Self

logger = logging.get_logger(__name__)


class XModelConfig(PretrainedConfig):
    model_type = "xmodel_32000"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=None,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        # self.intermediate_size = intermediate_size
        if intermediate_size is None:
            self.intermediate_size = find_multiple(int(8 * hidden_size / 3), 256)
        else:
            self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.auto_map = {
            "AutoConfig": "configuration_xmodel.XModelConfig",
            "AutoModelForCausalLM": "modeling_xmodel.XModelForCausalLM"
        }

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**xmodel_configs[name])


xmodel_configs = {
    "nano": dict(num_hidden_layers=6, num_attention_heads=6, num_key_value_heads=1, hidden_size=192),
    "micro": dict(num_hidden_layers=6, num_attention_heads=6, num_key_value_heads=1, hidden_size=384),
    "tiny": dict(num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2, hidden_size=512),
    "small": dict(num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=3, hidden_size=768),
    # GPT-1 & Bert-Base
    "medium": dict(num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=4, hidden_size=1024),  # Bert-Large
    "large": dict(num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=4, hidden_size=1536),
    "xl": dict(num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=4, hidden_size=2048),  # GPT-2
    "3B": dict(num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=4, hidden_size=2560),
    "7B": dict(num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32, hidden_size=4096),
    "13B": dict(num_hidden_layers=40, num_attention_heads=40, num_key_value_heads=40, hidden_size=5120),
    "34B": dict(num_hidden_layers=48, num_attention_heads=64, num_key_value_heads=8, hidden_size=8192),
    "70B": dict(num_hidden_layers=80, num_attention_heads=64, num_key_value_heads=8, hidden_size=8192),  # Llama
}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)
