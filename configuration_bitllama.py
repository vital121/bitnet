# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BitLlama model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

BITLLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class BitLlamaConfig(PretrainedConfig):
    """
    Configuration class for BitLlamaModel.

    This class stores the configuration of a BitLlamaModel and is used to instantiate the model
    with specific arguments, defining its architecture. The defaults are similar to the Llama-7B model configuration.

    Inherits from PretrainedConfig with additional model-specific parameters for controlling model outputs.

    Parameters:
        vocab_size (int, optional): Vocabulary size, default is 32000.
        hidden_size (int, optional): Dimension of hidden layers, default is 4096.
        intermediate_size (int, optional): Size of the intermediate layer in the feedforward network, default is 11008.
        num_hidden_layers (int, optional): Number of hidden layers in the Transformer model, default is 32.
        num_attention_heads (int, optional): Number of attention heads, default is 32.
        num_key_value_heads (int, optional): Number of key/value heads for Grouped Query Attention, defaults to num_attention_heads.
        hidden_act (str or callable, optional): Activation function, default is "silu".
        max_position_embeddings (int, optional): Maximum sequence length, default is 2048.
        initializer_range (float, optional): Standard deviation of the initializer, default is 0.02.
        rms_norm_eps (float, optional): Epsilon for RMS normalization, default is 1e-6.
        use_cache (bool, optional): If True, enables caching for faster inference, default is True.
        pad_token_id (int, optional): ID for the padding token.
        bos_token_id (int, optional): Beginning of stream token ID, default is 1.
        eos_token_id (int, optional): End of stream token ID, default is 2.
        pretraining_tp (int, optional): Tensor parallelism rank for pretraining, default is 1.
        tie_word_embeddings (bool, optional): If True, ties the word embeddings, default is False.
        rope_theta (float, optional): Base period for RoPE embeddings, default is 10000.0.
        rope_scaling (dict, optional): Configuration for RoPE embeddings scaling.
        attention_bias (bool, optional): If True, adds bias to attention layers, default is False.
        attention_dropout (float, optional): Dropout rate for attention probabilities, default is 0.0.
        layer_repeat (int, optional): Number of times each layer is repeated, default is 1.
    """
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        layer_repeat=1,
        **kwargs,
    ):
        # Setting num_key_value_heads to num_attention_heads if not specified
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        # Initialize parent class with all provided arguments
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Model-specific configurations
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = self.validate_rope_scaling(rope_scaling)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_repeat = layer_repeat

    def validate_rope_scaling(self, rope_scaling):
        """
        Validates the `rope_scaling` configuration.

        Parameters:
            rope_scaling (dict): Configuration for RoPE scaling.

        Returns:
            dict: Validated RoPE scaling configuration.
        """
        if rope_scaling is None:
            return None
        
        if not isinstance(rope_scaling, dict) or 'type' not in rope_scaling or 'factor' not in rope_scaling:
            raise ValueError("`rope_scaling` must be a dictionary with 'type' and 'factor' keys.")
        
        if rope_scaling['type'] not in ['linear', 'dynamic']:
            raise ValueError("`rope_scaling['type']` must be either 'linear' or 'dynamic'.")
        
        if not isinstance(rope_scaling['factor'], float) or rope_scaling['factor'] <= 1.0:
            raise ValueError("`rope_scaling['factor']` must be a float greater than 1.")
        
        return rope_scaling
