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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaRMSNorm,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .bitlinear import BitLinear
from .configuration_bitllama import BitLlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BitLlamaConfig"


class BitLlamaMLP(nn.Module):
    """
    This class implements the MLP (multilayer perceptron) component of the BitLlama model, 
    serving as a fully connected feed-forward network. It applies a series of transformations 
    to the input tensor, including linear projections and non-linear activations, 
    to produce the output tensor. This component is typically used within the transformer 
    decoder layers for additional processing of attention output.

    The MLP consists of three main operations:
    - A "gate" projection mapping from hidden size to intermediate size without bias.
    - An "up" projection also mapping from hidden size to intermediate size without bias.
    - A "down" projection mapping back from intermediate size to hidden size without bias.
    The non-linearity between the gate and up projection is defined by the activation function 
    specified in the configuration.

    Parameters:
        config (BitLlamaConfig): Configuration object containing model hyperparameters. 
        The configuration includes the hidden size, intermediate size, and the type of 
        activation function to use.

    Forward Pass Input:
        x (torch.Tensor): Input tensor to the MLP module.

    Forward Pass Output:
        torch.Tensor: Output tensor after applying the MLP transformations.

    Example:
        >>> mlp = BitLlamaMLP(config)
        >>> input_tensor = torch.rand(size=(batch_size, seq_length, config.hidden_size))
        >>> output_tensor = mlp(input_tensor)
    """
    def __init__(self, config: BitLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BitLlamaAttention(LlamaAttention):
    """
    Custom attention mechanism for the BitLlama model, extending the LlamaAttention mechanism. It is designed to
    process sequences for tasks requiring attention to previous tokens, such as language modeling.

    Parameters:
        config (BitLlamaConfig): Configuration class instance containing model hyperparameters.

    The attention mechanism supports different configurations for key/value head dimensionality, enabling Grouped
    Query Attention when `num_key_value_heads` differs from `num_attention_heads`. It incorporates position
    embeddings using RoPE (Rotary Position Embeddings) with a configurable `rope_theta`.

    Raises:
        ValueError: If the hidden size is not divisible by the number of attention heads.
    """
    def __init__(self, config: BitLlamaConfig):
        nn.Module.__init__(self)

        self.config = config
        self.layer_idx = None  # set at call time by BitLlamaDecoderLayer
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = BitLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = BitLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = BitLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = BitLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self._init_rope()


class BitLlamaFlashAttention2(BitLlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    forward = LlamaFlashAttention2.forward
    _flash_attention_forward = LlamaFlashAttention2._flash_attention_forward
    _upad_input = LlamaFlashAttention2._upad_input


BITLLAMA_ATTENTION_CLASSES = {
    "eager": BitLlamaAttention,
    "flash_attention_2": BitLlamaFlashAttention2,
}


class BitLlamaDecoderLayer(nn.Module):
    """
    Represents a single layer in the BitLlama decoder architecture. Each decoder layer is composed of a self-attention
    mechanism and a feed-forward network (MLP), with layer normalization applied before and after the self-attention
    and before the feed-forward network.

    Parameters:
        config (BitLlamaConfig): Configuration object containing model hyperparameters.
        layer_idx (Optional[int]): Index of the layer within the decoder. This is used for logging and any layer-specific behaviors.

    The layer supports optional caching of past key/value pairs to facilitate efficient decoding. It can output
    attention weights if requested. This layer can also utilize the FlashAttention mechanism for improved performance
    and efficiency, depending on the configuration.
    """
    def __init__(self, config: BitLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = (
            BitLlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else BitLlamaFlashAttention2(config)
        )
        self.mlp = BitLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        effective_idx: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if effective_idx is not None:
            self.self_attn.layer_idx = effective_idx

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


BITLLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`BitLlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare BitLlama Model outputting raw hidden-states without any specific head on top.",
    BITLLAMA_START_DOCSTRING,
)
class BitLlamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    It extends the `PreTrainedModel` by including BitLlama-specific configurations and weight initialization methods.

    Attributes:
        config_class (BitLlamaConfig): The configuration class used to initialize the BitLlama model configurations.
        base_model_prefix (str): A prefix used to differentiate the base model's parameters from those of the head(s).
        supports_gradient_checkpointing (bool): Indicates if the model supports gradient checkpointing to save memory during training.
        _no_split_modules (list[str]): Modules listed here will not be split among different GPUs in model parallel settings.
        _skip_keys_device_placement (list[str]): Keys listed here will skip device placement when loading pretrained weights.
        _supports_flash_attn_2 (bool): Indicates if the model supports Flash Attention version 2.
        _supports_cache_class (bool): Indicates if the model supports caching mechanism for improved performance in generation tasks.

    Methods:
        _init_weights(self, module): Initializes the weights of the model according to the specified configurations in `BitLlamaConfig`.
    """
    config_class = BitLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BitLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@add_start_docstrings(
    "The bare BitLlama Model outputting raw hidden-states without any specific head on top.",
    BITLLAMA_START_DOCSTRING,
)
class BitLlamaModel(BitLlamaPreTrainedModel):
    """
    The core BitLlama model transformer consisting of a series of `BitLlamaDecoderLayer` layers. This model is specifically designed
    for tasks that involve generating or transforming text, leveraging the Llama architecture's capabilities.

    Args:
        config (BitLlamaConfig): The configuration instance with all the necessary parameters to build the model.

    The model is composed of an embedding layer, multiple decoder layers as specified in the configuration, and a final normalization layer.
    It supports operations such as gradient checkpointing and Flash Attention optimization for efficient memory usage and computational performance.

    Attributes:
        padding_idx (int): The index of the padding token in the token vocabulary.
        vocab_size (int): The size of the token vocabulary.
        embed_tokens (nn.Embedding): The embedding layer for input tokens.
        layers (nn.ModuleList): The list of decoder layers (instances of `BitLlamaDecoderLayer`).
        _use_flash_attention_2 (bool): Indicates whether Flash Attention version 2 is used in this model configuration.
        norm (LlamaRMSNorm): The layer normalization applied to the output of the last decoder layer.

    Methods:
        get_input_embeddings(self): Returns the model's token embedding layer.
        set_input_embeddings(self, value): Sets the model's token embedding layer to the specified value.
        forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None): Defines the forward pass of the BitLlama model.
    """
    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                BitLlamaDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        attention_mask = None

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        _layer_idx = 0
        for decoder_layer in self.layers:
            for _ in range(max(1, self.config.layer_repeat)):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        _layer_idx,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        effective_idx=_layer_idx,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                _layer_idx += 1

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BitLlamaForCausalLM(BitLlamaPreTrainedModel):
    """
    BitLlamaForCausalLM is a model specifically tailored for causal language modeling tasks using the BitLlama architecture.
    Causal language modeling involves predicting the next token in a sequence given the preceding tokens, focusing on autoregressive generation.

    This model extends the capabilities of BitLlamaPreTrainedModel with an additional linear layer (lm_head) to map the output of the BitLlama model
    to the vocabulary space, thereby facilitating token predictions.

    Parameters:
        config (BitLlamaConfig): Configuration class instance containing all model hyperparameters.

    Attributes:
        _tied_weights_keys (List[str]): Specifies the parameters for which weight tying is applied, a common technique to reduce model size by sharing weights between the input embedding layer and the output prediction layer.
        model (BitLlamaModel): The core BitLlama model instance responsible for the bulk of the computation.
        vocab_size (int): The size of the model's vocabulary, determining the range of possible token predictions.
        lm_head (nn.Linear): A linear layer that projects the hidden state output by the model to the vocabulary size for token prediction.

    Methods:
        get_input_embeddings(): Retrieves the input embeddings layer from the BitLlamaModel, allowing for manipulation or inspection of the embedding weights.
        set_input_embeddings(value): Sets a new embeddings layer for the BitLlamaModel, enabling customization of the input embeddings.
        get_output_embeddings(): Accesses the output embeddings, specifically the weights used in the lm_head for token prediction.
        set_output_embeddings(new_embeddings): Allows setting a new output embeddings layer, facilitating customization of the prediction weights.
        set_decoder(decoder): Replaces the current BitLlamaModel (decoder) with a specified one, allowing for model updates or modifications.
        get_decoder(): Retrieves the current BitLlamaModel used for decoding, useful for inspecting the model's configuration or state.
        forward(): Implements the forward pass of the model for causal language modeling, handling input preparation, model computation, and output processing.

    The forward method accepts various inputs and options to control the model's behavior during training or inference, returning outputs tailored for causal language modeling tasks, including optional loss calculation when labels are provided.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BitLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

    """
    Processes input data through the BitLlama model for causal language modeling, predicting the next token in a sequence.

    Parameters:
        input_ids (Optional[torch.LongTensor]): Indices of input sequence tokens in the vocabulary.
        attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
        position_ids (Optional[torch.LongTensor]): Positions of tokens in the input sequence; defaults to sequential positions.
        past_key_values (Optional[List[torch.FloatTensor]]): List of past key and value states to enable efficient continuation of generation.
        inputs_embeds (Optional[torch.FloatTensor]): Optionally, input token embeddings instead of token IDs, providing a way to inject custom embeddings.
        labels (Optional[torch.LongTensor]): Labels for computing the language modeling loss. Tokens with `-100` are ignored (masked), focusing loss computation on the remaining tokens.
        use_cache (Optional[bool]): If set to True, enables caching of past key and value states for more efficient generation.
        output_attentions (Optional[bool]): Whether to return attention weights, providing insight into the model's focus during token prediction.
        output_hidden_states (Optional[bool]): Whether to return the hidden states of all layers, facilitating deep analysis of the model's internal processing.
        return_dict (Optional[bool]): Determines the return type of the method. If True, the output is returned as a custom object.

    Returns:
        Union[Tuple, CausalLMOutputWithPast]: A `CausalLMOutputWithPast` object (when `return_dict=True`) containing the loss (if labels are provided), logits, past key values (if caching is enabled), and optionally hidden states and attentions. When `return_dict=False`, these components are returned as a tuple.
    """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
