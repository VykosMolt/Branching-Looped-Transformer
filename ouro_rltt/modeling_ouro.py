"""
RLTT-Modified Ouro Model

This is a modified version of the Ouro model that exposes per-loop hidden states
and exit probabilities for use with the RLTT (Reward Latent Thought Trajectories)
training algorithm.

Key modifications:
1. OuroRLTTOutputWithPast: New output class that includes per-loop hidden states and exit PDFs
2. OuroForCausalLMRLTT: Modified forward pass that returns per-loop logits when requested

The original model already computes:
- hidden_states_list: Hidden states after each loop
- gate_list: Per-loop exit gate outputs
- stacked_exit_pdf: Exit probability distribution

This modified version makes these accessible for RLTT training.
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Tuple
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from .configuration_ouro import OuroConfig


logger = logging.getLogger(__name__)


# ============================================================================
# RLTT-specific output classes
# ============================================================================

@dataclass
class OuroRLTTModelOutput(ModelOutput):
    """
    Output class for OuroModel with RLTT-specific fields.

    Args:
        last_hidden_state: Hidden states from the final loop
        past_key_values: Cache for autoregressive generation
        hidden_states: Standard transformer hidden states (if output_hidden_states=True)
        attentions: Attention weights (if output_attentions=True)
        per_loop_hidden_states: List of hidden states, one per UT loop
        per_loop_exit_gates: List of exit gate logits, one per UT loop
        exit_pdf: Exit probability distribution [batch, seq, num_loops]
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_loop_hidden_states: Optional[List[torch.FloatTensor]] = None
    per_loop_exit_gates: Optional[List[torch.FloatTensor]] = None
    exit_pdf: Optional[torch.FloatTensor] = None


@dataclass
class OuroRLTTCausalLMOutput(ModelOutput):
    """
    Output class for OuroForCausalLM with RLTT-specific fields.

    Args:
        loss: Language modeling loss
        logits: Final logits (from last loop or weighted)
        past_key_values: Cache for autoregressive generation
        hidden_states: Standard transformer hidden states
        attentions: Attention weights
        per_loop_logits: List of logits, one per UT loop (for RLTT training)
        per_loop_log_probs: List of log-probs, one per UT loop (memory-efficient alternative)
        per_loop_entropies: List of entropies, one per UT loop (computed with per_loop_log_probs)
        per_loop_hidden_states: List of hidden states, one per UT loop
        exit_pdf: Exit probability distribution [batch, seq, num_loops]
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_loop_logits: Optional[List[torch.FloatTensor]] = None
    per_loop_log_probs: Optional[List[torch.FloatTensor]] = None
    per_loop_entropies: Optional[List[torch.FloatTensor]] = None
    per_loop_hidden_states: Optional[List[torch.FloatTensor]] = None
    exit_pdf: Optional[torch.FloatTensor] = None


# ============================================================================
# Original Ouro model components (unchanged)
# ============================================================================

def needs_universal_cache(
    cache: Optional[Cache], max_cache_size: Optional[int]
) -> bool:
    if cache is None:
        return True
    if isinstance(cache, UniversalTransformerCache):
        return False
    if not isinstance(cache, Cache):
        return False
    can_grow = getattr(cache, "layer_class_to_replicate", None) is not None
    if can_grow:
        return False
    cache_layers = getattr(cache, "layers", [])
    if max_cache_size is not None and len(cache_layers) < max_cache_size:
        try:
            cached_tokens = cache.get_seq_length()
        except Exception:
            cached_tokens = 0
        if cached_tokens > 0:
            raise ValueError(
                "The provided cache cannot store all Universal Transformer iterations. Please "
                "instantiate Ouro.modeling_ouro.UniversalTransformerCache and pass it as past_key_values."
            )
        return True
    return False


class OuroMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class UniversalTransformerCache:
    """Cache implementation that supports Ouro's multi-step Universal Transformer loops."""

    def __init__(self, max_cache_size: Optional[int] = None):
        self._key_cache: list[Optional[torch.Tensor]] = []
        self._value_cache: list[Optional[torch.Tensor]] = []
        self.layers: list[Any] = []
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds configured max_cache_size={self.max_cache_size}. "
                "Check total_ut_steps and num_hidden_layers."
            )

        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)

        cached_key = self._key_cache[layer_idx]
        cached_value = self._value_cache[layer_idx]

        if cached_key is None:
            self._key_cache[layer_idx] = key_states
            self._value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            self._key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self._value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self._key_cache[layer_idx]
        result_value = self._value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self._key_cache) <= layer_idx:
            return 0
        cached = self._key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]
    def get_mask_sizes(self, cache_position, layer_idx=0):
        kv_length = self.get_seq_length(layer_idx) + len(cache_position)
        return kv_length, 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for idx, (key_entry, value_entry) in enumerate(
            zip(self._key_cache, self._value_cache)
        ):
            if key_entry is None:
                continue
            assert value_entry is not None
            device = key_entry.device
            self._key_cache[idx] = key_entry.index_select(0, beam_idx.to(device))
            self._value_cache[idx] = value_entry.index_select(0, beam_idx.to(device))

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        logger.debug("Clearing UniversalTransformerCache")
        self._key_cache = []
        self._value_cache = []
        self._seen_tokens = 0


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OuroAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_ut: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                current_ut * self.config.num_hidden_layers + self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OuroDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)

        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class OuroPreTrainedModel(PreTrainedModel):
    config: OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OuroDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OuroDecoderLayer,
        "attentions": OuroAttention,
    }


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config: OuroConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                OuroDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        # Loop-level gradient checkpointing for memory efficiency during RLTT training
        self.loop_level_checkpointing = getattr(self.config, "rltt_loop_level_checkpointing", True)
        self.post_init()

    def _run_single_ut_loop(
        self,
        hidden_states: torch.Tensor,
        causal_mask_mapping: dict,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache],
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        current_ut: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single UT (Universal Transformer) loop.

        This is extracted as a separate method to enable gradient checkpointing
        at the loop level for memory efficiency during RLTT training.

        Returns:
            hidden_states: Output hidden states after this loop
            gate_output: Exit gate logits for this loop
        """
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                current_ut=current_ut,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        gate_output = self.early_exit_gate(hidden_states)
        return hidden_states, gate_output

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache

        max_cache_size: Optional[int] = None
        if use_cache:
            total_ut_steps = getattr(self.config, "total_ut_steps", 1) or 1
            total_layers = getattr(self.config, "num_hidden_layers", None)
            if total_layers is not None:
                max_cache_size = total_layers * total_ut_steps

            if needs_universal_cache(past_key_values, max_cache_size):
                past_key_values = UniversalTransformerCache(max_cache_size)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []
        gate_list = []

        # Determine if we should use loop-level gradient checkpointing
        # Only use during training when loop_level_checkpointing is enabled
        # Disable when FSDP is wrapping the model - FSDP's checkpointing conflicts with
        # torch.utils.checkpoint. Use the DISABLE_LOOP_CHECKPOINTING env var to disable.
        disable_loop_checkpointing = os.environ.get('DISABLE_LOOP_CHECKPOINTING', '0') == '1'
        use_loop_checkpointing = (
            self.training
            and self.loop_level_checkpointing
            and not use_cache  # Can't use checkpointing with cache
            and not disable_loop_checkpointing  # Disabled via env var (for FSDP compatibility)
        )

        for current_ut in range(self.total_ut_steps):
            if use_loop_checkpointing:
                # Use gradient checkpointing at the loop level
                # This saves memory by not storing intermediate activations for all loops
                # Instead, activations are recomputed during backward pass
                hidden_states, gate_output = torch_checkpoint(
                    self._run_single_ut_loop,
                    hidden_states,
                    causal_mask_mapping,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    current_ut,
                    use_reentrant=False,
                )
            else:
                # Standard forward pass without checkpointing
                hidden_states, gate_output = self._run_single_ut_loop(
                    hidden_states,
                    causal_mask_mapping,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    current_ut,
                    **kwargs,
                )

            hidden_states_list.append(hidden_states)
            gate_list.append(gate_output)

        output_hidden_states = kwargs.get("output_hidden_states", None)
        if output_hidden_states is None:
            output_hidden_states = getattr(self.config, "output_hidden_states", False)
        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=(inputs_embeds, *hidden_states_list) if output_hidden_states else None,
            ),
            hidden_states_list,
            gate_list,
        )


# ============================================================================
# RLTT-Modified OuroForCausalLM
# ============================================================================

@auto_docstring
class OuroForCausalLM(OuroPreTrainedModel, GenerationMixin):
    """
    RLTT-Modified Ouro model for causal language modeling.

    This version exposes per-loop logits and hidden states for RLTT training.
    Use `return_per_loop_logits=True` to get logits from each UT loop.

    For maximum memory efficiency during RLTT training, use:
        `return_per_loop_log_probs=True` with `streaming_log_probs=True`
    This computes log-probs during each UT loop and never stores full hidden states.
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OuroModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.chunk_size = getattr(config, "chunk_size", 2)
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = getattr(config, "early_exit_threshold", None)

        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _compute_log_probs_for_hidden_state(
        self,
        hidden: torch.Tensor,
        log_prob_labels: torch.Tensor,
        log_prob_temperature: float,
        chunk_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probs and entropy for a single hidden state tensor.

        This method processes the hidden state in chunks if chunk_size is specified,
        to minimize peak memory usage.

        Args:
            hidden: Hidden states tensor (batch, seq_len, hidden_size)
            log_prob_labels: Target token IDs (batch, seq_len)
            log_prob_temperature: Temperature for softmax
            chunk_size: If set, process in chunks of this size

        Returns:
            log_probs: Log probabilities for target tokens (batch, seq_len)
            entropy: Entropy of the distribution (batch, seq_len)
        """
        seq_len = hidden.shape[1]

        if chunk_size is not None and chunk_size > 0 and seq_len > chunk_size:
            # Chunked processing for memory efficiency
            all_log_probs = []
            all_entropies = []

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)

                # Get chunk of hidden states
                chunk_hidden = hidden[:, start:end, :]

                # Compute logits for this chunk only
                chunk_logits = self.lm_head(chunk_hidden)
                chunk_logits = chunk_logits / log_prob_temperature

                # Compute log-softmax for chunk
                chunk_log_softmax = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                del chunk_logits

                # Compute entropy for chunk
                chunk_probs = torch.exp(chunk_log_softmax)
                chunk_entropy = -torch.sum(chunk_probs * chunk_log_softmax, dim=-1)
                del chunk_probs
                all_entropies.append(chunk_entropy)

                # Gather log-probs for target tokens in this chunk
                chunk_labels = log_prob_labels[:, start:end]
                chunk_log_probs = torch.gather(
                    chunk_log_softmax,
                    dim=-1,
                    index=chunk_labels.unsqueeze(-1)
                ).squeeze(-1)
                del chunk_log_softmax
                all_log_probs.append(chunk_log_probs)

            # Concatenate all chunks
            log_probs = torch.cat(all_log_probs, dim=1)
            entropy = torch.cat(all_entropies, dim=1)
        else:
            # Non-chunked processing
            loop_logits = self.lm_head(hidden)
            loop_logits = loop_logits / log_prob_temperature
            loop_log_softmax = torch.nn.functional.log_softmax(loop_logits, dim=-1)
            del loop_logits

            loop_probs = torch.exp(loop_log_softmax)
            entropy = -torch.sum(loop_probs * loop_log_softmax, dim=-1)
            del loop_probs

            log_probs = torch.gather(
                loop_log_softmax,
                dim=-1,
                index=log_prob_labels.unsqueeze(-1)
            ).squeeze(-1)
            del loop_log_softmax

        return log_probs, entropy

    def _forward_streaming_log_probs(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        inputs_embeds: Optional[torch.FloatTensor],
        use_cache: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        log_prob_labels: torch.LongTensor,
        log_prob_temperature: float,
        return_exit_pdf: bool,
        **kwargs,
    ) -> OuroRLTTCausalLMOutput:
        """Memory-efficient forward pass that computes log-probs during UT loops.

        This method never stores all hidden states simultaneously. Instead, it:
        1. Runs each UT loop
        2. Immediately computes log-probs for that loop's hidden states
        3. Discards the hidden states before moving to the next loop

        This reduces peak memory from O(num_loops * hidden_size) to O(hidden_size).
        Combined with loop-level gradient checkpointing, this provides maximum
        memory efficiency for RLTT training.
        """
        # Get model components
        model = self.model

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = model.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = model.config.use_cache

        # Disable cache for streaming mode (incompatible with checkpointing)
        use_cache = False

        if cache_position is None:
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        # Get chunk size from config
        chunk_size = getattr(self.config, 'rltt_logprob_chunk_size', None)

        # Collect per-loop results (small tensors only)
        per_loop_log_probs = []
        per_loop_entropies = []
        gate_list = []

        # Determine if we should use loop-level gradient checkpointing
        # Disable when FSDP is wrapping the model - FSDP's checkpointing conflicts with
        # torch.utils.checkpoint. Use the DISABLE_LOOP_CHECKPOINTING env var to disable.
        disable_loop_checkpointing = os.environ.get('DISABLE_LOOP_CHECKPOINTING', '0') == '1'
        use_loop_checkpointing = (
            model.training
            and model.loop_level_checkpointing
            and not disable_loop_checkpointing  # Disabled via env var (for FSDP compatibility)
        )

        # Log checkpointing status once per forward pass (for debugging)
        if not hasattr(self, '_logged_checkpointing_status'):
            print(f"[RLTT MODEL] _forward_streaming_log_probs: DISABLE_LOOP_CHECKPOINTING={disable_loop_checkpointing}, "
                  f"use_loop_checkpointing={use_loop_checkpointing}, training={model.training}, "
                  f"loop_level_checkpointing={model.loop_level_checkpointing}", flush=True)
            self._logged_checkpointing_status = True

        for current_ut in range(model.total_ut_steps):
            if use_loop_checkpointing:
                # Use gradient checkpointing at the loop level
                hidden_states, gate_output = torch_checkpoint(
                    model._run_single_ut_loop,
                    hidden_states,
                    causal_mask_mapping,
                    position_ids,
                    None,  # past_key_values - disabled for checkpointing
                    False,  # use_cache - disabled for checkpointing
                    cache_position,
                    position_embeddings,
                    current_ut,
                    use_reentrant=False,
                )
            else:
                hidden_states, gate_output = model._run_single_ut_loop(
                    hidden_states,
                    causal_mask_mapping,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    current_ut,
                    **kwargs,
                )

            gate_list.append(gate_output)

            # Compute log-probs immediately for this loop's hidden states
            # This is the key optimization: we compute and store only the small
            # log-prob tensor, not the large hidden states tensor
            log_probs, entropy = self._compute_log_probs_for_hidden_state(
                hidden_states,
                log_prob_labels,
                log_prob_temperature,
                chunk_size,
            )
            per_loop_log_probs.append(log_probs)
            per_loop_entropies.append(entropy)

            # Note: hidden_states is reused for the next loop, so we don't delete it
            # The previous loop's hidden states are automatically freed when reassigned

        # Compute exit PDF from gates
        stacked_exit_pdf = None
        if return_exit_pdf and gate_list:
            pdf_list = []
            remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
            for idx, gate_tensor in enumerate(gate_list):
                lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
                if idx < len(gate_list) - 1:
                    p_i = lambda_i * remaining_prob
                    remaining_prob = remaining_prob * (1.0 - lambda_i)
                else:
                    p_i = remaining_prob
                pdf_list.append(p_i)
            stacked_exit_pdf = torch.stack(pdf_list, dim=2)

        # Skip final logits computation - RLTT only uses per_loop_log_probs
        # This avoids OOM from materializing the full (batch, seq, vocab) tensor
        logits = None

        return OuroRLTTCausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            per_loop_logits=None,
            per_loop_log_probs=per_loop_log_probs,
            per_loop_entropies=per_loop_entropies,
            per_loop_hidden_states=None,
            exit_pdf=stacked_exit_pdf,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_weighted_exit: Optional[bool] = False,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        # RLTT-specific arguments
        return_per_loop_logits: bool = False,
        return_per_loop_log_probs: bool = False,
        return_per_loop_hidden_states: bool = False,
        return_exit_pdf: bool = False,
        log_prob_labels: Optional[torch.LongTensor] = None,
        log_prob_temperature: float = 1.0,
        streaming_log_probs: bool = True,  # Use memory-efficient streaming mode by default
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[CausalLMOutputWithPast, OuroRLTTCausalLMOutput]:
        r"""
        RLTT-specific Args:
            return_per_loop_logits (`bool`, *optional*, defaults to `False`):
                If True, returns logits from each UT loop in `per_loop_logits`.
                This is essential for RLTT training but uses significant GPU memory.
            return_per_loop_log_probs (`bool`, *optional*, defaults to `False`):
                If True, computes and returns log-probs from each UT loop sequentially.
                Much more memory-efficient than return_per_loop_logits as logits are
                discarded immediately after computing log-probs for each loop.
                Requires `log_prob_labels` to specify which tokens to compute log-probs for.
            return_per_loop_hidden_states (`bool`, *optional*, defaults to `False`):
                If True, returns hidden states from each UT loop.
            return_exit_pdf (`bool`, *optional*, defaults to `False`):
                If True, returns the exit probability distribution.
            log_prob_labels (`torch.LongTensor`, *optional*):
                Token IDs to compute log-probs for. Required when return_per_loop_log_probs=True.
                Shape: (batch_size, sequence_length)
            log_prob_temperature (`float`, *optional*, defaults to 1.0):
                Temperature for log-prob computation.
            streaming_log_probs (`bool`, *optional*, defaults to `True`):
                If True and return_per_loop_log_probs is True, uses memory-efficient streaming
                mode that computes log-probs during each UT loop rather than storing all hidden
                states first. This significantly reduces peak memory usage for RLTT training.
                Set to False to use the original behavior (store all hidden states, then compute).
        """
        # Use streaming mode for maximum memory efficiency during RLTT training
        # This computes log-probs during each UT loop, never storing all hidden states
        if return_per_loop_log_probs and streaming_log_probs and log_prob_labels is not None:
            if return_per_loop_logits or return_per_loop_hidden_states:
                # Can't use streaming mode if we also need logits or hidden states
                pass  # Fall through to standard path
            else:
                return self._forward_streaming_log_probs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    log_prob_labels=log_prob_labels,
                    log_prob_temperature=log_prob_temperature,
                    return_exit_pdf=return_exit_pdf,
                    **kwargs,
                )

        exit_at_step = (
            exit_at_step if exit_at_step is not None else self.early_exit_step
        )
        exit_threshold = (
            exit_threshold if exit_threshold is not None else self.early_exit_threshold
        )

        outputs, hidden_states_list, gate_list = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        def _select_token_positions(tensor: torch.Tensor) -> torch.Tensor:
            if isinstance(slice_indices, slice):
                return tensor[:, slice_indices, ...]
            if isinstance(slice_indices, torch.Tensor):
                return tensor.index_select(1, slice_indices.to(tensor.device))
            raise TypeError(
                f"Unsupported index type for logits_to_keep: {type(slice_indices)}"
            )

        # Compute exit PDF from gates
        stacked_exit_pdf = None
        if gate_list:
            pdf_list = []
            remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
            for idx, gate_tensor in enumerate(gate_list):
                lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
                if idx < len(gate_list) - 1:
                    p_i = lambda_i * remaining_prob
                    remaining_prob = remaining_prob * (1.0 - lambda_i)
                else:
                    p_i = remaining_prob
                pdf_list.append(p_i)
            stacked_exit_pdf = torch.stack(pdf_list, dim=2)  # [batch, seq, num_loops]

        expected_logits_cache: Optional[torch.Tensor] = None

        def compute_expected_logits() -> Optional[torch.Tensor]:
            nonlocal expected_logits_cache
            if expected_logits_cache is not None:
                return expected_logits_cache
            if stacked_exit_pdf is None or not hidden_states_list:
                return None
            token_exit_pdf = _select_token_positions(stacked_exit_pdf)
            expected_logits = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_hidden = _select_token_positions(hidden)
                step_logits = self.lm_head(step_hidden)
                weight = (
                    token_exit_pdf[..., step_idx].unsqueeze(-1).to(step_logits.dtype)
                )
                expected_logits = (
                    step_logits * weight
                    if expected_logits is None
                    else expected_logits + step_logits * weight
                )
            expected_logits_cache = expected_logits
            return expected_logits_cache

        logits: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            logits = compute_expected_logits()
            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        else:
            if stacked_exit_pdf is not None and hidden_states_list:
                if exit_at_step is not None and 0 <= exit_at_step < len(
                    hidden_states_list
                ):
                    selected_hidden = hidden_states_list[exit_at_step]
                    logits = self.lm_head(_select_token_positions(selected_hidden))
                elif exit_threshold is not None:
                    cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
                    threshold_value = exit_threshold
                    if isinstance(threshold_value, torch.Tensor):
                        threshold_value = threshold_value.to(cumulative_probs.device)
                    threshold_mask = cumulative_probs >= threshold_value
                    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
                    last_step_idx = stacked_exit_pdf.shape[2] - 1
                    if last_step_idx >= 0:
                        never_exceeded = ~threshold_mask.any(dim=2)
                        exit_steps[never_exceeded] = last_step_idx
                    stacked_hidden = torch.stack(hidden_states_list, dim=2)
                    gather_index = (
                        exit_steps.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(-1, -1, 1, stacked_hidden.size(-1))
                    )
                    final_hidden_states = torch.gather(
                        stacked_hidden, 2, gather_index
                    ).squeeze(2)
                    logits = self.lm_head(_select_token_positions(final_hidden_states))
                elif use_weighted_exit:
                    logits = compute_expected_logits()

            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))

        # RLTT: Compute per-loop logits if requested (memory-intensive)
        per_loop_logits = None
        if return_per_loop_logits and hidden_states_list:
            per_loop_logits = []
            for hidden in hidden_states_list:
                loop_logits = self.lm_head(hidden)
                per_loop_logits.append(loop_logits)

        # RLTT: Compute per-loop log-probs and entropies sequentially (memory-efficient)
        # This computes logits for each loop, extracts log-probs and entropy, then immediately
        # deletes the logits to free GPU memory before processing the next loop.
        #
        # If rltt_logprob_chunk_size is set in config, we process the sequence in chunks
        # to further reduce peak memory usage. This is critical for long sequences.
        per_loop_log_probs = None
        per_loop_entropies = None
        if return_per_loop_log_probs and hidden_states_list:
            if log_prob_labels is None:
                raise ValueError(
                    "log_prob_labels must be provided when return_per_loop_log_probs=True"
                )
            per_loop_log_probs = []
            per_loop_entropies = []

            # Get chunk size from config (None means no chunking)
            chunk_size = getattr(self.config, 'rltt_logprob_chunk_size', None)

            for hidden in hidden_states_list:
                seq_len = hidden.shape[1]

                if chunk_size is not None and chunk_size > 0 and seq_len > chunk_size:
                    # Chunked processing for memory efficiency
                    all_log_probs = []
                    all_entropies = []

                    for start in range(0, seq_len, chunk_size):
                        end = min(start + chunk_size, seq_len)

                        # Get chunk of hidden states
                        chunk_hidden = hidden[:, start:end, :]

                        # Compute logits for this chunk only
                        chunk_logits = self.lm_head(chunk_hidden)
                        chunk_logits = chunk_logits / log_prob_temperature

                        # Compute log-softmax for chunk
                        chunk_log_softmax = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
                        del chunk_logits

                        # Compute entropy for chunk
                        chunk_probs = torch.exp(chunk_log_softmax)
                        chunk_entropy = -torch.sum(chunk_probs * chunk_log_softmax, dim=-1)
                        del chunk_probs
                        all_entropies.append(chunk_entropy)

                        # Gather log-probs for target tokens in this chunk
                        chunk_labels = log_prob_labels[:, start:end]
                        chunk_log_probs = torch.gather(
                            chunk_log_softmax,
                            dim=-1,
                            index=chunk_labels.unsqueeze(-1)
                        ).squeeze(-1)
                        del chunk_log_softmax
                        all_log_probs.append(chunk_log_probs)

                    # Concatenate all chunks
                    loop_log_probs = torch.cat(all_log_probs, dim=1)
                    loop_entropy = torch.cat(all_entropies, dim=1)
                    per_loop_log_probs.append(loop_log_probs)
                    per_loop_entropies.append(loop_entropy)
                else:
                    # Original non-chunked processing
                    # Compute logits for this loop
                    loop_logits = self.lm_head(hidden)
                    # Apply temperature
                    loop_logits = loop_logits / log_prob_temperature
                    # Compute log-softmax
                    loop_log_softmax = torch.nn.functional.log_softmax(loop_logits, dim=-1)
                    # Delete logits immediately to free memory
                    del loop_logits

                    # Compute entropy: H = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
                    # This is memory-efficient as we compute from log_softmax
                    loop_probs = torch.exp(loop_log_softmax)
                    loop_entropy = -torch.sum(loop_probs * loop_log_softmax, dim=-1)  # (batch, seq_len)
                    del loop_probs
                    per_loop_entropies.append(loop_entropy)

                    # Gather log-probs for the target tokens
                    # log_prob_labels shape: (batch, seq_len)
                    # loop_log_softmax shape: (batch, seq_len, vocab)
                    loop_log_probs = torch.gather(
                        loop_log_softmax,
                        dim=-1,
                        index=log_prob_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    # Delete log_softmax to free memory
                    del loop_log_softmax
                    per_loop_log_probs.append(loop_log_probs)

        # Return RLTT output if any RLTT fields requested
        if return_per_loop_logits or return_per_loop_log_probs or return_per_loop_hidden_states or return_exit_pdf:
            return OuroRLTTCausalLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                per_loop_logits=per_loop_logits,
                per_loop_log_probs=per_loop_log_probs,
                per_loop_entropies=per_loop_entropies,
                per_loop_hidden_states=hidden_states_list if return_per_loop_hidden_states else None,
                exit_pdf=stacked_exit_pdf if return_exit_pdf else None,
            )

        # Standard output for backward compatibility
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return result


class OuroForSequenceClassification(
    GenericForSequenceClassification, OuroPreTrainedModel
):
    pass


class OuroForTokenClassification(GenericForTokenClassification, OuroPreTrainedModel):
    pass


class OuroForQuestionAnswering(GenericForQuestionAnswering, OuroPreTrainedModel):
    base_model_prefix = (
        "transformer"
    )


__all__ = [
    "OuroPreTrainedModel",
    "OuroModel",
    "OuroForCausalLM",
    "OuroForSequenceClassification",
    "OuroForTokenClassification",
    "OuroForQuestionAnswering",
    "UniversalTransformerCache",
    "OuroRLTTModelOutput",
    "OuroRLTTCausalLMOutput",
]
