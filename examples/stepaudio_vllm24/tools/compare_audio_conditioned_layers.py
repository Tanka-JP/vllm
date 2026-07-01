#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import io
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


TRACE: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
TRACE_FINAL: dict[int, torch.Tensor] = {}
TRACE_SHAPES: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
TRACE_POSITIONS: dict[int, torch.Tensor] = {}
TRACE_SUBLAYERS: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
TRACE_ATTRIBUTION: dict[int, dict[str, object]] = defaultdict(dict)
TRACE_STATE = {"pass_idx": -1, "current_layer": None}
LAYER0_SUBLAYER_ORDER = [
    "layer_input",
    "input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attn_output",
    "attn_o_proj",
    "post_attention_layernorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "layer_output",
]


def _add_native_src(native_src: str) -> None:
    if native_src not in sys.path:
        sys.path.insert(0, native_src)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for idx in range(n):
        if a[idx] != b[idx]:
            return idx
    return n


def _load_row(parity_jsonl: Path, audio: Path | None) -> dict[str, Any]:
    wanted = str(audio) if audio is not None else None
    with parity_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if wanted is not None and row.get("audio") != wanted:
                continue
            if wanted is None and row.get("exact_token_match"):
                continue
            return row
    raise ValueError(f"no matching non-exact row found in {parity_jsonl}")


def _decode_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    with io.BytesIO(audio_bytes) as fp:
        data, sr = sf.read(fp, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(data.astype(np.float32))


def _compute_audio_patch_count(waveform: np.ndarray, modeling_helpers) -> int:
    wav = torch.from_numpy(waveform)
    mel = modeling_helpers.log_mel_spectrogram(wav)
    return int(modeling_helpers.compute_token_num(mel.shape[-1]))


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x = a.float()
    y = b.float()
    diff = x - y
    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y)
    denom = x_norm * y_norm
    cosine = float((torch.dot(x, y) / denom).item()) if denom > 0 else float("nan")
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rms": float(math.sqrt(torch.mean(diff * diff).item())),
        "cosine": cosine,
        "native_norm": float(x_norm.item()),
        "vllm_norm": float(y_norm.item()),
    }


def _stats_or_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    try:
        if tuple(a.shape) != tuple(b.shape):
            return {"error": f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"}
        return _stats(a.reshape(-1), b.reshape(-1))
    except Exception as exc:
        return {"error": repr(exc)}


def _last_vector(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (tuple, list)):
        tensor = None
        for item in value:
            tensor = _last_vector(item)
            if tensor is not None:
                return tensor
        return None
    else:
        return None

    if tensor.ndim >= 3:
        tensor = tensor[0, -1]
    elif tensor.ndim == 2:
        tensor = tensor[-1]
    return tensor.detach().cpu().float()


def _record_vllm_sublayer(name: str, value: object) -> None:
    pass_idx = int(TRACE_STATE["pass_idx"])
    if pass_idx < 0:
        return
    tensor = _last_vector(value)
    if tensor is not None:
        TRACE_SUBLAYERS[pass_idx][name] = tensor


def _layer0_native_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
    out: dict[str, torch.Tensor] = {}
    for prefix, module in [
        ("q_proj", layer.self_attn.q_proj),
        ("k_proj", layer.self_attn.k_proj),
        ("v_proj", layer.self_attn.v_proj),
        ("o_proj", layer.self_attn.o_proj),
        ("input_layernorm", layer.input_layernorm),
        ("post_attention_layernorm", layer.post_attention_layernorm),
    ]:
        for name, param in module.named_parameters(recurse=False):
            out[f"{prefix}.{name}"] = param.detach().cpu().clone()
    return out


def _linear_last(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_dev = x.to(device=device, dtype=dtype).reshape(1, -1)
    w_dev = weight.to(device=device, dtype=dtype)
    b_dev = None if bias is None else bias.to(device=device, dtype=dtype)
    return F.linear(x_dev, w_dev, b_dev)[0].detach().cpu().float()


def _native_linear_reference(
    native_weights: dict[str, torch.Tensor],
    native_sublayers: dict[str, torch.Tensor],
    vllm_sublayers: dict[str, torch.Tensor],
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None,
    q_size: int,
    kv_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    q_w = qkv_weight[:q_size].detach().cpu()
    k_w = qkv_weight[q_size : q_size + kv_size].detach().cpu()
    v_w = qkv_weight[q_size + kv_size : q_size + kv_size + kv_size].detach().cpu()
    q_b = k_b = v_b = None
    if qkv_bias is not None:
        q_b = qkv_bias[:q_size].detach().cpu()
        k_b = qkv_bias[q_size : q_size + kv_size].detach().cpu()
        v_b = qkv_bias[q_size + kv_size : q_size + kv_size + kv_size].detach().cpu()

    shards = {
        "q": ("q_proj", q_w, q_b),
        "k": ("k_proj", k_w, k_b),
        "v": ("v_proj", v_w, v_b),
    }
    report: dict[str, object] = {}
    native_input = native_sublayers.get("input_layernorm")
    vllm_input = vllm_sublayers.get("input_layernorm")
    for shard, (native_name, vllm_w, vllm_b) in shards.items():
        native_w = native_weights.get(f"{native_name}.weight")
        native_b = native_weights.get(f"{native_name}.bias")
        native_out = native_sublayers.get(native_name)
        vllm_out = vllm_sublayers.get(native_name)
        shard_report: dict[str, object] = {}
        if native_w is not None:
            shard_report["weight_native_vs_vllm"] = _stats_or_error(native_w, vllm_w)
        if native_b is not None and vllm_b is not None:
            shard_report["bias_native_vs_vllm"] = _stats_or_error(native_b, vllm_b)
        if native_input is not None and native_w is not None and native_out is not None:
            native_recomputed = _linear_last(
                native_input,
                native_w,
                native_b,
                device=device,
                dtype=dtype,
            )
            shard_report["native_flinear_vs_native_recorded"] = _stats_or_error(
                native_recomputed, native_out
            )
        if native_input is not None and native_w is not None:
            native_by_native = _linear_last(
                native_input,
                native_w,
                native_b,
                device=device,
                dtype=dtype,
            )
            native_by_vllm = _linear_last(
                native_input,
                vllm_w,
                vllm_b,
                device=device,
                dtype=dtype,
            )
            shard_report["same_native_input_native_weight_vs_vllm_weight"] = (
                _stats_or_error(native_by_native, native_by_vllm)
            )
        if vllm_input is not None and native_w is not None:
            vllm_by_native = _linear_last(
                vllm_input,
                native_w,
                native_b,
                device=device,
                dtype=dtype,
            )
            vllm_by_vllm = _linear_last(
                vllm_input,
                vllm_w,
                vllm_b,
                device=device,
                dtype=dtype,
            )
            shard_report["same_vllm_input_native_weight_vs_vllm_weight"] = (
                _stats_or_error(vllm_by_native, vllm_by_vllm)
            )
        if vllm_out is not None and vllm_input is not None:
            vllm_recomputed = _linear_last(
                vllm_input,
                vllm_w,
                vllm_b,
                device=device,
                dtype=dtype,
            )
            shard_report["vllm_flinear_vs_vllm_recorded"] = _stats_or_error(
                vllm_recomputed, vllm_out
            )
        report[shard] = shard_report
    return report


def _patch_vllm_rmsnorm_hf_style() -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: PLC0415

    def hf_style_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        y = x.to(torch.float32)
        variance = y.pow(2).mean(-1, keepdim=True)
        y = y * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            return self.weight * y.to(input_dtype)
        return y.to(input_dtype)

    def forward_hf_style(self, x, residual=None):
        if residual is None:
            return hf_style_rms_norm(self, x)
        residual = x + residual
        return hf_style_rms_norm(self, residual), residual

    RMSNorm.forward_native = forward_hf_style
    RMSNorm.forward_cuda = forward_hf_style
    RMSNorm.forward_xpu = forward_hf_style


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seqlen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)


def _hf_eager_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    seq_len = q.shape[0]
    q_states = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    k_states = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    v_states = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    n_rep = self.num_heads // self.num_kv_heads
    k_states = _repeat_kv(k_states, n_rep)
    v_states = _repeat_kv(v_states, n_rep)
    attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)).float() * self.scaling
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    attn_weights = attn_weights.masked_fill(causal_mask, torch.finfo(torch.float32).min)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_states)
    return attn_output.transpose(1, 2).reshape(seq_len, self.q_size).contiguous()


def _hf_sdpa_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    seq_len = q.shape[0]
    q_states = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    k_states = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    v_states = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    backend_flags = {
        "auto": (True, True, True, True),
        "math": (False, True, False, False),
        "flash": (True, False, False, False),
        "mem_efficient": (False, False, True, False),
        "cudnn": (False, False, False, True),
    }
    enable_flash, enable_math, enable_mem_efficient, enable_cudnn = backend_flags[backend]
    sdp_context = (
        contextlib.nullcontext()
        if backend == "auto"
        else torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient,
            enable_cudnn=enable_cudnn,
        )
    )
    with sdp_context:
        try:
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask=None,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=seq_len > 1,
                enable_gqa=True,
            )
        except TypeError:
            n_rep = self.num_heads // self.num_kv_heads
            attn_output = F.scaled_dot_product_attention(
                q_states,
                _repeat_kv(k_states, n_rep),
                _repeat_kv(v_states, n_rep),
                attn_mask=None,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=seq_len > 1,
            )
    return attn_output.transpose(1, 2).reshape(seq_len, self.q_size).contiguous()


def _rotate_half(x: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    if is_neox_style:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _hf_style_rope(
    rotary_emb,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[0]
    rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_dim))
    base = float(getattr(rotary_emb, "base", 10000.0))
    is_neox_style = bool(getattr(rotary_emb, "is_neox_style", True))

    q_shape = q.shape
    k_shape = k.shape
    q_states = q.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k_states = k.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, device=q.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    position_ids = positions.flatten().view(1, -1).float()
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    freqs = (inv_freq_expanded.float() @ position_ids[:, None, :].float()).transpose(
        1, 2
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=q.dtype).unsqueeze(1)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = (x_rot * cos) + (_rotate_half(x_rot, is_neox_style) * sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    q_out = apply(q_states).transpose(1, 2).reshape(q_shape)
    k_out = apply(k_states).transpose(1, 2).reshape(k_shape)
    return q_out.contiguous(), k_out.contiguous()


def _install_native_sublayer_hooks(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[Any]]:
    from transformers.models.qwen2 import modeling_qwen2  # noqa: PLC0415

    store: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []
    layer = model.model.layers[0]

    def record(name: str, value: object) -> None:
        tensor = _last_vector(value)
        if tensor is not None:
            store[name] = tensor

    def forward_hook(name: str):
        def hook(_module, _inputs, output):
            record(name, output)

        return hook

    def pre_hook(_module, inputs):
        if inputs:
            record("layer_input", inputs[0])

    def layer_hook(_module, _inputs, output):
        record("layer_output", output)

    def flatten_heads(value: torch.Tensor) -> torch.Tensor:
        return value.transpose(1, 2).reshape(*value.shape[:1], value.shape[2], -1)

    orig_self_attn_forward = layer.self_attn.forward

    def wrapped_self_attn_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        record("q_rope", flatten_heads(query_states))
        record("k_rope", flatten_heads(key_states))

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            modeling_qwen2.eager_attention_forward,
        )
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
        record("attn_output", attn_output)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    class RestoreForward:
        def remove(self) -> None:
            layer.self_attn.forward = orig_self_attn_forward

    hooks.append(layer.register_forward_pre_hook(pre_hook))
    hooks.append(layer.register_forward_hook(layer_hook))
    layer.self_attn.forward = types.MethodType(wrapped_self_attn_forward, layer.self_attn)
    hooks.append(RestoreForward())
    for name, module in [
        ("input_layernorm", layer.input_layernorm),
        ("q_proj", layer.self_attn.q_proj),
        ("k_proj", layer.self_attn.k_proj),
        ("v_proj", layer.self_attn.v_proj),
        ("attn_o_proj", layer.self_attn.o_proj),
        ("post_attention_layernorm", layer.post_attention_layernorm),
        ("mlp_gate_proj", layer.mlp.gate_proj),
        ("mlp_up_proj", layer.mlp.up_proj),
        ("mlp_down_proj", layer.mlp.down_proj),
    ]:
        hooks.append(module.register_forward_hook(forward_hook(name)))
    return store, hooks


def _install_vllm_layer_hooks(
    *,
    trace_sublayers: bool = False,
    native_weights: dict[str, torch.Tensor] | None = None,
    native_sublayers: dict[str, torch.Tensor] | None = None,
    hf_style_layer0_rope: bool = False,
    hf_style_layer0_attention: bool = False,
    hf_style_layer0_sdpa: bool = False,
    hf_style_layer0_sdpa_backend: str = "auto",
) -> None:
    from vllm.model_executor.models import qwen2  # noqa: PLC0415

    orig_forward = qwen2.Qwen2DecoderLayer.forward
    orig_model_forward = qwen2.Qwen2Model.forward
    orig_attention_forward = qwen2.Qwen2Attention.forward
    orig_mlp_forward = qwen2.Qwen2MLP.forward

    def layer_index(self) -> int:
        name = getattr(getattr(self.self_attn, "attn", None), "layer_name", "")
        match = re.search(r"\.layers\.(\d+)(?:\.|$)", str(name))
        return int(match.group(1)) if match else -1

    def wrapped_attention_forward(self, positions, hidden_states):
        if not trace_sublayers or TRACE_STATE.get("current_layer") != 0:
            return orig_attention_forward(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _record_vllm_sublayer("q_proj", q)
        _record_vllm_sublayer("k_proj", k)
        _record_vllm_sublayer("v_proj", v)
        if TRACE_STATE.get("current_layer") == 0:
            pass_idx = int(TRACE_STATE["pass_idx"])
            qkv_weight = self.qkv_proj.weight
            qkv_bias = self.qkv_proj.bias
            manual_qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
            manual_q, manual_k, manual_v = manual_qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            TRACE_ATTRIBUTION[pass_idx]["packed_qkv_vs_flinear"] = {
                "q": _stats_or_error(_last_vector(q), _last_vector(manual_q)),
                "k": _stats_or_error(_last_vector(k), _last_vector(manual_k)),
                "v": _stats_or_error(_last_vector(v), _last_vector(manual_v)),
            }
            if native_weights and native_sublayers:
                TRACE_ATTRIBUTION[pass_idx]["native_vs_vllm_linear_reference"] = (
                    _native_linear_reference(
                        native_weights,
                        native_sublayers,
                        TRACE_SUBLAYERS[pass_idx],
                        qkv_weight,
                        qkv_bias,
                        self.q_size,
                        self.kv_size,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )

        if self.qk_norm:
            total_tokens = q.shape[0]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)
            _record_vllm_sublayer("q_norm", q)
            _record_vllm_sublayer("k_norm", k)

        if hf_style_layer0_rope and TRACE_STATE.get("current_layer") == 0:
            q, k = _hf_style_rope(
                self.rotary_emb,
                positions,
                q,
                k,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            q, k = self.rotary_emb(positions, q, k)
        _record_vllm_sublayer("q_rope", q)
        _record_vllm_sublayer("k_rope", k)
        if hf_style_layer0_sdpa and TRACE_STATE.get("current_layer") == 0:
            attn_output = _hf_sdpa_attention(
                self,
                q,
                k,
                v,
                backend=hf_style_layer0_sdpa_backend,
            )
        elif hf_style_layer0_attention and TRACE_STATE.get("current_layer") == 0:
            attn_output = _hf_eager_attention(self, q, k, v)
        else:
            attn_output = self.attn(q, k, v)
        _record_vllm_sublayer("attn_output", attn_output)
        output, _ = self.o_proj(attn_output)
        _record_vllm_sublayer("attn_o_proj", output)
        return output

    def wrapped_mlp_forward(self, x):
        if not trace_sublayers or TRACE_STATE.get("current_layer") != 0:
            return orig_mlp_forward(self, x)

        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        _record_vllm_sublayer("mlp_gate_proj", gate)
        _record_vllm_sublayer("mlp_up_proj", up)
        x = self.act_fn(gate_up)
        _record_vllm_sublayer("mlp_act", x)
        x, _ = self.down_proj(x)
        _record_vllm_sublayer("mlp_down_proj", x)
        return x

    def wrapped_layer_forward(self, positions, hidden_states, residual):
        layer_idx = layer_index(self)
        if layer_idx == 0:
            TRACE_STATE["pass_idx"] += 1
            TRACE_POSITIONS[TRACE_STATE["pass_idx"]] = positions.detach().cpu().clone()
            if trace_sublayers:
                TRACE_STATE["current_layer"] = 0
                _record_vllm_sublayer("layer_input", hidden_states)
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm(
                        hidden_states, residual
                    )
                _record_vllm_sublayer("input_layernorm", hidden_states)
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
                _record_vllm_sublayer("post_attention_layernorm", hidden_states)
                hidden_states = self.mlp(hidden_states)
                post = hidden_states + residual if residual is not None else hidden_states
                _record_vllm_sublayer("layer_output", post)
                pass_idx = TRACE_STATE["pass_idx"]
                TRACE[pass_idx][layer_idx] = post[-1].detach().cpu().float()
                TRACE_SHAPES[pass_idx][layer_idx] = tuple(post.shape)
                TRACE_STATE["current_layer"] = None
                return hidden_states, residual
        out_hidden, out_residual = orig_forward(self, positions, hidden_states, residual)
        post = out_hidden + out_residual if out_residual is not None else out_hidden
        if layer_idx >= 0:
            pass_idx = TRACE_STATE["pass_idx"]
            TRACE[pass_idx][layer_idx] = post[-1].detach().cpu().float()
            TRACE_SHAPES[pass_idx][layer_idx] = tuple(post.shape)
        return out_hidden, out_residual

    def wrapped_model_forward(self, *args, **kwargs):
        out = orig_model_forward(self, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            TRACE_FINAL[TRACE_STATE["pass_idx"]] = out[-1].detach().cpu().float()
        elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            TRACE_FINAL[TRACE_STATE["pass_idx"]] = out[0][-1].detach().cpu().float()
        return out

    qwen2.Qwen2DecoderLayer.forward = wrapped_layer_forward
    qwen2.Qwen2Model.forward = wrapped_model_forward
    qwen2.Qwen2Attention.forward = wrapped_attention_forward
    qwen2.Qwen2MLP.forward = wrapped_mlp_forward


def _native_hidden_dump(
    *,
    row: dict[str, Any],
    model_path: str,
    native_src: str,
    device: str,
    trace_sublayers: bool,
    trace_weights: bool,
) -> dict[str, Any]:
    _add_native_src(native_src)
    from step_audio_finetune import _modeling_helpers  # noqa: PLC0415
    from step_audio_finetune.prompt import (  # noqa: PLC0415
        build_tagged_text_prompt,
        v5_v2three_thinking_instruction,
    )

    audio_path = Path(row["audio"])
    native_ids = [int(x) for x in row["native"]["token_ids"]]
    vllm_ids = [int(x) for x in row["vllm"]["token_ids"]]
    common = _common_prefix_len(native_ids, vllm_ids)
    if common >= len(native_ids) or common >= len(vllm_ids):
        raise ValueError("row is exact or lacks a next-token divergence")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    native_sublayers: dict[str, torch.Tensor] = {}
    native_hooks: list[Any] = []
    if trace_sublayers:
        native_sublayers, native_hooks = _install_native_sublayer_hooks(model)
    common_ids = native_ids[:common]
    common_text = tokenizer.decode(common_ids, skip_special_tokens=False)
    retok = tokenizer(common_text, add_special_tokens=False)["input_ids"]

    raw = audio_path.read_bytes()
    wav = _decode_audio_bytes(raw)
    n_patches = _compute_audio_patch_count(wav, _modeling_helpers)
    prompt_text = build_tagged_text_prompt(
        {},
        n_audio_patches=n_patches,
        instruction_template=v5_v2three_thinking_instruction,
        transcript=None,
        add_thinking_prefix=True,
    )
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)
    full_ids = prompt_ids[0].tolist() + common_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    wav_t = torch.from_numpy(wav).to(device)
    mel = _modeling_helpers.log_mel_spectrogram(wav_t).to(dtype)
    wavs = mel.unsqueeze(0)
    wav_lens = torch.tensor([mel.shape[-1]], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            wavs=wavs,
            wav_lens=wav_lens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for hook in native_hooks:
        hook.remove()
    hidden_last = torch.stack(
        [h[0, -1].detach().cpu().float() for h in out.hidden_states]
    )
    logits_last = out.logits[0, -1].detach().cpu().float()
    candidate_ids = [int(native_ids[common]), int(vllm_ids[common])]
    candidate_logits = {
        str(token_id): float(logits_last[token_id].item())
        for token_id in candidate_ids
    }
    top = torch.topk(logits_last, k=8)
    report = {
        "audio": str(audio_path),
        "prompt_token_count": int(prompt_ids.shape[-1]),
        "full_token_count": len(full_ids),
        "common_prefix_len": common,
        "common_prefix_retokenizes_exact": retok == common_ids,
        "common_prefix_text": common_text,
        "native_next": {
            "token_id": int(native_ids[common]),
            "token": tokenizer.decode([int(native_ids[common])], skip_special_tokens=False),
        },
        "vllm_next": {
            "token_id": int(vllm_ids[common]),
            "token": tokenizer.decode([int(vllm_ids[common])], skip_special_tokens=False),
        },
        "native_candidate_logits": candidate_logits,
        "native_top8": [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(value),
            }
            for value, token_id in zip(top.values.tolist(), top.indices.tolist())
        ],
        "hidden_last": hidden_last,
    }
    if trace_sublayers:
        report["layer0_sublayers"] = native_sublayers
    if trace_weights:
        report["layer0_weights"] = _layer0_native_weights(model)

    del model
    del out
    del logits_last
    torch.cuda.empty_cache()
    gc.collect()
    return report


def _vllm_hidden_dump(
    *,
    native_report: dict[str, Any],
    model_path: str,
    served_model_name: str,
    gpu_memory_utilization: float,
    trace_sublayers: bool,
    hf_style_rmsnorm: bool,
    hf_style_layer0_rope: bool,
    hf_style_layer0_attention: bool,
    hf_style_layer0_sdpa: bool,
    hf_style_layer0_sdpa_backend: str,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    if hf_style_rmsnorm:
        _patch_vllm_rmsnorm_hf_style()

    llm = LLM(
        model=model_path,
        served_model_name=served_model_name,
        tokenizer_mode="step_audio_2",
        trust_remote_code=False,
        interleave_mm_strings=True,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    _install_vllm_layer_hooks(
        trace_sublayers=trace_sublayers,
        native_weights=native_report.get("layer0_weights"),
        native_sublayers=native_report.get("layer0_sublayers"),
        hf_style_layer0_rope=hf_style_layer0_rope,
        hf_style_layer0_attention=hf_style_layer0_attention,
        hf_style_layer0_sdpa=hf_style_layer0_sdpa,
        hf_style_layer0_sdpa_backend=hf_style_layer0_sdpa_backend,
    )
    TRACE.clear()
    TRACE_FINAL.clear()
    TRACE_SHAPES.clear()
    TRACE_POSITIONS.clear()
    TRACE_SUBLAYERS.clear()
    TRACE_ATTRIBUTION.clear()
    TRACE_STATE["pass_idx"] = -1
    TRACE_STATE["current_layer"] = None

    audio_path = Path(native_report["audio"])
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Japanese audio emotion annotator. "
                "Listen to the audio and output tags and transcript."
            ),
        }
    ]

    # Use the exact instruction string from the native package when available.
    try:
        from step_audio_finetune.prompt import v5_v2three_thinking_instruction  # type: ignore

        messages[0]["content"] = v5_v2three_thinking_instruction({})
    except Exception:
        pass

    messages.extend(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "<think>\n" + native_report["common_prefix_text"],
            },
        ]
    )
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        stop_token_ids=[151665],
        ignore_eos=True,
        skip_special_tokens=False,
    )
    out = llm.chat(
        messages,
        sampling_params=params,
        add_generation_prompt=False,
        continue_final_message=True,
        use_tqdm=False,
    )[0]
    completion = out.outputs[0]

    pass_idx = max(TRACE.keys(), default=0)
    layer_trace = TRACE.get(pass_idx, {})
    final = TRACE_FINAL.get(pass_idx)
    positions = TRACE_POSITIONS.get(pass_idx)
    return {
        "prompt_token_count": len(out.prompt_token_ids or []),
        "generated_token_ids": [int(x) for x in completion.token_ids],
        "generated_text": completion.text,
        "pass_idx": pass_idx,
        "hf_style_layer0_sdpa_backend": hf_style_layer0_sdpa_backend
        if hf_style_layer0_sdpa
        else None,
        "positions_count": int(positions.numel()) if positions is not None else None,
        "layer_hidden_last": {
            int(layer): tensor for layer, tensor in layer_trace.items()
        },
        "final_hidden_last": final,
        "layer0_sublayers": dict(TRACE_SUBLAYERS.get(pass_idx, {})),
        "attribution": dict(TRACE_ATTRIBUTION.get(pass_idx, {})),
        "layer_shapes": {
            int(layer): shape for layer, shape in TRACE_SHAPES.get(pass_idx, {}).items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-jsonl", type=Path, default=Path("/data/liujun/tmp/stepaudio-vllm24_parity_20_rerun_current.jsonl"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--model-path", default="/data/liujun/stepaudio-merged/v25_C0_ep3")
    parser.add_argument("--served-model-name", default="stepaudio-v25-c0-ep3")
    parser.add_argument("--native-src", default="/data/cbhua/step-audio-finetune/src")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--out-json", type=Path, default=Path("/data/liujun/tmp/stepaudio-vllm24_layer_probe.json"))
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--native-pt-out", type=Path)
    parser.add_argument("--native-pt-in", type=Path)
    parser.add_argument("--trace-layer0-sublayers", action="store_true")
    parser.add_argument("--trace-layer0-weights", action="store_true")
    parser.add_argument("--hf-style-rmsnorm", action="store_true")
    parser.add_argument("--hf-style-layer0-rope", action="store_true")
    parser.add_argument("--hf-style-layer0-attention", action="store_true")
    parser.add_argument("--hf-style-layer0-sdpa", action="store_true")
    parser.add_argument(
        "--hf-style-layer0-sdpa-backend",
        choices=["auto", "math", "flash", "mem_efficient", "cudnn"],
        default="auto",
    )
    args = parser.parse_args()

    _add_native_src(args.native_src)
    if args.native_pt_in is not None:
        native = torch.load(args.native_pt_in, map_location="cpu", weights_only=False)
    else:
        row = _load_row(args.parity_jsonl, args.audio)
        native = _native_hidden_dump(
            row=row,
            model_path=args.model_path,
            native_src=args.native_src,
            device=args.device,
            trace_sublayers=args.trace_layer0_sublayers,
            trace_weights=args.trace_layer0_weights,
        )
    if args.native_pt_out is not None:
        args.native_pt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(native, args.native_pt_out)
    if args.native_only:
        print(
            json.dumps(
                {
                    "audio": native["audio"],
                    "prompt_tokens_native": native["full_token_count"],
                    "common_prefix_len": native["common_prefix_len"],
                    "native_next": native["native_next"],
                    "vllm_next": native["vllm_next"],
                    "native_candidate_logits": native["native_candidate_logits"],
                    "layer0_sublayers": sorted(
                        native.get("layer0_sublayers", {}).keys()
                    ),
                    "layer0_weights": sorted(native.get("layer0_weights", {}).keys()),
                    "native_pt_out": str(args.native_pt_out) if args.native_pt_out else None,
                },
                ensure_ascii=False,
            )
        )
        return 0

    vllm = _vllm_hidden_dump(
        native_report=native,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trace_sublayers=args.trace_layer0_sublayers,
        hf_style_rmsnorm=args.hf_style_rmsnorm,
        hf_style_layer0_rope=args.hf_style_layer0_rope,
        hf_style_layer0_attention=args.hf_style_layer0_attention,
        hf_style_layer0_sdpa=args.hf_style_layer0_sdpa,
        hf_style_layer0_sdpa_backend=args.hf_style_layer0_sdpa_backend,
    )

    native_hidden = native.pop("hidden_last")
    native_sublayers = native.pop("layer0_sublayers", None)
    native.pop("layer0_weights", None)
    layer_rows: list[dict[str, object]] = []
    layer_hidden = vllm.pop("layer_hidden_last")
    vllm_sublayers = vllm.pop("layer0_sublayers", None)
    attribution = vllm.pop("attribution", {})
    # HF hidden_states are: embeddings, after layer 0, ..., final norm. Compare
    # vLLM raw layer N to HF hidden_states[N + 1] while that HF slot is not the
    # final RMSNorm output. HF does not expose raw layer 63 via output_hidden_states.
    max_layer = min(native_hidden.shape[0] - 3, max(layer_hidden.keys(), default=-1))
    for layer in range(max_layer + 1):
        if layer not in layer_hidden:
            continue
        layer_rows.append({
            "layer": layer,
            **_stats(native_hidden[layer + 1], layer_hidden[layer]),
        })
    final_stats = None
    if vllm.get("final_hidden_last") is not None:
        final_stats = _stats(native_hidden[-1], vllm.pop("final_hidden_last"))

    first_large = next(
        (row for row in layer_rows if float(row["max_abs"]) > 1e-2),
        None,
    )
    sublayer_rows: list[dict[str, object]] = []
    first_large_sublayer = None
    if native_sublayers and vllm_sublayers:
        shared = set(native_sublayers) & set(vllm_sublayers)
        ordered_names = [name for name in LAYER0_SUBLAYER_ORDER if name in shared]
        ordered_names.extend(sorted(shared - set(ordered_names)))
        for name in ordered_names:
            sublayer_rows.append({"name": name, **_stats(native_sublayers[name], vllm_sublayers[name])})
        first_large_sublayer = next(
            (row for row in sublayer_rows if float(row["max_abs"]) > 1e-2),
            None,
        )
    report = {
        "native": native,
        "vllm": vllm,
        "layer_stats": layer_rows,
        "layer0_sublayer_stats": sublayer_rows,
        "attribution": attribution,
        "final_stats": final_stats,
        "first_layer_max_abs_gt_1e_2": first_large,
        "first_layer0_sublayer_max_abs_gt_1e_2": first_large_sublayer,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "audio": native["audio"],
                "prompt_tokens_native": native["full_token_count"],
                "prompt_tokens_vllm": vllm["prompt_token_count"],
                "vllm_generated": vllm["generated_token_ids"],
                "first_layer_max_abs_gt_1e_2": first_large,
                "first_layer0_sublayer_max_abs_gt_1e_2": first_large_sublayer,
                "final_stats": final_stats,
                "out_json": str(args.out_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
