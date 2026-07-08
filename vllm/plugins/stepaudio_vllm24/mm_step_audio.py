from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)

AUDIO_PATCH_TOKEN_ID = 151690


class Step1fAudioInputs(TypedDict):
    audio_waveforms: list[torch.Tensor]
    audio_lens: list[int]
    audio_mels: Optional[torch.Tensor]


def _as_int_list(x: object) -> list[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for item in x:
            out.extend(_as_int_list(item))
        return out
    return [int(x)]


def _collect_tensors(x: object) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, np.ndarray):
        return [torch.from_numpy(x)]
    if isinstance(x, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in x:
            out.extend(_collect_tensors(item))
        return out
    raise TypeError(f"Unsupported audio waveform input type: {type(x)!r}")


def _split_waveforms(
    audio_waveforms: object,
    audio_waveform_lens: Sequence[int],
) -> list[torch.Tensor]:
    if audio_waveforms is None:
        return []

    lens = [int(x) for x in audio_waveform_lens]
    if not lens:
        return []

    if isinstance(audio_waveforms, torch.Tensor):
        if audio_waveforms.ndim == 1:
            flat = audio_waveforms
            waveforms = []
            cur_idx = 0
            for wav_len in lens:
                waveforms.append(flat[cur_idx : cur_idx + wav_len])
                cur_idx += wav_len
            return waveforms

        rows = audio_waveforms.flatten(0, audio_waveforms.ndim - 2)
        if rows.size(0) == len(lens):
            return [rows[i, :wav_len] for i, wav_len in enumerate(lens)]

        flat = audio_waveforms.reshape(-1)
        waveforms = []
        cur_idx = 0
        for wav_len in lens:
            waveforms.append(flat[cur_idx : cur_idx + wav_len])
            cur_idx += wav_len
        return waveforms

    tensors = _collect_tensors(audio_waveforms)
    if len(tensors) == len(lens):
        return [tensor.reshape(-1)[:wav_len] for tensor, wav_len in zip(tensors, lens)]

    flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
    waveforms = []
    cur_idx = 0
    for wav_len in lens:
        waveforms.append(flat[cur_idx : cur_idx + wav_len])
        cur_idx += wav_len
    return waveforms


_MEL_FILTERS_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _stepaudio_mel_feature_len(num_samples: int) -> int:
    return (int(num_samples) + 479) // 160


def _stepaudio_mel_filters(n_mels: int, device: torch.device) -> torch.Tensor:
    key = (n_mels, str(device))
    cached = _MEL_FILTERS_CACHE.get(key)
    if cached is not None:
        return cached
    filters = torch.from_numpy(
        librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
    ).to(device=device)
    _MEL_FILTERS_CACHE[key] = filters
    return filters


def _stepaudio_log_mel_spectrogram(
    audio: torch.Tensor,
    n_mels: int = 128,
    padding: int = 479,
) -> torch.Tensor:
    audio = audio.float()
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _stepaudio_mel_filters(n_mels, audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class Step1fProcessor:
    def __init__(self, config: PretrainedConfig, tokenizer: TokenizerLike) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.audio_token = "<audio_patch>"
        self.n_mels = 128
        self.max_chunk_size = 29
        self.sampling_rate = 16000
        self._mel_filters = torch.from_numpy(
            librosa.filters.mel(sr=self.sampling_rate, n_fft=400, n_mels=self.n_mels)
        )

    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.audio_token]

    def _log_mel_spectrogram(
        self,
        audio: np.ndarray,
        padding: int = 0,
    ) -> torch.Tensor:
        audio_t = F.pad(torch.from_numpy(audio.astype(np.float32)), (0, padding))
        window = torch.hann_window(400).to(audio_t.device)
        stft = torch.stft(audio_t, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.t()

    def preprocess_audio(self, audio_tensor: np.ndarray) -> torch.Tensor:
        return self._log_mel_spectrogram(audio_tensor, padding=479)

    def get_audio_feature_len(self, audio_tensor: np.ndarray) -> int:
        return _stepaudio_mel_feature_len(len(audio_tensor))

    def get_num_audio_tokens(self, max_feature_len: int) -> int:
        native_feature_len = max_feature_len - 2
        encoder_output_dim = (native_feature_len + 1) // 2 // 2
        padding = 1
        kernel_size = 3
        stride = 2
        return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1

    def get_num_audio_embeddings(self, max_feature_len: int) -> int:
        encoder_output_dim = (max_feature_len + 1) // 2 // 2
        return (encoder_output_dim - 1) // 2 + 1

    def _get_audio_repl(self, audio_feat_len: int) -> tuple[str, list[int], list[bool]]:
        num_audio_tokens = self.get_num_audio_tokens(audio_feat_len)
        num_audio_embeddings = self.get_num_audio_embeddings(audio_feat_len)
        if not (num_audio_tokens <= num_audio_embeddings <= num_audio_tokens + 1):
            raise ValueError(
                "Unexpected StepAudio audio length relation: "
                f"patch_tokens={num_audio_tokens}, embeddings={num_audio_embeddings}"
            )
        text = "<audio_start>" + "<audio_patch>" * num_audio_tokens + "<audio_end>"
        token_ids = [
            self.tokenizer.convert_tokens_to_ids("<audio_start>"),
            *([self.audio_token_id] * num_audio_tokens),
            self.tokenizer.convert_tokens_to_ids("<audio_end>"),
        ]
        embed_mask = [False, *([True] * num_audio_tokens), num_audio_embeddings > num_audio_tokens]
        return text, token_ids, embed_mask

    @staticmethod
    def replace_placeholder(text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)
        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."
            )
        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])
        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audios: Union[np.ndarray, list[np.ndarray], None] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        if audios is None:
            audios = kwargs.pop("audio", None)
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]

        if len(audios) == 0:
            audio_inputs: dict[str, object] = {}
            text_inputs = self.tokenizer(text)
        else:
            audio_waveforms_lst = []
            audio_waveform_lens = []
            audio_lens = []
            audio_repl_str_lst = []
            for audio in audios:
                if isinstance(audio, tuple) and len(audio) == 2:
                    audio = audio[0]
                audio = np.asarray(audio, dtype=np.float32)
                audio_waveforms_lst.append(torch.from_numpy(audio))
                audio_waveform_lens.append(len(audio))
                audio_feat_len = self.get_audio_feature_len(audio)
                audio_lens.append(audio_feat_len)
                audio_repl_str, _, _ = self._get_audio_repl(audio_feat_len)
                audio_repl_str_lst.append(audio_repl_str)

            audio_inputs = {
                "audio_waveforms": torch.concat(audio_waveforms_lst),
                "audio_waveform_lens": torch.tensor(
                    audio_waveform_lens, dtype=torch.long
                ),
                "audio_lens": torch.tensor(audio_lens, dtype=torch.long),
            }
            text = [
                self.replace_placeholder(t, self.audio_token, audio_repl_str_lst)
                for t in text
            ]
            text_inputs = self.tokenizer(text)

        return BatchFeature({**text_inputs, **audio_inputs}, tensor_type=return_tensors)


class Step1fAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Step1fProcessor:
        return Step1fProcessor(self.get_hf_config(), self.get_tokenizer())

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=16000)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        mm_counts = mm_counts or {}
        if mm_counts.get("audio", 0) <= 0:
            return {}
        hf_processor = self.get_hf_processor()
        max_audio_length = int(hf_processor.sampling_rate * hf_processor.max_chunk_size)
        dummy_audio = np.zeros(max_audio_length, dtype=np.float32)
        dummy_audio_len = hf_processor.get_audio_feature_len(dummy_audio)
        return {"audio": len(hf_processor._get_audio_repl(dummy_audio_len)[1])}


class Step1fAudioDummyInputsBuilder(BaseDummyInputsBuilder[Step1fAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<audio_patch>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        hf_processor = self.info.get_hf_processor()
        audio_len = int(hf_processor.sampling_rate * hf_processor.max_chunk_size)
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=mm_counts.get("audio", 0),
                overrides=(mm_options or {}).get("audio"),
            )
        }


class Step1fAudioMultiModalProcessor(
    BaseMultiModalProcessor[Step1fAudioProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is None:
            audios = mm_data.pop("audio", None)
        processor = self.info.get_hf_processor(**mm_kwargs)
        return processor(
            text=prompt,
            audios=audios,
            return_tensors=tok_kwargs.get("return_tensors"),
        )

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_waveform_lens = hf_inputs.get("audio_waveform_lens", torch.empty(0))
        return {
            "audio_waveforms": MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_waveform_lens
            ),
            "audio_waveform_lens": MultiModalFieldConfig.batched("audio"),
            "audio_lens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()
        audio_lens = out_mm_data.get("audio_lens")
        if audio_lens is None:
            audio_items = mm_items.get_items("audio", AudioProcessorItems)
            batched_audio_lens = [
                _stepaudio_mel_feature_len(audio_items.get_audio_length(item_idx))
                for item_idx in range(audio_items.get_count())
            ]
        else:
            batched_audio_lens = _as_int_list(audio_lens)

        def get_replacement_step_audio(item_idx: int):
            audio_repl_ids, embed_mask = processor._get_audio_repl(
                batched_audio_lens[item_idx]
            )[1:]
            embed_mask_t = torch.tensor(embed_mask, dtype=torch.bool)
            return PromptUpdateDetails(
                full=audio_repl_ids,
                is_embed=lambda _tokenizer, _seq, mask=embed_mask_t: mask,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[processor.audio_token_id],
                replacement=get_replacement_step_audio,
            )
        ]


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return ~(seq_range_expand >= seq_length_expand)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask.to(dtype)
    return (1.0 - mask) * -1.0e10


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input).type(input.dtype)


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            None if self.bias is None else self.bias.to(input.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return super()._conv_forward(
            input,
            weight.to(input.dtype),
            None if bias is None else bias.to(input.dtype),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        _, _, dim = q.shape
        scale = (dim // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.attn_ln(x.contiguous()), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x.contiguous()))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = LayerNorm(n_state)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        mask = make_non_pad_mask(x_len, t).unsqueeze(1)
        mask = mask_to_bias(mask[:, :, (t + 1) % 2 :: 2], x.dtype)
        x = (x + self.positional_embedding.weight[: x.shape[1], :]).to(x.dtype).contiguous()
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x)
        return x, x_len


class Adaptor(nn.Module):
    def __init__(
        self,
        n_state: int = 1280,
        n_hidden: int = 3072,
        kernel_size: int = 7,
        stride: int = 4,
        adapter_state: int = 2048,
    ) -> None:
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            self.conv = Conv1d(n_state, n_state, kernel_size, stride, padding=1)
        self.linear1 = Linear(n_state, adapter_state)
        self.relu = nn.ReLU()
        self.linear2 = Linear(adapter_state, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != -1:
            x = x.permute(0, 2, 1)
            x = F.gelu(self.conv(x))
            x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@MULTIMODAL_REGISTRY.register_processor(
    Step1fAudioMultiModalProcessor,
    info=Step1fAudioProcessingInfo,
    dummy_inputs=Step1fAudioDummyInputsBuilder,
)
class StepAudio2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # PEFT adapters name language-model modules `model.layers...` /
    # `lm_head...`; the LoRA loader uses this mapper to place them onto
    # the wrapped `language_model` submodule.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<audio_patch>"
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.encoder = AudioEncoder(
                config.audio_encoder_config.n_mels,
                config.audio_encoder_config.n_audio_ctx,
                config.audio_encoder_config.n_audio_state,
                config.audio_encoder_config.n_audio_head,
                config.audio_encoder_config.n_audio_layer,
            )
            self.adapter = Adaptor(
                config.audio_encoder_config.n_audio_state,
                config.audio_encoder_config.llm_dim,
                config.audio_encoder_config.kernel_size,
                config.audio_encoder_config.adapter_stride,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="adapter.",
            tower_model="encoder.",
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[Step1fAudioInputs]:
        audio_waveforms = kwargs.pop("audio_waveforms", None)
        audio_waveform_lens = kwargs.pop("audio_waveform_lens", None)
        audio_mels = kwargs.pop("audio_mels", None)
        audio_lens = kwargs.pop("audio_lens", None)

        if audio_waveforms is not None:
            audio_waveform_lens_lst = _as_int_list(audio_waveform_lens)
            audio_lens_lst = _as_int_list(audio_lens)
            waveforms = [
                wav.to(device=self.device, dtype=torch.float32)
                for wav in _split_waveforms(audio_waveforms, audio_waveform_lens_lst)
            ]
            return Step1fAudioInputs(
                audio_waveforms=waveforms,
                audio_lens=audio_lens_lst,
                audio_mels=None,
            )

        if audio_mels is None:
            return None

        audio_mels = flatten_bn(audio_mels, concat=True)
        audio_lens = flatten_bn(audio_lens, concat=True).tolist()
        audio_mels_lst = []
        cur_idx = 0
        for audio_len in audio_lens:
            audio_mels_lst.append(audio_mels[cur_idx : cur_idx + audio_len])
            cur_idx += audio_len
        max_len = max(x.size(0) for x in audio_mels_lst)
        audio_mels = torch.stack(
            [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in audio_mels_lst],
            dim=0,
        )
        return Step1fAudioInputs(
            audio_waveforms=[],
            audio_lens=[int(x) for x in audio_lens],
            audio_mels=audio_mels.to(self.dtype).to(self.device),
        )

    def _process_audio_input(
        self, audio_input: Step1fAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        audio_mels = audio_input.get("audio_mels")
        if audio_mels is None:
            mel_list = [
                _stepaudio_log_mel_spectrogram(wav, n_mels=128).to(self.dtype)
                for wav in audio_input["audio_waveforms"]
            ]
            audio_lens = torch.tensor([mel.shape[-1] for mel in mel_list], device=self.device)
            max_len = max(mel.shape[-1] for mel in mel_list)
            audio_mels = torch.stack(
                [F.pad(mel, (0, max_len - mel.shape[-1])) for mel in mel_list],
                dim=0,
            )
        else:
            audio_lens = torch.tensor(audio_input["audio_lens"], device=self.device)
            audio_mels = audio_mels.permute(0, 2, 1)

        audio_features, audio_lens = self.encoder(audio_mels, audio_lens)
        audio_features = self.adapter(audio_features)
        audio_feature_lens = (audio_lens - 1) // 2 + 1
        return tuple(
            audio_features[i, : audio_feature_lens[i]]
            for i in range(audio_features.size(0))
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def maybe_remap_params(self, name: str) -> str:
        if name.startswith("model."):
            name = name.replace("model.", "language_model.model.", 1)
        if name.startswith("lm_head"):
            name = name.replace("lm_head", "language_model.lm_head", 1)
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        remapped = ((self.maybe_remap_params(name), weight) for name, weight in weights)
        try:
            loader = AutoWeightsLoader(self)
            return loader.load_weights(remapped)
        except Exception:
            loaded_params = set()
            stacked_params_mapping = [
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ]
            for name, loaded_weight in ((self.maybe_remap_params(n), w) for n, w in weights):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
            return loaded_params
