from __future__ import annotations

import json
from functools import cached_property
from typing import Any, Optional, Union

import torch
from transformers import Qwen2TokenizerFast


class StepAudio2Tokenizer(Qwen2TokenizerFast):
    _tts_start_token = "<tts_start>"
    _tts_end_token = "<tts_end>"
    _first_audio_token = "<audio_0>"
    _tts_pad_token = "<tts_pad>"
    _audio_pad_token = "<audio_6561>"

    @cached_property
    def max_token_id(self) -> int:
        return len(self.vocab) - 1

    @cached_property
    def max_chars_per_token(self) -> int:
        return max(len(token) for token in self.get_vocab())

    @cached_property
    def tts_start_token_id(self) -> int | None:
        return self.vocab.get(self._tts_start_token)

    @cached_property
    def tts_end_token_id(self) -> int | None:
        return self.vocab.get(self._tts_end_token)

    @cached_property
    def first_audio_token_id(self) -> int | None:
        return self.vocab.get(self._first_audio_token)

    @cached_property
    def tts_pad_token_id(self) -> int | None:
        return self.vocab.get(self._tts_pad_token)

    @cached_property
    def audio_pad_token_id(self) -> int | None:
        return self.vocab.get(self._audio_pad_token)

    def is_step_audio_token(self, token_id: int) -> bool:
        return self.first_audio_token_id is not None and token_id >= self.first_audio_token_id

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text"}:
                        parts.append(str(item.get("text", "")))
                    elif item_type in {"audio", "input_audio", "audio_url"}:
                        parts.append("<audio_patch>")
                    elif "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return "" if content is None else str(content)

    def _render_step_audio_chat(
        self,
        conversation: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible."
            )

        result: list[str] = []
        messages = conversation

        if tools:
            result.append("<|BOT|>system\n")
            if messages and messages[0].get("role") == "system":
                result.append(self._content_to_text(messages[0].get("content")) + "<|EOT|>")
            result.append("<|BOT|>tool_json_schemas\n")
            result.append(json.dumps(tools, ensure_ascii=False) + "<|EOT|>")
        elif messages and messages[0].get("role") == "system":
            result.append(
                "<|BOT|>system\n"
                + self._content_to_text(messages[0].get("content"))
                + "<|EOT|>"
            )

        last_content = ""
        for i, message in enumerate(messages):
            role = message.get("role")
            content = self._content_to_text(message.get("content"))
            last_content = content

            if role == "user":
                result.append("<|BOT|>human\n" + content + "<|EOT|>")
            elif role == "system":
                if i != 0:
                    result.append("<|BOT|>system\n" + content + "<|EOT|>")
            elif role == "assistant":
                result.append("<|BOT|>assistant\n")
                if content:
                    result.append(content)
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        result.append("<tool_call>function\n")
                        result.append(str(tool_call["name"]) + "\n")
                        result.append(
                            json.dumps(tool_call["arguments"], ensure_ascii=False)
                        )
                        result.append("</tool_call>")
                result.append("<|EOT|>")
            elif role == "tool":
                result.append("<|BOT|>function_output\ntool\n")
                result.append(content + "<|EOT|>")
            elif role == "function_output":
                result.append("<|BOT|>input\n" + content + "<|EOT|>")
            else:
                result.append(f"<|BOT|>{role}\n{content}<|EOT|>")

        if add_generation_prompt:
            result.append("<|BOT|>assistant\n")

        rendered = "".join(result)
        if continue_final_message and last_content:
            idx = rendered.rfind(last_content)
            if idx >= 0:
                rendered = rendered[: idx + len(last_content)]
        return rendered

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, Any]], list[list[dict[str, Any]]]],
        tools: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        del documents, chat_template, return_assistant_tokens_mask, kwargs
        if conversation and isinstance(conversation[0], list):
            rendered = [
                self._render_step_audio_chat(
                    conv,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
                for conv in conversation
            ]
        else:
            rendered = self._render_step_audio_chat(
                conversation,  # type: ignore[arg-type]
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )

        if not tokenize:
            return rendered

        tokenizer_kwargs = tokenizer_kwargs or {}
        encoded = self(
            rendered,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        if return_dict:
            return encoded
        input_ids = encoded["input_ids"]
        if return_tensors is None and isinstance(input_ids, list) and input_ids:
            if isinstance(rendered, str):
                return input_ids[0] if isinstance(input_ids[0], list) else input_ids
            return input_ids
        if isinstance(input_ids, torch.Tensor) and isinstance(rendered, str):
            return input_ids[0]
        return input_ids
