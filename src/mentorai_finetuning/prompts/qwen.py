"""
Qwen prompt formatter.
"""

from __future__ import annotations

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|im_start|>user\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class QwenPromptFormatter(ChatTemplateFormatter):
    """
    Prompt formatter for Qwen chat models.

    Uses the ChatML template shipped by the tokenizer. When the
    tokenizer has no chat template, falls back to the canonical Qwen
    template.
    """

    CHAT_TEMPLATE = QWEN_CHAT_TEMPLATE
