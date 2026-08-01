"""
Phi prompt formatter.
"""

from __future__ import annotations

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)

PHI_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>' ~ '\\n' ~ message['content'] ~ '<|end|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>' ~ '\\n' ~ message['content'] ~ '<|end|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>' ~ '\\n' ~ message['content'] ~ '<|end|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' ~ '\\n' }}{% endif %}"
)


class PhiPromptFormatter(ChatTemplateFormatter):
    """
    Prompt formatter for Microsoft Phi chat models.
    """

    CHAT_TEMPLATE = PHI_CHAT_TEMPLATE
