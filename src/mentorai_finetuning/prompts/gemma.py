"""
Gemma prompt formatter.
"""

from __future__ import annotations

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)

GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<start_of_turn>system\\n' ~ messages[0]['content'] ~ '<end_of_turn>\\n' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' ~ message['content'] ~ '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' ~ message['content'] ~ '<end_of_turn>\\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)


class GemmaPromptFormatter(ChatTemplateFormatter):
    """
    Prompt formatter for Google Gemma chat models.
    """

    CHAT_TEMPLATE = GEMMA_CHAT_TEMPLATE
