"""
Mistral prompt formatter.
"""

from __future__ import annotations

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)

MISTRAL_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}"
    "{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{% if system_message != false %}"
    "{% set content = system_message ~ '\\n\\n' ~ message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{{ '[INST] ' ~ content ~ ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] ~ '</s>' }}"
    "{% endif %}{% endfor %}"
)


class MistralPromptFormatter(ChatTemplateFormatter):
    """
    Prompt formatter for Mistral chat models.
    """

    CHAT_TEMPLATE = MISTRAL_CHAT_TEMPLATE
