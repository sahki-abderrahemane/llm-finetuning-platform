"""
Llama prompt formatter.
"""

from __future__ import annotations

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)

LLAMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}"
    "{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.first and system_message != false %}"
    "{% set content = system_message ~ '\\n\\n' ~ message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' ~ content ~ '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' ~ content ~ '<|eot_id|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


class LlamaPromptFormatter(ChatTemplateFormatter):
    """
    Prompt formatter for Llama 3 chat models.
    """

    CHAT_TEMPLATE = LLAMA_CHAT_TEMPLATE
