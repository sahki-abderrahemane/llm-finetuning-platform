"""
Model export utilities.

Saves a trained/merged model to safetensors and optionally pushes it
to the Hugging Face Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class ModelExporter:
    """
    Export a model to disk and optionally to the Hugging Face Hub.
    """

    def __init__(self) -> None:
        pass

    def export(
        self,
        model: PreTrainedModel,
        output_dir: Path | str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        repo_id: str | None = None,
        commit_message: str = "Export trained model",
    ) -> Path:
        """
        Save a model to ``output_dir`` as safetensors.

        If ``repo_id`` is given the model is also pushed to the
        Hugging Face Hub.
        """

        output_path = Path(output_dir)

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            str(output_path),
        )

        if tokenizer is not None:
            tokenizer.save_pretrained(
                str(output_path),
            )

        if repo_id is not None:
            cast(
                Any,
                model,
            ).push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
            )

            if tokenizer is not None:
                cast(
                    Any,
                    tokenizer,
                ).push_to_hub(
                    repo_id=repo_id,
                    commit_message=commit_message,
                )

        return output_path
