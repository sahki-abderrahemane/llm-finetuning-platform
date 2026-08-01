"""
Data collator.

Converts TokenizedSample objects into padded PyTorch tensors ready for
training.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.tokenization.schema import TokenizedSample

IGNORE_INDEX = -100


def _to_features(
    samples: list[TokenizedSample] | list[dict[str, Any]],
) -> list[dict[str, list[int]]]:
    """
    Normalize a batch of samples into feature dicts.

    Accepts either TokenizedSample objects or plain dicts (as yielded by
    Hugging Face Dataset instances).
    """

    features: list[dict[str, list[int]]] = []

    for sample in samples:
        if isinstance(sample, TokenizedSample):
            features.append(
                {
                    "input_ids": sample.input_ids,
                    "attention_mask": sample.attention_mask,
                    "labels": sample.labels,
                }
            )
        else:
            features.append(
                {
                    "input_ids": sample["input_ids"],
                    "attention_mask": sample["attention_mask"],
                    "labels": sample["labels"],
                }
            )

    return features


class DataCollator:
    """
    Dynamically padded data collator.

    Pads ``input_ids`` and ``attention_mask`` to the longest sequence in
    the batch and pads ``labels`` with ``-100`` so masked tokens stay
    ignored by the loss. Hugging Face's tokenizer ``pad`` only pads the
    model input names, so ``labels`` must be padded explicitly.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(
        self,
        samples: list[TokenizedSample] | list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """
        Collate a batch of tokenized samples into padded tensors.

        Parameters
        ----------
        samples:
            List of tokenized samples or feature dicts.

        Returns
        -------
        dict
            Dictionary containing padded tensors.
        """

        features = _to_features(samples)

        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])

            input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )

            attention_masks.append(
                feature["attention_mask"] + [0] * padding
            )

            labels.append(
                feature["labels"] + [IGNORE_INDEX] * padding
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }