"""
Deployment model loader.

"""

from __future__ import annotations

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import PeftModel

from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)
from mentorai_finetuning.lora.loader import (
    LoRALoader,
)
from mentorai_finetuning.qlora.config import (
    QLoRAConfig,
)
from mentorai_finetuning.qlora.loader import (
    QLoRAModelLoader,
)
from mentorai_finetuning.training.config import (
    TrainingConfig,
)


class DeploymentLoader:
    """
    High-level model loader used during deployment.

    Depending on the deployment configuration, it loads one of:

    • Base model
    • Base model + LoRA adapter
    • Merged model
    """

    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:
        self.config = config

    def load(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load the model and tokenizer for inference.
        """

        if self.config.merged_model_path is not None:
            return self._load_merged_model()

        if self.config.adapter_path is not None:
            return self._load_lora_model()

        return self._load_base_model()

    def _load_base_model(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load a standard Hugging Face model.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device,
        )

        return model, tokenizer

    def _load_lora_model(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load a base model and attach a LoRA adapter.
        """

        training_config = TrainingConfig(
            model_name=self.config.model_name,
        )

        qlora_loader = QLoRAModelLoader(
            training_config=training_config,
            qlora_config=QLoRAConfig(),
        )

        model, tokenizer = qlora_loader.load()

        model = PeftModel.from_pretrained(
            model,
            str(self.config.adapter_path),
        )

        return model, tokenizer

    def _load_merged_model(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load a merged model.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.merged_model_path),
            trust_remote_code=self.config.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(self.config.merged_model_path),
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device,
        )

        return model, tokenizer