"""
Dataset validation utilities.

"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mentorai_finetuning.dataset.schema import DatasetSample, MessageRole


class ValidationResult(BaseModel):
    """Result of validating a dataset sample."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class DatasetValidator:
    """Validates DatasetSample objects."""

    def validate(self, sample: DatasetSample) -> ValidationResult:
        errors: list[str] = []

        messages = sample.messages

        if len(messages) < 2:
            errors.append("Conversation must contain at least two messages.")

        if messages:
            if messages[0].role == MessageRole.ASSISTANT:
                errors.append("Conversation cannot start with an assistant message.")

            if messages[-1].role != MessageRole.ASSISTANT:
                errors.append("Conversation must end with an assistant message.")

        previous_role: MessageRole | None = None

        for index, message in enumerate(messages):

            if not message.content.strip():
                errors.append(f"Message {index} has empty content.")

            if previous_role is not None:

                if (
                    previous_role == MessageRole.USER
                    and message.role == MessageRole.USER
                ):
                    errors.append(
                        f"Messages {index - 1} and {index} are consecutive user messages."
                    )

                if (
                    previous_role == MessageRole.ASSISTANT
                    and message.role == MessageRole.ASSISTANT
                ):
                    errors.append(
                        f"Messages {index - 1} and {index} are consecutive assistant messages."
                    )

                if (
                    previous_role == MessageRole.SYSTEM
                    and message.role == MessageRole.SYSTEM
                ):
                    errors.append(
                        f"Messages {index - 1} and {index} are consecutive system messages."
                    )

            previous_role = message.role

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    def validate_many(
        self,
        samples: list[DatasetSample],
    ) -> list[ValidationResult]:
        """Validate multiple dataset samples."""

        return [self.validate(sample) for sample in samples]