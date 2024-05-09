from typing import Any, Dict, List, Optional, Union
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.messages import BaseMessage
import logging
import llm_guard

logger = logging.getLogger(__name__)
# code was adapted from https://llm-guard.com/tutorials/notebooks/langchain/#what-is-lcel to use
# Mistal-7B-instruct-v0.2 as the LLM instead of an OpenAI model.
class LLMGuardOutputException(Exception):
    """Exception to raise when llm-guard marks output invalid."""


class LLMGuardOutputChain(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    scanners: Dict[str, Dict] = {}
    """The scanners to use."""
    scanners_ignore_errors: List[str] = []
    """The scanners to ignore if they throw errors."""
    vault: Optional[llm_guard.vault.Vault] = None
    """The scanners to ignore errors from."""
    raise_error: bool = True
    """Whether to raise an error if the LLMGuard marks the output invalid."""

    initialized_scanners: List[Any] = []  #: :meta private:

    @root_validator(pre=True)
    def init_scanners(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initializes scanners

        Args:
            values (Dict[str, Any]): A dictionary containing configuration values.

        Returns:
            Dict[str, Any]: A dictionary with the updated configuration values,
                            including the initialized scanners.

        Raises:
            ValueError: If there is an issue importing 'llm-guard' or loading scanners.
        """

        if values.get("initialized_scanners") is not None:
            return values
        try:
            if values.get("scanners") is not None:
                values["initialized_scanners"] = []
                for scanner_name in values.get("scanners"):
                    scanner_config = values.get("scanners")[scanner_name]
                    if scanner_name == "Deanonymize":
                        scanner_config["vault"] = values["vault"]

                    values["initialized_scanners"].append(
                        llm_guard.output_scanners.get_scanner_by_name(scanner_name, scanner_config)
                    )

            return values
        except Exception as e:
            raise ValueError(
                "Could not initialize scanners. " f"Please check provided configuration. {e}"
            ) from e

    def _check_result(
            self,
            scanner_name: str,
            is_valid: bool,
            risk_score: float,
    ):
        if is_valid:
            return  # prompt is valid, keep scanning

        logger.warning(
            f"This output was determined as invalid by {scanner_name} scanner with risk score {risk_score}"
        )

        if scanner_name in self.scanners_ignore_errors:
            return  # ignore error, keep scanning

        if self.raise_error:
            raise LLMGuardOutputException(
                f"This output was determined as invalid based by {scanner_name} scanner with risk score {risk_score}"
            )

    def scan(
            self,
            prompt: str,
            output: Union[BaseMessage, str],
    ) -> Union[BaseMessage, str]:
        sanitized_output = output
        if isinstance(output, BaseMessage):
            sanitized_output = sanitized_output.content

        for scanner in self.initialized_scanners:
            sanitized_output, is_valid, risk_score = scanner.scan(prompt, sanitized_output)
            self._check_result(type(scanner).__name__, is_valid, risk_score)

        if isinstance(output, BaseMessage):
            output.content = sanitized_output
            return output

        return sanitized_output