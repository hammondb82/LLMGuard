
from typing import Any, Dict, List, Optional
from langchain.pydantic_v1 import root_validator
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain
import llm_guard

class LLMGuardPromptException(Exception):
    """Exception to raise when llm-guard marks prompt invalid."""
class LLMGuardPromptChain(Chain):
    scanners: Dict[str, Dict] = {}
    """The scanners to use."""
    scanners_ignore_errors: List[str] = []
    """The scanners to ignore if they throw errors."""
    vault: Optional[llm_guard.vault.Vault] = None
    """The scanners to ignore errors from."""
    raise_error: bool = True
    """Whether to raise an error if the LLMGuard marks the prompt invalid."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "sanitized_input"  #: :meta private:
    initialized_scanners: List[Any] = []  #: :meta private:
    # error_message: Optional[str] = None

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
                    if scanner_name == "Anonymize":
                        scanner_config["vault"] = values["vault"]

                    values["initialized_scanners"].append(
                        llm_guard.input_scanners.get_scanner_by_name(scanner_name, scanner_config)
                    )

            return values
        except Exception as e:
            raise ValueError(
                "Could not initialize scanners. " f"Please check provided configuration. {e}"
            ) from e

    @property
    def input_keys(self) -> List[str]:
        """
        Returns a list of input keys expected by the prompt.

        This method defines the input keys that the prompt expects in order to perform
        its processing. It ensures that the specified keys are available for providing
        input to the prompt.

        Returns:
           List[str]: A list of input keys.

        Note:
           This method is considered private and may not be intended for direct
           external use.
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """
        Returns a list of output keys.

        This method defines the output keys that will be used to access the output
        values produced by the chain or function. It ensures that the specified keys
        are available to access the outputs.

        Returns:
            List[str]: A list of output keys.

        Note:
            This method is considered private and may not be intended for direct
            external use.

        """
        return [self.output_key]

    def _check_result(
            self,
            scanner_name: str,
            is_valid: bool,
            risk_score: float,
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        if is_valid:
            return  # prompt is valid, keep scanning

        if run_manager:
            run_manager.on_text(
                text=f"This prompt was determined as invalid by {scanner_name} scanner with risk score {risk_score}",
                color="red",
                verbose=self.verbose,
            )

        if scanner_name in self.scanners_ignore_errors:
            return  # ignore error, keep scanning

        if self.raise_error:
            raise LLMGuardPromptException(
                f"This prompt was determined as invalid based by {scanner_name} scanner with risk score {risk_score}"
            )

        # if self.raise_error:
        #     raise LLMGuardPromptException(
        #         f"This prompt was determined as invalid based on configured policies with risk score {risk_score}"
        #     )

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented yet")

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """
        Executes the scanning process on the prompt and returns the sanitized prompt.

        This internal method performs the scanning process on the prompt. It uses the
        provided scanners to scan the prompt and then returns the sanitized prompt.
        Additionally, it provides the option to log information about the run using
        the provided `run_manager`.

        Args:
            inputs: A dictionary containing input values
            run_manager: A run manager to handle run-related events. Default is None

        Returns:
            Dict[str, str]: A dictionary containing the processed output.

        Raises:
            LLMGuardPromptException: If there is an error during the scanning process
        """
        if run_manager:
            run_manager.on_text("Running LLMGuardPromptChain...\n")

        sanitized_prompt = inputs[self.input_keys[0]]
        for scanner in self.initialized_scanners:
            sanitized_prompt, is_valid, risk_score = scanner.scan(sanitized_prompt)
            self._check_result(type(scanner).__name__, is_valid, risk_score, run_manager)

        return {self.output_key: sanitized_prompt}