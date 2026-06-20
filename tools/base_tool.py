"""
Abstract base class enforcing structured input/output for all tools.

Every tool must declare a Pydantic input schema and implement `execute()`
returning a ToolCallResult. This is what makes "structured tool usage"
an enforced contract rather than a description — a malformed call to any
tool fails at validation, not deep inside business logic.
"""

from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel, ValidationError

from core.schemas import ToolCallResult, ToolName
from utils.exceptions import ToolInputError, ToolExecutionError
from utils.logger import trace_step, log_event


class BaseTool(ABC):
    tool_name: ToolName
    input_schema: Type[BaseModel]

    def run(self, raw_input: dict) -> ToolCallResult:
        """
        Validates raw_input against the tool's schema, then executes.
        Always returns a ToolCallResult — never raises — so the calling
        agent can inspect `success` and `error` rather than needing a
        try/except around every tool invocation.
        """
        with trace_step("Tool", f"{self.tool_name.value}_run", raw_input=raw_input):
            try:
                validated_input = self.input_schema.model_validate(raw_input)
            except ValidationError as exc:
                log_event(
                    "Tool", "tool_input_validation_failed",
                    tool=self.tool_name.value, error=str(exc),
                )
                return ToolCallResult(
                    tool_name=self.tool_name, success=False,
                    error=f"Invalid input: {exc}",
                )

            try:
                output = self.execute(validated_input)
                log_event(
                    "Tool", "tool_executed",
                    tool=self.tool_name.value, output=str(output)[:200],
                )
                return ToolCallResult(tool_name=self.tool_name, success=True, output=output)
            except Exception as exc:
                log_event(
                    "Tool", "tool_execution_failed",
                    tool=self.tool_name.value, error=str(exc),
                )
                return ToolCallResult(
                    tool_name=self.tool_name, success=False, error=str(exc)
                )

    @abstractmethod
    def execute(self, validated_input: BaseModel):
        """Concrete tool logic. Receives already-validated typed input."""
        raise NotImplementedError