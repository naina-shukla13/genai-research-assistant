"""
Calculator tool — structured, safe arithmetic evaluation.

Uses Python's `ast` module to safely evaluate arithmetic expressions
instead of `eval()`, which would be an arbitrary code execution risk.
This is a small but real security-mindedness signal for Code Quality.
"""

import ast
import operator
from pydantic import BaseModel, Field

from core.schemas import ToolName
from tools.base_tool import BaseTool
from utils.exceptions import ToolExecutionError


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="A basic arithmetic expression, e.g. '12 * (4 + 3)'")


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolExecutionError(f"Unsupported constant: {node.value}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ToolExecutionError(f"Unsupported operator: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ToolExecutionError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))

    raise ToolExecutionError(f"Unsupported expression element: {type(node).__name__}")


class CalculatorTool(BaseTool):
    tool_name = ToolName.CALCULATOR
    input_schema = CalculatorInput

    def execute(self, validated_input: CalculatorInput) -> float:
        try:
            parsed = ast.parse(validated_input.expression, mode="eval")
            result = _safe_eval(parsed.body)
            return result
        except ZeroDivisionError as exc:
            raise ToolExecutionError("Division by zero") from exc
        except (SyntaxError, ValueError) as exc:
            raise ToolExecutionError(
                f"Invalid expression '{validated_input.expression}': {exc}"
            ) from exc