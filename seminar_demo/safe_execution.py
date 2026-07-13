from __future__ import annotations

import ast
import math
import multiprocessing as mp
import re
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any


ALLOWED_CALLS = {"abs", "float", "int", "max", "min", "pow", "round"}
ALLOWED_MATH_ATTRIBUTES = {
    "ceil",
    "e",
    "exp",
    "floor",
    "log",
    "log10",
    "pi",
    "sqrt",
}
ALLOWED_NODES = {
    ast.Module,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.Pass,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
}


@dataclass(frozen=True)
class SafeExecutionResult:
    status: str
    result: Any = None
    error: str | None = None
    duration_ms: int = 0
    execution_mode: str = "safe_ast_child_process"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnsafeCodeError(ValueError):
    pass


def strip_code_fences(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def validate_code(code: str) -> ast.Module:
    tree = ast.parse(strip_code_fences(code), mode="exec")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise UnsafeCodeError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCodeError("Dunder names are forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_CALLS:
                    raise UnsafeCodeError(f"Forbidden call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "math"
                    and node.func.attr in ALLOWED_MATH_ATTRIBUTES
                ):
                    raise UnsafeCodeError("Only approved math calls are allowed")
            else:
                raise UnsafeCodeError("Dynamic calls are forbidden")
        if isinstance(node, ast.Attribute):
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr in ALLOWED_MATH_ATTRIBUTES
            ):
                raise UnsafeCodeError("Attribute access is forbidden")
    if not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        for node in ast.walk(tree)
    ):
        raise UnsafeCodeError("Code must assign a value to result")
    return tree


def _worker(code: str, connection) -> None:
    safe_math = SimpleNamespace(
        **{name: getattr(math, name) for name in ALLOWED_MATH_ATTRIBUTES}
    )
    safe_builtins = {
        "abs": abs,
        "float": float,
        "int": int,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
    }
    namespace: dict[str, Any] = {"math": safe_math}
    try:
        tree = validate_code(code)
        exec(compile(tree, "<generated-calculation>", "exec"), {"__builtins__": safe_builtins}, namespace)
        result = namespace.get("result")
        if not isinstance(result, (int, float, str, bool, tuple, list)):
            raise ValueError(f"Unsupported result type: {type(result).__name__}")
        connection.send(("success", result, None))
    except BaseException as exc:
        connection.send(("error", None, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def execute_safely(code: str, timeout_seconds: float = 2.0) -> SafeExecutionResult:
    started = time.monotonic()
    try:
        validate_code(code)
    except (SyntaxError, UnsafeCodeError) as exc:
        return SafeExecutionResult(
            status="rejected",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(code, child))
    process.start()
    child.close()
    if parent.poll(timeout_seconds):
        status, result, error = parent.recv()
        process.join(timeout=0.2)
        return SafeExecutionResult(
            status=status,
            result=result,
            error=error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    process.terminate()
    process.join(timeout=0.5)
    return SafeExecutionResult(
        status="timeout",
        error=f"Execution exceeded {timeout_seconds} seconds",
        duration_ms=int((time.monotonic() - started) * 1000),
    )
