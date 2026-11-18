from typing import Any

from .ToolError import ToolArgumentError


def _get_parameters_schema(tool: Any) -> dict:
    if hasattr(tool, "get_openai_schema"):
        schema = tool.get_openai_schema() or {}
    elif hasattr(tool, "to_openai_schema"):
        schema = tool.to_openai_schema() or {}
    else:
        return {}

    if "function" in schema:
        return (schema.get("function") or {}).get("parameters") or {}
    return schema.get("parameters") or schema


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(tool: Any, kwargs: dict) -> None:
    """执行前做一次轻量校验，不合格直接抛 ToolArgumentError。"""
    schema = _get_parameters_schema(tool)
    if not schema:
        return

    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    missing = [
        name for name in required
        if name not in kwargs or kwargs[name] is None
    ]
    if missing:
        raise ToolArgumentError(
            f"缺少必填参数: {', '.join(missing)}",
            details={
                "missing": missing,
                "received": kwargs,
                "expected": properties,
            },
        )

    for name, value in kwargs.items():
        if name not in properties:
            if schema.get("additionalProperties") is False:
                raise ToolArgumentError(
                    f"不支持的参数: {name}",
                    details={
                        "received": kwargs,
                        "allowed": list(properties.keys()),
                    },
                )
            continue

        expected_type = (properties.get(name) or {}).get("type")
        if expected_type and not _type_ok(value, expected_type):
            raise ToolArgumentError(
                f"参数 {name} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}",
                details={
                    "received": kwargs,
                    "expected": {name: properties.get(name)},
                },
            )