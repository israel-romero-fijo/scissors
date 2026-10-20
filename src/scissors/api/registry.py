import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import best_match
from pydantic import ValidationError as PydanticError
from pydantic import create_model

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.persistence import json_bytes

logger = logging.getLogger(__name__)


def _validation_location(path: Any) -> str:
    """Render a validation path such as ``background[0]`` or ``rows.3.figure``."""
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        elif text:
            text += f".{part}"
        else:
            text = str(part)
    return text


def _invalid_message(
    subject: str,
    location: str,
    reason: str,
    expected: str | None = None,
    received: str | None = None,
) -> str:
    where = f" {location}" if location else ""
    if expected is not None:
        head = f"expected {expected}"
        if received is not None:
            head += f", received {received}"
        return f"Invalid {subject}{where}: {head}. {reason}"
    return f"Invalid {subject}{where}: {reason}"


def _short(value: Any, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _schema_expectation(match: Any) -> str | None:
    """Describe a failed JSON Schema constraint so callers see the accepted range."""
    validator = getattr(match, "validator", None)
    value = getattr(match, "validator_value", None)
    if validator == "maximum":
        return f"<= {value}"
    if validator == "minimum":
        return f">= {value}"
    if validator == "exclusiveMaximum":
        return f"< {value}"
    if validator == "exclusiveMinimum":
        return f"> {value}"
    if validator == "maxItems":
        return f"at most {value} items"
    if validator == "minItems":
        return f"at least {value} items"
    if validator == "maxLength":
        return f"at most {value} characters"
    if validator == "minLength":
        return f"at least {value} characters"
    if validator == "pattern":
        return f"text matching {value}"
    if validator == "enum" and isinstance(value, list):
        return "one of " + ", ".join(str(item) for item in value)
    if validator == "const":
        return f"exactly {value}"
    if validator == "type":
        return f"type {value}"
    if validator == "required":
        return f"field {value} is required"
    if validator == "additionalProperties":
        return "no unknown fields"
    return None


def _pydantic_expectation(first: dict[str, Any]) -> str | None:
    kind = str(first.get("type", ""))
    context = first.get("ctx") or {}
    table = {
        "greater_than": f"> {context.get('gt')}",
        "greater_than_equal": f">= {context.get('ge')}",
        "less_than": f"< {context.get('lt')}",
        "less_than_equal": f"<= {context.get('le')}",
        "string_too_long": f"at most {context.get('max_length')} characters",
        "string_too_short": f"at least {context.get('min_length')} characters",
        "too_short": f"at least {context.get('min_length')} items",
        "too_long": f"at most {context.get('max_length')} items",
        "missing": "the field is required",
        "extra_forbidden": "no unknown fields",
    }
    return table.get(kind)


@dataclass
class Operation:
    name: str
    description: str
    executor: Callable[..., Any]
    parameters: dict[str, Any]
    version: str = "1.0.0"
    result_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    examples: list[dict[str, Any]] = field(default_factory=list)
    category: str = ""
    plugin: str | None = None
    side_effects: bool = True
    permissions: list[str] = field(default_factory=lambda: ["scene.write"])
    scene_mutation: bool | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters,
            "result_schema": self.result_schema,
            "examples": self.examples,
            "category": self.category or self.name.split(".")[0],
            "plugin": self.plugin,
            "side_effects": self.side_effects,
            "permissions": self.permissions,
            "scene_mutation": self.side_effects
            if self.scene_mutation is None
            else self.scene_mutation,
        }


class OperationRegistry:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.operations: dict[str, Operation] = {}
        self.revision = 0
        self.batch_depth = 0
        self.transaction_depth = 0
        self._rollback_stack: list[list[Callable[[], None]]] = []

    def _on_rollback(self, cleanup: Callable[[], None]) -> None:
        """Enlist cleanup of files owned by the active native operation."""
        if not self._rollback_stack:
            raise RuntimeError("Rollback cleanup requires an active registry call")
        self._rollback_stack[-1].append(cleanup)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group many mutations behind one rollback point and one checkpoint.

        Per-operation rollback snapshots and state writes are suspended, so a script that
        builds a large scene pays for one snapshot and one commit instead of thousands.
        Any exception restores the state and catalog captured on entry, discarding every
        change made inside the block. A failed operation that leaves the engine dirty must
        therefore end the whole transaction rather than be swallowed by the caller.
        """
        with self.engine.mutex:
            if self.transaction_depth:
                raise ScissorsError("INVALID_PARAMETER", "Nested transactions are unsupported")
            snapshot = self.engine.snapshot()
            catalog = dict(self.operations)
            revision = self.revision
            self.transaction_depth = 1
            try:
                yield
                self.engine.check_state()
                self.engine.commit()
            except BaseException:
                self.engine.restore(snapshot)
                self.operations = catalog
                self.revision = revision
                raise
            finally:
                self.transaction_depth = 0

    def register(self, operation: Operation) -> None:
        with self.engine.mutex:
            if operation.scene_mutation is True and not operation.side_effects:
                raise ValueError("Scene mutation requires side_effects=true")
            if operation.name in self.operations:
                raise ScissorsError("OPERATION_COLLISION", "Operation name already registered")
            Draft202012Validator.check_schema(operation.parameters)
            Draft202012Validator.check_schema(operation.result_schema)
            self.operations[operation.name] = operation
            self.revision += 1

    def unregister(self, name: str) -> None:
        with self.engine.mutex:
            if name not in self.operations:
                raise ScissorsError("UNKNOWN_OPERATION", "Operation is not registered")
            del self.operations[name]
            self.revision += 1

    def list_operations(self, category: str | None = None) -> list[dict[str, Any]]:
        with self.engine.mutex:
            return [
                {
                    k: v
                    for k, v in op.describe().items()
                    if k not in {"parameters", "result_schema", "examples"}
                }
                for op in self.operations.values()
                if category is None or (op.category or op.name.split(".")[0]) == category
            ]

    def describe_operation(self, name: str) -> dict[str, Any]:
        with self.engine.mutex:
            if name not in self.operations:
                raise ScissorsError("UNKNOWN_OPERATION", "Operation is not registered")
            return self.operations[name].describe()

    def add_function(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        side_effects: bool = True,
        permissions: list[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
        scene_mutation: bool | None = None,
    ) -> None:
        hints = get_type_hints(function, include_extras=True)
        fields = {
            key: (
                hints[key],
                parameter.default if parameter.default is not inspect.Parameter.empty else ...,
            )
            for key, parameter in inspect.signature(function).parameters.items()
        }
        model = create_model(name.replace(".", "_"), **fields)  # type: ignore[call-overload]
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        if examples is None:
            values: dict[str, Any] = {}
            for key in schema.get("required", []):
                parameter = schema["properties"][key]
                kind = parameter.get("type")
                if key == "output":
                    values[key] = "exports/model.glb"
                elif key == "path":
                    values[key] = "scenes/SCENE_ID.json"
                elif key == "object_ids":
                    values[key] = ["OBJECT_ID_FROM_PREVIOUS_RESULT"]
                elif key == "vertices":
                    values[key] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
                elif key == "faces":
                    values[key] = [[0, 1, 2]]
                elif kind == "string":
                    values[key] = f"{key.upper()}_FROM_PREVIOUS_RESULT"
                elif kind == "array":
                    values[key] = []
                elif kind == "object":
                    values[key] = {}
                elif kind in {"number", "integer"}:
                    values[key] = 1
            examples = [values]
        self.register(
            Operation(
                name,
                description,
                function,
                schema,
                side_effects=side_effects,
                permissions=permissions or (["scene.write"] if side_effects else ["scene.read"]),
                examples=examples or [],
                scene_mutation=scene_mutation,
            )
        )

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.engine.mutex:
            cleanups: list[Callable[[], None]] = []
            cleanup_frame = False
            snapshot = None
            catalog = None
            checking_result = False
            revision = self.revision
            try:
                if self.engine.closed:
                    raise ScissorsError("WORKSPACE_CLOSED", "Scissors has been closed")
                if name not in self.operations:
                    raise ScissorsError("UNKNOWN_OPERATION", "Operation is not registered")
                op = self.operations[name]
                mutates_scene = op.side_effects and op.scene_mutation is not False
                if self.transaction_depth and any(
                    permission not in {"scene.read", "scene.write", "filesystem.read"}
                    for permission in op.permissions
                ):
                    raise ScissorsError(
                        "INVALID_PARAMETER", "Transactions allow scene changes and read-only files"
                    )
                args = arguments if arguments is not None else {}
                if len(json_bytes(args)) > self.engine.settings.max_file_bytes:
                    raise ScissorsError("RESOURCE_LIMIT", "Arguments exceed size limit")
                Draft202012Validator(op.parameters).validate(args)
                self._rollback_stack.append(cleanups)
                cleanup_frame = True
                if mutates_scene and not self.transaction_depth and not self.batch_depth:
                    snapshot = self.engine.snapshot()
                    catalog = dict(self.operations)
                checking_result = True
                result = op.executor(**args)
                Draft202012Validator(op.result_schema).validate(result)
                checking_result = False
                if (
                    mutates_scene
                    and not self.batch_depth
                    and not self.transaction_depth
                    and len(self._rollback_stack) == 1
                ):
                    self.engine.check_state()
                    self.engine.commit()
                self._rollback_stack.pop()
                cleanup_frame = False
                if self._rollback_stack:
                    self._rollback_stack[-1].extend(cleanups)
                return {
                    "success": True,
                    "operation": name,
                    "result": result,
                    "warnings": result.get("warnings", []) if isinstance(result, dict) else [],
                }
            except ValidationError as exc:
                failed_schema = exc.schema if isinstance(exc.schema, dict) else {}
                unavailable = (
                    failed_schema.get("x-unavailable-values", {}).get(exc.instance)
                    if exc.validator == "enum" and isinstance(exc.instance, str)
                    else None
                )
                if unavailable:
                    error = ScissorsError(
                        "BACKEND_UNAVAILABLE",
                        f"Requested value {exc.instance} requires {unavailable['requirement']}",
                        {
                            "parameter": list(exc.path),
                            "requested_value": exc.instance,
                            "available_values": failed_schema["enum"],
                            **unavailable,
                        },
                    )
                else:
                    # Optional parameters wrap their schema in anyOf; descend to the
                    # specific failure so the message names the field and the limit.
                    match = best_match([exc]) or exc
                    location = _validation_location(match.absolute_path)
                    reason = match.message[:500]
                    expected = _schema_expectation(match)
                    received = _short(match.instance) if expected is not None else None
                    subject = "result" if checking_result else "parameter"
                    error = ScissorsError(
                        "INVALID_PARAMETER",
                        _invalid_message(subject, location, reason, expected, received),
                        {
                            "reason": reason,
                            "parameter": list(match.absolute_path),
                            "expected": expected,
                            "received": match.instance if expected is not None else None,
                        },
                    )
            except PydanticError as exc:
                errors = exc.errors()
                first: dict[str, Any] = dict(errors[0]) if errors else {}
                location = _validation_location(first.get("loc", ()))
                reason = str(first.get("msg", str(exc)))[:500]
                expected = _pydantic_expectation(first)
                received = _short(first.get("input")) if expected is not None else None
                error = ScissorsError(
                    "INVALID_PARAMETER",
                    _invalid_message("parameter", location, reason, expected, received),
                    {
                        "reason": reason,
                        "parameter": list(first.get("loc", ())),
                        "expected": expected,
                        "received": first.get("input") if expected is not None else None,
                    },
                )
            except (ValueError, TypeError) as exc:
                reason = str(exc).strip()[:500] or type(exc).__name__
                error = ScissorsError("INVALID_PARAMETER", reason, {"reason": reason})
            except ScissorsError as exc:
                error = exc
            except Exception as exc:
                logger.error("Operation %s failed (%s)", name, type(exc).__name__)
                error = ScissorsError(
                    "INTERNAL_ERROR",
                    "Operation failed; consult local diagnostics",
                    {"exception_type": type(exc).__name__},
                )
            except BaseException:
                if cleanup_frame:
                    self._rollback_stack.pop()
                if snapshot is not None:
                    self.engine.restore(snapshot)
                    if catalog is not None:
                        self.operations = catalog
                    self.revision = revision
                for cleanup in reversed(cleanups):
                    try:
                        cleanup()
                    except Exception as exc:
                        logger.error("Interrupted native cleanup failed (%s)", type(exc).__name__)
                raise
            if cleanup_frame:
                self._rollback_stack.pop()
            if snapshot is not None:
                self.engine.restore(snapshot)
                if catalog is not None:
                    self.operations = catalog
                self.revision = revision
            for cleanup in reversed(cleanups):
                try:
                    cleanup()
                except Exception as exc:
                    logger.error("Native rollback cleanup failed (%s)", type(exc).__name__)
                    error.details.setdefault("cleanup_errors", []).append(type(exc).__name__)
            return {"success": False, "operation": name, "error": error.as_dict()}

    def batch(self, operations: list[dict[str, Any]], atomic: bool = True) -> dict[str, Any]:
        if self.batch_depth or len(operations) > 100:
            raise ScissorsError("RESOURCE_LIMIT", "Nested batches or more than 100 operations")
        for item in operations:
            name = item.get("operation", "")
            if name in {"operations.batch", "scene.edit"}:
                raise ScissorsError("INVALID_PARAMETER", "Nested batches are unsupported")
            op = self.operations.get(name)
            if (
                atomic
                and op
                and any(
                    p not in {"scene.read", "scene.write", "filesystem.read"}
                    for p in op.permissions
                )
            ):
                raise ScissorsError(
                    "INVALID_PARAMETER", "Atomic batches allow scene changes and read-only files"
                )
        self.batch_depth += 1
        results = []
        try:
            for item in operations:
                # Atomic batches already have the outer rollback snapshot. Non-atomic
                # batches retain each successful step but restore a failed step.
                step_state = self.engine.snapshot() if not atomic else None
                step_catalog = dict(self.operations) if not atomic else None
                step_revision = self.revision
                result = self.execute(item.get("operation", ""), item.get("arguments", {}))
                if not result["success"] and step_state is not None:
                    self.engine.restore(step_state)
                    if step_catalog is not None:
                        self.operations = step_catalog
                    self.revision = step_revision
                results.append(result)
                if atomic and not result["success"]:
                    raise ScissorsError(
                        "BATCH_FAILED", "Atomic batch rolled back", {"results": results}
                    )
        finally:
            self.batch_depth -= 1
        return {
            "results": results,
            "atomic": atomic,
            "success_count": sum(r["success"] for r in results),
        }
