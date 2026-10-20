import re
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PluginOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    examples: list[dict[str, Any]] = Field(default_factory=list)
    side_effects: bool = True

    @model_validator(mode="after")
    def schemas_valid(self) -> "PluginOperation":
        Draft202012Validator.check_schema(self.input_schema)
        Draft202012Validator.check_schema(self.output_schema)
        if self.input_schema.get("type") != "object":
            raise ValueError("Input schema must describe an object")
        return self


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    api_version: Literal["1.0"]
    description: str
    author: str | None = None
    entrypoint: str = "plugin.py"
    dependencies: list[str] = Field(default_factory=list)
    permissions: list[Literal["scene.read", "scene.write"]] = Field(default_factory=list)
    operations: list[PluginOperation] = Field(min_length=1)

    @model_validator(mode="after")
    def entrypoint_valid(self) -> "PluginManifest":
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*\.py", self.entrypoint):
            raise ValueError("Entrypoint must be a Python file in the plugin directory")
        names = [op.name for op in self.operations]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate operation names")
        if any(op.side_effects for op in self.operations) and "scene.write" not in self.permissions:
            raise ValueError("Mutating operations require scene.write permission")
        return self
