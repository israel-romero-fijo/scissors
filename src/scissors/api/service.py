from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scissors.api.errors import ScissorsError
from scissors.api.operations import register_native
from scissors.api.registry import OperationRegistry
from scissors.config import Settings
from scissors.core.engine import Engine
from scissors.plugins.manager import PluginManager


class Result(SimpleNamespace):
    """Compact Python results with attribute and mapping access."""

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class Namespace:
    def __init__(self, app: "Scissors", category: str):
        self.app, self.category = app, category

    def __getattr__(self, name: str) -> Any:
        def invoke(*values: Any, **arguments: Any) -> Result:
            if values:
                import inspect

                operation = self.app.registry.operations.get(f"{self.category}.{name}")
                if operation is None:
                    raise ScissorsError("UNKNOWN_OPERATION", "Operation is not registered")
                parameters = list(inspect.signature(operation.executor).parameters)
                if (
                    self.category == "mesh"
                    and name.startswith("create_")
                    and "scene_id" in parameters
                ):
                    parameters.remove("scene_id")
                    if "scene_id" not in arguments:
                        scenes = list(self.app.engine.scenes)
                        if len(scenes) != 1:
                            raise ScissorsError(
                                "INVALID_PARAMETER",
                                "Short figure calls need scene_id or exactly one scene",
                            )
                        arguments["scene_id"] = scenes[0]
                if len(values) > len(parameters):
                    raise ScissorsError("INVALID_PARAMETER", "Too many positional arguments")
                for parameter, value in zip(parameters, values, strict=False):
                    if parameter in arguments:
                        raise ScissorsError(
                            "INVALID_PARAMETER", f"Argument {parameter} supplied twice"
                        )
                    arguments[parameter] = value
            if self.category == "scene" and name == "create" and "name" not in arguments:
                arguments["name"] = "Untitled"
            return Result(**self.app.call(f"{self.category}.{name}", **arguments))

        return invoke


class Exports:
    def __init__(self, app: "Scissors"):
        self.app = app

    def glb(self, scene_id: str, output: str = "exports/model.glb", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="glb", **options
            )
        )

    def gltf(
        self, scene_id: str, output: str = "exports/gltf/model.gltf", **options: Any
    ) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="gltf", **options
            )
        )

    def obj(self, scene_id: str, output: str = "exports/model.obj", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="obj", **options
            )
        )

    def stl(self, scene_id: str, output: str = "exports/model.stl", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="stl", **options
            )
        )

    def ply(self, scene_id: str, output: str = "exports/model.ply", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="ply", **options
            )
        )

    def dae(self, scene_id: str, output: str = "exports/model.dae", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="dae", **options
            )
        )

    def fbx(self, scene_id: str, output: str = "exports/model.fbx", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="fbx", **options
            )
        )

    def usdz(self, scene_id: str, output: str = "exports/model.usdz", **options: Any) -> Result:
        return Result(
            **self.app.call(
                "scene.export", scene_id=scene_id, output=output, format="usdz", **options
            )
        )


class Scissors:
    def __init__(self, workspace: str | Path | None = None, settings: Settings | None = None):
        config = settings or Settings()
        if workspace is not None:
            config = config.model_copy(update={"workspace": Path(workspace)})
        self.engine = Engine(config)
        try:
            self.registry = OperationRegistry(self.engine)
            register_native(self.registry, self.engine)
            self.plugins = PluginManager(self.engine, self.registry)
            self._register_plugins()
        except Exception:
            self.engine.close()
            raise
        self.scenes = Namespace(self, "scene")
        self.meshes = Namespace(self, "mesh")
        self.materials = Namespace(self, "material")
        self.textures = Namespace(self, "texture")
        self.uv = Namespace(self, "uv")
        self.deform = Namespace(self, "deform")
        self.sculpt = Namespace(self, "sculpt")
        self.objects = Namespace(self, "object")
        self.references = Namespace(self, "reference")
        self.export = Exports(self)

    def _register_plugins(self) -> None:
        def plugin_list() -> dict[str, Any]:
            return self.plugins.list()

        def plugin_describe(plugin_id: str) -> dict[str, Any]:
            return self.plugins.describe(plugin_id)

        def plugin_validate(path: str) -> dict[str, Any]:
            return self.plugins.validate(self.engine.path(path))

        def plugin_enable(plugin_id: str) -> dict[str, Any]:
            return self.plugins._enable(plugin_id)

        def plugin_disable(plugin_id: str) -> dict[str, Any]:
            return self.plugins._disable(plugin_id)

        def plugin_reload(plugin_id: str) -> dict[str, Any]:
            return self.plugins._enable(plugin_id)

        registrations: list[tuple[str, Callable[..., Any], bool]] = [
            ("list", plugin_list, False),
            ("describe", plugin_describe, False),
            ("validate", plugin_validate, False),
            ("enable", plugin_enable, True),
            ("disable", plugin_disable, True),
            ("reload", plugin_reload, True),
        ]
        for name, function, write in registrations:
            self.registry.add_function(
                f"plugins.{name}",
                function,
                (
                    f"{name.capitalize()} plugins. Packages use application storage; "
                    "activation belongs to this workspace."
                ),
                write,
                ["plugins.manage"] if write else ["plugins.read"],
            )

    def call(self, operation: str, **arguments: Any) -> dict[str, Any]:
        response = self.registry.execute(operation, arguments)
        if not response["success"]:
            error = response["error"]
            raise ScissorsError(error["code"], error["message"], error["details"])
        return response["result"]

    def transaction(self) -> AbstractContextManager[None]:
        """Group many calls behind one rollback point and one state write.

        Use it for scripted builds that make hundreds of calls: the state file is written
        once at the end instead of after every call. An exception inside the block
        discards all of its changes.
        """
        return self.registry.transaction()

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> "Scissors":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
