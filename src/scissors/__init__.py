"""Scissors public Python API."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scissors.api.service import Scissors
    from scissors.config import Settings

__all__ = ["Scissors", "Settings"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    # Importing a public plugin module must not also load the engine in its worker.
    if name == "Scissors":
        from scissors.api.service import Scissors

        globals()[name] = Scissors
        return Scissors
    if name == "Settings":
        from scissors.config import Settings

        globals()[name] = Settings
        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
