import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def default_plugins_dir() -> Path:
    """Return the application plugin store shared by this user's workspaces."""
    return Path(os.environ.get("SCISSORS_PLUGINS_DIR", Path.home() / ".scissors" / "plugins"))


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: Path = Path("workspaces/agents")
    plugins_dir: Path = Field(default_factory=default_plugins_dir)
    max_vertices: int = Field(default=200_000, gt=0)
    max_faces: int = Field(default=400_000, gt=0)
    max_file_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    max_memory_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    operation_timeout: float = Field(default=30, gt=0)
    max_texture_pixels: int = Field(default=16_777_216, gt=0)
    max_geometry_pairs: int = Field(default=2_000_000, gt=0)
