from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scissors.core.materials import Material
from scissors.core.mesh import MeshObject


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    units: str = "meters"
    coordinates: str = "right-handed, Z-up; transforms baked into vertices"
    format_version: int = 1
    objects: dict[str, MeshObject] = Field(default_factory=dict)
    materials: dict[str, Material] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene_id": self.id,
            "name": self.name,
            "units": self.units,
            "coordinates": self.coordinates,
            "format_version": self.format_version,
            "metadata": self.metadata,
            "objects": [obj.summary() for obj in self.objects.values()],
            "materials": [mat.model_dump() for mat in self.materials.values()],
        }
