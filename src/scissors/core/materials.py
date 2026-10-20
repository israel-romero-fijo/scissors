from pydantic import BaseModel, ConfigDict, Field


class Material(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    name: str = "Material"
    base_color: list[float] = Field(
        default_factory=lambda: [0.7, 0.7, 0.7, 1.0], min_length=4, max_length=4
    )
    metallic: float = Field(default=0, ge=0, le=1)
    roughness: float = Field(default=0.7, ge=0, le=1)
    alpha: float = Field(default=1, ge=0, le=1)
    double_sided: bool = False
    base_color_texture: str | None = None
    normal_texture: str | None = None
    metallic_roughness_texture: str | None = None
    emissive: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    emissive_texture: str | None = None
    occlusion_texture: str | None = None
