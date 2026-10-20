"""Bounded linear-space tinting and content/factor reuse."""

import numpy as np
import pytest
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.exchange import ExchangeContext, tinted_image


def reference(image, factor):
    rgba = np.asarray(image.convert("RGBA"), dtype=float) / 255
    rgb = rgba[:, :, :3]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear *= factor[:3]
    rgba[:, :, :3] = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055
    )
    if len(factor) == 4:
        rgba[:, :, 3] *= factor[3]
    return np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)


def test_neutral_tint_reuses_original_2048_image():
    image = Image.new("RGBA", (2048, 2048), (25, 50, 75, 128))
    assert tinted_image(image, np.ones(4), Settings()) is image


@pytest.mark.parametrize("factor", [[0.25, 0.75, 0.9], [0.5, 0.2, 0.8, 0.4]])
def test_tint_matches_reference_with_blocks_and_alpha(factor):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (39, 1031, 4), dtype=np.uint8))
    # Enough for the output plus small blocks, not full-image floating arrays.
    settings = Settings(max_memory_bytes=image.width * image.height * 4 + 60000)
    result = tinted_image(image, np.array(factor), settings)
    np.testing.assert_array_equal(np.asarray(result), reference(image, factor))
    assert result.size == image.size


def test_tint_rejects_output_that_cannot_fit():
    with pytest.raises(ScissorsError) as failure:
        tinted_image(
            Image.new("RGBA", (256, 256)), np.array([0.5, 1, 1]), Settings(max_memory_bytes=1024)
        )
    assert failure.value.code == "RESOURCE_LIMIT"


def test_encoded_tint_cache_uses_content_and_effective_factors(monkeypatch):
    import scissors.core.exchange as exchange

    context = ExchangeContext(trimesh.Scene(), Settings(), "cache")
    original = exchange.tinted_image
    calls = []

    def tint(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(exchange, "tinted_image", tint)
    image = Image.new("RGBA", (32, 16), (80, 120, 40, 128))
    a = context.image(image, np.array([0.5, 0.5, 0.5]))
    b = context.image(image.copy(), np.array([0.5, 0.5, 0.5, 1]))
    assert a == b and len(calls) == 1
    assert len(context.assets) == 1
    c = context.image(image, np.array([0.8, 0.5, 0.5, 1]))
    assert c != a and len(calls) == 2
