"""Generate the formal plugin manifest JSON Schema from its public Pydantic model."""

import json
from pathlib import Path

from scissors.plugins.manifest import PluginManifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/plugin-manifest.schema.json"
    path.write_text(
        json.dumps(PluginManifest.model_json_schema(), indent=2) + "\n", encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
