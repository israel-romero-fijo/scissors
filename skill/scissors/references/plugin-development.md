# Public plugin API 1.0

Create an independent package with `manifest.json`, `plugin.py`, a README and
`tests/test_plugin.py`. The complete minimal example below requires only installed
Scissors and NumPy. Choose IDs and operation names that do not collide with the catalog.
Place the folder at `<workspace>/plugin_staging/stretch_mesh/` when workspace filesystem
access is available. With MCP-only access, prepare the package where you can write files
and have the operator transfer it to the server workspace.

`manifest.json` fields: `id`, `name`, `version` (semantic three-part version),
`api_version` (`1.0`), `description`, optional `author`, `entrypoint` (local `.py` basename),
`dependencies` (installed PEP 508 requirements), `permissions` (`scene.read`, `scene.write`),
`operations` (list of name, description, input_schema, output_schema, examples, side_effects).
Input schemas must be object schemas; use `additionalProperties:false` and appropriate
limits. IDs use lowercase letters, digits and underscores, starting with a letter,
up to 64 characters. Operation names have the form `namespace.operation`, with lowercase
letters, digits and underscores, each component starting with a letter.
Scissors never installs dependencies automatically.

## Complete example

`manifest.json`:

```json
{
  "id": "stretch_mesh",
  "name": "Stretch mesh",
  "version": "1.0.0",
  "api_version": "1.0",
  "description": "Scale world Z coordinates by a positive factor.",
  "entrypoint": "plugin.py",
  "dependencies": ["numpy>=2.2,<3"],
  "permissions": ["scene.read", "scene.write"],
  "operations": [{
    "name": "deform.stretch",
    "description": "Scale Z around the world origin; preserve faces and UV.",
    "input_schema": {
      "type": "object",
      "properties": {
        "object_id": {"type": "string"},
        "factor": {"type": "number", "exclusiveMinimum": 0, "maximum": 100}
      },
      "required": ["object_id"],
      "additionalProperties": false
    },
    "output_schema": {
      "type": "object",
      "required": ["object_id", "vertex_count", "face_count", "bounds"]
    },
    "examples": [{"object_id": "OBJECT_ID", "factor": 2}],
    "side_effects": true
  }]
}
```

`plugin.py`:

```python
import math

import numpy as np
from scissors.plugins.context import PluginContext


def stretch(context: PluginContext, object_id: str, factor: float = 2) -> dict:
    if not math.isfinite(factor) or not 0 < factor <= 100:
        raise ValueError("factor must be finite and in (0, 100]")
    mesh = context.get_mesh(object_id)
    vertices = np.asarray(mesh["vertices"], dtype=float)
    vertices[:, 2] *= factor
    context.replace_mesh(object_id, vertices, mesh["faces"])
    return context.statistics(object_id)


OPERATIONS = {"deform.stretch": stretch}
```

`tests/test_plugin.py` (the operator's test environment needs pytest):

```python
import importlib.util
from pathlib import Path

import pytest
from scissors.plugins.context import PluginContext

spec = importlib.util.spec_from_file_location(
    "stretch_example", Path(__file__).resolve().parents[1] / "plugin.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def context():
    return PluginContext(
        {
            "obj": {
                "vertices": [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
                "faces": [[0, 1, 2]],
                "uv": [[0, 0], [1, 0], [0, 1]],
            }
        },
        [],
        writable=True,
    )


def test_stretch_preserves_faces_and_uv():
    ctx = context()
    result = module.stretch(ctx, "obj", 2)
    mesh = ctx.get_mesh("obj")
    assert mesh["vertices"] == [[0, 0, 2], [1, 0, 2], [0, 1, 2]]
    assert mesh["faces"] == [[0, 1, 2]]
    assert mesh["uv"] == [[0, 0], [1, 0], [0, 1]]
    assert result["bounds"] == [[0, 0, 2], [1, 1, 2]]


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf"), 101])
def test_invalid_factor_leaves_geometry_unchanged(factor):
    ctx = context()
    original = ctx.get_mesh("obj")
    with pytest.raises(ValueError):
        module.stretch(ctx, "obj", factor)
    assert ctx.get_mesh("obj") == original
```

Document the operation's origin, units, limits, dependencies and expected results in
the package README. Extend tests for the geometry and edge cases of your actual plugin.

## Public context contract

Inputs and outputs are validated against the declared schemas. Context receives only
objects named by `object_id`/`object_ids` and the destination named by `scene_id`.
`get_mesh` returns a copy with vertices, triangular faces, normals, UV, transform and
metadata. `replace_mesh` stages changes and keeps existing UV by default.
`create_mesh(scene_id, vertices, faces, name)` stages a new object; a successful operation
returns its stable ID in `created_objects`. `statistics` returns counts and bounds.
Successful calls apply validated geometry changes together and recompute normals.
Read-only operations cannot stage writes. Do not assume Trimesh is present in
the context; you can declare a dependency and use it explicitly in your own plugin.

Write tests for known coordinates, preserved topology, zero extents and invalid inputs.
Static validation does not prove code is safe or tests pass. Local workflow:

```bash
scissors --workspace /absolute/workspace plugins validate /absolute/workspace/plugin_staging/stretch_mesh
scissors --workspace /absolute/workspace plugins install /absolute/workspace/plugin_staging/stretch_mesh
scissors --workspace /absolute/workspace plugins test stretch_mesh
```

Here relative CLI paths refer to the current directory. MCP `plugins_validate` paths
are workspace-relative: `{"path":"plugin_staging/stretch_mesh"}`. The validation
result's `valid` flag must be true, even if the tool call itself succeeded.
Install copies the package into the application plugin store (`plugins_dir` in
`modeler.info`, default `~/.scissors/plugins`), enables it in the current workspace and
registers its operations in one step. All workspaces using that store share the package;
activation settings remain per workspace. Reopen other sessions to discover additions or
removals. Store selection uses
`SCISSORS_PLUGINS_DIR`, CLI `--plugins-dir` or Python `Settings.plugins_dir`. Installation runs
Python with the operator's permissions; MCP does not expose installation. The operator must
stop any server owning the workspace before using these CLI commands and restart it
afterward. If a test fails, run `plugins disable ID` or `plugins uninstall ID` while you fix
the package. After restart, discover `deform.stretch`, describe it and invoke it with an
existing `object_id`; no application source changes are needed. Edited files in an installed
package take effect on the next invocation.
If the public context cannot express the operation, report the required capability and
the API limitation to the maintainers. Keep the plugin implemented against public APIs.
