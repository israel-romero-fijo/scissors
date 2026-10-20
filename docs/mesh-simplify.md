# Mesh simplification

`mesh.simplify` reduces the number of triangles in an existing object.
It is a native operation. Python, REST and MCP use the same registry.

## Terms

- **Face**: a triangle in the mesh.
- **UV**: coordinates that map the texture onto the mesh.
- **UV seam**: a boundary between regions with different texture coordinates.
- **Requested reduction**: the percentage of faces to try to remove.
- **Actual reduction**: the percentage of faces removed.

## Setup

VTK is included in the standard Scissors installation:

```bash
uv sync
uv run scissors info
```

Scissors requires VTK for its built-in surface tools. Its `vtkDecimatePro` filter runs on CPU without a window.
VTK has compiled packages for Windows and is included in the standard installation.
PyVista is not required.

`modeler.info` reports `backends.simplify`. The operation always appears in the catalog.
If VTK is missing, the call returns `BACKEND_UNAVAILABLE`. It does not install dependencies.
Missing VTK indicates an incomplete installation. Ask the local operator to repair
it.

## Usage

```python
result = app.call(
    "mesh.simplify",
    object_id=object_id,
    reduction_percent=50,
)
```

A value of 50 requests removal of half the faces. The percentage must be greater than 0
and less than 100. The target face count is rounded up and never falls below one face.

The operation uses one conservative method. It preserves the ID, material, object data
and diagnostic matrix. User metadata is preserved. Internal geometry revision and signature
fields change with the mesh, so previous geometry references become stale. The updated fields
persist across workspace restarts. It uses the vertices with their transforms already applied.
It recalculates normals. It preserves topology, disconnected components and open boundaries.

`uv_policy="preserve"` is the default. It welds duplicates with identical positions and UV.
It keeps UV seams separate. Surviving vertices retain their positions and UV.
The boundaries of UV regions remain protected.

The method prioritizes geometric error. It does not optimize a texture error metric.
Even with UV preserved, new triangulation can change their interpolation.
Identical appearance is not guaranteed. Meshes with many seams may achieve little reduction
or remain unchanged. There is no automatic switch to a destructive method.

`uv_policy="discard"` explicitly removes UV and allows coincident positions to be welded.
It returns a warning. The material is preserved, but UV must be generated before textured
export. Generating new UV does not automatically preserve the placement of the old texture.

`max_error_relative=0.01` limits VTK's accumulated geometric error estimate.
It is relative to mesh size according to VTK's normalization.
It accepts values from 0 to 1. It does not certify the maximum distance between surfaces
or the absence of intersections. A smaller value may prevent the requested reduction.

Decimation keeps creases sharper than 30 degrees, so hard mechanical edges survive while
curved areas reduce first. The backend sets this angle explicitly.

`require_target=False` allows partial reduction. With `True`, an unmet target returns
`GEOMETRY_ERROR` and restores the previous state.

## Results and errors

The result includes the usual statistics and these fields:

| Field | Meaning |
| --- | --- |
| `original_face_count` | Faces before simplification |
| `target_face_count` | Target number of faces |
| `requested_reduction_percent` | Requested removal percentage |
| `actual_reduction_percent` | Percentage removed |
| `target_reached` | Whether the requested reduction was reached or exceeded |
| `uv_policy` | Selected UV policy |
| `backend` | `vtk` |
| `warnings` | Notices about unmet targets or discarded UV |

The computation runs in a child process with the configured timeout.
It uses the existing JSON transport, bounded by `max_file_bytes`.
Memory usage is estimated in advance; this is not a hard operating system limit.
The parent validates the mesh, components, boundaries and protected UV before applying it.
Failures restore the scene through the registry. The operation supports atomic batches.

Main errors: `INVALID_PARAMETER`, `BACKEND_UNAVAILABLE`, `GEOMETRY_ERROR`,
`RESOURCE_LIMIT` and `OPERATION_TIMEOUT`.

After simplification, inspect `actual_reduction_percent`, validate the mesh and review
a preview. If you export a file, reopen it to verify the result.
