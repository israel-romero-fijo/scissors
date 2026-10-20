# Anatomical modeling

These built-in operations use the same registry in Python, CLI, REST and MCP.
They edit ordinary triangle meshes on the CPU. Scissors does not infer anatomy
from a photograph. Choose brush positions, regions and projection axes explicitly.

## Stable terms

| Term | Meaning |
| --- | --- |
| Region | A sphere in world coordinates, defined by center and radius. |
| Falloff | The smooth reduction of an edit from the region center to its boundary. |
| Voxel size | Distance between samples of a signed-distance grid, in scene units. |
| Blend radius | Distance over which nearby volumes blend during smooth union. |
| Face group | A list of triangle indices that defines an unwrap region. |
| Atlas | UV regions packed in one image without intentional overlap. |
| Density weight | Relative image area allocated per unit of surface area. |
| Anisotropy | The ratio of maximum to minimum local UV stretch; 1 means equal stretch. |

## Dependencies and discovery

Standard Scissors installation includes manifold3d, xatlas and VTK. In a source
checkout use `uv sync`. VTK supplies CPU signed-distance sampling, surface extraction
and simplification. xatlas supplies chart generation and packing. No extra flags are
needed. Workers run under the configured operation timeout.

Check `modeler.info` and describe each operation. `backends.organic` reports VTK
availability. Operations remain discoverable when their backend is absent and return
`BACKEND_UNAVAILABLE` with the required standard installation. Never install dependencies
automatically when consuming Scissors. Native geometry libraries run in child processes for failure
containment, not as a full security sandbox.

## Operations

| Operation | Purpose | Main parameters |
| --- | --- | --- |
| `mesh.organic_union` | Blend closed volumes into a new surface. | scene_id, object_ids, voxel_size, blend_radius |
| `deform.local` | Move, scale or rotate a region with smooth falloff. | object_id, center, radius, displacement, scale, axis, angle_degrees |
| `sculpt.brush` | Add/remove volume, move vertices, smooth or sharpen. | object_id, center, radius, mode, strength, displacement, iterations |
| `mesh.remesh` | Reconstruct a closed volume and optionally simplify it. | object_id, voxel_size, reduction_percent, max_error_relative |
| `uv.unwrap` | Pack an atlas with region cuts and density control. | object_id, face_groups, density_weights, resolution, padding, max_anisotropy |
| `texture.project` | Bake and blend an image in a surface region. | object_id, texture_id, center, radius, size, u_axis, v_axis, opacity, resolution, front_only |
| `scene.compare_reference` | Place a photo beside a preview or overlay the model. | scene_id, reference_path, output, mode, opacity, camera parameters |

All distances use scene units. All vector parameters contain exactly three finite
numbers, except texture projection `size`, which contains two positive distances.
Geometry operations participate in atomic batches. Texture projection and reference
comparison write files and cannot participate in atomic batches. Failures restore scene
state through the registry. As with other filesystem operations, a file already written
before a checkpoint failure may remain unreferenced in the workspace.

### Smooth union and reconstruction

Use closed volumes with consistent outward winding. Smooth union creates a new object
and retains its sources. Delete sources after checking the result so exported assets
do not contain both versions. UV and materials do not transfer to the new object.
Disjoint inputs can still produce separate connected components. Move anatomical
volumes into contact before blending.

The union samples each mesh's signed distance and combines fields using a compact
polynomial smooth minimum. `blend_radius` controls the transition and can increase
volume near a junction. Smaller `voxel_size` improves resolution and costs more memory
and time. The grid is checked against estimated memory before allocation. Geometry
counts are checked before accepting the result.

`mesh.remesh` keeps the object ID and material, clears UV and leaves the recorded
transform unchanged. Vertices already contain baked transforms. Reconstruction does
not apply the recorded matrix again. Optional reduction uses `mesh.simplify`'s existing
conservative topology/error rules and can stop short of its requested target.

**Limit:** reconstruction uses a uniform grid, not adaptive sampling. Features near or
below voxel size can disappear. Closed output does not prove fidelity to every fold.
The input must already be closed and oriented. Arbitrary holes, inverted faces and
self-intersections are not automatically repaired. Simplification's error estimate
does not bound reconstruction error or guarantee preservation of each anatomical detail.
Keep a duplicate before reconstruction and compare the silhouette afterward.

### Local deformation and sculpting

`deform.local` applies scale, then rotation about `center`, then displacement. A cubic
smoothstep falloff blends that target into the original positions. Vertices at or
outside the radius remain unchanged. UV, triangle indices, material and ID remain.

`sculpt.brush` modes are `inflate`, `deflate`, `move`, `smooth` and `sharpen`.
Inflate/deflate move along angle-weighted normals. Strength is a positive distance.
Move uses the displacement vector. Smooth/sharpen use an adjacency Laplacian with
strength in `(0,1]`; iterations controls repeated smoothing/sharpening only.
Brushes weld exact positions while computing normals and adjacency. Render vertices
remain split, so existing UV seams stay closed. Smoothing weights stay fixed during
iterations. Repeated small brushes can form a lip groove or neck fold.

**Limit:** brushes only move existing vertices. Subdivide before small details.
A region can affect both sides of a thin surface. Large edits can create
self-intersections; current validation does not detect all of them. Sharpening can
amplify noise. Use small strokes and inspect previews after each group of edits.

### UV cuts, density and distortion

`face_groups` must partition every input face exactly once. Their boundaries force
cuts. xatlas can add cuts within a group to unwrap a curved region. Without groups,
xatlas chooses all cuts. There is no automatic anatomical seam placement.
Triangle order can change during unwrapping; read current face indices before grouping.

Provide one positive `density_weights` value per group. A face weight of 4 gives
roughly four times the image area of an equal-sized region with weight 1, or twice
the linear texel density. Packing and pixel rounding can change the exact ratio.
`padding` gives atlas gutter pixels. The requested resolution guides packing; returned
`atlas_width` and `atlas_height` report actual dimensions.

The result and `uv.inspect` report degenerate UV triangles and maximum/95th-percentile
anisotropy. `max_anisotropy` rejects an unwrap above the chosen limit and restores
the previous mesh. This measures directional stretch, not texel density or all forms
of distortion. It does not measure image overlap for existing UV.
Collapsed UV triangles also reject the unwrap. Very thin reconstruction triangles
can collapse when xatlas converts positions to float32. Try conservative
`mesh.simplify` before unwrapping and check the resulting silhouette.

### Texture projection

Assign a material and unwrap an atlas before projection. Load a source image with
`texture.load`. `center` locates the projection plane and brush region. `u_axis`
and `v_axis` define image directions and must be nonzero and perpendicular. They
are normalized; `size` defines the projected width and height in scene units.
This lets you orient fur along the neck or muzzle.

The image is bilinearly sampled. Its alpha, opacity and region falloff blend it into
the base-color texture. Each call writes a new managed PNG and assigns it to the
material. Existing images remain available. With no base-color image, the material
color is baked into the new image and its factor becomes white to avoid multiplying
the color twice. A four-pixel gutter extends border colors to reduce filtering seams.

Use a material dedicated to this object: changing a shared material affects all objects
that use it. `front_only` rejects faces pointing away from the projection normal
`cross(u_axis, v_axis)`. It does not test occlusion. Projection can reach hidden surfaces
inside the region. UV outside the unit square and degenerate UV are rejected. Raster
overlap is checked at output pixel centers; subpixel overlaps may be missed. Projection
does not repair arbitrary old UV seams or bake normal/roughness maps.

### Reference comparison

Place the image under the workspace, for example `references/camel.png`.
`side_by_side` writes a PNG with the photo on the left and the model on the right.
Each panel has the requested width/height. `overlay` uses a transparent model silhouette
over the photo; `opacity=0` shows the photo alone. The photo keeps its aspect ratio.

`elevation_degrees` tilts about X, then `azimuth_degrees` rotates about Z.
Defaults match `scene.preview`. `zoom` scales the fitted model. Offsets move it in
fractions of panel width/height; positive X moves right, positive Y moves down.
Use the same camera values across comparisons. The renderer remains orthographic
with solid material colors. It does not sample textures or solve photographic perspective.

## Example

Run `python examples/anatomical_workflow.py --workspace PATH` with the standard installation.
The example blends a head and muzzle, makes local edits, unwraps, projects a checker,
and exports a GLB. It demonstrates the tools, rather than claiming a finished animal.

```python
app.call("sculpt.brush", object_id=head_id, center=[0.8, 0, 0.1],
         radius=0.12, mode="deflate", strength=0.02)
app.call("scene.compare_reference", scene_id=scene_id,
         reference_path="references/camel.png", mode="overlay",
         azimuth_degrees=0, elevation_degrees=90, opacity=0.4)
```

VTK's [signed-distance function](https://vtk.org/doc/nightly/html/classvtkImplicitPolyDataDistance.html)
and [sampling example](https://examples.vtk.org/site/Java/ImplicitFunctions/SampleFunction/)
describe the native reconstruction building blocks.
