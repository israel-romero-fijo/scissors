# FBX rig retention

A **rig** is the skeleton, bind matrices and vertex weights used to deform a mesh.
An **animation clip** is a set of time curves that changes scene properties.
The **bind pose** is the undeformed mesh used as the basis for skinning.
A **retained document** is an immutable binary FBX stored inside the workspace.

## Import and export

```python
result = app.call("scene.import_mesh", path="assets/character.fbx")
print(result["rig"])
app.call(
    "scene.export", scene_id=result["scene_id"],
    output="exports/character.fbx", format="fbx",
)
```

The default `fbx_mode="preserve"` retains binary FBX rig and animation data when present.
The source document keeps its skeleton hierarchy, control-point weights, bind matrices,
curves, interpolation, clip timing and embedded media. External images are read only
through the workspace resolver and embedded into the retained document.
File textures need standard Video connections; their original image format is retained.
`scene.inspect` exposes summaries in `rigs`. A file with a rig can have zero animation clips.

The Scissors scene shows a static base-mesh view in right-handed Z-up scene units.
The retained export keeps source axes and units so the bind matrices and animation
properties do not change coordinate systems. Export writes binary FBX 7.4 and preserves
64-bit animation time arrays. This path uses no new dependency or external executable.

Use a new or empty scene, `source_up="auto"`, `scale=1` and no `position`.
Keep scene membership, hierarchy, names, visibility, placement and materials unchanged.
Scene/subtree copying is rejected for retained rigs. `scene.clear` removes retention.
Export the complete scene as FBX. Other formats, partial scenes, material splitting,
export LODs, root renaming and export transforms fail with `UNSUPPORTED_OPERATION`.
They do not silently produce a static version.

## Conservative mesh reduction

`mesh.simplify` can reduce a retained rig when surviving vertex positions and UV stay exact.
The FBX writer rebuilds the mesh and gives each surviving render vertex the exact weights
of its original control point. Bones, bind matrices and animation curves remain unchanged.
Check the operation's actual face count and warnings; preservation constraints can stop
reduction before the requested target or reject an invalid candidate.

Mesh edits require one Scissors mesh per source geometry. Edits to shared instances,
split material parts, blend shapes, extra deformers or dual-quaternion skinning are
unsupported. Original documents containing these data can still be retained unchanged.
Ambiguous coincident control points and new/moved vertex positions or changed UV fail
explicitly. Arbitrary remeshing, automatic weight transfer and rig editing are unavailable.
A failed export leaves scene state unchanged. As with other filesystem operations,
separate artifact writes are not an atomic scene batch.

## Static workflows and persistence

Set `fbx_mode="static"` to deliberately discard rig and animation data and use the normal
editable static import workflow. Static binary and ASCII FBX imports remain supported.
Rig retention requires binary input. Scissors has no animation player or animation editor.
Imported normalized previews do not evaluate skinning, constraints or animation.

Keep `assets/fbx_documents` with `state.json`, `.geometry` and `textures` when relocating
a workspace. Retained documents use content-addressed paths and are protected from public
artifact writes. Exports verify their SHA-256 before use. State references and their bounds
are validated on load. New document files are removed if their import rolls back.

An export is still subject to file and memory budgets. Embedded images can make rigged
FBX exports much larger than static files with separate PNGs. Reopen the exported FBX in
the intended consumer before delivery.
