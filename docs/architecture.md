# Architecture

Native operations are registered once in api/operations.py. Pydantic derives parameter
schemas from typed signatures; JSON Schema validates Python, REST and MCP calls identically.
Plugins supply manifest schemas. Catalog names are category.operation; direct MCP names
are category__operation plus convenience aliases.

Engine owns scenes, materials and texture/plugin metadata. Registry snapshots mutating
state, runs the executor, validates results/geometry/references, then atomically replaces
state.json using a flushed/fsynced same-directory file. Exceptions restore state and catalog.
Nested native registry calls checkpoint at the outermost successful mutation. Native
operations can enlist cleanup for files they own; cleanup propagates to the enclosing
call and runs on failure, including failed checkpoints. Texture transfer uses this
for its newly generated maps. Cleanup failures are reported in error details.
Interrupted native calls restore their snapshots and unwind cleanup before propagating
the interruption.
Workspace format 2 keeps metadata in that manifest and geometry in private numeric NPZ files.
Geometry files publish first; the manifest changes last. Failed publication keeps the previous
checkpoint usable. Unchanged geometry reuses its file. Legacy format 1 still loads and upgrades
on successful mutation. In-memory transaction snapshots remain full models.
Atomic batches suppress intermediate checkpoints and reject file writes/plugin lifecycle
effects. Read-only files are allowed. Non-atomic batches retain successful steps.
Artifact sidecars are individually atomic,
not part of a multi-file transaction; unreferenced files can remain after filesystem failure.

FileLock gives lifetime exclusive process ownership. RLock serializes registry calls from
HTTP threads and plugin execution. Context-manager exit/close releases the workspace.
FastAPI mounts SDK Streamable HTTP and enters its lifespan once; both transports share
one Scissors instance. Stdio owns a separate process/workspace.

Scene format 1 uses right-handed Z-up and named units. IDs use UUIDs independent of order.
Mesh stores scene-space vertices, triangular faces, normals, per-render-vertex UV, material
ID, cumulative baked transform and metadata. PBR materials reference workspace texture IDs.
Scene snapshots reference relative texture files; relocate whole workspaces for portability.
Scenes can also retain an immutable binary FBX document with control-point bindings. The
normalized scene view stays static. A separate FBX retention writer preserves source rig
and animation records, including source axes/units. It remaps exact surviving vertex weights
for supported conservative edits. Unsupported edits/exports fail explicitly. Retained
documents live under assets/fbx_documents and must move with the workspace.
See [FBX rig retention](fbx-rig.md).

UV seams duplicate render vertices. Topology diagnostics weld exact matching positions;
they do not identify self-intersections or intentional coincident components. Degenerate
triangles, bad indices, nonfinite data, malformed UV and exceeded counts are errors. Open
meshes and UV outside 0..1 are warnings unless closure is explicitly requested.

Plugins import only in child processes, receive targeted geometry copies and stage scoped
changes. Parent-side validation/recomputed normals precede one checkpoint. Every valid
package found in the application plugin store is enabled when a workspace opens unless
that workspace explicitly disabled it. Settings.plugins_dir selects shared package storage
(default ~/.scissors/plugins); state.json retains only workspace activation metadata.
A separate FileLock serializes package changes and worker access across workspaces. Workers give
failure containment, not OS security isolation. Native operations are trusted core code;
no separately privileged builtin plugin packages are included.

Registry revision changes trigger modern ToolsListChanged subscription events or legacy
notifications/tools/list_changed. REST changes are observed too. operations_invoke always
reads the current registry, independent of client tool caching.
