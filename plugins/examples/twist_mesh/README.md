# twist_mesh

`deform.twist_mesh` implements a progressive, deterministic rotation about the bounds center.
Use cylinders with `height_segments >= 8`; the defaults have enough cross-sections.
`falloff` is `linear` or `smoothstep`. The operation preserves faces and UV; the
parent recomputes normals and validates all staged geometry before committing.

From the repository: `uv run scissors plugins install plugins/examples/twist_mesh`.
Install enables the package. It runs Python with your OS account permissions. This example
is external code, not a privileged built-in. Test with `uv run scissors plugins test twist_mesh`.
