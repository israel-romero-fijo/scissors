"""Private child-process OpenGL probe. Native driver failures stay in the child."""

import json


def main() -> None:
    import vtk

    window = vtk.vtkRenderWindow()
    try:
        window.SetOffScreenRendering(1)
        window.SetSize(16, 16)
        window.AddRenderer(vtk.vtkRenderer())
        window.Render()
        report = window.ReportCapabilities() or ""
        device = ""
        for line in report.splitlines():
            if line.lower().startswith("opengl renderer string:"):
                device = line.split(":", 1)[1].strip()
        print(json.dumps({"device": device}))
    finally:
        window.Finalize()


if __name__ == "__main__":
    main()
