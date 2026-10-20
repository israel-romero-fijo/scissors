from scissors import Scissors

with Scissors("workspaces/cube") as app:
    scene = app.scenes.create(name="Example")
    cube = app.meshes.create_cube(scene_id=scene.id, size=2)
    app.meshes.translate(object_id=cube.id, x=1)
    print(app.export.glb(scene_id=scene.id).as_dict())
