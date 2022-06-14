import blenderproc as bproc

import random
from functools import reduce
import operator

import numpy as np

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj("mat_plane.obj")[0]
# obj.set_location(np.zeros(3))
obj.set_location([0, 0, 0.001])
obj.set_scale([0.42, 0.42, 0.25])
# scale = obj.get_scale()
# print(f"scale={scale}")
obj.set_cp("category_id", 1)
### target_objs = [obj]
poi = bproc.object.compute_poi([obj])


cc_textures = bproc.loader.load_ccmaterials("datasets")
for c in cc_textures:
    print(c.get_name())
print("########################")


# for i in range(1):
#     c = random.choice(cc_textures)
#     print(c.get_name())

obj = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
mat = random.choice(cc_textures)
print(mat.get_name())
obj.add_material(mat)

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

bproc.camera.set_resolution(512, 512)

# location = np.random.uniform([-1.35, -1.35, 1.95], [1.35, 1.35, 1.95])
# rotation_matrix = bproc.camera.rotation_from_forward_vec(location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
# matrix_world = bproc.math.build_transformation_mat(location, rotation_matrix)
location = np.random.uniform([-1.35, -1.35, 1.95], [1.35, 1.35, 1.95])
inplane_rot = np.random.uniform(-0.7854, 0.7854)
b = poi - location
rotation_matrix = bproc.camera.rotation_from_forward_vec(b, inplane_rot=inplane_rot)

# bproc.camera.add_camera_pose(matrix_world)
cam_pose = bproc.math.build_transformation_mat([1, 1, 3], rotation_matrix)
bproc.camera.add_camera_pose(cam_pose)

data = bproc.renderer.render()
bproc.writer.write_hdf5("output", data)

