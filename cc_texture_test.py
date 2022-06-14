import blenderproc as bproc

import pprint
import random
from functools import reduce
import pathlib

import numpy as np


bproc.init()


def make_camera_position(poi):
    location = np.random.uniform([-1.35, -1.35, 1.95], [1.35, 1.35, 1.95])
    inplane_rot = np.random.uniform(-0.7854, 0.7854)
    forward_vec = poi - location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec, inplane_rot=inplane_rot)
    cam_pose = bproc.math.build_transformation_mat([1, 1, 3], rotation_matrix)
    return cam_pose


def make_mat_plane():
    # load the objects into the scene
    obj = bproc.loader.load_obj("data/mat_plane.obj")[0]
    # obj.set_location(np.zeros(3))
    obj.set_location([0, 0, 0.001])
    obj.set_scale([0.42, 0.42, 0.25])
    # scale = obj.get_scale()
    # print(f"scale={scale}")
    obj.set_cp("category_id", 1)
    ### target_objs = [obj]


def setup_light():
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)


def render(ground, poi, texture):
    ground.set_material(0, texture)

    bproc.camera.add_camera_pose(make_camera_position(poi))
    bproc.camera.add_camera_pose(make_camera_position(poi))

    data = bproc.renderer.render()
    bproc.writer.write_hdf5("output", data, append_to_existing_output=True)
    bproc.utility.reset_keyframes()


def main():
    ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    poi = bproc.object.compute_poi([ground])

    cc_textures = bproc.loader.load_ccmaterials("datasets")

    ground.add_material(cc_textures[0])

    setup_light()

    bproc.camera.set_resolution(512, 512)

    for i in range(0, 3):
        t = cc_textures[i]
        print("########################################")
        print(t.get_name())
        render(ground, poi, t)

    # data = {"colors": data1["colors"] + data2["colors"]}


if __name__ == '__main__':
    main()
