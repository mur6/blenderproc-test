import blenderproc as bproc

import sys

sys.path.append(".")

import json
import random
from functools import reduce
import operator
import time

import numpy as np
import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

from src.common import MATERIALS_PATH, OUTPUT_DATA_DIR


def create_room_ground():
    return bproc.object.create_primitive("PLANE", scale=[5, 5, 1])


def make_light_2():
    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    # loc = bproc.sampler.shell(
    #     center=[5, -5, 5],
    #     radius_min=4,
    #     radius_max=7,
    #     elevation_min=2,
    #     elevation_max=10,
    # )
    # light.set_location(loc)
    light.set_color([1, 1, 1])
    light.set_energy(1200)


def make_target_mat_obj():
    floor_z = 0

    obj = bproc.loader.load_obj("data/mat_plane.obj")[0]
    # obj.set_location(np.zeros(3))
    obj.set_location([0, 0, floor_z + 0.001])
    obj.set_scale([0.42, 0.42, 0.25])
    obj.set_cp("category_id", 1)
    return obj


def mat_keypoints_locations():
    p1 = [0.0, 0.218, 0.002]
    p2 = [-0.316, -0.107, 0.002]
    p3 = [0.316, -0.107, 0.002]
    keypoints_locations = list(map(mathutils.Vector, [p1, p2, p3]))
    return keypoints_locations


def set_resolution_and_get_render_size():
    bproc.camera.set_resolution(512, 512)
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return render_size


def _write_keypoints_and_bbox_data(*, output_dir, keypoints_list, bbox_list):
    d = dict(keypoints_list=keypoints_list, bbox_list=bbox_list)
    js_str = json.dumps(d, indent=4)
    (output_dir / "keypoints_and_bbox.json").write_text(js_str)


def _iter_class_segmaps_conv_to_bbox(class_segmap_list):
    def calc_bbox(*, x_ary, y_ary):
        x_min = x_ary.min()
        x_max = x_ary.max()
        y_min = y_ary.min()  # Y_MAX_PIXELS - y_ary.max()
        y_max = y_ary.max()  # Y_MAX_PIXELS - y_ary.min()
        width = x_max - x_min
        height = y_max - y_min
        return x_min, y_min, width, height

    for item in class_segmap_list:
        y_ary, x_ary = np.where(item == 1)
        x, y, width, height = map(int, calc_bbox(x_ary=x_ary, y_ary=y_ary))
        yield x, y, width, height


def _render_and_save(*, count, list_of_keypoints):
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    bbox_list = list(_iter_class_segmaps_conv_to_bbox(seg_data["class_segmaps"]))

    output_dir = OUTPUT_DATA_DIR / f"{count}"

    bproc.writer.write_coco_annotations(
        str(output_dir),
        instance_segmaps=seg_data["instance_segmaps"],
        instance_attribute_maps=seg_data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    _write_keypoints_and_bbox_data(
        output_dir=output_dir, keypoints_list=list_of_keypoints, bbox_list=bbox_list
    )


Y_MAX_PIXELS = 512


def _convert_camera_coord(coord_2d, *, render_size):
    x = round(coord_2d.x * render_size[0])
    y = Y_MAX_PIXELS - round(coord_2d.y * render_size[1])
    return x, y


def sample_random_camera(obj, ground, poi, *, keypoints_builder, texture, sample_count):
    ground.set_material(0, texture)
    list_of_keypoints = []

    for _ in range(sample_count):
        # Sample random camera location above objects
        location = np.random.uniform([-1.35, -1.35, 0.95], [1.35, 1.35, 2.35])
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854)
        )
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )
        bproc.camera.add_camera_pose(cam2world_matrix)
        keypoints = keypoints_builder(bpy.context.scene, bpy.context.scene.camera)
        list_of_keypoints.append(keypoints)

    # Add dust to all materials of the loaded object
    # for material in obj.get_materials():
    #     bproc.material.add_dust(material, strength=0.8, texture_scale=0.05)

    return list_of_keypoints


bproc.init()


def render_all(obj, ground, poi, *, cc_textures, texture_count, sample_count):
    render_size = set_resolution_and_get_render_size()

    def keypoints_builder(scene, camera):
        def _iter_tpls():
            for keypoints_coord in mat_keypoints_locations():
                coord_2d = world_to_camera_view(scene, camera, keypoints_coord)
                coord_2d = _convert_camera_coord(coord_2d, render_size=render_size)
                yield coord_2d

        lis = [[*tp, 2] for tp in _iter_tpls()]
        return reduce(operator.add, lis, [])

    for count, texture in enumerate(cc_textures[:texture_count], 1):
        list_of_keypoints = sample_random_camera(
            obj,
            ground,
            poi,
            keypoints_builder=keypoints_builder,
            texture=texture,
            sample_count=sample_count,
        )
        _render_and_save(count=count, list_of_keypoints=list_of_keypoints)
        bproc.utility.reset_keyframes()
        time.sleep(10)


def main():
    obj = make_target_mat_obj()
    poi = bproc.object.compute_poi([obj])
    ground = create_room_ground()

    make_light_2()

    cc_textures = bproc.loader.load_ccmaterials(MATERIALS_PATH)
    ground.add_material(cc_textures[0])

    render_all(
        obj, ground, poi, cc_textures=cc_textures, texture_count=1, sample_count=5
    )


if __name__ == "__main__":
    main()
