import blenderproc as bproc

import json
import pathlib
from functools import reduce
import operator

import numpy as np
import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view



def create_room_ground():
    return bproc.object.create_primitive("PLANE", scale=[5, 5, 1])

def make_light_2():
    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_color([1, 1, 1])
    light.set_energy(1200)#random.uniform(100, 550))


def make_target_mat_obj():
    floor_z = 0

    obj = bproc.loader.load_obj("data/mat_plane.obj")[0]
    # obj.set_location(np.zeros(3))
    obj.set_location([0, 0, floor_z + 0.001])
    obj.set_scale([0.42, 0.42, 0.25])
    obj.set_cp("category_id", 1)
    return obj

def make_mat_keypoints():
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


OUTPUT_DATA_DIR = pathlib.Path("data/outputs/coco")

def _write_keypoints(*, output_dir, list_of_keypoints):
    js_str = json.dumps(list_of_keypoints, indent=4)
    (output_dir / "list_of_keypoints.json").write_text(js_str)


def _render_and_save(*, count, list_of_keypoints):
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

    output_dir = OUTPUT_DATA_DIR / f"{count}"

    bproc.writer.write_coco_annotations(
        str(output_dir),
        instance_segmaps=seg_data["instance_segmaps"],
        instance_attribute_maps=seg_data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    _write_keypoints(output_dir=output_dir, list_of_keypoints=list_of_keypoints)


def _convert_camera_coord(coord_2d, *, render_size):
    x = round(coord_2d.x * render_size[0])
    y = 512 - round(coord_2d.y * render_size[1])
    return x, y


def sample_random_camera(ground, poi, keypoints_builder, texture):
    ground.set_material(0, texture)
    list_of_keypoints = []

    for _ in range(2):
        # Sample random camera location above objects
        location = np.random.uniform([-1.35, -1.35, 1.95], [1.35, 1.35, 1.95])
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        keypoints = keypoints_builder(bpy.context.scene, bpy.context.scene.camera)
        list_of_keypoints.append(keypoints)

    return list_of_keypoints


bproc.init()


def main():
    obj = make_target_mat_obj()
    keypoints_locations = make_mat_keypoints()

    poi = bproc.object.compute_poi([obj])
    ground = create_room_ground()

    make_light_2()

    cc_textures = bproc.loader.load_ccmaterials("datasets")
    ground.add_material(cc_textures[0])

    render_size = set_resolution_and_get_render_size()

    def keypoints_builder(scene, camera):
        def _iter_tpls():
            for keypoints_coord in keypoints_locations:
                coord_2d = world_to_camera_view(scene, camera, keypoints_coord)
                coord_2d = _convert_camera_coord(coord_2d, render_size=render_size)
                yield coord_2d
        lis = [[*tp, 2] for tp in _iter_tpls()]
        return reduce(operator.add, lis, [])

    for count, texture in enumerate(cc_textures[:2], 1):
        list_of_keypoints = sample_random_camera(ground, poi, keypoints_builder, texture)
        _render_and_save(count=count, list_of_keypoints=list_of_keypoints)
        bproc.utility.reset_keyframes()


if __name__ == '__main__':
    main()
