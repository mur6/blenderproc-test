import blenderproc as bproc

import random
import pathlib
from functools import reduce
import operator

import numpy as np
import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view


floor_z = 0

def create_room_planes():
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[5, 5, 0]),
        # bproc.object.create_primitive(
        #     "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
        # ),
        # bproc.object.create_primitive(
        #     "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
        # ),
        # bproc.object.create_primitive(
        #     "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
        # ),
        # bproc.object.create_primitive(
        #     "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
        # ),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape="BOX",
            mass=1.0,
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
    return room_planes[0]

def make_light_planes():
    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive(
        "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
    )
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # Sample two light sources
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=1,
        radius_max=1.5,
        elevation_min=5,
        elevation_max=89,
    )
    light_point.set_location(location)


def make_light_2():
    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    #light.set_energy(1000)
    # light.set_location(bproc.sampler.shell(
    #     center=obj.get_location(),
    #     radius_min=1,
    #     radius_max=5,
    #     elevation_min=1,
    #     elevation_max=89
    # ))
    #light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    light.set_color([1, 1, 1])
    light.set_energy(1200)#random.uniform(100, 550))


bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj("mat_plane.obj")[0]
# obj.set_location(np.zeros(3))
obj.set_location([0, 0, floor_z + 0.001])
obj.set_scale([0.42, 0.42, 0.25])
# scale = obj.get_scale()
# print(f"scale={scale}")
obj.set_cp("category_id", 1)
target_objs = [obj]

def make_keypoints():
    p1 = [0.0, 0.218, 0.002]
    p2 = [-0.316, -0.107, 0.002]
    p3 = [0.316, -0.107, 0.002]
    keypoints_locations = list(map(mathutils.Vector, [p1, p2, p3]))
    return keypoints_locations

keypoints_locations = make_keypoints()

def show_keypoints_sphere(p1, p2, p3):
    print(keypoints_locations)
    print("###########")
    for k in keypoints_locations:
        print(k)

    marker1 = bproc.object.create_primitive(
        "SPHERE", scale=[0.03, 0.03, 0.03], location=p1
    )
    marker_l_bottom = bproc.object.create_primitive(
        "SPHERE", scale=[0.03, 0.03, 0.03], location=p2
    )
    marker_r_bottom = bproc.object.create_primitive(
        "SPHERE", scale=[0.03, 0.03, 0.03], location=p3
    )

floor_plane = create_room_planes()
#make_light_planes()
make_light_2()
cc_textures = bproc.loader.load_ccmaterials("datasets")

for cc_texture in cc_textures:
    floor_plane.add_material(cc_texture)

poi = bproc.object.compute_poi([obj])

# activate depth rendering without antialiasing and set amount of samples for color rendering
#bproc.renderer.enable_depth_output(activate_antialiasing=False)

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)
scene = bpy.context.scene
render_scale = scene.render.resolution_percentage / 100
render_size = (
    int(scene.render.resolution_x * render_scale),
    int(scene.render.resolution_y * render_scale),
)

def iter_cam_coords(coords_2d):
    for co_2d in coords_2d:
        x = round(co_2d.x * render_size[0])
        y = 512 - round(co_2d.y * render_size[1])
        #print(x, y)
        yield (x, y)


def build_keypoints(coords_2d):
    lis = [[*tp, 2] for tp in iter_cam_coords(coords_2d)]
    return reduce(operator.add, lis, [])

for i in range(len(floor_plane.get_materials())):
    # In 50% of all cases
    if np.random.uniform(0, 1) <= 0.5:
        # Replace the material with a random one
        floor_plane.set_material(i, random.choice(cc_textures))

# #floor_plane.replace_materials(cc_texture)
# selected_material = random.choice(cc_textures)
# if floor_plane.has_materials():
#     floor_plane.set_material(0, selected_material)
# else:
#     floor_plane.add_material(selected_material)



list_of_keypoints = []

# Sample five camera poses
for i in range(3):
    # Sample random camera location above objects
    location = np.random.uniform([-1.35, -1.35, 1.95], [1.35, 1.35, 1.95])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    coords_2d = [world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, coord) for coord in keypoints_locations]
    keypoints = build_keypoints(coords_2d)
    list_of_keypoints.append(keypoints)

import json
js_str = json.dumps(list_of_keypoints, indent=4)
pathlib.Path("list_of_keypoints.json").write_text(js_str)

# Enable transparency so the background becomes transparent
#bproc.renderer.set_output_format(enable_transparency=True)

data = bproc.renderer.render()
#print(data)
seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
# print(seg_data["instance_attribute_maps"])
# print(seg_data["instance_segmaps"])
bproc.writer.write_coco_annotations(
    "output_coco_data",
    instance_segmaps=seg_data["instance_segmaps"],
    instance_attribute_maps=seg_data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG",
)
