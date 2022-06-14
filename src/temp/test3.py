import blenderproc as bproc
import argparse
import os
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
# parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
# parser.add_argument('output_dir', help="Path to where the final files will be saved ")
# parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
# args = parser.parse_args()

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj("mat_plane.obj")[0]
# obj.set_location(np.zeros(3))
obj.set_location([0, 0, 15])
obj.set_cp("category_id", 1)
target_objs = [obj]

# # load bop objects into the scene
# target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'), mm2m = True)

# # load distractor bop objects
# tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
# ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True)
# tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tyol'), mm2m = True)

# # load BOP datset intrinsics
# bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'))

# set shading and hide objects
# for obj in (target_bop_objs + tless_dist_bop_objs + ycbv_dist_bop_objs + tyol_dist_bop_objs):
#     obj.set_shading_mode('auto')
#     obj.hide(True)

# create room
room_planes = [
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
    ),
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

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive(
    "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
)
light_plane.set_name("light_plane")
light_plane_material = bproc.material.create("light_material")

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials("datasets")

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

#for i in range(2):

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

# sample CC Texture and assign to room planes
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# Sample object poses and check collisions
# bproc.object.sample_poses(
#     objects_to_sample=target_objs,
#     sample_pose_func=sample_pose_func,
#     max_tries=1000,
# )

# Physics Positioning
bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=3,
    max_simulation_time=10,
    check_object_interval=1,
    substeps_per_frame=20,
    solver_iters=25,
)


cam_poses = 0
while cam_poses < 25:
    # Sample location
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=0.44,
        radius_max=1.42,
        elevation_min=5,
        elevation_max=89,
    )
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(target_objs)
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159)
    )
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(
        location, rotation_matrix
    )


# # render the whole pipeline
# data = bproc.renderer.render()

# # Write data in bop format
# bproc.writer.write_bop(
#     os.path.join(args.output_dir, "bop_data"),
#     target_objects=target_objs,
#     dataset="hb",
#     depth_scale=0.1,
#     depths=data["depth"],
#     colors=data["colors"],
#     color_file_format="JPEG",
#     ignore_dist_thres=10,
# )

data = bproc.renderer.render()
seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

bproc.writer.write_coco_annotations(
    "output_coco_data",
    instance_segmaps=seg_data["instance_segmaps"],
    instance_attribute_maps=seg_data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG",
)

for obj in target_objs:
    obj.disable_rigidbody()
    obj.hide(True)
