import blenderproc as bproc
import numpy as np
import random
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('scene', nargs='?', default="examples/resources/scene.obj", help="Path to the scene.obj file")
# parser.add_argument('output_dir', nargs='?', default="examples/basics/entity_manipulation/output", help="Path to where the final files, will be saved")
# args = parser.parse_args()

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj("mat_plane.obj")[0]
# obj.set_location(np.zeros(3))
obj.set_cp("category_id", 1)


# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
#light.set_energy(1000)
light.set_location(bproc.sampler.shell(
    center=obj.get_location(),
    radius_min=1,
    radius_max=5,
    elevation_min=1,
    elevation_max=89
))
# Randomly set the color and energy
light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
light.set_energy(random.uniform(100, 1000))

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# # Add two camera poses via location + euler angles
# bproc.camera.add_camera_pose(
#     bproc.math.build_transformation_mat([0, -13.741, 4.1242], [1.3, 0, 0])
# )
# bproc.camera.add_camera_pose(
#     bproc.math.build_transformation_mat([1.9488, -6.5202, 0.23291], [1.84, 0, 0.5])
# )

poi = bproc.object.compute_poi([obj])

# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# Enable transparency so the background becomes transparent
bproc.renderer.set_output_format(enable_transparency=True)

data = bproc.renderer.render()
seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

# # activate normal and depth rendering
# bproc.renderer.enable_normals_output()
# bproc.renderer.enable_depth_output(activate_antialiasing=False)

# # render the whole pipeline

# write the data to a .hdf5 container
# bproc.writer.write_hdf5("output", data)
# import pprint
# pprint.pprint(data.keys())
# bproc.writer.write_coco_annotations("output", )
bproc.writer.write_coco_annotations(
    "output_coco_data",
    instance_segmaps=seg_data["instance_segmaps"],
    instance_attribute_maps=seg_data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG",
)
