from tools.visualize_gs import load_gaussian_splats_from_ply, register_pc
import polyscope as ps
import numpy as np
from plyfile import PlyData

# Store transformation state
state = {
    "translation": np.array([0.0, 0.0, 0.0]),
    "scale": 1.0,
}

positions_o = None

def apply_transformations():
    global positions_o
    # Create a 4x4 transformation matrix (
    transform_matrix = np.eye(4)
    transform_matrix[0, 0] = transform_matrix[1, 1] = transform_matrix[2, 2] = state["scale"]
    transform_matrix[:3, 3] = state["translation"]

    # 2. Apply matrix to original positions (requires homogeneous coordinates)
    # Add a '1' to each position vector to make it (x,y,z,1)
    homogeneous_coords = np.hstack((positions_o, np.ones((positions_o.shape[0], 1))))
    transformed_coords = (transform_matrix @ homogeneous_coords.T).T

    # 3. Update the Polyscope point cloud with the new positions
    ps.get_point_cloud("Object").update_point_positions(transformed_coords[:, :3])


def callback():
    # Create a slider for scale
    changed_scale, new_scale = ps.imgui.SliderFloat("Object Scale", state["scale"], v_min=0.1, v_max=10.0)
    if changed_scale:
        state["scale"] = new_scale

    # Create a 3D input vector for translation
    changed_trans, new_trans = ps.imgui.InputFloat3("Object Translation", state["translation"])
    if changed_trans:
        state["translation"] = new_trans

    # If any value changed, apply the new transformations
    if changed_scale or changed_trans:
        apply_transformations()


def main():
    object_ply_file_path = "object_models/hotdog/point_cloud/iteration_30000/point_cloud.ply"
    scene_ply_file_path = "scene_models/bicycle/point_cloud/iteration_30000/point_cloud.ply"

    global positions_o
    positions_o, colors_o, opacities_o, scales_o = load_gaussian_splats_from_ply(object_ply_file_path)
    positions_s, colors_s, opacities_s, scales_s = load_gaussian_splats_from_ply(scene_ply_file_path)

    # Initialize Polyscope
    ps.init()
    register_pc(name="Object", positions=positions_o, colors=colors_o, opacities=opacities_o, scales=scales_o)
    register_pc(name="Scene", positions=positions_s, colors=colors_s, opacities=opacities_s, scales=scales_s)

     # Set the callback function to be executed each frame
    ps.set_user_callback(callback)

    # Show Visualization
    ps.show()

if __name__ == "__main__":
    main()