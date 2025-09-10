import json
import numpy as np

def get_camera_positions_from_npy(npy_path):
    """Extracts camera positions from a Mip-NeRF 360 pose_bounds.npy file."""
    poses_bounds = np.load(npy_path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    
    positions = []
    for pose in poses:
        R_mat = pose[:3, :3]
        t_vec = pose[:3, 3]
        position = -np.transpose(R_mat) @ t_vec
        positions.append(position)
    return np.array(positions)

def get_camera_positions_from_json(json_path):
    """Extracts camera positions from a NeRF-Synthetic transforms.json file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    positions = []
    for frame in data['frames']:
        position = np.array(frame['transform_matrix'])[:3, 3]
        positions.append(position)
    return np.array(positions)

def get_camera_bbox_diagonal(camera_positions):
    """Calculates the diagonal length of the bounding box of camera positions."""
    min_coords = np.min(camera_positions, axis=0)
    max_coords = np.max(camera_positions, axis=0)
    return np.linalg.norm(max_coords - min_coords)

def calculate_initial_scale(scene_npy_path, object_json_path):
    scene_cam_positions = get_camera_positions_from_npy(scene_npy_path)
    object_cam_positions = get_camera_positions_from_json(object_json_path)

    scene_diagonal = get_camera_bbox_diagonal(scene_cam_positions)
    object_diagonal = get_camera_bbox_diagonal(object_cam_positions)

    if scene_diagonal < object_diagonal:
        initial_scale = scene_diagonal / object_diagonal
    else:
        initial_scale = object_diagonal / scene_diagonal
    return initial_scale

if __name__ == "__main__":
    scene_npy_path = "dataset/360_v2/bicycle/poses_bounds.npy"
    object_json_path = "dataset/nerf_synthetic/chair/transforms_train.json"

    initial_scale = calculate_initial_scale(scene_npy_path, object_json_path)

    print(f"Calculated Initial Scale (from BBox): {initial_scale:.4f}")