import polyscope as ps
import numpy as np
from plyfile import PlyData

def load_gs_from_ply(ply_path):
    """Loads the xyz positions and base colors (f_dc) from a 3DGS .ply file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Extract and convert base color (SH degree 0)
    SH_C0 = 0.28209479177387814
    colors = np.vstack([
        0.5 + SH_C0 * vertex[f'f_dc_0'],
        0.5 + SH_C0 * vertex[f'f_dc_1'],
        0.5 + SH_C0 * vertex[f'f_dc_2']
    ]).T
    colors = np.clip(colors, 0.0, 1.0)
    
    print(f"Loaded {len(xyz)} points from {ply_path}")
    return xyz, colors

def register_pc(name, positions, colors):
    # Register the Point Cloud 
    pc = ps.register_point_cloud(
        name,
        positions,
        transparency = 0.5)

    pc.set_radius(0.001, relative=True)
    pc.add_color_quantity("colors", colors, enabled=True)
    return pc

def main():
    ply_file_path = "object_models/hotdog/point_cloud/iteration_30000/point_cloud.ply"
    ply_file_path = "scene_models/bicycle/point_cloud/iteration_30000/point_cloud.ply"
    ply_file_path = "scene_models/bicycle/point_cloud/iteration_30000/point_cloud_clean.ply"

    positions, colors, opacities, scales = load_gs_from_ply(ply_file_path)
    print(positions.shape, colors.shape, opacities.shape, scales.shape)

    # Initialize Polyscope
    ps.init()
    register_pc("TEST", positions, colors, opacities, scales)

    # Show Visualization
    ps.show()

if __name__ == "__main__":
    main()