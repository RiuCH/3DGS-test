import polyscope as ps
import numpy as np
from plyfile import PlyData

def load_gaussian_splats_from_ply(ply_path):
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']

        # Extract positions
        positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

        # Extract and convert colors (RGB)
        SH_C0 = 0.28209479177387814
        sh_features = np.vstack([vertex[f'f_dc_{i}'] for i in range(3)]).T
        colors = 0.5 + SH_C0 * sh_features
        colors = np.clip(colors, 0.0, 1.0)

        # Extract and convert opacities (Alpha)
        if 'opacity' in vertex.data.dtype.names:
            opacities = 1 / (1 + np.exp(-vertex['opacity']))
        else:
            opacities = np.ones(len(positions))

        # Extract scales
        scales = np.vstack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2'])
        ]).T

        print(f"Loaded {len(positions)} Gaussian splats.")
        return positions, colors, opacities, scales

    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None, None, None, None


def register_pc(name, positions, colors, opacities, scales):
    # Register the Point Cloud 
    pc = ps.register_point_cloud(
        name,
        positions,
        transparency = 0.5,
    )

    # Set a starting point radius. 
    pc.set_radius(0.00005, relative=True)

    # Add Scalar Quantities
    avg_scale = np.mean(scales, axis=1)
    pc.add_scalar_quantity("opacity", opacities, enabled=True)
    pc.add_scalar_quantity("average_scale", avg_scale, enabled=True, cmap='viridis')

    # Add Colors
    pc.add_color_quantity("colors", colors, enabled=True)

def main():
    ply_file_path = "object_models/hotdog/point_cloud/iteration_30000/point_cloud.ply"
    ply_file_path = "scene_models/bicycle/point_cloud/iteration_30000/point_cloud.ply"

    positions, colors, opacities, scales = load_gaussian_splats_from_ply(ply_file_path)
    print(positions.shape, colors.shape, opacities.shape, scales.shape)

    # Initialize Polyscope
    ps.init()
    register_pc("TEST", positions, colors, opacities, scales)

    # Show Visualization
    ps.show()

if __name__ == "__main__":
    main()