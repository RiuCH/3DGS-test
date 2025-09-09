import polyscope as ps
import numpy as np
from plyfile import PlyData

def load_gaussian_splats_from_ply(ply_path):
    """
    Loads 3D Gaussian Splatting data from a .ply file.

    Args:
        ply_path (str): The path to the .ply file.

    Returns:
        tuple: A tuple containing arrays for positions, colors, opacities, and scales.
               Returns (None, None, None, None) if the file cannot be loaded.
    """
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']

        # Extract positions
        positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

        # --- Extract and convert colors ---
        # The base color is stored in the first three spherical harmonic coefficients (f_dc_*)
        # We need to apply the SH_C0 constant and shift from [-1, 1] to [0, 1]
        SH_C0 = 0.28209479177387814
        sh_features = np.vstack([vertex[f'f_dc_{i}'] for i in range(3)]).T
        colors = 0.5 + SH_C0 * sh_features
        colors = np.clip(colors, 0.0, 1.0) # Ensure colors are in the [0, 1] range

        # --- Extract and convert opacities ---
        # The opacity is often stored after a sigmoid activation.
        if 'opacity' in vertex.data.dtype.names:
            opacities = 1 / (1 + np.exp(-vertex['opacity']))
        else:
            opacities = np.ones(len(positions)) # Default to fully opaque if not found

        # --- Extract scales ---
        # Scales are often stored after an exponential activation.
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

def main():
    """
    Main function to load and visualize the Gaussian splats.
    """
    ply_file_path = "object_models/hotdog/point_cloud/iteration_30000/point_cloud.ply"
    # ----------------------------------------------

    positions, colors, opacities, scales = load_gaussian_splats_from_ply(ply_file_path)

    if positions is None:
        print("Could not load data. Exiting.")
        return

    # Initialize Polyscope
    ps.init()

    # --- Register the Point Cloud ---
    # We combine colors and opacities into an RGBA array for transparency.
    rgba_colors = np.hstack([colors, opacities[:, np.newaxis]])

    # Register the point cloud with its RGBA colors.
    # 'pretty' transparency mode gives a nice depth-aware feel.
    pc = ps.register_point_cloud(
        "Gaussian Splats",
        positions,
        color=rgba_colors,
        transparency_mode='pretty'
    )

    # Set a starting point radius. You can adjust this in the UI.
    pc.set_radius(0.002, relative=True)

    # --- Add Scalar Quantities for Inspection ---
    # This allows you to color the points by their properties in the UI.
    avg_scale = np.mean(scales, axis=1)
    pc.add_scalar_quantity("opacity", opacities, enabled=False)
    pc.add_scalar_quantity("average_scale", avg_scale, enabled=True, cmap='viridis')

    # Show the interactive viewer
    ps.show()

if __name__ == "__main__":
    main()