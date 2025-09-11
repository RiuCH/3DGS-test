import polyscope as ps
import numpy as np
from plyfile import PlyData
import json
from scipy.spatial.transform import Rotation as R
from tools.initial_scale import calculate_initial_scale
from tools.visualize_gs import register_pc, load_gs_from_ply
from tools.utils import save_composition, GaussianData

# ===============================================================
# --- USER CONFIGURATION ---
# ===============================================================
# --- File Paths ---
SCENE_FILE_PATH = "scene_models/bicycle/point_cloud/iteration_30000/point_cloud_clean.ply"
OBJECT_FILE_PATH = "object_models/chair/point_cloud/iteration_30000/point_cloud_clean.ply"

# --- Camera Info for Scale Calculation ---
SCENE_CAM_FILE_PATH = "dataset/360_v2/bicycle/poses_bounds.npy" # .npy for Mip-NeRF 360
OBJECT_CAM_FILE_PATH = "dataset/nerf_synthetic/chair/transforms_train.json" # .json for NeRF-Synthetic
# ===============================================================

class InteractiveAdjuster:
    def __init__(self, scene_file_path, object_file_path):
        # --- Load 3DGS Data ---
        scene_xyz, scene_colors = load_gs_from_ply(scene_file_path)
        object_xyz, object_colors = load_gs_from_ply(object_file_path)
        
        # Keep original copies for transformations
        self.original_object_xyz = object_xyz.copy()
        self.original_object_colors = object_colors.copy()

        # Load full data for saving
        self.scene_full_data = GaussianData(scene_file_path)
        self.object_full_data = GaussianData(object_file_path)

        # --- Initialize Polyscope Viewer ---
        ps.init()
        ps.set_background_color((0,0,0,0))
        
        register_pc("Scene", scene_xyz, scene_colors)
        self.obj_ps = register_pc("Object", object_xyz, object_colors)
        
        ps.look_at(camera_location=(-10, -5, -10), target=(0, 0, 0))

        # -- Set Scene Range for Slider --
        self.set_scene_range(scene_xyz)

        # -- Define State and Interactive Logic ---
        # self.state = {
        #     "translation": np.array([0.0, 0.0, 0.0]),
        #     "rotation_euler": np.array([0.0, 0.0, 0.0]),
        #     "scale": 1.0,
        #     "brightness": 0.0,
        #     "contrast": 1.0,
        #     "temperature": 0.0,
        #     "output_path": "merged_pointcloud.ply"
        # }
        self.state = {
            "translation": np.array([1.35, 1.205, 1.757]),
            "rotation_euler": np.array([107.191, 28.0, 8.0]),
            "scale": 0.268,
            "brightness": -0.173,
            "contrast": 1.68,
            "temperature": 0.03,
            "output_path": "merged_pointcloud.ply"
        }

    def set_scene_range(self, scene_xyz):
        scene_min_coords = np.min(scene_xyz, axis=0)
        scene_max_coords = np.max(scene_xyz, axis=0)
        # Add a 20% buffer 
        scene_range = scene_max_coords - scene_min_coords
        self.slider_min = scene_min_coords - scene_range * 0.2
        self.slider_max = scene_max_coords + scene_range * 0.2

    def set_initial_scale(self, scene_npy_path, object_json_path):
        self.initial_scale = calculate_initial_scale(scene_npy_path, object_json_path)
        self.state["scale"] = self.initial_scale

    def apply_transformations_and_color(self):
        """Applies all transformations and color corrections from the state dictionary."""
        
        # --- Apply Transformations ---
        scale_matrix = np.diag([self.state["scale"]] * 3)
        rotation_matrix = R.from_euler('xyz', self.state["rotation_euler"], degrees=True).as_matrix()
        
        # T * R * S
        transformed_xyz = (self.original_object_xyz @ scale_matrix.T @ rotation_matrix.T) + self.state["translation"]
        self.obj_ps.update_point_positions(transformed_xyz)

        # --- Apply Color Correction ---
        corrected_colors = self.original_object_colors.copy()
        corrected_colors = (corrected_colors - 0.5) * self.state["contrast"] + 0.5
        corrected_colors += self.state["brightness"]
        
        temp_factor = self.state["temperature"] * 0.1
        corrected_colors[:, 0] += temp_factor 
        corrected_colors[:, 2] -= temp_factor 
        
        # Update the point cloud's color
        self.obj_ps.add_color_quantity("Corrected Color", np.clip(corrected_colors, 0, 1), enabled=True)

    def callback(self):

        # Transformation Controls
        ps.imgui.Text("Transformations")
        changed_tx, tx_val = ps.imgui.SliderFloat("Translate X", self.state["translation"][0], v_min=self.slider_min[0], v_max=self.slider_max[0])
        changed_ty, ty_val = ps.imgui.SliderFloat("Translate Y", self.state["translation"][1], v_min=self.slider_min[1], v_max=self.slider_max[1])
        changed_tz, tz_val = ps.imgui.SliderFloat("Translate Z", self.state["translation"][2], v_min=self.slider_min[2], v_max=self.slider_max[2])
        changed_r, r_val = ps.imgui.SliderFloat3("Rotation", self.state["rotation_euler"], v_min=-180, v_max=180)
        changed_s, s_val = ps.imgui.SliderFloat("Scale", self.state["scale"], v_min=0.01, v_max=5.0)

        ps.imgui.Separator()

        # Color Correction Controls
        ps.imgui.Text("Color & Appearance")
        changed_b, b_val = ps.imgui.SliderFloat("Brightness", self.state["brightness"], v_min=-0.5, v_max=0.5)
        changed_c, c_val = ps.imgui.SliderFloat("Contrast", self.state["contrast"], v_min=0.0, v_max=3.0)
        changed_tmp, tmp_val = ps.imgui.SliderFloat("Temperature", self.state["temperature"], v_min=-1.0, v_max=1.0)

        # Save Controls
        changed_path, path_val = ps.imgui.InputText("Output Filename", self.state["output_path"])
        if changed_path: self.state["output_path"] = path_val
        
        if ps.imgui.Button("Save Merged PLY"):
            save_composition(self.object_full_data, self.scene_full_data, self.state, self.state["output_path"])

        # Check if any value was changed
        if any([changed_tx, changed_ty, changed_tz, changed_r, changed_s, changed_b, changed_c, changed_tmp]):
            if changed_tx: self.state["translation"][0] = tx_val
            if changed_ty: self.state["translation"][1] = ty_val
            if changed_tz: self.state["translation"][2] = tz_val
            if changed_r: self.state["rotation_euler"] = r_val
            if changed_s: self.state["scale"] = s_val
            if changed_b: self.state["brightness"] = b_val
            if changed_c: self.state["contrast"] = c_val
            if changed_tmp: self.state["temperature"] = tmp_val
            
            # Re-apply all transformations and colors
            self.apply_transformations_and_color()

    def run(self):
        ps.set_user_callback(self.callback)
        self.apply_transformations_and_color() 
        ps.show()

def main():
    interactive_adjuster = InteractiveAdjuster(SCENE_FILE_PATH, OBJECT_FILE_PATH)
    interactive_adjuster.set_initial_scale(SCENE_CAM_FILE_PATH, OBJECT_CAM_FILE_PATH)
    interactive_adjuster.run()

if __name__ == "__main__":
    main()