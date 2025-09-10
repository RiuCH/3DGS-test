from e3nn import o3
import einops
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

class GaussianData:
    """A helper class to load and hold all 3DGS attributes."""
    def __init__(self, ply_path):
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        def load_prop(prefix):
            names = sorted([p.name for p in vertex.properties if p.name.startswith(prefix)],
                           key=lambda x: int(x.split('_')[-1]))
            if not names: return np.empty([len(vertex), 0])
            return np.vstack([vertex[name] for name in names]).T

        self.xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        self.opacities = vertex['opacity'][..., np.newaxis]
        self.scales = load_prop("scale_")
        self.rotations = load_prop("rot_")
        self.features_dc = load_prop("f_dc_")[..., np.newaxis]
        self.features_rest = load_prop("f_rest_").reshape((len(self.xyz), -1, 3))

def transform_shs(shs, rotation_matrix):
    if shs.shape[1] == 0: return shs
    shs_torch = torch.from_numpy(shs).float()
    rot_torch = torch.from_numpy(rotation_matrix).float()
    rotated_shs = torch.zeros_like(shs_torch)
    for l in range(o3.deg_from_n(shs.shape[1] + 1)):
        D = o3.wigner_D(l, o3.matrix_to_angles(rot_torch)).to(shs_torch.device)
        indices = slice(l**2, (l+1)**2)
        shs_l = einops.rearrange(shs_torch[:, indices, :], 'n c rgb -> n rgb c')
        rotated_shs_l = einops.einsum(D, shs_l, 'i j, ... j -> ... i')
        rotated_shs[:, indices, :] = einops.rearrange(rotated_shs_l, 'n rgb c -> n c rgb')
    return rotated_shs.numpy()

def quat_multiply(q1, q2_wxyz):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2_wxyz
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.vstack([w, x, y, z]).T

def save_composition(object_full_data, scene_full_data, state, output_path):
    print("Applying final transformations for saving...")
    obj = object_full_data
    
    # --- Create a transformed copy of the object's full data ---
    transformed_obj = GaussianData.__new__(GaussianData)
    transformed_obj.xyz = obj.xyz.copy() * state["scale"]
    transformed_obj.scales = obj.scales.copy() + np.log(state["scale"])
    transformed_obj.rotations = obj.rotations.copy()
    transformed_obj.features_dc = obj.features_dc.copy()
    transformed_obj.features_rest = obj.features_rest.copy()
    transformed_obj.opacities = obj.opacities.copy()
    
    rotation = R.from_euler('xyz', state["rotation_euler"], degrees=True)
    rot_matrix = rotation.as_matrix()
    rot_quat = rotation.as_quat()
    rot_quat_wxyz = np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])
    
    transformed_obj.xyz = transformed_obj.xyz @ rot_matrix.T
    transformed_obj.rotations = quat_multiply(transformed_obj.rotations, rot_quat_wxyz)
    
    full_shs = np.concatenate((transformed_obj.features_dc, transformed_obj.features_rest), axis=1)
    rotated_shs = transform_shs(full_shs, rot_matrix)
    transformed_obj.features_dc = rotated_shs[:, 0:1, :]
    transformed_obj.features_rest = rotated_shs[:, 1:, :]
    
    transformed_obj.xyz += state["translation"]
    
    transformed_obj.features_dc = (transformed_obj.features_dc - 0.5) * state["contrast"] + 0.5
    transformed_obj.features_dc += state["brightness"]
    temp = state["temperature"] * 0.1
    transformed_obj.features_dc[:, 0, 0] += temp
    transformed_obj.features_dc[:, 0, 2] -= temp

    # --- Merge and Save ---
    merged = GaussianData.__new__(GaussianData)
    for attr in ['xyz', 'opacities', 'scales', 'rotations', 'features_dc', 'features_rest']:
        setattr(merged, attr, np.concatenate((getattr(scene_full_data, attr), getattr(transformed_obj, attr)), axis=0))

    xyz, f_dc, f_rest = merged.xyz, merged.features_dc.reshape(len(merged.xyz), -1), merged.features_rest.reshape(len(merged.xyz), -1)
    attrs = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), ('opacity', 'f4')]
    for i in range(merged.scales.shape[1]): attrs.append((f'scale_{i}', 'f4'))
    for i in range(merged.rotations.shape[1]): attrs.append((f'rot_{i}', 'f4'))
    for i in range(f_rest.shape[1]): attrs.append((f'f_rest_{i}', 'f4'))
    
    elements = np.empty(len(xyz), dtype=attrs)
    data = np.concatenate((xyz, np.zeros_like(xyz), f_dc, merged.opacities, merged.scales, merged.rotations, f_rest), axis=1)
    for i, name in enumerate(elements.dtype.names): elements[name] = data[:, i]
    
    PlyData([PlyElement.describe(elements, 'vertex')]).write(output_path)
    print(f"Successfully saved {len(xyz)} merged Gaussians to {output_path}")