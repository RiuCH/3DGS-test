from e3nn import o3
import einops
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

class GaussianData:
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
        self.features_rest = load_prop("f_rest_").reshape((len(self.xyz), 3, -1))

def transform_shs(features, rotation_matrix):
    """
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    """

    try:
        from e3nn import o3
        import einops
        from einops import einsum
    except:
        print("Please run `pip install e3nn einops` to enable SHs rotation")
        return features

    if features.shape[1] == 1:
        return features

    features = torch.from_numpy(features)
    rotation_matrix = torch.from_numpy(rotation_matrix).to(torch.float32)

    features = features.clone()

    shs_feat = features[:, 1:, :]

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
    inversed_P = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=shs_feat.dtype, device=shs_feat.device)
    
    permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return features.numpy()

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
    obj = filter_background_gaussians(obj)
    
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

    print(transformed_obj.features_dc.shape, transformed_obj.features_rest.shape)
    
    full_shs = np.concatenate((transformed_obj.features_dc, transformed_obj.features_rest), axis=2).transpose((0, 2, 1))
    rotated_shs = transform_shs(full_shs, rot_matrix)
    transformed_obj.features_dc = rotated_shs[:, 0:1, :]
    transformed_obj.features_rest = rotated_shs[:, 1:, :]
    transformed_obj.features_rest = transformed_obj.features_rest.transpose((0, 2, 1))
    
    transformed_obj.xyz += state["translation"]
    
    transformed_obj.features_dc = (transformed_obj.features_dc - 0.5) * state["contrast"] + 0.5
    transformed_obj.features_dc += state["brightness"]
    temp = state["temperature"] * 0.1
    transformed_obj.features_dc[:, 0, 0] += temp
    transformed_obj.features_dc[:, 0, 2] -= temp
    transformed_obj.features_dc = transformed_obj.features_dc.transpose((0, 2, 1))

    # --- Merge and Save ---
    merged = GaussianData.__new__(GaussianData)
    for attr in ['xyz', 'opacities', 'scales', 'rotations', 'features_dc', 'features_rest']:
        print(attr, getattr(scene_full_data, attr).shape, getattr(transformed_obj, attr).shape)
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


def filter_background_gaussians(gaussian_data, color_threshold=0.8, opacity_threshold=0.4):

    print(f"Original object point count: {len(gaussian_data.xyz)}")

    SH_C0 = 0.2820947917738781
    colors = 0.5 + SH_C0 * gaussian_data.features_dc.squeeze()
    colors = np.clip(colors, 0.0, 1.0)

    opacities = 1 / (1 + np.exp(-gaussian_data.opacities.squeeze()))

    mask_is_white = (colors[:, 0] > color_threshold) & \
                    (colors[:, 1] > color_threshold) & \
                    (colors[:, 2] > color_threshold)
    
    mask_is_opacity = opacities > opacity_threshold

    mask_to_remove =  mask_is_opacity | mask_is_white
    mask_to_keep = ~mask_to_remove

    cleaned_data = GaussianData.__new__(GaussianData)
    for attr in ['xyz', 'opacities', 'scales', 'rotations', 'features_dc', 'features_rest']:
        original_attr = getattr(gaussian_data, attr)
        if original_attr.shape[0] > 0:
            setattr(cleaned_data, attr, original_attr[mask_to_keep])
            
    print(f"Filtered opacity object count: {len(cleaned_data.xyz)}")
    return cleaned_data
