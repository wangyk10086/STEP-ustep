"""Minimal stand-in for the pytorch3d.transforms functions used by RotationTransformer
(axis_angle <-> matrix <-> rotation_6d), matching pytorch3d conventions."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as _R


def _np(x):
    return x.detach().cpu().double().numpy()


def axis_angle_to_matrix(aa):
    shp = aa.shape[:-1]
    m = _R.from_rotvec(_np(aa).reshape(-1, 3)).as_matrix()
    return torch.as_tensor(m, dtype=aa.dtype, device=aa.device).reshape(*shp, 3, 3)


def matrix_to_axis_angle(m):
    shp = m.shape[:-2]
    aa = _R.from_matrix(_np(m).reshape(-1, 3, 3)).as_rotvec()
    return torch.as_tensor(aa, dtype=m.dtype, device=m.device).reshape(*shp, 3)


def matrix_to_rotation_6d(m):
    return m[..., :2, :].clone().reshape(*m.shape[:-2], 6)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
