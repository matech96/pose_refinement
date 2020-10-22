import torch
import numpy as np

from databases.joint_sets import MuPoTSJoints


def th_sp2cart(r, t, p):
    x = r * torch.cos(p) * torch.cos(t)
    y = r * torch.sin(p) * torch.cos(t)
    z = r * torch.sin(t)
    return x, y, z


def orient2pose(pred_bo, bl, root):
    joint_set = MuPoTSJoints()
    connected_joints = joint_set.LIMBGRAPH
    cj = np.array(connected_joints)
    cj_index = [2, 1, 0, 5, 4, 3, 9, 8, 12, 11, 10, 15, 14, 13, 7, 6]
    ordered_cj = cj[cj_index, :]

    pred_bx, pred_by, pred_bz = th_sp2cart(bl, pred_bo[:, 0, :], pred_bo[:, 1, :])
    pred_bxyz = torch.stack((pred_bx, pred_by, pred_bz), dim=-1)
    res = torch.zeros((pred_bo.shape[0], 17, 3)).to("cuda")
    res[:, 14, :] = root
    for (a, b), i in zip(ordered_cj, cj_index):
        res[:, a, :] = res[:, b, :] + pred_bxyz[:, i, :]
    return res
