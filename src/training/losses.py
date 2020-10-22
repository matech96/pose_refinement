import torch
import numpy as np

from databases.joint_sets import MuPoTSJoints
from util.geom import th_sp2cart


def orientation_loss(pred_bo, gt_3d, bl, root):
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
    res = res.reshape([pred_bo.shape[0], 1, -1])
    return torch.nn.functional.l1_loss(res, gt_3d)
