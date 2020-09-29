from comet_ml import Experiment
import sys

sys.path.append("/workspace/src")

import os
import torch
import numpy as np
import copy
from collections import defaultdict

from util.misc import load
from model.videopose import TemporalModel
from model.pose_refinement import (
    abs_to_hiprel,
    add_back_hip,
    gmloss,
    capped_l2,
    capped_l2_euc_err,
    step_zero_velocity_loss,
)
from databases.joint_sets import MuPoTSJoints
from databases.datasets import PersonStackedMuPoTsDataset
from training.preprocess import get_postprocessor, SaveableCompose, MeanNormalize3D
from training.callbacks import TemporalMupotsEvaluator
from training.loaders import UnchunkedGenerator
from training.torch_tools import get_optimizer
from model.pose_refinement import StackedArrayAllMupotsEvaluator
from scripts.eval import unstack_mupots_poses
from databases import mupots_3d


LOG_PATH = "../models"


def load_model(model_folder):
    config = load(os.path.join(LOG_PATH, model_folder, "config.json"))
    path = os.path.join(LOG_PATH, model_folder, "model_params.pkl")

    # Input/output size calculation is hacky
    weights = torch.load(path)
    num_in_features = weights["expand_conv.weight"].shape[1]

    m = TemporalModel(
        num_in_features,
        MuPoTSJoints.NUM_JOINTS,
        config["model"]["filter_widths"],
        dropout=0,
        channels=config["model"]["channels"],
        layernorm=config["model"]["layernorm"],
    )

    m.cuda()
    m.load_state_dict(weights)
    m.eval()

    return config, m


def get_dataset(config):
    return PersonStackedMuPoTsDataset(
        config["pose2d_type"],
        config.get("pose3d_scaling", "normal"),
        pose_validity="all",
    )


def extract_post(model_name, test_set, config):
    params_path = os.path.join(LOG_PATH, str(model_name), "preprocess_params.pkl")
    transform = SaveableCompose.from_file(params_path, test_set, globals())
    test_set.transform = transform

    assert isinstance(transform.transforms[1].normalizer, MeanNormalize3D)
    normalizer3d = transform.transforms[1].normalizer

    return get_postprocessor(config, test_set, normalizer3d)


def run(**kwargs):
    refine_config = load("scripts/nn_refine_config.json")
    for k, v in kwargs.items():
        refine_config[k] = v
    exp = Experiment(
        workspace="pose-refinement",
        project_name="06-nn-refine-large",
        display_summary_level=0,
    )
    exp.log_parameters(refine_config)

    model_name = refine_config["model_name"]
    config, model = load_model(model_name)
    test_set = get_dataset(config)
    post_process_func = extract_post(model_name, test_set, config)

    joint_set = MuPoTSJoints()

    pad = (model.receptive_field() - 1) // 2
    generator = UnchunkedGenerator(test_set, pad, True)
    seqs = sorted(np.unique(test_set.index.seq))

    optimized_preds_list = defaultdict(list)
    max_batch = len(generator)
    exp.log_parameter("max_batch", max_batch)
    for curr_batch, (pose2d, valid) in enumerate(generator):
        exp.log_parameter("curr_batch", curr_batch)
        exp.log_parameter("curr_batch%", curr_batch / max_batch)
        if refine_config["full_batch"]:
            max_item = 1
        else:
            max_item = valid.shape[-1]
        for curr_item in range(max_item):
            if not refine_config["full_batch"]:
                exp.log_parameter("curr_item", curr_item)
                exp.log_parameter("curr_item%", curr_item / max_item)
                if (curr_item + 1) > (
                    max_item - refine_config["smoothness_loss_hip_largestep"]
                ):
                    reverse = True
                    f = curr_item - refine_config["smoothness_loss_hip_largestep"]
                    t = curr_item + 1
                else:
                    reverse = False
                    f = curr_item
                    t = f + refine_config["smoothness_loss_hip_largestep"] + 1
            model_ = copy.deepcopy(model)
            optimizer = get_optimizer(model_.parameters(), refine_config)
            max_iter = refine_config["num_iter"]
            for curr_iter in range(max_iter):
                exp.log_parameter("curr_iter", curr_iter)
                exp.log_parameter("curr_iter%", curr_iter / max_iter)
                optimizer.zero_grad()

                seq = seqs[curr_batch]
                if refine_config["full_batch"]:
                    nn_input = pose2d
                    valid_ = valid[0]
                else:
                    nn_input = pose2d[:, f : t + 2 * pad, :]
                    valid_ = valid[0][f:t]
                pred3d = model_(
                    torch.from_numpy(nn_input).cuda()
                )  # [2, 401, 42] -> [2, 21+2*13, 42], pred3d: [21, 16, 3]

                pred_real_pose = post_process_func(
                    pred3d[0], seq
                )  # unnormalized output

                pred_real_pose_aug = post_process_func(pred3d[1], seq)
                pred_real_pose_aug[:, :, 0] *= -1
                pred_real_pose_aug = test_set.pose3d_jointset.flip(pred_real_pose_aug)
                pred_real_pose = (pred_real_pose + pred_real_pose_aug) / 2

                pred = pred_real_pose[valid_]

                inds = test_set.index.seq == seq

                poses_pred = abs_to_hiprel(pred, joint_set) / 1000  # (201, 17, 3)
                if refine_config["reinit"] or (curr_iter == 0):
                    poses_init = poses_pred.detach().clone()
                    poses_init.requires_grad = False
                    if not refine_config["full_batch"]:
                        kp_score = np.mean(test_set.poses2d[inds, :, 2], axis=-1)[
                            f:t
                        ]  # (201,)
                    else:
                        kp_score = np.mean(
                            test_set.poses2d[inds, :, 2], axis=-1
                        )  # (201,)
                    #     if refine_config['smooth_visibility']:
                    #         kp_score = ndimage.median_filter(kp_score, 9)
                    kp_score = torch.from_numpy(kp_score).cuda()  # [201]
                    scale = torch.ones((len(kp_score), 1, 1))  # torch.Size([201, 1, 1])

                    kp_score.requires_grad = False
                    scale.requires_grad = False

                # smoothing formulation

                if refine_config["pose_loss"] == "gm":
                    pose_loss = kp_score.view(-1, 1, 1) * gmloss(
                        poses_pred - poses_init, refine_config["gm_alpha"]
                    )
                elif refine_config["pose_loss"] == "capped_l2":
                    pose_loss = kp_score.view(-1, 1, 1) * capped_l2(
                        poses_pred - poses_init,
                        torch.tensor(refine_config["l2_cap"]).float().cuda(),
                    )
                elif refine_config["pose_loss"] == "capped_l2_euc_err":
                    pose_loss = kp_score.view(-1, 1) * capped_l2_euc_err(
                        poses_pred,
                        poses_init,
                        torch.tensor(refine_config["l2_cap"]).float().cuda(),
                    )
                else:
                    raise NotImplementedError(
                        "Unknown pose_loss" + refine_config["pose_loss"]
                    )

                velocity_loss_hip = globals()[refine_config["smoothness_loss_hip"]](
                    poses_pred[:, [0], :], 1
                )

                step = refine_config["smoothness_loss_hip_largestep"]
                vel_loss = globals()[refine_config["smoothness_loss_hip"]](
                    poses_pred[:, [0], :], step
                )
                velocity_loss_hip_large = (1 - kp_score[-len(vel_loss) :]) * vel_loss

                velocity_loss_rel = globals()[refine_config["smoothness_loss_rel"]](
                    poses_pred[:, 1:, :], 1
                )
                vel_loss = globals()[refine_config["smoothness_loss_rel"]](
                    poses_pred[:, 1:, :], step
                )
                velocity_loss_rel_large = (1 - kp_score[-len(vel_loss) :]) * vel_loss

                prefix = f"{curr_batch}_{curr_item}"
                if refine_config["full_batch"]:
                    total_loss = (
                        torch.sum(pose_loss)
                        + refine_config["smoothness_weight_hip"]
                        * torch.sum(velocity_loss_hip)
                        + refine_config["smoothness_weight_hip_large"]
                        * torch.sum(velocity_loss_hip_large)
                        + refine_config["smoothness_weight_rel"]
                        * torch.sum(velocity_loss_rel)
                        + refine_config["smoothness_weight_rel_large"]
                        * torch.sum(velocity_loss_rel_large)
                    )
                    m = {
                        f"{prefix}_total_loss": total_loss,
                        f"{prefix}_pose_loss": torch.sum(pose_loss),
                        f"{prefix}_velocity_loss_hip": torch.sum(velocity_loss_hip),
                        f"{prefix}_velocity_loss_hip_large": torch.sum(
                            velocity_loss_hip_large
                        ),
                        f"{prefix}_velocity_loss_rel": torch.sum(velocity_loss_rel),
                        f"{prefix}_velocity_loss_rel_large": torch.sum(
                            velocity_loss_rel_large
                        ),
                    }
                else:
                    neighbour_dist_idx = 0 if not reverse else -1
                    total_loss = (
                        torch.sum(pose_loss[neighbour_dist_idx,])
                        + refine_config["smoothness_weight_hip"]
                        * velocity_loss_hip[[neighbour_dist_idx]]
                        + refine_config["smoothness_weight_hip_large"]
                        * velocity_loss_hip_large
                        + refine_config["smoothness_weight_rel"]
                        * velocity_loss_rel[[neighbour_dist_idx]]
                        + refine_config["smoothness_weight_rel_large"]
                        * velocity_loss_rel_large
                    )
                    m = {
                        f"{prefix}_total_loss": total_loss[0],
                        f"{prefix}_pose_loss": torch.sum(
                            pose_loss[neighbour_dist_idx,]
                        ),
                        f"{prefix}_velocity_loss_hip": velocity_loss_hip[
                            neighbour_dist_idx
                        ],
                        f"{prefix}_velocity_loss_hip_large": velocity_loss_hip_large[0],
                        f"{prefix}_velocity_loss_rel": velocity_loss_rel[
                            neighbour_dist_idx
                        ],
                        f"{prefix}_velocity_loss_rel_large": velocity_loss_rel_large[0],
                    }

                total_loss.backward()
                optimizer.step()

                # m = {k: v.detach().cpu().numpy() for k, v in m.items()}
                # exp.log_metrics(m, step=curr_iter)

            if refine_config["full_batch"]:
                optimized_preds_list[seq].append(
                    add_back_hip(poses_pred.detach().cpu().numpy() * 1000, joint_set)
                )
            else:
                optimized_preds_list[seq].append(
                    add_back_hip(
                        poses_pred[[neighbour_dist_idx]].detach().cpu().numpy() * 1000,
                        joint_set,
                    )
                )

    pred = {k: np.concatenate(v) for k, v in optimized_preds_list.items()}
    pred = TemporalMupotsEvaluator._group_by_seq(pred)
    pred = np.concatenate([pred[i] for i in range(1, 21)])

    l = StackedArrayAllMupotsEvaluator(pred, test_set, True, prefix="R")
    l.eval(calculate_scale_free=True, verbose=True)
    exp.log_metrics(l.losses_to_log)

    pred_by_seq = {}
    for seq in range(1, 21):
        inds = test_set.index.seq_num == seq
        pred_by_seq[seq] = pred[inds]
    pred_2d, pred_3d = unstack_mupots_poses(test_set, pred_by_seq)

    print("\nR-PCK  R-AUC  A-PCK  A-AUC")
    keys = ["R-PCK", "R-AUC", "A-PCK", "A-AUC"]
    values = []
    for relative in [True, False]:
        pcks, aucs = mupots_3d.eval_poses(
            False,
            relative,
            "annot3" if config["pose3d_scaling"] == "normal" else "univ_annot3",
            pred_2d,
            pred_3d,
            keep_matching=True,
        )
        pck = np.mean(list(pcks.values()))
        auc = np.mean(list(aucs.values()))
        values.append(pck)
        values.append(auc)

        print(" %4.1f   %4.1f  " % (pck, auc), end="")
    print()
    exp.log_metrics({curr_iter: v for curr_iter, v in zip(keys, values)})


if __name__ == "__main__":
    reinit = False
    full_batch = True
    num_iter = 100
    smoothness_loss_hip_largestep = 20
    large_mult = 0.1
    learning_rate = 0.001
    rel_mult = 0.1
    model_name = "e665b873d3954dd19c2cf427cc61b6e9"
    run(    
        full_batch=full_batch,
        reinit=reinit,
        num_iter=num_iter,
        learning_rate=learning_rate,
        smoothness_loss_hip_largestep=smoothness_loss_hip_largestep,
        smoothness_weight_hip=1,
        smoothness_weight_hip_large=large_mult,
        smoothness_weight_rel=rel_mult,
        smoothness_weight_rel_large=rel_mult * large_mult,
        model_name=model_name,
    )
