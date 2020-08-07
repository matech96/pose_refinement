#!/usr/bin/python3
"""
Evaluates a (not end2end) model on MuPo-TS
"""
from comet_ml import Experiment, ExistingExperiment
import argparse
import os
import sys

sys.path.append("/workspace/src")

import numpy as np
import torch
from util.misc import load

from databases import mupots_3d, mpii_3dhp
from databases.datasets import PersonStackedMuPoTsDataset, Mpi3dTestDataset
from databases.joint_sets import MuPoTSJoints, CocoExJoints
from model.pose_refinement import optimize_poses, StackedArrayAllMupotsEvaluator
from model.videopose import TemporalModel
from training.callbacks import TemporalMupotsEvaluator, TemporalTestEvaluator
from training.preprocess import get_postprocessor, SaveableCompose, MeanNormalize3D

LOG_PATH = "../models"


def unstack_mpi3dhp_poses(dataset, logger):
    """
    Converts output of the logger to dict of ndarrays that are aligned with gt, adding back
    frames where 2D/depth was invalid.
    """

    pred_3d = {}
    for seq in range(1, 7):
        gt = mpii_3dhp.test_ground_truth(seq)
        gt_len = len(gt["annot2"])

        pred_3d[seq] = np.full(
            (gt_len, 17, 3), 1e6
        )  # Fill preds with large values to be ignored by pck calculation
        seq_inds = dataset.index.seq == seq
        valid = dataset.good_poses[seq_inds]
        frame_inds = dataset.index.frame[seq_inds][valid]
        assert len(frame_inds) == len(logger.preds[seq])

        pred_3d[seq][frame_inds] = logger.preds[seq]
        # for i in range(gt_len):
        #     seq_inds = (dataset.index.seq == seq)
        #     frame_inds = (dataset.index.frame == i)
        #     pred_3d[seq].append(logger.preds[seq][frame_inds[seq_inds]])

    return pred_3d


def unstack_mupots_poses(dataset, predictions):
    """ Converts output of the logger to dict of list of ndarrays. """
    COCO_TO_MUPOTS = []
    for i in range(MuPoTSJoints.NUM_JOINTS):
        try:
            COCO_TO_MUPOTS.append(CocoExJoints().index_of(MuPoTSJoints.NAMES[i]))
        except:
            COCO_TO_MUPOTS.append(-1)
    COCO_TO_MUPOTS = np.array(COCO_TO_MUPOTS)
    assert np.all(COCO_TO_MUPOTS[1:14] >= 0)

    pred_2d = {}
    pred_3d = {}
    for seq in range(1, 21):
        gt = mupots_3d.load_gt_annotations(seq)
        gt_len = len(gt["annot2"])

        pred_2d[seq] = []
        pred_3d[seq] = []

        seq_inds = dataset.index.seq_num == seq
        for i in range(gt_len):
            frame_inds = dataset.index.frame == i
            valid = dataset.good_poses & seq_inds & frame_inds

            pred_2d[seq].append(dataset.poses2d[valid, :, :2][:, COCO_TO_MUPOTS])
            pred_3d[seq].append(
                predictions[seq][frame_inds[dataset.good_poses & seq_inds]]
            )

    return pred_2d, pred_3d


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
        dropout=config["model"]["dropout"],
        channels=config["model"]["channels"],
        layernorm=config["model"]["layernorm"],
    )

    m.cuda()
    m.load_state_dict(weights)
    m.eval()

    return config, m


def get_dataset(config):
    # data = PersonStackedMuPoTsDataset(
    #     config["pose2d_type"],
    #     config.get("pose3d_scaling", "normal"),
    #     pose_validity="all",
    # )
    data = Mpi3dTestDataset(
        config["pose2d_type"],
        config.get("pose3d_scaling", "normal"),
        eval_frames_only=True,
    )
    return data


def main(model_name, pose_refine, exp: Experiment):
    config, m = load_model(model_name)
    test_set = get_dataset(config)

    params_path = os.path.join(LOG_PATH, str(model_name), "preprocess_params.pkl")
    transform = SaveableCompose.from_file(params_path, test_set, globals())
    test_set.transform = transform

    assert isinstance(transform.transforms[1].normalizer, MeanNormalize3D)
    normalizer3d = transform.transforms[1].normalizer

    post_process_func = get_postprocessor(config, test_set, normalizer3d)

    prefix = "R" if pose_refine else "NR"
    prefix = f"mpi_{prefix}"
    # logger = TemporalMupotsEvaluator(
    #     m,
    #     test_set,
    #     config["model"]["loss"],
    #     True,
    #     post_process3d=post_process_func,
    #     prefix=prefix,
    # )
    logger = TemporalTestEvaluator(
        m,
        test_set,
        config["model"]["loss"],
        True,
        post_process3d=post_process_func,
        prefix=prefix,
    )
    logger.eval(calculate_scale_free=not pose_refine, verbose=not pose_refine)
    exp.log_metrics(logger.losses_to_log)

    pred_3d = unstack_mpi3dhp_poses(test_set, logger)
    print("\n%13s  R-PCK  R-AUC  A-PCK  A-AUC" % "")
    print("%13s: " % "all poses", end="")
    keys = ["R-PCK", "R-AUC", "A-PCK", "A-AUC"]
    values = []
    for relative in [True, False]:
        pcks, aucs = mpii_3dhp.eval_poses(
            relative,
            "annot3" if config["pose3d_scaling"] == "normal" else "univ_annot3",
            pred_3d,
        )
        pck = np.mean(list(pcks.values()))
        auc = np.mean(list(aucs.values()))
        values.append(pck)
        values.append(auc)

        print(" %4.1f   %4.1f  " % (pck, auc), end="")
    print()
    exp.log_metrics({f"{prefix}-{k}": v for k, v in zip(keys, values)})

    # if pose_refine:
    #     refine_config = load("../models/pose_refine_config.json")
    #     pred = np.concatenate([logger.preds[i] for i in range(1, 21)])
    #     pred = optimize_poses(pred, test_set, refine_config)
    #     l = StackedArrayAllMupotsEvaluator(pred, test_set, True, prefix="R")
    #     l.eval(calculate_scale_free=True, verbose=True)
    #     exp.log_metrics(l.losses_to_log)

    #     pred_by_seq = {}
    #     for seq in range(1, 21):
    #         inds = test_set.index.seq_num == seq
    #         pred_by_seq[seq] = pred[inds]
    #     pred_2d, pred_3d = unstack_mupots_poses(test_set, pred_by_seq)
    # else:
    #     pred_2d, pred_3d = unstack_mupots_poses(test_set, logger.preds)
    #     exp.log_metrics(logger.losses_to_log)

    # print("\nR-PCK  R-AUC  A-PCK  A-AUC")
    # keys = ["R-PCK", "R-AUC", "A-PCK", "A-AUC"]
    # values = []
    # for relative in [True, False]:
    #     pcks, aucs = mupots_3d.eval_poses(
    #         False,
    #         relative,
    #         "annot3" if config["pose3d_scaling"] == "normal" else "univ_annot3",
    #         pred_2d,
    #         pred_3d,
    #         keep_matching=True,
    #     )
    #     pck = np.mean(list(pcks.values()))
    #     auc = np.mean(list(aucs.values()))
    #     values.append(pck)
    #     values.append(auc)

    #     print(" %4.1f   %4.1f  " % (pck, auc), end="")
    # print()
    # exp.log_metrics({f"{prefix}-{k}": v for k, v in zip(keys, values)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", help="Name of the model (either 'normal' or 'universal')"
    )
    parser.add_argument(
        "-r",
        "--pose-refine",
        action="store_true",
        help="Apply pose-refinement after TPN",
    )
    args = parser.parse_args()

    exp = ExistingExperiment(previous_experiment=args.model_name)

    main(args.model_name, args.pose_refine, exp)
