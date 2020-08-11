import argparse
import os
import sys
import numpy as np
from comet_ml import Experiment

sys.path.append("/workspace/src")

from databases.datasets import (
    Mpi3dTestDataset,
    Mpi3dTrainDataset,
    PersonStackedMucoTempDataset,
    ConcatPoseDataset,
)
from model.videopose import TemporalModel, TemporalModelOptimized1f
from model.pose_refinement import capped_l2_euc_err, step_zero_velocity_loss
from training.callbacks import preds_from_logger, ModelCopyTemporalEvaluator
from training.loaders import ChunkedGenerator
from training.preprocess import *
from training.torch_tools import torch_train
from util.misc import save, ensuredir
import copy
from collections import Iterable

from scripts import eval


def flatten_params(params):
    res = copy.copy(params)
    to_flatten = [k for k, v in params.items() if isinstance(v, dict)]
    for k in to_flatten:
        poped = res.pop(k)
        for pk, pv in poped.items():
            new_key = f"{k}_{pk}"
            if isinstance(pv, Iterable) and not isinstance(pv, str):
                for i, pvi in enumerate(pv):
                    res[f"{new_key}_{i}"] = pvi
            else:
                res[new_key] = pv
    return res


def calc_loss(model, batch, config, mean_2d, std_2d):
    loss_type = config["model"]["loss"]
    if loss_type == "l1_nan":
        pose2d = batch["temporal_pose2d"]
        gt_3d = batch["pose3d"]

        # different handling for numpy and PyTorch inputs
        if isinstance(pose2d, torch.Tensor):
            inds = torch.all(torch.all(1 - (pose2d != pose2d), dim=(-1)), dim=-1)
            pose2d = pose2d[inds]
            gt_3d = gt_3d[inds]
            pose2d = pose2d.to("cuda")
            gt_3d = gt_3d.to("cuda")
        else:
            inds = np.all(~np.isnan(pose2d), axis=(-1, -2))
            pose2d = pose2d[inds]
            gt_3d = gt_3d[inds]
            pose2d = torch.from_numpy(pose2d).to("cuda")
            gt_3d = torch.from_numpy(gt_3d).to("cuda")

    elif (loss_type == "l1") or (loss_type == "smooth"):
        pose2d = batch["temporal_pose2d"]  # [1024, 81, 42]
        gt_3d = batch["pose3d"]  # [1024, 1, 51]
        pose2d = pose2d.to("cuda")
        gt_3d = gt_3d.to("cuda")

    # forward pass
    pred_3d = model(pose2d)  # [1024, 1, 51]
    assert (pose2d.shape[1] % 2) == 1

    if loss_type == "l1":
        loss_3d = torch.nn.functional.l1_loss(pred_3d, gt_3d)
    elif loss_type == "l1_nan":
        loss_3d = torch.nn.functional.l1_loss(pred_3d, gt_3d)
    elif loss_type == "smooth":
        _conf_l2_cap = 1
        _conf_large_step = 20
        _conf_alpha_1 = 0# 0.1
        _conf_alpha_2 = 0# 1
        middle_channel = pose2d.shape[1] // 2 + 1
        normalized_probs = pose2d[:, middle_channel, 2::3] # [1024, 14]
        unnormalized_probs = normalized_probs * std_2d + mean_2d # [1024, 14]
        v = torch.mean(unnormalized_probs, dim=1)

        e_pred = capped_l2_euc_err(
            pred_3d, gt_3d, torch.tensor(_conf_l2_cap).float().cuda()
        )
        e_smooth_small = step_zero_velocity_loss(pred_3d[:, [0], :], 1)
        # e_smooth_large = step_zero_velocity_loss(pred_3d[:, [0], :], _conf_large_step)
        # if len(e_smooth_large) == 0:
        #     e_smooth_large = torch.tensor([0.0]).cuda()

        loss_3d = (
            torch.mean(v * e_pred)
            # + _conf_alpha_1 * torch.mean((1 - v[-len(e_smooth_large):]) * e_smooth_large)
            + _conf_alpha_2 * torch.mean(e_smooth_small)
        )
        # loss_3d = torch.mean(v * e_pred)
        # loss_3d = torch.nn.functional.l1_loss(pred_3d, gt_3d)
    else:
        raise Exception("Unknown pose loss: " + str(config["model"]["loss"]))

    return loss_3d, {"loss_3d": loss_3d.item()}


def run_experiment(output_path, _config, exp: Experiment):
    exp.log_parameters(flatten_params(_config))
    save(os.path.join(output_path, "config.json"), _config)
    ensuredir(output_path)

    if _config["train_data"] == "mpii_train":
        print("Training data is mpii-train")
        train_data = Mpi3dTrainDataset(
            _config["pose2d_type"],
            _config["pose3d_scaling"],
            _config["cap_25fps"],
            _config["stride"],
        )

    elif _config["train_data"] == "mpii+muco":
        print("Training data is mpii-train and muco_temp concatenated")
        mpi_data = Mpi3dTrainDataset(
            _config["pose2d_type"],
            _config["pose3d_scaling"],
            _config["cap_25fps"],
            _config["stride"],
        )

        muco_data = PersonStackedMucoTempDataset(
            _config["pose2d_type"], _config["pose3d_scaling"]
        )
        train_data = ConcatPoseDataset(mpi_data, muco_data)

    elif _config["train_data"].startswith("muco_temp"):
        train_data = PersonStackedMucoTempDataset(
            _config["pose2d_type"], _config["pose3d_scaling"]
        )

    test_data = Mpi3dTestDataset(
        _config["pose2d_type"], _config["pose3d_scaling"], eval_frames_only=True
    )

    if _config["simple_aug"]:
        train_data.augment(False)

    # Load the preprocessing steps
    train_data.transform = None
    transforms_train = [
        decode_trfrm(_config["preprocess_2d"], globals())(train_data, cache=False),
        decode_trfrm(_config["preprocess_3d"], globals())(train_data, cache=False),
    ]

    normalizer2d = transforms_train[0].normalizer
    normalizer3d = transforms_train[1].normalizer

    transforms_test = [
        decode_trfrm(_config["preprocess_2d"], globals())(test_data, normalizer2d),
        decode_trfrm(_config["preprocess_3d"], globals())(test_data, normalizer3d),
    ]

    transforms_train.append(RemoveIndex())
    transforms_test.append(RemoveIndex())

    train_data.transform = SaveableCompose(transforms_train)
    test_data.transform = SaveableCompose(transforms_test)

    # save normalisation params
    save(output_path + "/preprocess_params.pkl", train_data.transform.state_dict())

    len_train = len(train_data)
    len_test = len(test_data)
    print("Length of training data:", len_train)
    print("Length of test data:", len_test)
    exp.log_parameter("train data length", len_train)
    exp.log_parameter("test data length", len_test)

    model = TemporalModelOptimized1f(
        train_data[[0]]["pose2d"].shape[-1],
        MuPoTSJoints.NUM_JOINTS,
        _config["model"]["filter_widths"],
        dropout=_config["model"]["dropout"],
        channels=_config["model"]["channels"],
        layernorm=_config["model"]["layernorm"],
    )
    test_model = TemporalModel(
        train_data[[0]]["pose2d"].shape[-1],
        MuPoTSJoints.NUM_JOINTS,
        _config["model"]["filter_widths"],
        dropout=_config["model"]["dropout"],
        channels=_config["model"]["channels"],
        layernorm=_config["model"]["layernorm"],
    )

    model.cuda()
    test_model.cuda()

    save(output_path + "/model_summary.txt", str(model))

    pad = (model.receptive_field() - 1) // 2
    train_loader = ChunkedGenerator(
        train_data,
        _config["batch_size"],
        pad,
        _config["train_time_flip"],
        shuffle=_config["shuffle"],
        ordered_batch=_config["ordered_batch"],
    )
    tester = ModelCopyTemporalEvaluator(
        test_model,
        test_data,
        _config["model"]["loss"],
        _config["test_time_flip"],
        post_process3d=get_postprocessor(_config, test_data, normalizer3d),
        prefix="test",
    )

    torch_train(
        exp,
        train_loader,
        model,
        lambda m, b: calc_loss(m, b, _config, torch.tensor(normalizer2d.mean[2::3]).cuda(), torch.tensor(normalizer2d.std[2::3]).cuda()),
        _config,
        callbacks=[tester],
    )

    model_path = os.path.join(output_path, "model_params.pkl")
    torch.save(model.state_dict(), model_path)
    exp.log_model("model", model_path)

    save(
        output_path + "/test_results.pkl",
        {
            "index": test_data.index,
            "pred": preds_from_logger(test_data, tester),
            "pose3d": test_data.poses3d,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="folder to save the model to")
    args = parser.parse_args()
    
    layernorm = "instancenorm"
    for _ in range(2):
        for ordered_batch in [False, True]:
            exp = Experiment(workspace="pose-refinement", project_name="03-batch-shuffle-norm-selection")

            if args.output is None:
                output_path = f"../models/{exp.get_key()}"
            else:
                output_path = args.output

            params = {
                "num_epochs": 15,
                "preprocess_2d": "DepthposeNormalize2D",
                "preprocess_3d": "SplitToRelativeAbsAndMeanNormalize3D",
                "shuffle": True,
                "ordered_batch": ordered_batch,
                # training
                "optimiser": "adam",
                "adam_amsgrad": True,
                "learning_rate": 1e-3,
                "sgd_momentum": 0,
                "batch_size": 1024,
                "train_time_flip": True,
                "test_time_flip": True,
                "lr_scheduler": {"type": "multiplicative", "multiplier": 0.95, "step_size": 1,},
                # dataset
                "train_data": "mpii_train",
                "pose2d_type": "hrnet",
                "pose3d_scaling": "normal",
                "megadepth_type": "megadepth_at_hrnet",
                "cap_25fps": True,
                "stride": 2,
                "simple_aug": True,  # augments data by duplicating each frame
                "model": {
                    "loss": "l1",
                    "channels": 512,
                    "dropout": 0.25,
                    "filter_widths": [3, 3, 3],
                    "layernorm": layernorm, #False,
                },
            }
            run_experiment(output_path, params, exp)
            eval.main(output_path, False, exp)
            eval.main(output_path, True, exp)
