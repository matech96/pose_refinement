import argparse
import os
import sys
import numpy as np
import copy
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

from scripts import eval, train

LOG_PATH = "../models"


def run_experiment(output_path, _config, exp: Experiment):
    config, m = eval.load_model(_config["weights"])
    # config.update(_config)
    config["model"].update(_config["model"])
    _config["model"] = config["model"]

    # tmp = _config["model"]["loss"]
    # _config["model"]["loss"] = "v * mse + e_smooth_small"
    exp.log_parameters(train.flatten_params(_config))
    # _config["model"]["loss"] = tmp
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
    params_path = os.path.join(LOG_PATH, _config["weights"], "preprocess_params.pkl")
    transform = SaveableCompose.from_file(params_path, test_data, globals())

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
    # train_data.transform = SaveableCompose.from_file(params_path, train_data, globals())
    # test_data.transform = SaveableCompose.from_file(params_path, test_data, globals())

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
        config["model"]["filter_widths"],
        dropout=config["model"]["dropout"],
        channels=config["model"]["channels"],
        layernorm=config["model"]["layernorm"],
    )
    model.load_state_dict(m.state_dict())
    test_model = TemporalModel(
        train_data[[0]]["pose2d"].shape[-1],
        MuPoTSJoints.NUM_JOINTS,
        config["model"]["filter_widths"],
        dropout=config["model"]["dropout"],
        channels=config["model"]["channels"],
        layernorm=config["model"]["layernorm"],
    )

    model.cuda()
    test_model.cuda()

    save(output_path + "/model_summary.txt", str(model))

    # normalizer2d = train_data.transform.transforms[0].normalizer
    # normalizer3d = train_data.transform.transforms[1].normalizer

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
        config["model"]["loss"],
        _config["test_time_flip"],
        post_process3d=get_postprocessor(_config, test_data, normalizer3d),
        prefix="test",
    )

    torch_train(
        exp,
        train_loader,
        model,
        lambda m, b: train.calc_loss(
            m,
            b,
            _config,
            torch.tensor(normalizer2d.mean[2::3]).cuda(),
            torch.tensor(normalizer2d.std[2::3]).cuda(),
            torch.tensor(normalizer3d.std).cuda(),
        ),
        _config,
        callbacks=[tester],
    )

    model_path = os.path.join(output_path, "model_params.pkl")
    torch.save(model.state_dict(), model_path)
    exp.log_model("model", model_path)

    # save(
    #     output_path + "/test_results.pkl",
    #     {
    #         "index": test_data.index,
    #         "pred": preds_from_logger(test_data, tester),
    #         "pose3d": test_data.poses3d,
    #     },
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="folder to save the model to")
    args = parser.parse_args()

    def run(loss, lr, loss_a1, loss_a2):
        exp = Experiment(
            workspace="pose-refinement", project_name="02-batch-shuffle-pretrained"
        )

        if args.output is None:
            output_path = f"../models/{exp.get_key()}"
        else:
            output_path = args.output

        params = {
            "num_epochs": 1,
            "preprocess_2d": "DepthposeNormalize2D",
            "preprocess_3d": "SplitToRelativeAbsAndMeanNormalize3D",
            "shuffle": True,
            "ordered_batch": True,
            # training
            "optimiser": "adam",
            "adam_amsgrad": True,
            "learning_rate": lr,
            "sgd_momentum": 0,
            "batch_size": 1024,
            "train_time_flip": False, # True,
            "test_time_flip": True,
            "lr_scheduler": {
                "type": "multiplicative",
                "multiplier": 0.95,
                "step_size": 1,
            },
            # dataset
            "train_data": "mpii+muco",
            "pose2d_type": "hrnet",
            "pose3d_scaling": "normal",
            "megadepth_type": "megadepth_at_hrnet",
            "cap_25fps": True,
            "stride": 2,
            "simple_aug": False, # True,  # augments data by duplicating each frame
            "weights": "29cbfa0fc1774b9cbb06a3573b7fb711",
            "model": {"loss": loss, "loss_a1": loss_a1, "loss_a2": loss_a2},
        }
        run_experiment(output_path, params, exp)
        eval.main(output_path, False, exp)
        # eval.main(output_path, True, exp)

    device = 1
    if device == 0:
        lr_range = [1e-5, 1e-3]
    else:
        lr_range = [1e-4, 1e-2]
    for lr in lr_range:
        for loss_a1 in [1, 0.1, 0.01, 0.001]:
            for loss_a2 in [1, 0.1, 0.01, 0.001]:
                run("smooth", lr, loss_a1, loss_a2)
