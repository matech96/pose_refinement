import os

import cv2

from util.misc import load

# MUCO_TEMP_PATH = '../datasets/MucoTemp'


def __get_path(v):
    if v == "v1":
        return "../datasets/MucoTemp"
    elif v == "v1":
        return "../datasets/MucoTempV2"
    else:
        raise "Not supported version"


def get_frame(cam, vid_id, frame_ind, rgb=True, v="v1"):
    path = os.path.join(
        __get_path(v),
        "frames/cam_%d/vid_%d" % (cam, vid_id),
        "img_%04d.jpg" % frame_ind,
    )
    img = cv2.imread(path)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_metadata(v="v1"):
    """ Returns metadata for each video in the video. It contains the original videos and starting frames each video contains. """
    return load(os.path.join(__get_path(v), "sequence_meta.pkl"))


def load_gt(cam, v="v1"):
    return load(os.path.join(__get_path(v), "frames", "cam_%d" % cam, "gt.pkl"))


def load_hrnet(cam, vid, v="v1"):
    return load(
        os.path.join(
            __get_path(v),
            "hrnet_keypoints",
            "cam_%d" % cam,
            "gt_match_posedist_80",
            "vid_%d.pkl" % vid,
        )
    )
