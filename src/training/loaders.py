import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from itertools import chain
import torch

from databases.datasets import (
    pose_grid_from_index,
    Mpi3dTrainDataset,
    PersonStackedMucoTempDataset,
    ConcatPoseDataset,
)


class ConcatSampler(torch.utils.data.Sampler):
    """ Concatenates two samplers. """

    def __init__(self, sampler1, sampler2):
        self.sampler1 = sampler1
        self.sampler2 = sampler2

    def __iter__(self):
        return chain(iter(self.sampler1), iter(self.sampler2))

    def __len__(self):
        return len(self.sampler1) + len(self.sampler2)


class UnchunkedGenerator:
    """
    Loader that can be used with VideoPose3d model to load all frames of a video at once.
    Useful for testing/prediction.
    """

    def __init__(self, dataset, pad, augment):
        self.seqs = sorted(np.unique(dataset.index.seq))
        self.dataset = dataset
        self.pad = pad
        self.augment = augment

    def __iter__(self):
        for seq in self.seqs:
            inds = np.where(self.dataset.index.seq == seq)[0]
            batch = self.dataset.get_samples(inds, False)
            batch_2d = np.expand_dims(
                np.pad(batch["pose2d"], ((self.pad, self.pad), (0, 0)), "edge"), axis=0
            )
            batch_3d = np.expand_dims(batch["pose3d"], axis=0)
            batch_valid = np.expand_dims(batch["valid_pose"], axis=0)

            if self.augment:
                flipped_batch = self.dataset.get_samples(inds, True)
                flipped_batch_2d = np.expand_dims(
                    np.pad(
                        flipped_batch["pose2d"], ((self.pad, self.pad), (0, 0)), "edge"
                    ),
                    axis=0,
                )
                flipped_batch_3d = np.expand_dims(flipped_batch["pose3d"], axis=0)

                batch_2d = np.concatenate((batch_2d, flipped_batch_2d), axis=0)
                batch_3d = np.concatenate((batch_3d, flipped_batch_3d), axis=0)
                batch_valid = np.concatenate((batch_valid, batch_valid), axis=0)

            #             yield {'pose2d': batch_2d, 'pose3d':batch_3d}
            yield self._to_yield(batch_2d, batch_3d, batch_valid)

    def __len__(self):
        return len(self.seqs)

    def _to_yield(self, batch_2d, batch_3d, batch_valid):
        return batch_2d, batch_valid


class UnchunkedGeneratorWithGT(UnchunkedGenerator):
    def _to_yield(self, batch_2d, batch_3d, batch_valid):
        return batch_2d, batch_valid, batch_3d


class ChunkedGenerator:
    """
    Generator to be used with temporal model, during training.
    """

    def __init__(
        self, dataset, batch_size, pad, augment, shuffle=True, ordered_batch=False
    ):
        """
        pad: 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
               it is usually (receptive_field-1)/2
        augment: turn on random horizontal flipping for training
        shuffle: randomly shuffle the dataset before each epoch
        ordered_batch: the frames inside the batch are continueous
        """
        assert isinstance(
            dataset,
            (Mpi3dTrainDataset, PersonStackedMucoTempDataset, ConcatPoseDataset),
        ), "Only works with Mpi datasets"
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad = pad
        self.shuffle = shuffle
        self.ordered_batch = ordered_batch
        self.augment = augment

        N = len(dataset.index)
        frame_start = (
            np.arange(N) - pose_grid_from_index(dataset.index.seq)[1]
        )  # index of the start of the frame
        frame_end = np.arange(N) - pose_grid_from_index(dataset.index.seq[::-1])[1]
        frame_end = (
            N - frame_end[::-1] - 1
        )  # index of the end of the frame (last frame)

        self.frame_start = frame_start
        self.frame_end = frame_end

        assert np.all(frame_start <= frame_end)
        assert np.all(dataset.index.seq[frame_start] == dataset.index.seq[frame_end])
        assert np.all(dataset.index.seq[frame_start] == dataset.index.seq)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        SUB_BATCH = 1

        N = len(self.dataset)
        num_batch = N // self.batch_size

        if not self.ordered_batch:
            indices = np.arange(N)
        else:
            idxs = np.arange(len(self.frame_start))
            sub_batch_size = self.batch_size // SUB_BATCH
            indices = []
            for start_idx in np.unique(self.frame_start):
                frame_idx = idxs[self.frame_start == start_idx]
                if len(frame_idx) > sub_batch_size:
                    split_idx = np.arange(len(frame_idx))[
                        sub_batch_size::sub_batch_size
                    ]
                    batch_start_end_idx = zip(
                        chain([0], split_idx), chain(split_idx, [None])
                    )
                    indices += list(
                        frame_idx[i:j]
                        for i, j in batch_start_end_idx
                        if ((j is not None) and (j - i > 1))
                        or ((j is None) and (len(frame_idx) - i > 1))
                    )
                else:
                    indices.append(frame_idx)

        if self.shuffle:
            np.random.shuffle(indices)

        assert self.batch_size % SUB_BATCH == 0, "SUB_BATCH must divide batch_size"

        class LoadingDataset:
            def __len__(iself):
                return num_batch * SUB_BATCH

            def __getitem__(iself, ind):
                if not self.ordered_batch:
                    sub_batch_size = self.batch_size // SUB_BATCH
                    batch_inds = indices[
                        ind * sub_batch_size : (ind + 1) * sub_batch_size
                    ]  # (nBatch,)
                else:
                    batch_inds = indices[ind]  # (nBatch,)

                batch_frame_start = self.frame_start[batch_inds][:, np.newaxis]
                batch_frame_end = self.frame_end[batch_inds][:, np.newaxis]
                if self.ordered_batch:
                    assert len(np.unique(batch_frame_start)) == 1
                    assert len(np.unique(batch_frame_end)) == 1

                if self.augment:
                    flip = np.random.random(len(batch_inds)) < 0.5
                else:
                    flip = np.zeros(len(batch_inds), dtype="bool")
                flip = np.tile(flip[:, np.newaxis], (1, 2 * self.pad + 1))

                # expand batch_inds such that it includes lower&upper bound indices for every element
                chunk_inds = (
                    batch_inds[:, np.newaxis]
                    + np.arange(-self.pad, self.pad + 1)[np.newaxis, :]
                )
                chunk_inds = np.clip(chunk_inds, batch_frame_start, batch_frame_end)
                assert np.all(chunk_inds >= batch_frame_start)
                assert np.all(chunk_inds <= batch_frame_end)

                chunk = self.dataset.get_samples(chunk_inds.ravel(), flip.ravel())
                chunk_pose2d = chunk["pose2d"].reshape(  # (82944, 42) -> (1024, 81, 42)
                    chunk_inds.shape + chunk["pose2d"].shape[1:]
                )
                chunk_pose3d = chunk["pose3d"].reshape(  # (82944, 51) -> (1024, 81, 51)
                    chunk_inds.shape + chunk["pose3d"].shape[1:]
                )
                chunk_org_pose3d = chunk["org_pose3d"].reshape(  # (82944, 51) -> (1024, 81, 51)
                    chunk_inds.shape + chunk["org_pose3d"].shape[1:]
                )
                chunk_valid = chunk["valid_pose"].reshape(
                    chunk_inds.shape + chunk["valid_pose"].shape[1:]
                )
                chunk_length = chunk["bone_length"].reshape(chunk_inds.shape + chunk["bone_length"].shape[1:])
                chunk_orientation = chunk["bone_orientation"].reshape(chunk_inds.shape + chunk["bone_orientation"].shape[1:])
                chunk_root = chunk["root"].reshape(chunk_inds.shape + chunk["root"].shape[1:])

                # for non temporal values select the middle item:
                chunk_pose3d = chunk_pose3d[:, self.pad]
                chunk_valid = chunk_valid[:, self.pad]
                chunk_length = chunk_length[:, self.pad]
                chunk_orientation = chunk_orientation[:, self.pad]
                chunk_root = chunk_root[:, self.pad]
                chunk_org_pose3d = chunk_org_pose3d[:, self.pad]

                chunk_pose3d = np.expand_dims(chunk_pose3d, 1)

                return chunk_pose2d, chunk_pose3d, chunk_valid, chunk_length, chunk_orientation, chunk_root, chunk_org_pose3d

        wrapper_dataset = LoadingDataset()
        loader = DataLoader(
            wrapper_dataset,
            sampler=SequentialSampler(wrapper_dataset),
            batch_size=SUB_BATCH,
            num_workers=4
        )

        for chunk_pose2d, chunk_pose3d, chunk_valid, chunk_length, chunk_orientation, chunk_root, chunk_org_pose3d in loader:
            chunk_pose2d = chunk_pose2d.reshape((-1,) + chunk_pose2d.shape[2:])
            chunk_pose3d = chunk_pose3d.reshape((-1,) + chunk_pose3d.shape[2:])
            chunk_valid = chunk_valid.reshape(-1)
            yield {
                "temporal_pose2d": chunk_pose2d,
                "pose3d": chunk_pose3d,
                "valid_pose": chunk_valid,
                "length": chunk_length[0, ],
                "orientation": chunk_orientation[0, ],
                "root": chunk_root[0, ],
                "org_pose3d": chunk_org_pose3d[0, ]
            }
