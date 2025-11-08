# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random

import h5py
import mmcv
import numpy as np
import pandas as pd

from .base import BaseDataset
from .builder import DATASETS
from ..utils import get_root_logger

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 **kwargs):
        modality = 'Pose'
        self.split = split

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None and isinstance(self.valid_ratio, float) and self.valid_ratio > 0:
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                assert 'box_score' in item, 'if valid_ratio is a positive number, item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data

@DATASETS.register_module()
class ASDetDataset(PoseDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 atypes,
                 target_fps,
                 sequence_length_seconds,
                 step_size_seconds,
                 valid_threshold,
                 njoints_threshold,
                 random_sampling_state=None,
                 min_length_prop=0.75,
                 fold=None,
                 split='train',
                 balance=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 **kwargs):
        if ann_file.endswith('.h5'):
            self.files = pd.Series({'filepath': ann_file, 'split': 'test', 'fold': np.nan}).to_frame().T
        else:
            files = pd.read_csv(ann_file)
            files = files[files['atype'].isin(atypes)].reset_index(drop=True)
            self.files = files
        self.target_fps = target_fps
        self.sequence_length_seconds = sequence_length_seconds
        self.sequence_length = self.sequence_length_seconds * self.target_fps
        self.step_size = step_size_seconds * self.target_fps
        self.valid_threshold = valid_threshold
        self.njoints_threshold = njoints_threshold
        self.min_length_prop = min_length_prop
        self.min_segment_length = self.min_length_prop * self.sequence_length_seconds
        self.random_sampling_state = random_sampling_state
        self.fold = fold
        self.split = split
        self.balance = balance
        self.ds, self.metadata = self.load_dataset()
        super().__init__(ann_file, pipeline, split, valid_ratio, box_thr, class_prob, memcached, mc_cfg, **kwargs)

    @staticmethod
    def generate_intervals(sequence_length, step_size, original_fps, target_fps, video_length):
        sequence_time = sequence_length / target_fps # in seconds
        step_time = step_size / target_fps # in seconds
        sequence_frames = int(np.round(sequence_time * original_fps))
        step_frames = int(np.round(step_time * original_fps))
        F = video_length - sequence_frames + step_frames
        starts = (np.arange(0, F / original_fps, step_time) * original_fps).round().astype(int)
        intervals = [(x, min(x + sequence_frames, video_length)) for x in starts]
        return intervals

    def read_segments(self, file):
        with h5py.File(file, "r") as f:
            vi_attrs = dict(f["video_info"].attrs)
            video_info = pd.Series({k: v.decode() if isinstance(v, bytes) else v for k, v in vi_attrs.items()})
            video_info['filepath'] = file
            vf = f['valid_frames'][()]
            if 'valid_keypoints' in f:
                vkp = f['valid_keypoints'][()].astype(int)
            else:
                vkp = np.ones(vf.shape).astype(int) * 17
            F = vf.shape[0]
            _F = f['kp'].shape[0]
            intervals = self.generate_intervals(self.sequence_length, self.step_size, video_info['fps'], self.target_fps, F)
            df = pd.DataFrame(intervals, columns=['start', 'end'])
            df = df[df['end'] <= _F]
            starts = df['start'].astype(int).to_numpy()
            ends = df['end'].astype(int).to_numpy()
            df['valid%'] = [np.mean(vf[s:e]) for s, e in zip(starts, ends)]
            df['mean_valid_joints'] = [np.mean(vkp[s:e]) for s, e in zip(starts, ends)]
            above_t_vkp = vkp >= self.njoints_threshold
            df['above_t_valid_joints'] = [np.mean(above_t_vkp[s:e]) for s, e in zip(starts, ends)]
            df['segment_length_seconds'] = (df['end'] - df['start']) / video_info['fps']
            df['basename'] = video_info['basename']
        return df, video_info

    def load_dataset(self):
        metadata = {}
        segments = []
        files = self.files[self.files['split'] == self.split]
        if self.fold is not None:
            files = files[files['fold'].isin(self.fold)]
        for i, row in files.iterrows():
            f = row['filepath']
            if not osp.exists(f):
                print(f"File {f} does not exist, skipping.")
                continue
            seg, meta = self.read_segments(f)
            seg['split'] = row['split']
            seg['fold'] = row['fold']
            metadata[meta['basename']] = meta
            segments.append(seg)
        _df = pd.concat(segments)
        _df['length'] = _df['end'] - _df['start']
        df = _df[(_df['valid%'] >= self.valid_threshold) &
                 # (_df['valid_joints'] >= self.njoints_threshold) &
                 (_df['above_t_valid_joints'] >= self.valid_threshold) &
                 (_df['segment_length_seconds'] >= self.min_segment_length)].reset_index(drop=True)
        df['assessment'] = df['basename'].apply(lambda x: metadata[x]['assessment'])
        df['cid'] = df['basename'].apply(lambda x: metadata[x]['cid'])
        df['final_diagnosis'] = df['basename'].apply(lambda x: metadata[x]['final_diagnosis'])
        df['label'] = df['final_diagnosis'].map({'ASD': 1, 'Control': 0, 'ASD denied': 0})
        # print(f'Final dataset size: {len(segments)} segments from {segments["basename"].nunique()} videos.')
        # print(f'Label distribution:\n{segments["final_diagnosis"].value_counts()}\n({segments["final_diagnosis"].value_counts(normalize=True)})')
        return df, metadata

    def load_annotations(self):
        """Load annotation file to get video information."""
        df = self.ds
        if self.balance in ['downsample', 'upsample']:
            counts = df['final_diagnosis'].value_counts()
            n = counts.min() if self.balance == 'downsample' else counts.max()
            def safe_sample(group):
                rep = n > len(group)  # only oversample minority with replacement
                return group.sample(n=n, random_state=self.random_sampling_state, replace=rep)
            df = (
                df.groupby('final_diagnosis', group_keys=False)
                .apply(safe_sample)
                .reset_index(drop=True)
            )
        elif self.balance is not None:
            raise ValueError(f'Unknown balance method: {self.balance}')
        ds = [{
                'start': row['start'],
                'end': row['end'],
                'basename': row['basename'],
                'label': row['label'],
                'valid%': row['valid%'],
                'mean_valid_joints': row['mean_valid_joints'],
                'above_t_valid_joints': row['above_t_valid_joints'],
                'split': row['split'],
                'filepath': self.metadata[row['basename']]['filepath'],
                'img_shape': (int(self.metadata[row['basename']]['height']),
                              int(self.metadata[row['basename']]['width']))
               }
            for _, row in df.iterrows()]
        return ds
