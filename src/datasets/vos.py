from typing import Callable, List
import os
import pandas as pd
import glob
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict

from src.datasets.base import BaseDataset


class VosDataset(BaseDataset):
    
    def __init__(
            self,
            path: str,
            img_loader: Callable,
            split: str,
            ):
        """
        split: If split='train', the official train split (protocol-II)
               is used for training. Note: Only one of
               vid_ids or split option can be used at a time.
        """
        super().__init__(path, img_loader)

        self.sequence_list = self._build_sequence_list(split)
        self.frame_names_dict, self.mask_names_dict = self._build_frames_list()

    @property
    def name(self):
        return 'VOS'

    def _build_sequence_list(self, split: str):
        if split != 'train' and split != 'valid':
            raise ValueError("Unknown yotube-vos dataset split argument")

        file_path = os.path.join(self.path, 'vos-list-' + split + '.txt')

        sequence_list = pd.read_csv(file_path, header=None,
                squeeze=True).values.tolist()
        return sequence_list

    def _build_frames_list(self):
        fr_names_dict, msk_names_dict = dict(), dict()
        for name in self.sequence_list:
            folder = name.split('-')[0]
            fr_path = os.path.join(self.path, 'JPEGImages', folder)
            msk_path = os.path.join(self.path, 'Annotations', folder)
            fr_names_dict[name] = sorted(
                    [fname for fname in glob.glob(os.path.join(fr_path, '*.jpg'))]
                    )
            msk_names_dict[name] = sorted(
                    [fname for fname in glob.glob(os.path.join(msk_path, '*.png'))]
                    )
        return fr_names_dict, msk_names_dict

    def _anno (self, path, obj_id):
        anno_file = os.path.join(path, "groundtruth-%s.txt" % obj_id)
        gt = pd.read_csv(anno_file, delimiter=',', header=None,
                dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)


    def get_sequence_info(self, seq_id):
        name = self.sequence_list[seq_id]
        split = name.split('-')[0]
        dir_name, obj_id = split[0], split[1]

        path = os.path.join(self.path, 'boxes', dir_name)

        anno = self._anno(path, obj_id)
        target_visible = (anno[:, 0] > -1) & (anno[:, 1] > -1) & (anno[:, 2] > -1) & (anno[:, 3] > -1)
        return anno, target_visible

    def _get_mask(self, path, fr_id, obj_id):
        m = np.asarray(Image.open(path[fr_id])).astype(np.float32)
        mask = (m == float(obj_id)).astype(np.float32)
        return np.expand_dims(mask, axis=-1)

    def get_frames(self, seq_id, frame_ids, anno=None):
        sname = self.sequence_list[seq_id]
        split = sname.split('-')
        folder, obj_id = split[0], split[1]
        frpath, mskpath = self.frame_names_dict[sname], self.mask_names_dict[sname]

        frlist = [self.image_loader(frpath[i]) for i in frame_ids]
        msklist = [self._get_mask(mskpath, i, obj_id) for i in frame_ids]
        
        if anno is None:
            anno = self._anno(mskpath, obj_id)

        anno_frames = [anno[i, :] for i in frame_ids]

        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frlist, msklist, anno_frames, object_meta
