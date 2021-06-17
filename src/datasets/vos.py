from typing import Callable, List
import os
import pandas as pd

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
       super().__init__(path, image_loader)

       self.sequence_list = self._build_sequence_list(vid_ids, split)
       self.frame_names_dict, self.mask_names_dict = self._build_frames_list()

    @property
    def name(self):
        return 'VOS'

    def _build_sequence_list(split: str):
        if split != 'train' and split != 'valid':
            raise ValueError("Unknown yotube-vos dataset split argument")

        file_path = os.path.join(self.path, 'vos-list-' + split + '.txt')

        sequence_list = pd.read_csv(file_path, header=None, squeeze=True).values.tolist()
        return sequence_list

    def _build_frames_list(self):
        fr_names_dict, msk_names_dict = dict(), dict()
        for name in self.sequence_list:
            folder = name.split(-)[0]
            fr_path = os.path.join(self.path, 'JPEGImages', name)
            msk_path = os.path.join(self.path, 'Annotations', name)

