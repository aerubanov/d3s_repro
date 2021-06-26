from abc import ABC, abstractmethod
from typing import Callable, List
import torch.utils.data


class BaseDataset(torch.utils.data.Dataset, ABC):

    def __init__(self, path: str, image_loader: Callable):
        self.path = path
        self.image_loader = image_loader
        self.sequence_list = []

    def __len__(self):
        return len(sequence_list)

    @property
    def is_video_sequence(self):
        return True

    @property
    @abstractmethod
    def name(self):
        """name of the dataset"""

    @abstractmethod
    def get_sequence_info(self, seq_id: int):
        """ Returns information about a particular sequences
        returns:
            Tensor - Annotation for the sequence. A 2d tensor of shape
            (num_frames, 4). Format [top_left_x, top_left_y, width, height]
            Tensor - 1d Tensor specifying whether target is present (=1 )
            for each frame. shape (num_frames,)
            """
    
    @abstractmethod
    def get_frames(self, seq_id: int, frame_ids: List[int], anno=None):
        """ Get a set of frames from a particular sequence
        args:
            seq_id      - index of sequence
            frame_ids   - a list of frame numbers
            anno(None)  - The annotation for the sequence (see get_sequence_info)
            . If None, they will be loaded.
        returns:
            list - List of frames corresponding to frame_ids
            list - List of annotations (tensor of shape (4,)) for each frame
            dict - A dict containing meta information
            about the sequence, e.g. class of the target object.
        """
