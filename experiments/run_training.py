import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
import cv2 as cv
import multiprocessing
import torch

from src.datasets.vos import VosDataset
from src.data.loader import LTRLoader
from src.data import segm_processing, segm_sampler
import src.model.d3s as segm_models
from src.utils.trainer import LTRTrainer
import src.data.transforms as dltransforms
from src.datasets.image_loader import opencv_loader
import src.actors as actors


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.use_gpu = True


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'SegmentationNet with default settings.'
    settings.print_interval = 50  # How often to print loss and other info
    settings.batch_size = 64  # Batch size
    settings.num_workers = 8  # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 4.0  # Image patch size relative to target size
    settings.feature_sz = 24  # Size of feature map
    settings.output_sz = settings.feature_sz * 16  # Size of input image patches

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 1.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}

    settings.segm_topk_pos = 3
    settings.segm_topk_neg = 3

    settings.segm_use_distance = True

    mixer_channels = 3

    # check if debug folder exists
    settings.workspace_dir = 'wdir'
    if not os.path.isdir(settings.workspace_dir):
        os.mkdir(settings.workspace_dir)

    # Train datasets
    vos_train = VosDataset(
            path='DATA/youtube-vos/train/',
            img_loader=opencv_loader,
            split='train')

    # Validation datasets
    vos_val = VosDataset(
            path='DATA/youtube-vos/valid/',
            img_loader=opencv_loader,
            split='valid')

    # The joint augmentation transform, that is applied to the pairs jointly
    # No need for grayscale transformation since we are doing color segmentation
    # transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set
    # (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose(
            [dltransforms.ToTensorAndJitter(0.2),
             torchvision.transforms.Normalize(mean=settings.normalize_mean,
             std=settings.normalize_std)]
             )

    # The augmentation transform applied to the validation set
    # (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=settings.normalize_mean,
             std=settings.normalize_std)]
             )

    # Data processing to do on the training pairs
    data_processing_train = segm_processing.SegmProcessing(
            search_area_factor=settings.search_area_factor,
            output_sz=settings.output_sz,
            center_jitter_factor=settings.center_jitter_factor,
            scale_jitter_factor=settings.scale_jitter_factor,
            mode='pair',
            transform=transform_train,
            use_distance=settings.segm_use_distance)

    # Data processing to do on the validation pairs
    data_processing_val = segm_processing.SegmProcessing(
            search_area_factor=settings.search_area_factor,
            output_sz=settings.output_sz,
            center_jitter_factor=settings.center_jitter_factor,
            scale_jitter_factor=settings.scale_jitter_factor,
            mode='pair',
            transform=transform_val,
            use_distance=settings.segm_use_distance)

    # The sampler for training
    dataset_train = segm_sampler.SegmSampler(
            [vos_train], [1], 
            samples_per_epoch=1000 * settings.batch_size,
            max_gap=50,
            processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True,
            batch_size=settings.batch_size, num_workers=settings.num_workers,
            shuffle=True, drop_last=True, stack_dim=1)

    # # The sampler for validation
    dataset_val = segm_sampler.SegmSampler(
            [vos_val], [1], samples_per_epoch=10 * settings.batch_size, max_gap=50,
            processing=data_processing_val)

    # # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=10, stack_dim=1)

    # Create network
    # resnet50 or resnet18
    net = segm_models.segm_resnet50(
            backbone_pretrained=True,
            topk_pos=settings.segm_topk_pos,
            topk_neg=settings.segm_topk_neg,
            mixer_channels=mixer_channels)

    # Set objective
    objective = nn.BCEWithLogitsLoss()

    # Create actor, which wraps network and objective
    actor = actors.SegmActor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(actor.net.segm_predictor.parameters(), lr=1e-3)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(40, load_latest=True, fail_safe=False)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    settings = Settings()
    settings.project_path = ''
    settings.images_dir = os.path.join(settings.project_path, 'images')

    cv.setNumThreads(0)
    print(torch.cuda.is_available())
    dev = torch.cuda.current_device()
    print(torch.cuda.device(dev))
    print(torch.cuda.get_device_name(dev))
    run(settings)
