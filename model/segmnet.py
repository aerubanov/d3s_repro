from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F


class SegmNet(nn.Module):
    def __init__(
            self,
            segm_input_dim: Tuple[int] = (128, 256),
            segm_inter_dim: Tuple[int] = (256, 256),
            segm_dim: Tuple[int] = (64, 64),
            mixer_channels: int = 2,
            topk_pos: int = 3,
            topk_neg: int = 3,
            )
        super().__init__()

        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])

        self.mixer = conv(mixer_channels, segm_inter_dim[3])
        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])

        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, test_feat, train_feat, mask, test_dist=None):
        # reduce dimensionality of backbone features to 64 and apply 3x3 conv
        f_test = self.segment1(self.segment0(test_feat[3]))
        f_train = self.segment1(self.segment0(train_feat[3]))

        # reshape mask to the feature size
        mask_pos = F.interpolate(mask_train[0],
                size=(f_train.shape[-2], f_train.shape[-1]))
        mask_neg = 1 - mask_pos

        pred_pos, pred_neg = self.gim(f_test, f_train,
                mask_pos, mask_neg)

    def gim(self, f_test, f_train, mask_pos, mask_neg):
        # Normalize features to have L2 norm equal 1
        f_test = F.normalize(f_test, p=2, dim=1)
        f_train = F.normalize(f_train, p=2, dim=1)

        # dot product and reshape
        sim = torch.einsum('ijkl,ijmn->iklmn', f_test, f_train)
        sim_resh = sim.view(
                sim.shape[0],
                sim.shape[1],
                sim.shape[2],
                sim.shape[3] * sim.shape[4]
                )

        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)
        sim_neg = sim_resh * mask_neg.view(mask_neg.shape[0], 1, 1, -1)


        # take top k positive and negative examples
        # mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_pos, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(sim_neg, self.topk_neg, dim=-1).values, dim=-1)

        return pos_map, neg_map
