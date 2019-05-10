import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, identity
from rlkit.torch import pytorch_util as ptu

from copy import deepcopy


# Attention version v1, no spatial softmax
class ImageProcessor(PyTorchModule):
    def __init__(
        self,
    ):
        self.save_init_params(locals())
        super().__init__()

        CH = 64
        self.output_dim = 2*CH
        k = 5
        s = 2
        p = 2
        self.conv_part_1 = nn.Sequential(
            nn.Conv2d(3, CH, k, stride=s, padding=p),
            # nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            # nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(CH, CH, 1, stride=1, padding=0),
        )
        self.conv_part_2 = nn.Sequential(
            nn.Conv2d(2*CH, CH, k, stride=1, padding=p),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=1, padding=p),
            nn.ReLU(),
            nn.Conv2d(CH, CH, 1, stride=1, padding=0),
        )

        x_map, y_map = np.meshgrid(
            np.arange(-8, 8),
            np.arange(-8, 8),
        )
        x_map = (x_map.flatten()[None][None] + 0.5) / 8.0
        y_map = (y_map.flatten()[None][None] + 0.5) / 8.0
        self.x_map = Variable(ptu.from_numpy(x_map))
        self.y_map = Variable(ptu.from_numpy(y_map))


    def cuda(self):
        self.x_map = self.x_map.cuda()
        self.y_map = self.y_map.cuda()
        super().cuda()
    

    def cpu(self):
        self.x_map = self.x_map.cpu()
        self.y_map = self.y_map.cpu()
        super().cpu()


    def forward(self, z, image, return_softmax_output=False):
        # reshape z into CHx1x1
        z = z.view(z.size(0), z.size(1), 1, 1)

        conv_part_1_output = self.conv_part_1(image)

        # multiplicative interaction and concatenate to self
        # kinda like FiLM layers
        mult_feats = z * conv_part_1_output
        concat_output = torch.cat([conv_part_1_output, mult_feats], dim=1)

        output = self.conv_part_2(concat_output)
        output = output.view(output.size(0), output.size(1), output.size(2)*output.size(3))
        softmax_output = F.softmax(output, dim=-1)
        x_pos = torch.sum(softmax_output * self.x_map, dim=-1)
        y_pos = torch.sum(softmax_output * self.y_map, dim=-1)
        feature_positions = torch.cat([x_pos, y_pos], dim=-1)

        if return_softmax_output:
            return feature_positions, softmax_output
        return feature_positions


# Version where z's are the kernels for the convolutions
# class ImageProcessor(PyTorchModule):
#     '''
#     Given a batch of z's and a batch of images
#     processes it into a representation similar to
#     https://arxiv.org/pdf/1709.04905.pdf with
#     spatial softmax used
#     '''
#     def __init__(
#         self,
#     ):
#         self.save_init_params(locals())
#         super().__init__()

#         # self.Z_STRIDE = 2

#         Z_CONV_CH = 8
#         CORE_CONV_CH = 8
#         NUM_OUTPUT_FEATURES = 16
#         self.output_dim = 2*NUM_OUTPUT_FEATURES
#         kernel = (5, 5)
#         stride = (2, 2)
#         self.core_convs = nn.ModuleList(
#             [
#                 nn.Conv2d(3, CORE_CONV_CH, kernel, stride=stride, padding=2),
#                 nn.Conv2d(CORE_CONV_CH + Z_CONV_CH, CORE_CONV_CH, kernel, stride=stride, padding=2),
#                 nn.Conv2d(CORE_CONV_CH + Z_CONV_CH, CORE_CONV_CH, kernel, stride=stride, padding=2),
#                 nn.Conv2d(CORE_CONV_CH + Z_CONV_CH, NUM_OUTPUT_FEATURES, kernel, stride=stride, padding=2)
#             ]
#         )

#         # REST_CONV_CH = 32
#         # self.output_dim = REST_CONV_CH * 2
#         # self.rest_conv = nn.Sequential(
#         #     nn.ReLU(),
#         #     nn.Conv2d(FIRST_CONV_CH + Z_CONV_CH, REST_CONV_CH, kernel, stride=stride, padding=2),
#         #     nn.ReLU(),
#         #     nn.Conv2d(REST_CONV_CH, REST_CONV_CH, kernel, stride=stride, padding=2),
#         #     nn.ReLU(),
#         #     # made the last padding 3 so output H,W is odd
#         #     # nn.Conv2d(REST_CONV_CH, REST_CONV_CH, kernel, stride=stride, padding=3)
#         #     nn.Conv2d(REST_CONV_CH, REST_CONV_CH, kernel, stride=stride, padding=0)
#         # )

#         # CONV_OUTPUT_H = CONV_OUTPUT_W = 6
#         # x_map, y_map = np.meshgrid(
#         #     np.arange(-int(CONV_OUTPUT_W/2), int(CONV_OUTPUT_W/2)+1),
#         #     np.arange(-int(CONV_OUTPUT_H/2), int(CONV_OUTPUT_H/2)+1),
#         # )
        
#         x_map, y_map = np.meshgrid(
#             np.arange(-4, 4),
#             np.arange(-4, 4),
#         )
#         x_map = (x_map.flatten()[None][None] + 0.5) / 8.0
#         y_map = (y_map.flatten()[None][None] + 0.5) / 8.0
#         self.x_map = Variable(ptu.from_numpy(x_map))
#         self.y_map = Variable(ptu.from_numpy(y_map))


#     def cuda(self):
#         self.x_map = self.x_map.cuda()
#         self.y_map = self.y_map.cuda()
#         super().cuda()
    

#     def cpu(self):
#         self.x_map = self.x_map.cpu()
#         self.y_map = self.y_map.cpu()
#         super().cpu()


#     def forward(self, z, image):
#         num_images = image.size(0)

#         kernel_dims = [
#             [8, 3, 5, 5],
#             [8, 16, 5, 5],
#             [8, 16, 5, 5]
#         ]
#         flat_dims = [np.prod(k) for k in kernel_dims]
#         thus_far = 0
#         zs = []
#         # get the flat kernels
#         for fd in flat_dims:
#             zs.append(z[:,thus_far:thus_far+fd])
#             thus_far += fd
#         # reshape them into the kernel shapes we want
#         z_kernels = []
#         for i in range(len(zs)):
#             flat_z = zs[i].contiguous()
#             z_dims = kernel_dims[i]
#             z_kernels.append(flat_z.view(num_images*z_dims[0], z_dims[1], z_dims[2], z_dims[3]))

#         h = image
#         for i in range(3):
#             # weirdness to process each image with its own z
#             z_shaped_prev_h = h.view(1, num_images*h.size(1), h.size(2), h.size(3))
#             z_h = F.conv2d(z_shaped_prev_h, z_kernels[i], bias=None, stride=2, padding=2, groups=num_images)
#             z_h = z_h.view(num_images, kernel_dims[i][0], z_h.size(2), z_h.size(3))

#             # common network part
#             core_h = self.core_convs[i](h)

#             # concat channel-wise and non-linearity
#             h = torch.cat([z_h, core_h], dim=1)
#             h = F.relu(h)
#         # do the last conv
#         h = self.core_convs[-1](h)

#         # time for spatial softmax
#         h = h.view(
#             h.size(0),
#             h.size(1),
#             h.size(2) * h.size(3)
#         )
#         softmax_output = F.softmax(h, dim=-1)
#         x_pos = torch.sum(softmax_output * self.x_map, dim=-1)
#         y_pos = torch.sum(softmax_output * self.y_map, dim=-1)
#         feature_positions = torch.cat([x_pos, y_pos], dim=-1)

#         return feature_positions

#         # # print('\n\n\n\n\nImage Processor Input')
#         # # print(z)
#         # # print(image)
#         # '''
#         # z: N x out_ch x in_ch x kH x kW
#         # image: N x in_ch x h x w
#         # '''
#         # # convert z batch to a very large kernel
#         # # (Nxout_ch) x in_ch x kH x kW
#         # z_out_ch = z.size(1)
#         # z = z.view(z.size(0)*z.size(1), z.size(2), z.size(3), z.size(4))

#         # first_conv_out = self.first_conv(image)
#         # # reshape the image batch to an image with a lot of channels
#         # # 1 x (Nxin_ch) x kH x kW
#         # num_images = image.size(0)
#         # image = image.view(1, image.size(0)*image.size(1), image.size(2), image.size(3))

#         # # the z convolve
#         # # 1 x (Nxout_ch) x h x w
#         # z_conv_out = F.conv2d(image, z, bias=None, stride=self.Z_STRIDE, padding=2, groups=num_images)
#         # # N x out_ch x h x w
#         # z_conv_out = z_conv_out.view(num_images, z_out_ch, z_conv_out.size(2), z_conv_out.size(3))

#         # # do the rest
#         # concat_out = torch.cat([first_conv_out, z_conv_out], dim=1)
#         # conv_output = self.rest_conv(concat_out)

#         # conv_output = conv_output.view(
#         #     conv_output.size(0),
#         #     conv_output.size(1),
#         #     conv_output.size(2) * conv_output.size(3)
#         # )
#         # softmax_output = F.softmax(conv_output, dim=-1)
#         # x_pos = torch.sum(softmax_output * self.x_map, dim=-1)
#         # y_pos = torch.sum(softmax_output * self.y_map, dim=-1)
#         # feature_positions = torch.cat([x_pos, y_pos], dim=-1)

#         # # nan_check = (softmax_output != softmax_output).type(torch.FloatTensor)
#         # # nan_check = (conv_output != conv_output).type(torch.FloatTensor)
#         # # print(nan_check)
#         # # if nan_check.mean().data[0] > 0:
#         #     # print(softmax_output)
#         #     # print(conv_output)
#         #     # 1/0

#         # return feature_positions


class PusherDisc(PyTorchModule):
    def __init__(
        self,
        Z_CONV_CH,
        mode='video_only', # or 'video_and_state' or 'video_and_state_and_action'
        state_dim=None,
        action_dim=None
    ):
        self.save_init_params(locals())
        super().__init__()

        self.image_processor = ImageProcessor(Z_CONV_CH)

        assert mode in ['video_only', 'video_and_state', 'video_and_state_and_action']
        self.mode = mode
        if mode != 'video_and_state_and_action':
            raise NotImplementedError()

    
    def forward(self, z_batch, image_batch, state_batch=None, action_batch=None):
        feature_positions = self.image_processor(z_batch, image_batch)
        if self.mode == 'video_and_state_and_action':
            concat_fc_input = torch.cat([feature_positions, state_batch, action_batch], dim=-1)
        else:
            raise NotImplementedError()
        
        preds = self.fc_part(concat_fc_input)
        return preds
