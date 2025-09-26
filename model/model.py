# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    UNet model definition in here
"""

import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models
import timm


class UNet_down_block(BaseModel):
    """
        Encoder class
    """

    def __init__(self, input_channel, output_channel, conv_1=None, conv_2=None):
        super(UNet_down_block, self).__init__()
        if conv_1:
            print('LOG: Using pretrained convolutional layer', conv_1)
        if conv_2:
            print('LOG: Using pretrained convolutional layer', conv_2)
        self.input_channels = input_channel
        self.output_channels = output_channel
        self.conv1 = conv_1 if conv_1 else nn.Conv2d(
            input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = conv_2 if conv_2 else nn.Conv2d(
            output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(BaseModel):
    """
        Decoder class
    """

    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.output_channels = output_channel
        self.tr_conv_1 = nn.ConvTranspose2d(
            input_channel, input_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(
            prev_channel+input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        x = self.activate(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.activate(self.bn1(self.conv_1(x)))
        x = self.activate(self.bn2(self.conv_2(x)))
        return x


class UNet(BaseModel):
    def __init__(self, topology, input_channels, num_classes):
        super(UNet, self).__init__()
        # these topologies are possible right now
        self.topologies = {
            "ENC_1_DEC_1": self.ENC_1_DEC_1,
            "ENC_2_DEC_2": self.ENC_2_DEC_2,
            "ENC_3_DEC_3": self.ENC_3_DEC_3,
            "ENC_4_DEC_4": self.ENC_4_DEC_4,
        }
        assert topology in self.topologies
        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.6)
        self.activate = nn.ReLU()
        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(
            128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(
            256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])
        self.mid_conv_64_64_a = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_64_64_b = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_128_128_a = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_128_128_b = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_256_256_a = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_256_256_b = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.decoder_4 = UNet_up_block(prev_channel=self.encoder_4.output_channels,
                                       input_channel=self.mid_conv_1024_1024.out_channels, output_channel=256)
        self.decoder_3 = UNet_up_block(prev_channel=self.encoder_3.output_channels,
                                       input_channel=self.decoder_4.output_channels, output_channel=128)
        self.decoder_2 = UNet_up_block(prev_channel=self.encoder_2.output_channels,
                                       input_channel=self.decoder_3.output_channels, output_channel=64)
        self.decoder_1 = UNet_up_block(prev_channel=self.encoder_1.output_channels,
                                       input_channel=self.decoder_2.output_channels, output_channel=64)
        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.forward = self.topologies[topology]
        print('\n\n' + "#" * 100)
        print("(LOG): The following Model Topology will be Utilized: {}".format(
            self.forward.__name__))
        print("#" * 100 + '\n\n')

    def ENC_1_DEC_1(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1_cat_1 = self.dropout(x1_cat)
        x1 = self.max_pool(x1_cat_1)
        x_mid = self.mid_conv_64_64_a(x1)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_64_64_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_1(x1_cat, x_mid)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_2_DEC_2(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x_mid = self.mid_conv_128_128_a(x2)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_128_128_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_2(x2_cat, x_mid)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_3_DEC_3(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)
        x_mid = self.mid_conv_256_256_a(x3)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_256_256_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_3(x3_cat, x_mid)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_4_DEC_4(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)
        x4_cat = self.encoder_4(x3)
        x4_cat_1 = self.dropout(x4_cat)
        x4 = self.max_pool(x4_cat_1)
        x_mid = self.mid_conv_512_1024(x4)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_1024_1024(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_4(x4_cat, x_mid)
        x = self.decoder_3(x3_cat, x)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)


class SegFormerB5(BaseModel):
    def __init__(self, input_channels=18, num_classes=2):
        super(SegFormerB5, self).__init__()
        # Load the SegFormer B5 model from timm
        self.model = timm.create_model('segformer_b5', pretrained=False, num_classes=num_classes, in_chans=input_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return x, self.softmax(x)


class UNet3Plus(BaseModel):
    """
    UNet 3+ variant using the same VGG11-pretrained encoder blocks as UNet for fair comparison.
    This implementation performs full-scale skip aggregation at the highest resolution (level 1).

    Encoder: identical to UNet encoders (with VGG11 seeded layers where applicable)
    Decoder: aggregates multi-scale features (x1, x2, x3, x4, x_mid) to level-1 resolution
    Output: returns (logits, softmax(logits)) to match existing training loop
    """

    def __init__(self, input_channels: int, num_classes: int):
        super(UNet3Plus, self).__init__()
        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)

        # encoder (reuse same as UNet for apples-to-apples comparison)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.6)
        self.activate = nn.ReLU()

        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(
            128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(
            256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])

        # bottleneck as in UNet
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)

        # channel adjust layers to unify to the same channel width for aggregation
        agg_channels = 64
        self.conv_adj_x1 = nn.Conv2d(64, agg_channels, kernel_size=3, padding=1)
        self.conv_adj_x2 = nn.Conv2d(128, agg_channels, kernel_size=3, padding=1)
        self.conv_adj_x3 = nn.Conv2d(256, agg_channels, kernel_size=3, padding=1)
        self.conv_adj_x4 = nn.Conv2d(512, agg_channels, kernel_size=3, padding=1)
        self.conv_adj_xm = nn.Conv2d(1024, agg_channels, kernel_size=3, padding=1)

        self.bn_adj_x1 = nn.BatchNorm2d(agg_channels)
        self.bn_adj_x2 = nn.BatchNorm2d(agg_channels)
        self.bn_adj_x3 = nn.BatchNorm2d(agg_channels)
        self.bn_adj_x4 = nn.BatchNorm2d(agg_channels)
        self.bn_adj_xm = nn.BatchNorm2d(agg_channels)

        # fusion after concatenation (5 scales * agg_channels)
        fusion_in = agg_channels * 5
        self.fuse_conv1 = nn.Conv2d(fusion_in, 64, kernel_size=3, padding=1)
        self.fuse_bn1 = nn.BatchNorm2d(64)
        self.fuse_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fuse_bn2 = nn.BatchNorm2d(64)

        # output
        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # upsampler
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def _act_bn(self, bn_layer, x):
        return self.activate(bn_layer(x))

    def forward(self, x_in):
        # encoder pathway
        x1 = self.encoder_1(x_in)          # C=64, H/1
        x1_p = self.max_pool(self.dropout(x1))

        x2 = self.encoder_2(x1_p)          # C=128, H/2
        x2_p = self.max_pool(self.dropout(x2))

        x3 = self.encoder_3(x2_p)          # C=256, H/4
        x3_p = self.max_pool(x3)           # we drop dropout here to match UNet order above

        x4 = self.encoder_4(x3_p)          # C=512, H/8
        x4_p = self.max_pool(self.dropout(x4))

        # bottleneck
        xm = self.activate(self.mid_conv_512_1024(x4_p))
        xm = self.activate(self.mid_conv_1024_1024(xm))   # C=1024, H/16
        xm = self.dropout(xm)

        # adjust channels to agg_channels and upsample to level-1 spatial size
        x1_adj = self._act_bn(self.bn_adj_x1, self.conv_adj_x1(x1))
        x2_adj = self._act_bn(self.bn_adj_x2, self.conv_adj_x2(self.upsample2(x2)))
        x3_adj = self._act_bn(self.bn_adj_x3, self.conv_adj_x3(self.upsample4(x3)))
        x4_adj = self._act_bn(self.bn_adj_x4, self.conv_adj_x4(self.upsample8(x4)))
        xm_adj = self._act_bn(self.bn_adj_xm, self.conv_adj_xm(self.upsample16(xm)))

        # concatenate full-scale features at level-1 resolution
        x_cat = torch.cat([x1_adj, x2_adj, x3_adj, x4_adj, xm_adj], dim=1)

        # fuse
        x = self._act_bn(self.fuse_bn1, self.fuse_conv1(x_cat))
        x = self._act_bn(self.fuse_bn2, self.fuse_conv2(x))

        # logits and softmax
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

@torch.no_grad()
def check_model(topology, input_channels, num_classes, input_shape):
    model = UNet(topology=topology, input_channels=input_channels,
                 num_classes=num_classes)
    model.eval()
    in_tensor = torch.Tensor(*input_shape)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(in_tensor.shape, out_tensor.shape)


if __name__ == '__main__':
    # check_model
    check_model(topology="ENC_1_DEC_1", input_channels=7,
                num_classes=2, input_shape=[4, 7, 64, 64])
