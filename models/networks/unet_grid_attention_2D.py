import torch.nn as nn
from .utils import unetConv2, unetUp, UnetGridGatingSignal2
import torch.nn.functional as F
from models.layers.grid_attention_layer import GridAttentionBlock2D  # Assuming a 2D version exists
from models.networks_other import init_weights

class unet_grid_attention_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2), is_batchnorm=True):
        super(unet_grid_attention_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal2(filters[4], filters[3], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=nonlocal_mode)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
