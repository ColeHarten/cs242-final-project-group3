import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_CT_multi_att_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_CT_multi_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # self.early_exit_up4 = nn.Conv3d(filters[3], n_classes, kernel_size=1)
        # self.early_exit_up3 = nn.Conv3d(filters[2], n_classes, kernel_size=1)
        # self.early_exit_up2 = nn.Conv3d(filters[1], n_classes, kernel_size=1)

        self.thresholds = None
        self.layer_outputs = {}
        self.n_classes = n_classes

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        if self.thresholds is None:
            # raise ValueError("Thresholds are None in forward method")
            print(f"No threshold values loaded.")
            print(f"Learning thresholds through regular_forward pass...")
            return self.regular_forward(inputs)
        else:
            print(f"Threshold values loaded: {self.thresholds}")
            print(f"Using thresholds in forward pass ...")

        # Encoder
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        gating = self.gating(center)

        # Decoder
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)

        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)

        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)

        # Compute logits for each layer
        # logits_up_concat4 = self.early_exit_up4(up4)
        # logits_up_concat3 = self.early_exit_up3(up3)
        # logits_up_concat2 = self.early_exit_up2(up2)
        # self.layer_outputs['up_concat4'] = logits_up_concat4
        # self.layer_outputs['up_concat3'] = logits_up_concat3
        # self.layer_outputs['up_concat2'] = logits_up_concat2

        # Early exit logic for multiple layers
        for layer_name, feature_map in zip(
            ["up_concat2"],
            [up2],
        ):
            print(f"Checking early exit for {layer_name}")

            if self.thresholds is None:
                print(f"Thresholds not set for {layer_name}. Proceeding without early exit.")
                continue

            unconfident_map, confident_map, pixel_mask = self.apply_early_exit(feature_map, self.thresholds, (160, 160, 96))
            
            print(f"{pixel_mask.sum().item()}/{pixel_mask.numel()} pixels masked as confident.")
            print(f"Continuing with {(~pixel_mask).sum().item()} unconfident pixels at {layer_name}.")
            
        up1 = self.up_concat1(conv1, unconfident_map)
        
        print(f"up1 shape: {up1.shape}")
        print(f"mask shape: {pixel_mask.shape}")
        print(f"confident map shape: {confident_map.shape}")
        
        up1 = (up1 * pixel_mask) + confident_map

        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))

        return final
    
    def regular_forward(self, inputs):
        # Encoder
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        gating = self.gating(center)

        # Decoder
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        self.layer_outputs['up_concat4'] = up4

        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        self.layer_outputs['up_concat3'] = up3

        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        self.layer_outputs['up_concat2'] = up2

        # Compute logits for each layer
        logits_up_concat4 = self.exit_up2(up2)
        # self.layer_outputs['up_concat4'] = logits_up_concat4
        # self.layer_outputs['up_concat3'] = logits_up_concat3
        self.layer_outputs['up_concat2'] = logits_up_concat2

        up1 = self.up_concat1(conv1, up2)

        print(f"Final layer_outputs keys: {self.layer_outputs.keys()}")

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        
        return final
    
    import torch.nn.functional as F

    def apply_early_exit(self, feature_map, thresholds, target_shape):
        """
        Apply early exit logic and interpolate the pixel mask to match the target shape.

        Args:
            feature_map (torch.Tensor): Input feature map, shape [B, C, D, H, W].
            thresholds (list or None): Class-specific thresholds for confidence.
            target_shape (tuple): Desired output shape (spatial dimensions) for upsampling.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]: 
                - Unconfident map.
                - Confident map.
                - Interpolated pixel mask.
        """
        
        # If thresholds are not set, skip early exit
        if thresholds is None:
            print("Thresholds not set, skipping early exit.")
            pixel_mask = torch.ones_like(feature_map[:, 0], dtype=torch.bool)  # Default mask: no pixels excluded
        else:
            # Compute confidence and predicted classes
            softmax_confidence = F.softmax(feature_map, dim=1)
            early_exit_confidence, predicted_classes = torch.max(softmax_confidence, dim=1)

            # Create pixel mask based on class-specific thresholds
            pixel_mask = torch.zeros_like(early_exit_confidence, dtype=torch.bool)
            for cls in range(len(thresholds)):
                class_mask = (predicted_classes == cls)
                pixel_mask[class_mask] = early_exit_confidence[class_mask] < thresholds[cls]

            # Expand dimensions for broadcasting
            pixel_mask = pixel_mask.unsqueeze(1)

        # Separate confident and unconfident feature maps
        unconfident_map = feature_map * pixel_mask.float()
        confident_map = feature_map * (~pixel_mask).float()

        # Interpolate the pixel mask and feature maps to the target shape
        interpolated_mask = F.interpolate(pixel_mask.float(), size=target_shape, mode='trilinear', align_corners=False).bool()
        # unconfident_map_interpolated = F.interpolate(unconfident_map, size=target_shape, mode='trilinear', align_corners=False)
        confident_map_interpolated = F.interpolate(confident_map, size=target_shape, mode='trilinear', align_corners=False)

        return unconfident_map, confident_map_interpolated, interpolated_mask


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


