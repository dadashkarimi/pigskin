import torch
import torch.nn as nn
from . import layers


class GroupNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 do_batchnorm=True):
        super(GroupNet, self).__init__()

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = layers.group.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feat in features:
            self.downs.append(
                layers.group.MeanConv2d(
                    in_channels, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                )
            )
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(layers.group.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(
                layers.group.MeanConv2d(
                    feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                )
            )

        self.bottleneck = layers.group.MeanConv2d(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
            )
        self.final_conv = layers.group.MeanConv2d(
            features[0], out_channels, kernel_size=1, padding=0, do_activation=False, do_batchnorm=False
            )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

#             if x.shape != skip_connection.shape:
#                 print('interpolating')
#                 x = nn.functional.interpolate(x,
#                                               size=skip_connection.shape[2:],
#                                               mode='bilinear',
#                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=2)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class SimpleUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3):
        super(SimpleUNet, self).__init__()

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

        # Down part of U-Net
        for feat in features:
            self.downs.append(self.conv_block(in_channels,
                                              feat,
                                              kernel_size=conv_kernel_size,
                                              padding=padding))
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(self.conv_block(
                feat * 2, feat, kernel_size=conv_kernel_size, padding=padding))

        self.bottleneck = self.conv_block(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding)
        self.final_conv = self.conv_block(features[0], out_channels,
                                          kernel_size=1, padding=0)

    def conv_block(self, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(*args, **kwargs),
            self.act)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

#             if x.shape != skip_connection.shape:
#                 print('interpolating')
#                 x = nn.functional.interpolate(x,
#                                               size=skip_connection.shape[2:],
#                                               mode='bilinear',
#                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
