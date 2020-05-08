import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


class IdentityExpansion(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityExpansion, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.act7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.act8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.act9 = nn.ReLU()
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.act10 = nn.ReLU()
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act11 = nn.ReLU()
        self.act12 = nn.ReLU()
        # Here lies the latent space, with m feature maps of size n x n, total parameters m * n ^ 4
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.act13 = nn.ReLU()
        # The channel-wise fully-connected layer is followed by a 1 stride convolution layer to propagate information across channels
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act14 = nn.ReLU()
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.act15 = nn.ReLU()
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.act16 = nn.ReLU()
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()

    def forward(self, x):
        residual = x
        residual = IdentityExpansion(60,32,1).forward(residual)

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))

        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))

        x = self.act7(self.conv7(x))
        x = self.act8(self.conv8(x))

        x = self.act9(self.conv9(x))
        x = self.act10(self.conv10(x))

        x = self.act11(self.conv11(x))
        x = self.act12(self.conv12(x))

        x = self.act13(self.deconv13(x))
        x = self.act14(self.conv14(x))
        x = self.act15(self.deconv15(x))
        x = self.act16(self.conv16(x))
        x += residual
        x = self.act17(self.conv17(x))
        return x


class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = layers.Flatten()
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape, arc='places2'):
        super(GlobalDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        if arc == 'celeba':
            in_features = 512 * (self.img_h//32) * (self.img_w//32)
            self.flatten6 = layers.Flatten()
            self.linear6 = nn.Linear(in_features, 1024)
            self.act6 = nn.ReLU()
        elif arc == 'places2':
            self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
            self.bn6 = nn.BatchNorm2d(512)
            self.act6 = nn.ReLU()
            in_features = 512 * (self.img_h//64) * (self.img_w//64)
            self.flatten7 = layers.Flatten()
            self.linear7 = nn.Linear(in_features, 1024)
            self.act7 = nn.ReLU()
        else:
            raise ValueError('Unsupported architecture \'%s\'.' % self.arc)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        if self.arc == 'celeba':
            x = self.act6(self.linear6(self.flatten6(x)))
        elif self.arc == 'places2':
            x = self.bn6(self.act6(self.conv6(x)))
            x = self.act7(self.linear7(self.flatten7(x)))
        return x
