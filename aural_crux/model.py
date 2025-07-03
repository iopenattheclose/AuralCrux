import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride = 1):
        super().__init__
        self.conv1 = nn.Conv2d(in_channels = input_channels,out_channels= output_channels,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels = output_channels,out_channels= output_channels,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
