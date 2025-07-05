import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride = 1):
        super().__init__
        self.conv1 = nn.Conv2d(in_channels = input_channels,out_channels= output_channels,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels = output_channels,out_channels= output_channels,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.skip_connection = nn.Sequential() #initializing instance variable
        self.use_skip_connection = stride != 1 and input_channels != output_channels
        if self.use_skip_connection:
            self.skip_connection = nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,stride=stride, bias=False),
                                                 nn.BatchNorm2d(output_channels)
                                                 )

    def forwad(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        skip_connection = self.skip_connection(x) if self.use_skip_connection else x

        out_add = out + skip_connection
        out = torch.relu(out_add)

        return out


 #NN class for training and inference   

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        

