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
        #in_channels = 1 as we have single input channel for mel spectrogram
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels = 64, kernel_size = 7, stride=2, padding=3, bias = False)
                                    ,nn.BatchNorm2d(num_features = 64)
                                    ,nn.ReLU(inplace=True)
                                    ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1))   
        #module list is used because it is trainable
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range (3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128,128) for i in range (4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256,256) for i in range (6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512,512) for i in range (3)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))#flatten??
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forwad(self,x):

        x = self.conv1(x)

        for block in self.layer1:
            x = block(x)

        for block in self.layer2:
            x = block(x)

        for block in self.layer3:
            x = block(x)

        for block in self.layer4:
            x = block(x)

        x = self.avg_pool(x)

        #what
        x = x.view(x.size(0),-1)

        x = self.dropout(x)

        x = self.fc(x)


        return x



