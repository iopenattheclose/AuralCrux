import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride = 1):
        super().__init__()
        #self.conv1 is an instance variable that holds an instance (an object) of the nn.Conv2d class.
        #self.conv1 is an object. It's an instance of the nn.Conv2d class.
        self.conv1 = nn.Conv2d(in_channels = input_channels,out_channels= output_channels,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels = output_channels,out_channels= output_channels,kernel_size=3,
                               padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.skip_connection = nn.Sequential() #initializing instance variable
        self.use_skip_connection = stride != 1 and input_channels != output_channels
        if self.use_skip_connection:
            self.skip_connection = nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,
                                                           kernel_size=1,stride=stride, bias=False)
                                                           ,nn.BatchNorm2d(output_channels)
                                                )
    #In PyTorch, every nn.Module subclass must implement a forward method. 
    # This method defines how the input data (x) is processed through the layers defined in the __init__ method.
    def forward(self, x, fmap_dict=None, prefix=""):

        #You won't call forward directly.
        #When you use an instance of this class like a function, for example, my_residual_block(x), 
        # the __call__ method of nn.Module is implicitly invoked. This __call__ method is what handles 
        # the internal PyTorch logic and then, in turn, calls the forward method to perform the computations.

        #when you write out = self.conv1(x),you are invoking the __call__ magic method of the self.conv1 object (which is an instance of nn.Conv2d)
        #self.conv1(x) -> This effectively executes the forward method defined within the nn.Conv2d class (the code that performs the actual convolution operation) on your input x
        #objects can be made callable if __call__ is defined in the class (here it is defined in Conv2d class)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        skip_connection = self.skip_connection(x) if self.use_skip_connection else x
        out_add = out + skip_connection

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

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
        #module list is used because it tells Pytorch that ResidualBlocks are trainable
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range (3)])
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128,128, stride=2 if i == 0 else 1) for i in range (4)])
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i == 0 else 256,256, stride=2 if i == 0 else 1) for i in range (6)])
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512,512, stride=2 if i == 0 else 1) for i in range (3)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))#Global avg pool
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, return_feature_maps=False):
            if not return_feature_maps:
                x = self.conv1(x)
                for block in self.layer1:
                    x = block(x)
                for block in self.layer2:
                    x = block(x)
                for block in self.layer3:
                    x = block(x)
                for block in self.layer4:
                    x = block(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc(x)
                return x
            else:
                feature_maps = {}
                x = self.conv1(x)
                feature_maps["conv1"] = x

                for i, block in enumerate(self.layer1):
                    x = block(x, feature_maps, prefix=f"layer1.block{i}")
                feature_maps["layer1"] = x

                for i, block in enumerate(self.layer2):
                    x = block(x, feature_maps, prefix=f"layer2.block{i}")
                feature_maps["layer2"] = x

                for i, block in enumerate(self.layer3):
                    x = block(x, feature_maps, prefix=f"layer3.block{i}")
                feature_maps["layer3"] = x

                for i, block in enumerate(self.layer4):
                    x = block(x, feature_maps, prefix=f"layer4.block{i}")
                feature_maps["layer4"] = x

                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc(x)
                return x, feature_maps



if __name__=="__main__":
    print("gpu" if torch.cuda.is_available() else 'cpu')