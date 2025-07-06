# AuralCrux

This project is an attempt to classify sounds like a dog barking or birds chirping or wolves howling from an audio file. Uses Residual Neural Netwroks for classification. The frontend is built using ReactJS and Tailwind CSS.

summary of how self.conv1 works:

You create self.conv1 in the __init__ method, making it an instance variable that holds an object (an instance) of the nn.Conv2d class. This object represents your convolutional layer.

When you invoke **self.conv1(x)** with an input tensor x, PyTorch's nn.Module machinery (which nn.Conv2d inherits from) internally calls the forward method defined within the nn.Conv2d class on that input x. This is how the actual convolution operation is performed.


Extending it like:

self.bn1 = nn.BatchNorm2d(output_channels) in __init__: This line creates an instance (an object) of the nn.BatchNorm2d class and assigns it to the instance variable self.bn1. This self.bn1 object represents your batch normalization layer.

out = self.bn1(out) in forward: When you pass the tensor out through self.bn1 using the call syntax (out), it internally invokes the forward method defined within the nn.BatchNorm2d class. This forward method then performs the actual batch normalization operation on the input tensor out.

This consistent pattern (create an nn.Module object in __init__, call its forward method implicitly by calling the object in forward) is fundamental to how PyTorch networks are built.