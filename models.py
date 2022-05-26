# Remove warning messages
import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
GRAYSCALE = 1
RGB = 3
NUM_CLASSES = 1000

# the VGG11 architecture
class VGG11(nn.Module):
    #in_channels are the number of color channels 1 = grayscale 3 = RGB
    #num_classes are the output classes, I think it should be 3
    def __init__(self, in_channels = GRAYSCALE, num_classes=NUM_CLASSES):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers, 8 CNNs 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers, 3 Fully connected layers
        self.linear_layers = nn.Sequential(
            #5 max pool layers with stride of 2, images are 224 x 224 so 224/(2^5) = 7
            #final Convolutional layer is 512 so in is 512*7*7 = 25,088
            nn.Linear(in_features=512*7*7, out_features=4096),
           
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

if __name__ == '__main__':
    # initialize the VGG model for RGB images with 3 channels
    vgg11 = VGG11(in_channels=GRAYSCALE)
    # total parameters in the model
    total_params = sum(p.numel() for p in vgg11.parameters())
    print(f"[INFO]: {total_params:,} total parameters.")
    
    # forward pass check
    # a dummy (random) input tensor to feed into the VGG11 model
    #(inchannels, classes, dimension, dimension)
    image_tensor = torch.randn(GRAYSCALE, NUM_CLASSES, 224, 224) # a 3 image batch 1, 1 = grayscale 1, 3 = RGB
    outputs = vgg11(image_tensor)
    print(outputs.shape)