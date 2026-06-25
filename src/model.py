"""cnn architecture for cifar-10 classification"""
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias= False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))
    
class CIFAR10CNN(nn.Module):
    """simple vgg style cnn architecture for cifar-10 classification"""
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64, pool=True),
            
            ConvBlock(64, 128),
            ConvBlock(128, 128, pool=True),
            
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
