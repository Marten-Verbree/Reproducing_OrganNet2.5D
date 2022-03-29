#imports
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

class ConvBlock2D(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock2D, self).__init__()
    self.conv1 = nn.Conv3d(in_channels,out_channels, kernel_size=(1,3,3), padding='same') #unsure whether padding is used, assuming that it is
    self.relu1 = nn.ReLU()
    self.batchnorm = nn.BatchNorm3d(out_channels)
    self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), padding='same') #not clearly mentioned in paper that it is out_channels to out_channels
    self.relu2 = nn.ReLU()
    self.batchnorm2 = nn.BatchNorm3d(out_channels)
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.batchnorm(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.batchnorm2(x)
    return x

class ConvBlock3DResse(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock3DResse, self).__init__()
    self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same')
    self.relu1 = nn.ReLU()
    self.batchnorm1 = nn.BatchNorm3d(out_channels)
    self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same')
    self.relu2 = nn.ReLU()
    self.batchnorm2 = nn.BatchNorm3d(out_channels)
    self.globalpool = nn.AdaptiveAvgPool3d(output_size=1)
    self.flatten1 = nn.Flatten() #might have to change start_dim
    self.linear1 = nn.Linear(1, 1)
    self.relu3 = nn.ReLU()
    self.linear2 = nn.Linear(1, 1)
    self.sigmoid1 = nn.Sigmoid()
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.batchnorm1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.batchnorm2(x)
    x1 = self.globalpool(x)
    x1 = self.linear1(x1)
    x1 = self.relu3(x1)
    x1 = self.linear2(x1)
    x1 = self.sigmoid1(x1)
    print(x.shape)
    print(x1.shape)
    return x1*x + x

class HybridDilatedConv3DResse(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(HybridDilatedConv3DResse, self).__init__()
    #different dilation rates? but how different?
    self.hdc = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same', dilation = 2)
    self.relu1 = nn.ReLU()
    self.batchnorm1 = nn.BatchNorm3d(out_channels)
    self.globalpool = nn.AdaptiveAvgPool3d(output_size=1)
    self.flatten1 = nn.Flatten() #might not be necessary, let's see what the output is of the globalpooling. I suppose this is already flattened (output_size=1).
    self.linear1 = nn.Linear(1, 1)
    self.relu2 = nn.ReLU()
    self.linear2 = nn.Linear(1, 1)
    self.sigmoid1 = nn.Sigmoid()
  def forward(self, x):
    x = self.hdc(x)
    x = self.relu1(x)
    x = self.batchnorm1(x)
    x1 = self.globalpool(x)
    x1 = self.linear1(x1)
    x1 = self.relu2(x1)
    x1 = self.linear2(x1)
    x1 = self.sigmoid1(x1)
    return x1*x + x
class Conv3Dfine(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Conv3Dfine, self).__init__()
    self.conv3D = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding='same')
    self.relu = nn.ReLU()
    self.batchnorm = nn.BatchNorm3d(out_channels)
  def forward(self, x):
    x = self.conv3D(x)
    x = self.relu(x)
    x = self.batchnorm(x)
    return x

class TorchCNN(nn.Module):
  def __init__(self, in_channels = None, hidden_channels = None, out_features = None):
    super(TorchCNN, self).__init__()
    if in_channels is None:
        in_channels = 1
    if hidden_channels is None:
        hidden_channels = [16, 32, 64, 128, 256]
    if out_features is None:
        out_features = 10
    self.conv2D1 = ConvBlock2D(in_channels, hidden_channels[0])
    
    self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2)) 
    self.conv3D_coarse1 = ConvBlock3DResse(hidden_channels[0],hidden_channels[1])
    self.pool2 = nn.MaxPool3d(kernel_size=2, stride =2)
    self.conv3D_coarse2 = ConvBlock3DResse(hidden_channels[1],hidden_channels[2])
    self.hdc1 = HybridDilatedConv3DResse(hidden_channels[2], hidden_channels[3])
    self.hdc2 = HybridDilatedConv3DResse(hidden_channels[3], hidden_channels[4])
    self.conv3D_fine1 = Conv3Dfine(hidden_channels[4], hidden_channels[3])
    self.hdc3 = HybridDilatedConv3DResse(hidden_channels[4], hidden_channels[3])
    self.conv3D_fine2 = Conv3Dfine(hidden_channels[3], hidden_channels[2])
    self.conv3D_coarse3 = ConvBlock3DResse(hidden_channels[3], hidden_channels[2])
    self.transpose1 = nn.ConvTranspose3d(hidden_channels[2], hidden_channels[1], kernel_size = 2, stride=2)
    self.conv3D_coarse4 = ConvBlock3DResse(hidden_channels[2], hidden_channels[1])
    self.transpose2 = nn.ConvTranspose3d(hidden_channels[1], hidden_channels[0], kernel_size = (1,2,2), stride=(1,2,2))
    self.conv2D2 = ConvBlock2D(hidden_channels[1], hidden_channels[1])
    self.conv3D_fine3 = Conv3Dfine(hidden_channels[1], out_features)
  def forward(self, x):
    x1 = self.conv2D1(x)
    x2 = self.pool1(x1)
    x2 = self.conv3D_coarse1(x2)
    x3 = self.pool2(x2)
    x3 = self.conv3D_coarse2(x3)
    x4 = self.hdc1(x3)
    x5 = self.hdc2(x4)
    x5 = self.conv3D_fine1(x5)
    x5 = torch.cat((x4, x5), dim=1)
    x5 = self.hdc3(x5)
    x5 = self.conv3D_fine2(x5)
    x5 = torch.cat((x3, x5), dim=1)
    x5 = self.conv3D_coarse3(x5)
    x5 = self.transpose1(x5)
    x5 = torch.cat((x2, x5), dim=1)
    x5 = self.conv3D_coarse4(x5)
    x5 = self.transpose2(x5)
    x5 = torch.cat((x1, x5), dim=1)
    x5 = self.conv2D2(x5)
    x5 = self.conv3D_fine3(x5)
    return x5

def main():
    in_channels = 1
    hidden_channels = [16, 32, 64, 128, 256]
    out_channels = 10 # for Miccai data set
    CNN = TorchCNN(in_channels, hidden_channels, out_channels)
    x = torch.randn((1, 1, 240, 240, 80))
    out = CNN.forward(x)
    print(out.shape)

if __name__=='__main__':
    main()