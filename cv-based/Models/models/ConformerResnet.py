import sys
sys.path.append('./models/')
import torch
from torch import nn
from torchaudio.models import Conformer
from Encoding import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Resnet from Pytorch: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# Resnet pretrain pytorch: https://pytorch.org/hub/pytorch_vision_resnet/
# https://medium.com/ching-i/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1-cnn-%E7%B6%93%E5%85%B8%E6%A8%A1%E5%9E%8B-googlelenet-resnet-densenet-with-pytorch-code-1688015808d9
class bottleneck_block(nn.Module):
    # 輸出通道乘的倍數
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(bottleneck_block, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Conformer_Resnet(nn.Module):
    #  Conformer_Resnet(bottleneck_block, [3, 4, 23, 3], num_class)
    def __init__(self, num_class, net_block = bottleneck_block, layers = [3, 4, 23, 3]):
        super(Conformer_Resnet, self).__init__()
        self.model_type = 'Conformer-Resnet'
        # =======
        # Unet
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)
        
        self.avgpooling = nn.AvgPool2d(3, stride=1)        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.fc1 = nn.Linear(2048*2*2, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.ln1 = nn.LayerNorm((6, 100, 100))
        
        
        # =======
        # Conformer
        self.positional_encode = PositionalEncoding(100)
        self.patch_embedding = PatchEmbedding(in_channels=6, patch_size=10, emb_size=600)
        self.conformer = Conformer(
            input_dim=600,
            num_heads=5,
            ffn_dim=128,
            num_layers=1,
            depthwise_conv_kernel_size=31)


    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):
        downsample = None

      # 在 shortcut 時，若維度不一樣，要更改維度
        if stride != 1 or self.in_channels != out_channels * net_block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride, bias=False),
                      nn.BatchNorm2d(out_channels * net_block.expansion))

        layers = []
        layers.append(net_block(self.in_channels, out_channels, stride, downsample))
        if net_block.expansion != 1:
            self.in_channels = out_channels * net_block.expansion
        else:
            self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(net_block(self.in_channels, out_channels, 1, None))

        return nn.Sequential(*layers)
    
    def forward(self, x):        
        x_i = x.clone()
        x_s = x.size()
        # =======
        # Conformer
        # x = x.view(x_s[0], x_s[1] * x_s[3], x_s[2])        
        # x = self.positional_encode(x)
        x = self.patch_embedding(x)
        lengths = torch.tensor([x.shape[1] for i in range(len(x))]).to(device)
        x, len_ = self.conformer(x, lengths)
        x = x.permute(0, 2, 1).view(x_s)
        
        x = self.ln1(x)
        x = x + x_i
        
        # =======
        # Res
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x