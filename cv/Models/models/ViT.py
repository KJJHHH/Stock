import torch
import torchvision
import torch.nn as nn 
import sys 
sys.path.append('../')
try:
    from models.Encoding import *
except:
    from Encoding import *

# Vision Transformer: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# try pretrain, same link above
class ViT(nn.Module):
    def __init__(self, num_class=2):
        super(ViT, self).__init__()
        
        self.model_type = 'Vision-Transformer'

        # =======
        # Unet
        self.conv_init = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.VisionTransformer = torchvision.models.VisionTransformer(
            image_size = 100,
            patch_size = 10,
            num_layers = 16,
            num_heads = 5,
            hidden_dim = 250,
            mlp_dim = 520
        )
        self.ln_init = nn.LayerNorm(1000)
        self.fc = nn.Linear(1000, num_class)
        self.fc_res = nn.Linear(3*100*100, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input scale: (0, 255)
        Output scale: (0, 255)
        """
        
        x = self.conv_init(x)
        x_ = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.VisionTransformer(x)
        x = self.ln_init(x)
        x = self.relu(x)        
        res = self.fc_res(x_)
        x = self.fc(x)
        x = x + res

        return x