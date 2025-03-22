import torch
import torch.nn as nn

import math 

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        
        # Create matrix for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    """From language model to time sequence

    Args:
        d_model: embed dim -> features
        ntokens: seq length -> window length
        src_mask: True
    """
    __name__ = "Transformer"
    
    def __init__(
        self, 
        d_model: int = 6,  
        dropout: float = 0.5, 
        d_hid: int = 128, 
        nhead: int = 2,  
        nlayers_e: int = 64, 
        nlayers_d: int = 16, 
        ntoken: int = 10, # Window
        src_len: int = 1960, # Src seq len
        ):        
        super().__init__()
        
        
        # Len of windows
        self.d_model = d_model
        self.ntoken = ntoken
        self.model_type = f'Transformer-Window{ntoken}-EL{nlayers_e}-DL{nlayers_d}-Hid{d_hid}-NHead{nhead}'
        self.dropout = dropout
        self.src_len = src_len
        
        # Positional encoding
        # self.pos_enc = PositionalEncoding(d_model, src_len)
        # self.pos_dec = PositionalEncoding(d_model, ntoken)
        
        # Mask
        # self.src_mask = nn.Transformer.generate_square_subsequent_mask(self.src_len)
        # self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.ntoken)

        # Transformer
        """
        - Output of encoder: (1, seq, d_model)
        - batch_first (bool) - If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
        - attention mask code: in generate_square_subsequent_mask
            torch.triu(
                torch.full((10, 10), float("-inf"), dtype=torch.float32, device=device),
                diagonal=1,)
        - About attention mask and key feature mask: https://www.zhihu.com/question/455164736
        """
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_hid, 
            dropout=dropout, 
            batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_e)
        
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_hid, 
            dropout=dropout, 
            batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers_d)        
        
        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(d_model, 1)

    def forward(
        self, 
        tgt: torch.tensor, 
        src: torch.tensor = None, 
        memory: torch.tensor = None,
        ) -> torch.tensor:
        
        """
        Input:
        src: (batch, seq, feature) = (1, seq_src, d_model), a long seq
        tgt: (batch, seq, feature) = (batch, seq_tgt = window, d_model)
        """     
        
        # Encoder
        """
        NOTE:
        When validating, memory are passed by train memory
        ===
        - Padding for different src len
        if memory is None and src.size(1) != self.src_len:
            self.src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(device)
            src = F.pad(src, (0, 0, 0, self.src_len - src.size(1)))
            self.src_mask = ...
        """
        if memory is None: 
            # Positional encode
            # src = self.pos_enc(src)
            # Encoder
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(device)
            memory = self.transformer_encoder(src, src_mask) 
            
        # For decoder input
        memory_ = memory[0].repeat(tgt.size(0), 1, 1)
        
        # Positional encode
        # tgt = self.pos_dec(tgt)
        # Decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        output = self.transformer_decoder(tgt=tgt, tgt_mask=tgt_mask, memory=memory_) 
        
        # Linear
        ...
        
        return memory, output

# Resnet

# Define the basic block with skip connection
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.tanh(x)
