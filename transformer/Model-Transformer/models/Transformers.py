from curses import window
import torch
import torchvision
import torch.nn as nn

import math 
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Why odd numer of d_model not work
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderDecoder(nn.Module):

    def __init__(
        self, 
        d_model: int = 6,  dropout: float = 0.5, num_class: int = 1,
        d_hid: int = int(256/2), nhead: int = 1,  nlayers_e: int = int(64/64), 
        nlayers_d: int = int(16/16), windows: int = 10, ntoken: int = 100):        
        super().__init__()
        
        self.window = windows
        self.model_type = f'TransEnDecoder-Window{windows}-EL{nlayers_e}-DL{nlayers_d}-Hid{d_hid}-NHead{nhead}'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_e)
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers_d)        
        
        # Fin
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, num_class)
        self.linear2 = nn.Linear(d_model, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, train: bool, 
                memory: Tensor = None, src_mask: Tensor = None) -> Tensor:
        
        assert train or memory is not None, "Test mode but no memory" # Check if train is False and memory is None
        assert self.window != tgt.size(0), 'Window size wrong!!' # Check window size
        
        # Positional encode
        """
        Input of pos_encoding:
        src: (totallen, seqlen, d_model)
        tgt: (batch, seqlen, d_model)
        """        
        src = self.pos_encoder(src) 
        if tgt.size(1) > 1:
            tgt = self.pos_encoder(tgt)
        
        # Lengths
        N = src.size(0)
        L_src = src.size(1)
        L_tgt = tgt.size(1)
        D = src.size(2)
        
        # Encoder
        if train is True: 
            """
            Output of encoder: (1, seqlen, d_model)
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(L_src).to(device)
            memory = self.transformer_encoder(src, src_mask) # Memory  
            memory = memory.repeat(tgt.size(0), 1, 1)
        
        # Decoder
        # =====
        """
        # If padding mask:
        tgt_padding_mask = self.padding_mask(tgt).to(device)
        output = self.transformer_decoder(
            tgt=tgt, memory=memory , tgt_key_padding_mask = tgt_padding_mask) # 
        """
        output = self.transformer_decoder(tgt=tgt, memory=memory) 
        output = tgt + output
        
        output = self.linear1(output[:, -1, :].reshape(output.size(0), -1))
        tgt = self.linear2(tgt[:, -1, :].reshape(output.size(0), -1))
        output = tgt + output
        
        return memory, output
        
    def transform_patch_len_to_1(self, tgt):
        """
        Use in old version
        """
        tgt = tgt.permute(0, 2, 1)
        tgt1 = tgt.view(tgt.size(0), -1)
        tgt = self.conv1(tgt)        
        tgt = self.bn1(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt2 = tgt.view(tgt.size(0), -1)        
        tgt = self.conv2(tgt)
        tgt = self.bn2(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt3 = tgt.view(tgt.size(0), -1)
        tgt = self.conv3(tgt)
        tgt = self.bn3(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt4 = tgt.view(tgt.size(0), -1)
        tgt = self.maxpool(tgt)
        tgt5 = tgt.view(tgt.size(0), -1)
        tgt = self.maxpool2(tgt)
        tgt = tgt.view(tgt.size(0), -1)
        tgt = self.conv_linear(tgt)
        tgt1 = self.conv_linear1(tgt1)
        tgt2 = self.conv_linear2(tgt2)
        tgt3 = self.conv_linear3(tgt3)
        tgt4 = self.conv_linear4(tgt4)
        tgt5 = self.conv_linear5(tgt5)
        tgt = tgt + tgt1 + tgt2 + tgt3 + tgt4 + tgt5        
        
        tgt = tgt.unsqueeze(1)
        
        return tgt
        
    def padding_mask(self, x):
        N = x.size(0) # batch size
        L = x.size(1) # sequence length        
        patch_mask = torch.randint(0, int(L*0.5), (x.size(0),), dtype=torch.int)
        padding_mask = torch.zeros(N, L, dtype=torch.float32)
        for r in range(N):
            padding_mask[r, :patch_mask[r]] = torch.tensor(float('-inf'))
        return padding_mask