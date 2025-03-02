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

# Decoder
class TransformerDecoderOnly(nn.Module):

    def __init__(self, ntoken: int = 100, d_model: int = 6, nhead: int = 2, d_hid: int = 128, num_class: int = 1,
                 nlayers: int = 64, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer-Decoder-Only'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model*ntoken, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def padding_mask(self, x):
        N = x.size(0) # batch size
        L = x.size(1) # sequence length        
        patch_mask = torch.randint(0, 50, (x.size(0),), dtype=torch.int)
        padding_mask = torch.zeros(N, L, dtype=torch.float32)
        for r in range(N):
            padding_mask[r, :patch_mask[r]] = torch.tensor(float('-inf'))
        return padding_mask
    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        NOTE: tgt = src
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = self.embedding(src) * math.sqrt(self.d_model)
        N = src.size(0)
        L = src.size(1)
        D = src.size(2)
        src = self.pos_encoder(src)
        if src_mask is None:
            # Check mask correct?
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)

        padding_mask = self.padding_mask(src).to(device)
        output = self.transformer_decoder(
            src, src, tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)
        output = output + src
        output = output.reshape(output.size(0), -1)
        output = self.linear(output)
        return output

class TransformerEncoderOnly(nn.Module):

    def __init__(self, ntoken: int = 100, d_model: int = 6, nhead: int = 2, d_hid: int = 128, num_class: int = 1,
                 nlayers: int = 16, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer-Encoder-Only'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model*ntoken, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def padding_mask(self, x):
        N = x.size(0) # batch size
        L = x.size(1) # sequence length        
        patch_mask = torch.randint(0, 5, (x.size(0),), dtype=torch.int)
        padding_mask = torch.zeros(N, L, dtype=torch.float32)
        for r in range(N):
            padding_mask[r, :patch_mask[r]] = torch.tensor(float('-inf'))
        return padding_mask

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = self.embedding(src) * math.sqrt(self.d_model)
        N = src.size(0)
        L = src.size(1)
        D = src.size(2)
        
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
            
        padding_mask = self.padding_mask(src).to(device)
        output = self.transformer_encoder(src, src_mask, padding_mask)
        output = output.reshape(output.size(0), -1)
        output = self.linear(output)
        return output
