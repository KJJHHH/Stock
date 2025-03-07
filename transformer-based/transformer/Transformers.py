import torch
import torch.nn as nn

import math 

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int, 
        dropout: float = 0.1, 
        max_len: int = 5000):
        
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Input:
        x: (batch, seq, feature)
        """     
        print(x.shape, self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    """From language model to time sequence

    Args:
        d_model: embed dim -> features
        ntokens: seq length -> window length
        src_mask: True
    """
    def __init__(
        self, 
        d_model: int = 6,  
        dropout: float = 0.5, 
        d_hid: int = 128, 
        nhead: int = 2,  
        nlayers_e: int = 64, 
        nlayers_d: int = 16, 
        ntoken: int = 10,
        test = False):        
        super().__init__()
        
        # When testing
        if test:
            global device
            device = torch.device("cpu")
        
        self.ntoken = ntoken
        self.model_type = f'Transformer-Window{ntoken}-EL{nlayers_e}-DL{nlayers_d}-Hid{d_hid}-NHead{nhead}'
        
        # Pos enc
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
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
        train: bool,
        src: torch.tensor = None, 
        memory: torch.tensor = None,
        pos_enc: bool = False) -> torch.tensor:
        
        """
        Input:
        src: (batch, seq, feature) = (1, seq_src, d_model), a long seq
        tgt: (batch, seq, feature) = (batch, seq_tgt = window, d_model)
        """     
        
        assert train or memory is not None, "Testing but no memory" 
        
        # Positional encode   
        if pos_enc:
            src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
            
        
        # Encoder
        if train: 
            Ls = src.size(1) 
            src_mask = nn.Transformer.generate_square_subsequent_mask(Ls).to(device)
            memory = self.transformer_encoder(src, src_mask) 
        
        # For decoder input
        Lt = tgt.size(1) 
        memory_ = memory[0].repeat(tgt.size(0), 1, 1)
        
        # Decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Lt).to(device)
        output = self.transformer_decoder(tgt=tgt, tgt_mask=tgt_mask, memory=memory_) 
        output = tgt + output
        
        # Linear
        """
        output = self.linear1(output[:, -1, :].reshape(output.size(0), -1))
        tgt = self.linear2(tgt[:, -1, :].reshape(output.size(0), -1))
        output = tgt + output
        """
        
        return memory, output
    
"""
    def transform_patch_len_to_1(self, tgt):
        Use in old version
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
"""

