import torch
import torch.nn as nn

import math 

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

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
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, src_len)
        self.pos_dec = PositionalEncoding(d_model, ntoken)

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
        """
        if memory is None: 
            # Positional encode
            # src = self.pos_enc(src)
            # Encoder
            Ls = src.size(1) 
            src_mask = nn.Transformer.generate_square_subsequent_mask(Ls).to(device)
            memory = self.transformer_encoder(src, src_mask) 
            
        # For decoder input
        Lt = tgt.size(1) 
        memory_ = memory[0].repeat(tgt.size(0), 1, 1)
        
        # Positional encode
        # tgt = self.pos_dec(tgt)
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
    