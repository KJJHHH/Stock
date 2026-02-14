import math
import torch.nn as nn

class DecoderOnly(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(DecoderOnly, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        return self.decoder(src[:, -1, :])
