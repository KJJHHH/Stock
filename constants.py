from enum import Enum


class ModelName(str, Enum):
    DECODER_ONLY = "DecoderOnly"
    LSTM = "LSTM"
    GRU = "GRU"
    TRANSFORMER_ENCODER = "TransformerEncoder"
    TCN = "TCN"
    GBDT = "GBDT"
    MSKF = "MSKF"
    IGA_SVR = "IGA_SVR"
