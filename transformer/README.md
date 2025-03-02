# Transformer Models to Predict Stock 
## Goals
Predict the daily percentile change for open and close price

## Experiments
### Performance with Transformer
Asset using transformer model comparing to buy-and-hold strategy
- [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_5871_backtest.png) | 中租，5871
- [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2454_backtest.png) | 聯發科技，2454
- [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2884_backtest.png) | 玉山金融控股公司，2884

## Models 
- [x] Transformer-Encoder-Decoder
- [ ] Decoder-only
```python
L: Total length for all train dates
S: Sequence length for each patch
P: Patch numbers = L - S + 1
D: Input dim
W: window
B: Batch size
```
### Transformer Model Structure
- Positional Encoding   
- Transformer: encoder and decoder
- Linear transformation 
```python
Input of encoder: src shape = (B, T, D)
Input of decoder: tgt shape = (B, W, D), memory shape = (B, T, D)
```

## Data Preprocess

1. Download data with Open, Close, High, Low, Volume
2. Transform the data to percentile change
3. Normalise
4. Variables: normalised percentile change of Open, Close, High, Low, and Volume
5. Predict with last 10 dates data
6. Predict the daily change of open and close
