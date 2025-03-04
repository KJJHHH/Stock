# Transformer Models to Predict Stock 
## Goals
Predict the daily percentile change for open and close price

## Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform the data to percentile change
3. Normalise
4. Variables: normalised percentile change of Open, Close, High, Low, and Volume
5. Predict with last 10 dates data
6. Predict the daily change of open and close

## Models 
- [x] Transformer
- [ ] Decoder-only
    - processing...

## Experiments
### Performance with Decoder-Only
- [Decoder-only for time series](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- Asset backtracking, test from 2024-05 to 2024-12
    - [中租，5871](https://github.com/KJJHHH/Stock/blob/main/transformer/Model_Decoder/Model_Result/Decoder-only_class2_5871_backtest.png)
    - [聯發科技，2454](https://github.com/KJJHHH/Stock/blob/main/transformer/Model_Decoder/Model_Result/Decoder-only_class2_2454_backtest.png) 
    - [玉山金控，2884](https://github.com/KJJHHH/Stock/blob/main/transformer/Model_Decoder/Model_Result/Decoder-only_class2_2884_backtest.png) 
### Performance with Transformer
- Asset with transformer model, comparing to buy-and-hold strategy
- Asset backtracking
    - ...




