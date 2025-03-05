# Computer Vision Methods to Predict Stock 

## Data Preprocess
1. Download data from `yfinance`. Columns: Open, Close, High, Low, Volume
2. Use percentile change
3. Predict with last 100 dates data
4. Expand the variables shape (5, 100) to (5, 100, 100) by difference between each dates

## Models
- [x] ResNet
- [x] Conformer
- [x] Conformer + ResNet (ConRes)
- [x] VisionTransformer
- [x] Pretrained VisionTransformer (ViT pretrained)

## Experiments
Asset using computer vision model comparing to buy-and-hold strategy
### 中租，5871
- Conformer: [Asset backtest](https://github.com/KJJHHH/Stock/blob/main/cv-based/models/result/Conformer-CNN_class2_5871_backtest.png) 
- Resnet: [Asset backtest](https://github.com/KJJHHH/Stock/blob/main/cv-based/models/result/ResNet_class2_5871_backtest.png) 
- Conformer Resnet: [Asset backtest](https://github.com/KJJHHH/Stock/blob/main/cv-based/models/result/Conformer-Resnet_class2_5871_backtest.png) 
- VisionTransformer: [Asset backtest](https://github.com/KJJHHH/Stock/blob/main/cv-based/models/result/Vision-Transformer_class2_5871_backtest.png)
- ViT Pretrained model: [Asset backtest](https://github.com/KJJHHH/Stock/blob/main/cv-based/model-pretrains/result/ViT_b_16_class2_5871_backtest.png) 
### 玉山金控
...




