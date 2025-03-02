# Computer Vision Methods to Predict Stock 
## Models
- [x] ResNet
- [x] Conformer
- [x] Conformer + ResNet (ConRes)
- [x] VisionTransformer
- [ ] Pretrained VisionTransformer
## Experiments
Asset using computer vision model comparing to buy-and-hold strategy
### 中租，5871
- Conformer: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_CV/Models/Model_Result/Conformer-CNN_class2_5871_backtest.png) 
- Resnet: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_CV/Models/Model_Result/ResNet_class2_5871_backtest.png) 
- Conformer Resnet: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_CV/Models/Model_Result/Conformer-Resnet_class2_5871_backtest.png) 
- VisionTransformer: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Individual_CV/Models/Model_Result/Vision-Transformer_class2_5871_backtest.png) 
## Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change
3. Predict with last 100 dates data
4. Expand the variables shape (5, 100) to (5, 100, 100) by difference between each dates
5. Predict the next dates’ Open change and Close change



