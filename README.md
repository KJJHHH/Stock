# Stock
Predict the daily percentile change for open and close price

## Usage
### Train
```
cd Stock
python task.py train --model Transformer --stock 2884.TW
```
### Backtest
```
cd Stock
python task.py test --model Transformer --stock 2884.TW  
```

## Result
### Transformer
- Stock 2454
![2454 performance](https://github.com/KJJHHH/Stock/blob/main/transformer_based/Transformer-result/2454.TW.png)
- Stock 2884
![2884 performance](https://github.com/KJJHHH/Stock/blob/main/transformer_based/Transformer-result/2884.TW.png)
### Decoder only
...

### Resnet
...

