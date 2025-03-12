# Stock
Predict the daily percentile change for open and close price

## Usage
### Train
```
cd Stock
python task.py train --model Transformer --stock 2330.TW
```
### Backtest
```
cd Stock
python task.py test --model Transformer --stock 2330.TW  
```


## Result
### Transformer
- Stock 2882
![2882 performance](https://github.com/KJJHHH/Stock/blob/main/transformer_based/transformer-result/2882.TW.png)
- Stock 2884
![2884 performance](https://github.com/KJJHHH/Stock/blob/main/transformer_based/transformer-result/2884.TW.png)
### Decoder only
...

### Resnet
...

## Directories
project_root/
│── base_trainer/        # 🏋️ Base trainer module
│   ├── trainer/         # 🎯 Training logic for models
│
│── transformer_based/   # 🤖 Transformer-based models
│   ├── models.py        # 🏗️ Model definitions
│   ├── datas.py         # 📊 Data processing scripts
│   ├── trainer.py       # 🏋️ Training pipeline
│   ├── backtestor.py    # 📈 Backtesting implementation
│   ├── utils.py         # 🔧 Helper functions
│
│── cv_based/            # 🎥 Computer Vision-based models
│   ├── ...             # 📂 (Files for CV models go here)
│
│── README.md            # 📘 Project documentation
│── task.py              # 🚀 Main task execution script





