# Stock
Predict the daily percentile change for open and close price.

## Usage

### Environment
```bash
# Python 3.12
pip install -r requirements.txt
```

### Train and backtest
- `task`: `train` or `test`
- `--model`: currently supports `Transformer`
- `--stock_target`: target stock symbol (required)
- `--stock_pool`: optional pretrain stock symbols
- training arguments are defined in `configs/Transformer.json`

#### Train
```bash
cd Stock

# Basic train
python3 main.py train --model Transformer --stock_target 2330.TW

# Train with multiple stocks
python3 main.py train --model Transformer --stock_target 2330.TW --stock_pool 2454.TW

# Train 2884.TW with top 9 bank holdings
python3 main.py train --model Transformer --stock_target 2884.TW --stock_pool 2881.TW 2882.TW 2891.TW 2885.TW 2883.TW 2890.TW 2887.TW 2888.TW
```

#### Backtest
```bash
cd Stock
python3 main.py test --model Transformer --stock_target 2884.TW --stock_pool 2881.TW 2882.TW 2891.TW 2885.TW 2883.TW 2890.TW 2887.TW 2888.TW
```

## Result
### Transformer
- Stock 2882
<img src="./results/Transformer-result/2882.TW.png" alt="Alt Text" width="600" height="400">

- Stock 2884
<img src="./results/Transformer-result/2884.TW.png" alt="Alt Text" width="600" height="400">

### Transformer: single stock vs multiple stock training
- Stock 2884 with top 9 bank holding companies
<img src="./results/Transformer-result/2884.TW%3A%20%5B'2884.TW'%2C%20'2881.TW'%2C%20'2882.TW'%2C%20'2891.TW'%2C%20'2885.TW'%2C%20'2883.TW'%2C%20'2890.TW'%2C%20'2887.TW'%2C%20'2888.TW'%5D.png" alt="Stock Result" width="600" height="400">

## Project Structure
```text
project_root/
|-- base/                 # Core components
|   |-- base_data.py      # Data base class
|   |-- base_trainer.py   # Training base class
|   |-- base_testor.py    # Backtesting base class
|
|-- configs/              # Model configs
|   |-- Transformer.json
|
|-- results/              # Outputs
|   |-- {model}-temp/     # Checkpoints
|   |-- {model}-result/   # Performance plots
|
|-- scrape/
|   |-- scrape.py         # Industry stock scraping helper
|
|-- models.py             # Model definitions
|-- datas.py              # Data preparation
|-- trainers.py           # Training implementations
|-- backtestors.py        # Backtesting implementations
|-- main.py               # Entrypoint
|-- README.md
```
