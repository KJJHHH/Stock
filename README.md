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
|-- configs/              # Model configs
|   |-- Transformer.json
|-- config/               # StockAI migrated YAML configs
|
|-- results/              # Outputs
|   |-- {model}-temp/     # Checkpoints
|   |-- {model}-result/   # Performance plots
|
|-- scrape/
|   |-- scrape.py         # Industry stock scraping helper
|-- sa_models/            # Migrated StockAI model family
|-- rl/                   # Migrated RL modules
|-- sentiment/            # Sentiment utilities
|-- stockai_main.py       # Migrated StockAI app entrypoint
|
|-- main.py               # Transformer pipeline entrypoint
|-- datas.py              # Transformer data pipeline
|-- models.py             # Transformer model definitions
|-- trainers.py           # Transformer training implementation
|-- backtestors.py        # Transformer backtesting implementation
|-- base/                 # Transformer base abstractions
|-- README.md
```
