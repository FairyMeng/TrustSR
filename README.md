# TrustSR-PyTorch

- PyTorch Implementation of the TrustSR model.

## Requirements

- PyTorch 2.0.1
- Python 3.8
- pandas 1.5.3
- numpy  1.24.3
- pillow 10.0.0

## Usage

### Dataset

Dataset can be retreived from [Amazon review data]([Amazon review data (ucsd.edu)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/))

### Pre processing data

- You need to run Toys_and_Games_dataset_generator.py and dataloader.py to obtain training data and testing data. 
- The training set itself is divided into training and testing where the testing split is the last day sessions.

### Training and Testing

In TrustSR

Training

```
python main.py
```

Testing

```
python main.py --is_eval --load_model checkpoint/CHECKPOINT#/model_EPOCH#.pt
```
