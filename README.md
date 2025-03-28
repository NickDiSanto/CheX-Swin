# CheX-Swin: Chest X-ray Classification Using Swin Transformer

CheX-Swin is a deep learning framework for chest X-ray image classification, leveraging the power of Swin Transformers and other state-of-the-art architectures. This repository provides scripts for training, testing, and experimenting with various models and datasets.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NickDiSanto/CheX-Swin.git
   cd CheX-Swin
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Experiments

To run experiments with the specified dataset and hyperparameters, use the following command:

```bash
python experiments.py \
--dataset_name ChestXray14 \
--batch_size 24 \
--epochs 100 \
--num_trial 1
```

### Training and Testing

To train and test a model, use the `train_test.py` script:

```bash
python train_test.py \
--dataset_name ChestXray14 \
--model_name resnet18 \
--normalization imagenet \
--batch_size 24 \
--lr 0.01 \
--epochs 100 \
--num_trial 10 \
--in_chans 3 \
--train_list ./Xray14_train_official.txt \
--val_list ./Xray14_val_official.txt \
--test_list ./Xray14_test_official.txt
```

---

## Results

After training, results such as accuracy, loss, and model checkpoints will be saved in the output directory. You can customize the output path in the scripts.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
