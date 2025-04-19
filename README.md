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
--epochs 40 \
--batch_size 24 \
--num_trial 1
--lr 1e-3
```

---

## Results

After training, results such as accuracy, loss, and model checkpoints will be saved in the output directory. You can customize the output path in the scripts.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
