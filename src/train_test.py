import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import matplotlib as mpl
import matplotlib.pyplot as plt

from dataloader import ChestXray14
from models import build_model
from utils import (
    seed_it_all, train_one_epoch, validation, save_checkpoint, my_transform,
    plot_performance, test_classification, metric_AUROC
)
from sklearn.metrics import accuracy_score

# Configure matplotlib
mpl.rcParams["text.usetex"] = True

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ChestMNIST", help="ChestXray14|JSRT|ChestMNIST")
    parser.add_argument("--model_name", type=str, default="resnet18", help="swin_base|swin_tiny|resnet18|resnet50")
    parser.add_argument("--isinit", type=bool, default=True, help="False for Random| True for ImageNet")
    parser.add_argument("--normalization", type=str, default="imagenet", help="how to normalize data (imagenet|chestx-ray)")
    parser.add_argument("--num_classes", type=int, default=14, help="number of labels")
    parser.add_argument("--output_dir", type=str, help="output dir")
    parser.add_argument("--max_epochs", type=int, default=100, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size per GPU")
    parser.add_argument("--base_lr", type=float, default=0.001, help="classification network learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--exp_name", type=str, default="", help="experiment name")
    parser.add_argument("--num_trial", type=int, default=10, help="number of trials")
    parser.add_argument("--device", type=str, default="cuda", help="cpu|cuda")
    parser.add_argument("--train_list", type=str, default=None, help="file for training list")
    parser.add_argument("--val_list", type=str, default=None, help="file for validation list")
    parser.add_argument("--test_list", type=str, default=None, help="file for test list")
    parser.add_argument("--in_chans", type=int, default=1, help="input data channel numbers")
    parser.add_argument("--dataset_path", type=str, default="./images", help="dataset path")
    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()
    args.init = "ImageNet" if args.isinit else "Random"
    args.exp_name = f"{args.model_name}_{args.init}_{args.exp_name}"

    model_path = Path("./models") / args.dataset_name / args.exp_name
    output_path = Path("./outputs") / args.dataset_name / args.exp_name
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    seed_it_all(args.seed)

    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and DataLoader
    train_set = ChestXray14(split="train", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="train"))
    val_set = ChestXray14(split="val", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="val"))
    test_set = ChestXray14(split="test", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="test"))

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    # Visualize random samples
    data, _ = next(iter(train_loader))
    img_grid = make_grid(data, nrow=8, normalize=True).permute(1, 2, 0)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_grid)
    plt.axis('off')
    plt.savefig("RandomSamples.pdf", dpi=800)

    # Model setup
    torch.cuda.empty_cache()
    model = build_model(args).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    accuracy = []
    mean_auc = []

    # Training and evaluation
    for idx in range(args.num_trial):
        print(f"Run: {idx + 1}")
        experiment = f"{args.exp_name}_run_{idx}"
        save_model_path = model_path / experiment
        args.plot_path = model_path / f"{experiment}.pdf"

        log_file = model_path / f"run_{idx}.log"
        logging.basicConfig(
            filename=log_file, level=logging.INFO, filemode='w',
            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
        )
        logging.info(str(args))

        loss_train_hist, loss_valid_hist = [], []
        acc_train_hist, acc_valid_hist = [], []
        best_loss_valid = float('inf')

        for epoch in range(args.max_epochs):
            model, loss_train, acc_train = train_one_epoch(args, model, train_loader, loss_fn, optimizer)
            logging.info(f"Epoch:{epoch + 1}, TrainLoss:{loss_train:.4f}, TrainAcc:{acc_train:.4f}")

            print("Start validation...")
            loss_valid, acc_valid = validation(args, model, val_loader, loss_fn)
            logging.info(f"Epoch:{epoch + 1}, ValidLoss:{loss_valid:.4f}, ValidAcc:{acc_valid:.4f}")

            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)
            acc_train_hist.append(acc_train)
            acc_valid_hist.append(acc_valid)

            if loss_valid < best_loss_valid:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lossMIN': best_loss_valid,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=str(save_model_path))
                best_loss_valid = loss_valid
                print('Model Saved!')

        plot_performance(args, loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist, args.max_epochs)

        print("Start testing...")
        saved_model = model_path / f"{experiment}.pth.tar"
        y_test, p_test = test_classification(args, str(saved_model), test_loader)

        if args.dataset_name == "RSNAPneumonia":
            acc = accuracy_score(np.argmax(y_test.cpu().numpy(), axis=1), np.argmax(p_test.cpu().numpy(), axis=1))
            print(f">>{experiment}: ACCURACY = {acc}")
            logging.info(f"{experiment}: ACCURACY = {acc:.4f}")
            accuracy.append(acc)

        individual_results = metric_AUROC(y_test, p_test, args.num_classes)
        print(f">>{experiment}: AUC = {np.array2string(np.array(individual_results), precision=4, separator=',')}")
        logging.info(f"{experiment}: AUC = {np.array2string(np.array(individual_results), precision=4, separator=',')}")

        mean_over_all_classes = np.mean(individual_results)
        print(f">>{experiment}: Mean AUC = {mean_over_all_classes:.4f}")
        logging.info(f"{experiment}: Mean AUC = {mean_over_all_classes:.4f}")

        mean_auc.append(mean_over_all_classes)

    # Final results
    mean_auc = np.array(mean_auc)
    print(f">> All trials: mAUC = {np.array2string(mean_auc, precision=4, separator=',')}")
    logging.info(f"All trials: mAUC = {np.array2string(mean_auc, precision=4, separator=',')}")
    print(f">> Mean AUC over All trials: {np.mean(mean_auc):.4f}")
    logging.info(f"Mean AUC over All trials: {np.mean(mean_auc):.4f}")
    print(f">> STD over All trials: {np.std(mean_auc):.4f}")
    logging.info(f"STD over All trials: {np.std(mean_auc):.4f}")

if __name__ == "__main__":
    main()
    