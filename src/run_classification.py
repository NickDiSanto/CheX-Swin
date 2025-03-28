import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.functional.classification import multilabel_auroc
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

from models import build_model
from utils import (
    train_one_epoch,
    validation,
    save_checkpoint,
    plot_performance,
    test_classification,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def run_experiments(args, train_loader, val_loader, test_loader, model_path, output_path):

    accuracy = []
    mean_auc_all_runs = []

    for idx in range(args.num_trial):
        torch.cuda.empty_cache()
        model = build_model(args).to(args.device)
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        if args.loss_type == "focal":
            loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        
        print(f"Run: {idx + 1}\nStarting training...")
        experiment = f"{args.exp_name}_run_{idx + 1}"
        save_model_path = model_path / experiment
        args.plot_path = output_path / f"{experiment}.pdf"

        # Setup training logger
        log_file_train = output_path / f"run_{idx + 1}.log"
        logger1 = setup_logger("training_logger", log_file_train)
        logger1.info(str(args))

        # Initialize metrics
        loss_train_hist, loss_valid_hist = [], []
        acc_train_hist, acc_valid_hist = [], []
        best_loss_valid = float("inf")
        epoch_counter = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(epoch_counter, args.epochs):
            print(f"Epoch: {epoch + 1} out of {args.epochs}")
            model, loss_train, acc_train = train_one_epoch(
                args, model, train_loader, loss_fn, optimizer
            )
            logger1.info(
                f"Run:{idx + 1}-Epoch:{epoch + 1}, TrainLoss:{loss_train:.4f}, TrainAcc:{acc_train:.4f}"
            )

            loss_train_hist.append(loss_train)
            acc_train_hist.append(acc_train)

            print("Starting validation...")
            loss_valid, acc_valid = validation(args, model, val_loader, loss_fn)
            logger1.info(
                f"Run:{idx + 1}-Epoch:{epoch + 1}, ValidLoss:{loss_valid:.4f}, ValidAcc:{acc_valid:.4f}"
            )

            loss_valid_hist.append(loss_valid)
            acc_valid_hist.append(acc_valid)

            # Save the best model
            if loss_valid < best_loss_valid:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "lossMIN": best_loss_valid,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },
                    filename=str(save_model_path),
                )
                best_loss_valid = loss_valid
                print("Model Saved!")
                patience_counter = 0
            else:
                print(f"Epoch {epoch + 1}: val_loss did not improve from {best_loss_valid}")
                patience_counter += 1

            epoch_counter += 1
            lr_scheduler.step(epoch)

            # Early stopping
            if patience_counter > args.patience:
                print("Early Stopping")
                close_logger(logger1)
                break

        close_logger(logger1)

        # Plot performance
        plot_performance(
            args,
            loss_train_hist,
            loss_valid_hist,
            acc_train_hist,
            acc_valid_hist,
            epoch_counter,
        )

        print("Starting testing...")
        log_file_results = output_path / f"results_run_{idx + 1}.log"
        logger2 = setup_logger("results_logger", log_file_results)
        
        saved_model = model_path / f"{experiment}.pth.tar"
        y_test, p_test = test_classification(args, str(saved_model), test_loader)

        # Compute metrics
        mla_acc_indv = MultilabelAccuracy(num_labels=args.num_classes, average=None).to(args.device)
        mla_acc = MultilabelAccuracy(num_labels=args.num_classes).to(args.device)

        y_test_int = y_test.int().to(args.device)
        p_test = p_test.to(args.device)

        individual_acc = mla_acc_indv(p_test, y_test_int).cpu().numpy()
        acc = mla_acc(p_test, y_test_int).cpu().numpy()

        log_and_print_metrics(
            logger2,
            experiment,
            "ACC_ClassWise",
            individual_acc,
            precision=4,
            separator="\t",
        )
        log_and_print_metrics(
            logger2,
            experiment,
            "ACC_All_Classes",
            acc,
            precision=4,
            separator="\t",
        )

        auroc_mean = multilabel_auroc(
            p_test, y_test_int, num_labels=args.num_classes, average="macro"
        ).cpu().numpy()
        auroc_individual = multilabel_auroc(
            p_test, y_test_int, num_labels=args.num_classes, average=None
        ).cpu().numpy()

        log_and_print_metrics(
            logger2,
            experiment,
            "AUC_ClassWise",
            auroc_individual,
            precision=4,
            separator=",",
        )
        logger2.info(f"{experiment}: AUC_All_Classes = {auroc_mean:.4f}")

        # F1 Score and PR AUC
        y_true = y_test.cpu().numpy()
        y_prob = p_test.cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)

        f1_per_class = []
        pr_auc_per_class = []

        for i in range(args.num_classes):
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1_per_class.append(f1)
            try:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                pr_auc = np.trapz(recall, precision)
            except ValueError:
                pr_auc = 0.0

            pr_auc_per_class.append(pr_auc)

        log_and_print_metrics(
            logger2,
            experiment,
            "F1_ClassWise",
            np.array(f1_per_class),
            precision=4,
            separator=","
        )

        log_and_print_metrics(
            logger2,
            experiment,
            "PR_AUC_ClassWise",
            np.array(pr_auc_per_class),
            precision=4,
            separator=","
        )

        accuracy.append(acc.tolist())
        mean_auc_all_runs.append(auroc_mean.tolist())

    # Final metrics
    log_final_metrics(logger2, "ACC", accuracy)
    log_final_metrics(logger2, "AUC", mean_auc_all_runs)

    close_logger(logger2)


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler(str(log_file), mode="a")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def log_and_print_metrics(logger, experiment, metric_name, values, **kwargs):
    formatted_values = np.array2string(values, **kwargs)
    print(f">>{experiment}: {metric_name} = {formatted_values}")
    logger.info(f"{experiment}: {metric_name} = {formatted_values}")


def log_final_metrics(logger, metric_name, values):
    values = np.array(values)
    print(f">> All trials on all classes: {metric_name} = {np.array2string(values, precision=4, separator=',')}")
    logger.info(f"All trials on all classes: {metric_name} = {np.array2string(values, precision=4, separator=',')}")
    print(f">> Mean {metric_name} over All trials: = {np.mean(values):.4f}")
    logger.info(f"Mean {metric_name} over All trials = {np.mean(values):.4f}")
    print(f">> {metric_name}_STD over All trials:  = {np.std(values):.4f}")
    logger.info(f"{metric_name}_STD over All trials:  = {np.std(values):.4f}")
