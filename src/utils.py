import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, ConfusionMatrix, AUROC
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from models import build_model

# Configure matplotlib
mpl.rcParams["text.usetex"] = True


def seed_it_all(seed=1234):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_it_all()


def my_transform(normalize, crop_size=224, resize=224, mode="train", test_augment=False):
    transformations = []

    if normalize.lower() == "imagenet":
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
        normalize = T.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
        normalize = None
    else:
        raise ValueError(f"Mean and std for normalization '{normalize}' do not exist!")

    if mode == "train":
        transformations.extend([
            T.Grayscale(num_output_channels=3),
            T.Resize((resize, resize)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(7),
            T.ToTensor()
        ])
        if normalize:
            transformations.append(normalize)
    elif mode == "val":
        transformations.extend([
            T.Grayscale(num_output_channels=3),
            T.Resize((resize, resize)),
            T.CenterCrop(crop_size),
            T.ToTensor()
        ])
        if normalize:
            transformations.append(normalize)
    elif mode == "test":
        transformations.append(T.Grayscale(num_output_channels=3))
        if test_augment:
            transformations.extend([
                T.Resize((resize, resize)),
                T.TenCrop(crop_size),
                T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops]))
            ])
            if normalize:
                transformations.append(T.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        else:
            transformations.extend([
                T.Resize((resize, resize)),
                T.CenterCrop(crop_size),
                T.ToTensor()
            ])
            if normalize:
                transformations.append(normalize)

    return T.Compose(transformations)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(args, model, train_loader, loss_fn, optimizer, epoch=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = Accuracy(
        task="multilabel" if args.dataset_name != "JSRT" else "multiclass",
        num_labels=args.num_classes if args.dataset_name != "JSRT" else args.num_classes
    ).to(args.device)

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}" if epoch else "Training") as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(args.device, dtype=torch.float), targets.to(args.device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter(outputs, targets.int())

            tepoch.set_postfix(loss=loss_meter.avg, accuracy=100. * acc_meter.compute().item())

    return model, loss_meter.avg, acc_meter.compute().item()


def validation(args, model, val_loader, loss_fn):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = Accuracy(
        task="multilabel" if args.dataset_name != "JSRT" else "multiclass",
        num_labels=args.num_classes if args.dataset_name != "JSRT" else args.num_classes
    ).to(args.device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(args.device, dtype=torch.float), targets.to(args.device, dtype=torch.float)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter(outputs, targets.int())

    return loss_meter.avg, acc_meter.compute().item()


def plot_performance(args, loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist, epoch_counter):
    # Disable LaTeX
    plt.rcParams["text.usetex"] = False

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_title("Accuracy and Loss", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="black")
    ax1.plot(range(epoch_counter), loss_train_hist, lw=2, color="deepskyblue", label="Train Loss")
    ax1.plot(range(epoch_counter), loss_valid_hist, lw=2, color="yellow", label="Validation Loss")
    ax1.legend(loc="upper center")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", fontsize=14, color="green")
    ax2.plot(range(epoch_counter), acc_train_hist, lw=2, color="turquoise", label="Train Accuracy")
    ax2.plot(range(epoch_counter), acc_valid_hist, lw=2, color="red", label="Validation Accuracy")
    ax2.legend(loc="upper left")

    fig.savefig(args.plot_path, dpi=800)


def save_checkpoint(state, filename='model'):
    torch.save(state, f"{filename}.pth.tar")


def test_classification(args, checkpoint, data_loader_test):
    model = build_model(args)
    # print(model)

    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint_data["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    y_test, p_test = torch.FloatTensor(), torch.FloatTensor()

    with torch.no_grad():
        for samples, targets in tqdm(data_loader_test, desc="Testing"):
            targets = targets.to(args.device)
            y_test = torch.cat((y_test, targets.cpu()), dim=0)

            if samples.dim() == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif samples.dim() == 5:
                bs, n_crops, c, h, w = samples.size()

            inputs = samples.view(-1, c, h, w).to(args.device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1) if args.dataset_name == "JSRT" else torch.sigmoid(outputs)
            outputs_mean = outputs.view(bs, n_crops, -1).mean(dim=1)
            p_test = torch.cat((p_test, outputs_mean.cpu()), dim=0)

    return y_test, p_test


def metric_AUROC(target, output, nb_classes=14):
    target, output = target.cpu().numpy(), output.cpu().numpy()
    return [roc_auc_score(target[:, i], output[:, i]) for i in range(nb_classes)]
