import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from medmnist import ChestMNIST

from run_classification import run_experiments
from utils import seed_it_all, my_transform


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments for medical image classification.")
    
    # Dataset and model parameters
    parser.add_argument("--dataset_name", type=str, default="ChestXray14", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name: swin_base|resnet18")
    parser.add_argument("--init", type=str, default="ImageNet", help="Initialization weights")
    parser.add_argument("--normalization", type=str, default="imagenet", help="Normalization type")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of output classes")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_trial", type=int, default=1, help="Number of trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu|cuda")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # Optimizer parameters
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'], help='Loss function type')
    # parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer type")
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer type")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    # Scheduler parameters
    parser.add_argument("--sched", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_lr", type=float, default=1e-6, help="Warmup learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs")
    parser.add_argument("--cooldown_epochs", type=int, default=10, help="Number of cooldown epochs")
    parser.add_argument("--decay_epochs", type=float, default=30, help="Epoch interval for learning rate decay")
    parser.add_argument("--decay_rate", type=float, default=0.5, help="Learning rate decay rate")

    # Output paths
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--exp_name", type=str, default="exper", help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_arguments()
    args.device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    seed_it_all(args.seed)

    if args.dataset_name == "ChestXray14":
        
        # Load dataset
        train_set = ChestMNIST(split="train", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="train"))
        val_set = ChestMNIST(split="val", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="val"))
        test_set = ChestMNIST(split="test", download=True, size=args.img_size, transform=my_transform(normalize=args.normalization, mode="test"))

        # Define class names
        class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
        ]

        # Print class distributions
        print_class_distribution(train_set, "train_set", class_names)
        print_class_distribution(val_set, "val_set", class_names)
        print_class_distribution(test_set, "test_set", class_names)
        
        # Use a subset of the dataset for quick experimentation
        # subset_size = 500
        # subset_size = 10000
        # train_set, _ = random_split(train_set, [subset_size, len(train_set) - subset_size])
        # val_set, _ = random_split(val_set, [subset_size, len(val_set) - subset_size])
        # test_set, _ = random_split(test_set, [subset_size, len(test_set) - subset_size])

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            # shuffle=True,
            sampler=get_sampler(train_set),  # Use sampler for class imbalance
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=1,  # Prefetch batches to reduce CPU wait time
            persistent_workers=True  # Avoid reloading workers per epoch
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        
        model_names = ["vgg19", "resnet18", "swin_base"]
        # model_names = ["vgg19"]
        # model_names = ["swin_base"]
        # model_names = ["resnet18"]

        for model in model_names:
            args.model_name = model
            # args.init = "ImageNet"
            args.weights = "ImageNet"
            # print(f"\nRunning model {args.model_name} with {args.init} weights and {args.loss_type} loss.\n")
            print(f"\nRunning model {args.model_name} with {args.weights} weights and {args.loss_type} loss.\n")


            # args.exp_name = f"{args.model_name}_{args.init}_{args.exp_name}"
            args.exp_name = f"{args.model_name}_{args.weights}_{args.exp_name}"
            model_path = Path("./models") / args.dataset_name / args.exp_name
            output_path = Path("./outputs") / args.dataset_name / args.exp_name
            model_path.mkdir(parents=True, exist_ok=True)
            output_path.mkdir(parents=True, exist_ok=True)

            run_experiments(args, train_loader, val_loader, test_loader, model_path, output_path)

    elif args.dataset_name == "JSRT":
        print("JSRT dataset is not implemented yet.")
    else:
        raise ValueError(f"Dataset {args.dataset_name} is not supported.")
    

def print_class_distribution(dataset, set_name, class_names):
    # Handle Subset objects
    if isinstance(dataset, torch.utils.data.Subset):
        labels = dataset.dataset.labels[dataset.indices]
    else:
        labels = dataset.labels  # Direct access if not a Subset

    class_counts = labels.sum(axis=0)  # Sum along axis 0 to count per class
    print(f"Class distribution in {set_name}:")
    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name}: {int(count)} samples")
    print(f"Total number of samples in {set_name}: {labels.shape[0]}\n")


def get_sampler(train_set):
    # If train_set is a Subset, access the original dataset and use the indices
    if isinstance(train_set, torch.utils.data.Subset):
        targets = train_set.dataset.labels[train_set.indices]
    else:
        targets = train_set.labels  # Direct access if it's not a Subset

    class_freq = targets.sum(axis=0)
    class_weights = 1.0 / (class_freq + 1e-6)
    sample_weights = (targets * class_weights).sum(axis=1)
    sample_weights = sample_weights / sample_weights.sum()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    

if __name__ == "__main__":
    main()
    