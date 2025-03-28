import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from medmnist import INFO, ChestMNIST

from run_classification import run_experiments
from utils import seed_it_all, my_transform


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments for medical image classification.")
    
    # Dataset and model parameters
    parser.add_argument("--dataset_name", type=str, default="ChestXray14", help="Dataset name: ChestXray14|JSRT|ChestMNIST")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name: swin_base|swin_tiny|resnet18|resnet50")
    parser.add_argument("--init", type=str, default="ImageNet", help="Initialization: ImageNet|Random")
    parser.add_argument("--normalization", type=str, default="imagenet", help="Normalization type: imagenet|chestx-ray")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of output classes")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--dataset_path", type=str, default="/cabinet/reza/datasets/NIH_Chest_X_rays/images/", help="Path to the dataset")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_trial", type=int, default=5, help="Number of trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu|cuda")
    parser.add_argument("--valid_start_epoch", type=int, default=79, help="Epoch to start validation")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Optimizer parameters
    parser.add_argument("--opt", type=str, default="sgd", help="Optimizer type")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=None, help="Gradient clipping norm")

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


def print_class_distribution(dataset, set_name, class_names):
    labels = dataset.labels  # Shape: (N, num_classes)
    class_counts = labels.sum(axis=0)  # Sum along axis 0 to count per class
    print(f"Class distribution in {set_name}:")
    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name}: {int(count)} samples")
    print(f"Total number of samples in {set_name}: {labels.shape[0]}\n")


def main():
    args = parse_arguments()

    # Set device
    args.device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    seed_it_all(args.seed)
    
    # if args.dataset_name == "ChestMNIST":

    #     seed_it_all(args.seed)

    #     data_flag = args.dataset_name.lower()
    #     info = INFO[data_flag]
    #     task = info["task"]
    #     n_channels = info['n_channels']
    #     n_classes = len(info['label'])
    #     samples = info["n_samples"]
    #     DataClass = getattr(medmnist, info["python_class"])
    
    #     size = 224
    #     data_transform = T.Compose([
    #         T.ToTensor(),
    #         T.Normalize(mean=[.5], std=[.5]),
    #         T.Resize((size, size))
    #     ])
    #     download = False
    #     train_set = DataClass(split="train", transform=data_transform ,download=download)
    #     val_set = DataClass(split="val", transform=data_transform, download=download)
    #     test_set = DataClass(split="test", transform=data_transform, download=download)
        
    #     _, mini_train_set = random_split(val_set, (0.9, 0.1))
        
    #     train_loader = DataLoader(dataset=mini_train_set, batch_size=16, shuffle=True)
        
    #     model_names = ["resnet18", "resnet50"]
    #     init_weights = ["ImageNet", "Random"]
        
    #     for model in model_names:
    #         for init in init_weights:
    #             args.model_name = model
    #             args.init = init
    #             print(f"\n\nRunning model {args.model_name} with {args.init} weights.\n")
    #             args.exp_name = "exper"
    #             args.exp_name = args.model_name + "_" + args.init + "_" + args.exp_name
    #             model_path = Path("./models").joinpath(args.dataset_name, args.exp_name)
    #             output_path = Path("./outputs").joinpath(args.dataset_name, args.exp_name)
    #             model_path.mkdir(parents=True, exist_ok=True)
    #             output_path.mkdir(parents=True, exist_ok=True)
                
    #             run_experiments(args, train_loader, train_loader, train_loader, model_path, output_path)
        
    if args.dataset_name == "ChestXray14":
        
        seed_it_all(args.seed)
        # train_set = ChestXray14(split="train", download=True, size=args.img_size,
        #                         transform=my_transform(normalize=args.normalization, mode="train"))
        # val_set = ChestXray14(split="val", download=True, size=args.img_size,
        #                       transform=my_transform(normalize=args.normalization, mode="val"))
        # test_set = ChestXray14(split="test", download=True, size=args.img_size,
        #                        transform=my_transform(normalize=args.normalization, mode="test"))

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
        # subset_size = 100
        # train_set, _ = random_split(train_set, [subset_size, len(train_set) - subset_size])
        # val_set, _ = random_split(val_set, [subset_size, len(val_set) - subset_size])
        # test_set, _ = random_split(test_set, [subset_size, len(test_set) - subset_size])

        # Optimized DataLoader
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=128,  # Try reducing to 64 if CPU remains overloaded
            shuffle=True,
            # num_workers=min(8, torch.get_num_threads() - 2),  # Increase for better CPU utilization
            num_workers=6,  # Increase for better CPU utilization
            pin_memory=True,  # Enables faster GPU transfers
            prefetch_factor=1,  # Prefetch batches to reduce CPU wait time
            persistent_workers=True  # Avoid reloading workers per epoch
        )

        val_loader = DataLoader(
            val_set, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True,
            prefetch_factor=1, persistent_workers=True
        )

        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True,
            prefetch_factor=1, persistent_workers=True
        )

        # train_loader = DataLoader(dataset=train_set, batch_size=24, shuffle=True)
        # val_loader = DataLoader(dataset=val_set, batch_size=24, shuffle=False)
        # test_loader = DataLoader(dataset=test_set, batch_size=24, shuffle=False)
        
        # model_names = ["swin_tiny", "resnet18", "resnet50", "swin_base"]
        # init_weights = ["ImageNet", "Random"]
        model_names = ["resnet50", "swin_base"]
        # model_names = ["swin_base"]
        init_weights = ["ImageNet"]
        
        for model in model_names:
            for init in init_weights:
                args.model_name = model
                args.init = init
                print(f"\nRunning model {args.model_name} with {args.init} weights.\n")

                args.exp_name = f"{args.model_name}_{args.init}_{args.exp_name}"
                model_path = Path("./models") / args.dataset_name / args.exp_name
                output_path = Path("./outputs") / args.dataset_name / args.exp_name
                model_path.mkdir(parents=True, exist_ok=True)
                output_path.mkdir(parents=True, exist_ok=True)

                run_experiments(args, train_loader, val_loader, test_loader, model_path, output_path)

    elif args.dataset_name == "JSRT":
        print("JSRT dataset is not implemented yet.")
    else:
        raise ValueError(f"Dataset {args.dataset_name} is not supported.")


if __name__ == "__main__":
    main()
    