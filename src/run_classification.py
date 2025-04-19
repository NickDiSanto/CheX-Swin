import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.functional.classification import multilabel_auroc
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import f1_score, precision_recall_curve, auc, multilabel_confusion_matrix

from models import build_model
from utils import (
    train_one_epoch,
    validation,
    save_checkpoint,
    plot_performance,
    test_classification,
)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
        # self

    def save_activations(self, module, input, output):
        self.activations = output.detach().clone()  # <== Clone to avoid view-related issues

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()  # <== Clone to prevent view/in-place error

    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Backward pass for the target class
        self.model.zero_grad()
        target = output[:, target_class]  # Select the score for the target class
        target = target.sum()  # Ensure the target is a scalar
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam[0]  # Remove batch dimension

        # Normalize CAM
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            print("Warning: CAM output is all zeros.")
            cam = np.zeros_like(cam)

        return cam

    def visualize(self, input_image, cam, alpha=0.5, save_path=None):
        # Resize CAM to match the input image size
        cam_resized = cv2.resize(cam, (input_image.shape[2], input_image.shape[1]))  # Resize to (width, height)

        # Convert CAM to heatmap
        heatmap = plt.cm.jet(cam_resized)[..., :3]  # Use jet colormap
        heatmap = np.uint8(255 * heatmap)

        # Overlay heatmap on the original image
        input_image = np.array(to_pil_image(input_image.cpu().squeeze()))
        overlay = np.uint8(alpha * heatmap + (1 - alpha) * input_image)

        # Save or display the result
        if save_path:
            plt.imsave(save_path, overlay)
        else:
            plt.figure(figsize=(8, 8))
            plt.imshow(overlay)
            plt.axis("off")
            plt.show()

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

def remove_inplace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            remove_inplace_relu(child)


def run_experiments(args, train_loader, val_loader, test_loader, model_path, output_path):

    accuracy = []
    mean_auc_all_runs = []

    for idx in range(args.num_trial):
        torch.cuda.empty_cache()
        model = build_model(args).to(args.device)
        remove_inplace_relu(model)
        if args.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
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

        class_labels = args.class_labels if hasattr(args, "class_labels") else [f"Class {i}" for i in range(args.num_classes)]

        log_and_print_metrics(
            logger2,
            experiment,
            "ACC_ClassWise",
            individual_acc,
            class_labels=class_labels,
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


        # Compute macro-average F1 score
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Compute macro-average PR AUC
        precision_macro, recall_macro, _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
        pr_auc_macro = auc(recall_macro, precision_macro)

        log_and_print_metrics(
            logger2,
            experiment,
            "F1_Macro",
            np.array([f1_macro]),
            precision=4,
            separator=","
        )

        log_and_print_metrics(
            logger2,
            experiment,
            "PR_AUC_Macro",
            np.array([pr_auc_macro]),
            precision=4,
            separator=","
        )


        # Compute multilabel confusion matrix (shape: [num_classes, 2, 2])
        conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

        # Log class-wise confusion matrices
        for i, cm in enumerate(conf_matrices):
            label_name = class_labels[i] if class_labels else f"Class_{i}"
            logger2.info(f"{experiment}: ConfusionMatrix_{label_name} =\n{cm}")
            print(f">>{experiment}: ConfusionMatrix_{label_name} =\n{cm}")

        # Sum across all class confusion matrices
        overall_cm = np.sum(conf_matrices, axis=0)

        logger2.info(f"{experiment}: ConfusionMatrix_Overall =\n{overall_cm}")
        print(f">>{experiment}: ConfusionMatrix_Overall =\n{overall_cm}")


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


    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Move data to the appropriate device
    images = images.to(args.device)
    labels = labels.to(args.device)

    # --- Grad-CAM target layer selector ---
    target_layer = None
    if hasattr(model, "features"):  # VGG
        # Avoid ReLU or pooling layers
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                target_layer = layer
                break

    elif hasattr(model, "layer4"):  # ResNet
        target_layer = list(model.layer4.children())[-1].conv2

    elif hasattr(model, "stages"):  # Swin Transformer (may not work for all variants)
        try:
            from timm.models.swin_transformer import SwinTransformerBlock
            for block in reversed(model.stages[-1]):
                if isinstance(block, SwinTransformerBlock):
                    target_layer = block.norm1  # You can also try block.attn
                    break
        except Exception:
            pass

    # Fallback for Swin models using different structure
    if target_layer is None and "swin" in args.model_name.lower():
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                target_layer = module  # Pick last one
        if target_layer is None:
            raise ValueError("Could not find suitable target_layer in Swin for Grad-CAM.")

    if target_layer is None:
        raise ValueError("Unsupported model architecture for Grad-CAM.")


    # Initialize Grad-CAM and generate the CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(images, target_class=0)  # Change target_class as needed
    # gradcam.visualize(images[0], cam)

    # Generate a unique file name for the Grad-CAM output
    model_name = args.model_name if hasattr(args, "model_name") else "model"
    experiment_name = args.exp_name if hasattr(args, "exp_name") else "experiment"
    run_index = idx + 1
    save_path = output_path / f"{model_name}_{experiment_name}_run_{run_index}_gradcam.png"

    # Save the Grad-CAM visualization
    gradcam.visualize(images[0], cam, save_path=str(save_path))

    

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


def log_and_print_metrics(logger, experiment, metric_name, values, class_labels=None, **kwargs):
    if class_labels and metric_name == "ACC_ClassWise":
        print(f">>{experiment}: {metric_name}")
        logger.info(f"{experiment}: {metric_name}")
        for label, value in zip(class_labels, values):
            print(f"  {label}: {value:.4f}")
            logger.info(f"  {label}: {value:.4f}")
    else:
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
