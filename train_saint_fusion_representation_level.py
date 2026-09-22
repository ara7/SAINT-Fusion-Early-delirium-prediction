"""
SAINT + Bioformer representation-level late fusion training.

This script:
1. Loads tabular clinical data and text tokens.
2. Uses a frozen, fine-tuned Bioformer to extract 384-dim CLS embeddings.
3. Uses SAINTWithCrossAttention to encode the tabular data and project the
   text representation into the SAINT embedding space.
4. Performs representation-level fusion by element-wise addition:
       fused_representation = saint_cls + text_representation
5. Applies the SAINT prediction head to the fused representation.

The code is organized for reproducibility and GitHub readability while
preserving the core training setup of the original experiment.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from augmentations import embed_data_mask
from data_openml_saint_fusion import (
    DataSetCatCon,
    data_prep_openml,
)
from models_saint_fusion import SAINTWithCrossAttention
from utils import count_parameters, get_scheduler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAINT + Bioformer representation-level late fusion."
    )

    # Dataset
    parser.add_argument("--dset_id", required=True, type=int)
    parser.add_argument(
        "--task",
        default="binary",
        type=str,
        choices=["binary"],
        help="This experiment is implemented for binary classification.",
    )
    parser.add_argument("--vision_dset", action="store_true")
    parser.add_argument("--dset_seed", default=5, type=int)

    # SAINT
    parser.add_argument("--cont_embeddings", default="MLP", type=str)
    parser.add_argument("--embedding_size", default=32, type=int)
    parser.add_argument("--transformer_depth", default=2, type=int)
    parser.add_argument("--attention_heads", default=8, type=int)
    parser.add_argument("--attention_dropout", default=0.2, type=float)
    parser.add_argument("--ff_dropout", default=0.2, type=float)
    parser.add_argument(
        "--attentiontype",
        default="colrow",
        type=str,
        choices=["col", "colrow", "row", "justmlp", "attn", "attnmlp"],
    )
    parser.add_argument(
        "--final_mlp_style",
        default="sep",
        type=str,
        choices=["common", "sep"],
    )

    # Optimization
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        type=str,
        choices=["AdamW", "Adam", "SGD"],
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        choices=["cosine", "linear"],
    )
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)

    # Reproducibility / output
    parser.add_argument("--set_seed", default=1, type=int)
    parser.add_argument("--savemodelroot", default="./rerun_models", type=str)
    parser.add_argument("--run_name", default="saint_bioformer_fusion", type=str)
    parser.add_argument("--active_log", action="store_true")

    # Bioformer
    parser.add_argument(
        "--bioformer_path",
        default=(
            "/.../Delirium/"
            "Delirium-Workspace/ara-lena/new_algorithm/"
            "paper_3_again/saint/models_huggingface/bioformer-16L"
        ),
        type=str,
    )
    parser.add_argument(
        "--bioformer_checkpoint",
        default="Bioformer/bioformer_new_work/model_cam_icd_label9.pth", #fine-tuned-Bioformer
        type=str,
    )

    # Evaluation
    parser.add_argument("--eval_every", default=1, type=int)

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility at the possible cost of speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FrozenBioformer(nn.Module):
    """Frozen fine-tuned Bioformer used only for text representation extraction."""

    def __init__(self, model_path, checkpoint_path, device):
        super().__init__()

        self.device = device

        # The base architecture is used by the saved fine-tuned model.
        # Loading the checkpoint preserves the original experiment setup.
        self.model = torch.load(
            checkpoint_path,
            map_location=device,
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(device)

        print("Bioformer parameters frozen:", self.is_frozen())

    def is_frozen(self):
        return all(
            not parameter.requires_grad
            for parameter in self.model.parameters()
        )

    @torch.no_grad()
    def get_embeddings(self, encoded_text):
        """
        Extract the CLS representation from the frozen Bioformer.

        Returns
        -------
        torch.Tensor
            Shape: [batch_size, 384]
        """
        input_ids = encoded_text["input_ids"].squeeze(1).to(self.device)
        attention_mask = encoded_text["attention_mask"].squeeze(1).to(
            self.device
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Fine-tuned Bioformer representation used in the experiment.
        return outputs.last_hidden_state[:, 0, :]

def build_model(cat_dims, con_idxs, args, y_dim):
    """Create the SAINT model used for representation-level fusion."""

    model = SAINTWithCrossAttention(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=args.embedding_size,
        dim_out=1,
        depth=args.transformer_depth,
        heads=min(4, args.attention_heads),
        attn_dropout=args.attention_dropout,
        ff_dropout=args.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=args.cont_embeddings,
        attentiontype=args.attentiontype,
        final_mlp_style=args.final_mlp_style,
        y_dim=y_dim,
    )

    return model


def build_optimizer(model, args):
    """Create optimizer and scheduler."""

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = get_scheduler(args, optimizer)

    return optimizer, scheduler

def forward_fusion(
    model,
    bioformer,
    batch,
    device,
    vision_dset=False,
):
    """
    Run one forward pass through the tabular/text representation-level fusion.

    The fusion is performed after SAINT produces the tabular CLS
    representation and after the text representation has been projected
    into the same 32-dimensional space by SAINTWithCrossAttention.
    """

    (
        x_categ,
        x_cont,
        y_true,
        cat_mask,
        con_mask,
        text_tokens,
    ) = batch

    x_categ = x_categ.to(device)
    x_cont = x_cont.to(device)
    y_true = y_true.to(device)
    cat_mask = cat_mask.to(device)
    con_mask = con_mask.to(device)

    # Frozen Bioformer representation: [B, 384].
    text_embedding = bioformer.get_embeddings(text_tokens)

    # SAINT input embeddings.
    _, x_categ_enc, x_cont_enc = embed_data_mask(
        x_categ,
        x_cont,
        cat_mask,
        con_mask,
        model,
        vision_dset,
    )

    # SAINTWithCrossAttention returns:
    #   reps          -> tabular transformer representations
    #   text_embed_32 -> text representation projected to SAINT dimension
    reps, text_embed_32 = model.transformer(
        x_categ_enc,
        x_cont_enc,
        text_embedding,
    )

    # Tabular CLS representation: [B, 32].
    saint_cls = reps[:, 0, :]

    # Representation-level fusion used in the experiment.
    fused_representation = saint_cls + text_embed_32.to(saint_cls.device)

    # Final prediction head.
    logits = model.mlpfory(fused_representation)

    return logits, y_true

def train_one_epoch(
    model,
    bioformer,
    loader,
    optimizer,
    criterion,
    device,
    vision_dset,
):
    """Train SAINT fusion model for one epoch."""

    model.train()
    bioformer.eval()

    running_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        logits, y_true = forward_fusion(
            model=model,
            bioformer=bioformer,
            batch=batch,
            device=device,
            vision_dset=vision_dset,
        )

        loss = criterion(logits, y_true.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    #validation work
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    bioformer,
    loader,
    criterion,
    device,
    vision_dset,
):
    """
    Evaluate the binary classifier and return predictions/probabilities.
    """

    model.eval()
    bioformer.eval()

    running_loss = 0.0
    labels = []
    probabilities = []
    predictions = []

    for batch in loader:
        logits, y_true = forward_fusion(
            model=model,
            bioformer=bioformer,
            batch=batch,
            device=device,
            vision_dset=vision_dset,
        )

        loss = criterion(logits, y_true.squeeze())
        running_loss += loss.item()

        # conversion to class probabilities.
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        labels.extend(y_true.squeeze().cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

    labels = np.asarray(labels).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions).reshape(-1).astype(int)

    class_1_probability = probabilities[:, 1]

    metrics = {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, class_1_probability),
        "average_precision": average_precision_score(
            labels,
            class_1_probability,
        ),
        "brier": brier_score_loss(
            labels,
            class_1_probability,
        ),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    return metrics

def save_evaluation_plots(metrics, output_dir, epoch):
    """Save precision-recall and calibration curves."""

    labels = metrics["labels"]
    class_1_probability = metrics["probabilities"][:, 1]

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(
        labels,
        class_1_probability,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve "
        f"(AP={metrics['average_precision']:.3f})"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"precision_recall_epoch_{epoch}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        labels,
        class_1_probability,
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Model",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title(
        f"Calibration Curve "
        f"(Brier={metrics['brier']:.3f})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"calibration_epoch_{epoch}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_loss_curve(train_losses, test_losses, output_dir):
    """Save training and test loss curves."""

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
        linewidth=2,
    )
    plt.plot(
        epochs,
        test_losses,
        label="Test Loss",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Test Loss — SAINT/Bioformer Late Fusion")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    path = os.path.join(output_dir, "train_test_loss_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def print_evaluation(metrics, epoch):
    """Print evaluation results."""

    print(f"\nEpoch {epoch}")
    print(f"Test loss:       {metrics['loss']:.4f}")
    print(f"Test accuracy:   {metrics['accuracy']:.4f}")
    print(f"Test AUROC:      {metrics['auroc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Brier score:     {metrics['brier']:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            metrics["labels"],
            metrics["predictions"],
        )
    )

def main():
    args = parse_args()
    set_seed(args.set_seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output_dir = os.path.join(
        os.getcwd(),
        args.savemodelroot,
        args.run_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    print("Downloading and processing the dataset...")

    (
        cat_dims,
        cat_idxs,
        con_idxs,
        X_train,
        y_train,
        X_test,
        y_test,
        train_mean,
        train_std,
        df_train,
        df_test,
    ) = data_prep_openml(
        args.dset_id,
        args.dset_seed,
        args.task,
        datasplit=[0.65, 0.15, 0.20],
    )

    continuous_mean_std = np.asarray(
        [train_mean, train_std],
        dtype=np.float32,
    )

    _, n_features = X_train["data"].shape

    if n_features > 100:
        args.batchsize = min(64, args.batchsize)

    print(f"Number of tabular features: {n_features}")
    print(f"Batch size: {args.batchsize}")
    print(args)

    # -------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------

    train_ds = DataSetCatCon(
        X_train,
        y_train,
        cat_idxs,
        "clf",
        continuous_mean_std,
    )

    test_ds = DataSetCatCon(
        X_test,
        y_test,
        cat_idxs,
        "clf",
        continuous_mean_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
    )
    #Validation loaders

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
    )

    # CLS token category dimension required by SAINT.
    cat_dims = np.append(
        np.asarray([1]),
        np.asarray(cat_dims),
    ).astype(int)

    y_dim = len(np.unique(y_train["data"][:, 0]))

    if y_dim != 2:
        raise ValueError(
            f"Expected binary classification with 2 classes; "
            f"found y_dim={y_dim}."
        )

    model = build_model(
        cat_dims=cat_dims,
        con_idxs=con_idxs,
        args=args,
        y_dim=y_dim,
    ).to(device)

    print(f"SAINT parameters: {count_parameters(model):,}")

    # Class weighting
    class_weights = torch.tensor(
        [1.0, 1.5],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    ).to(device)

    optimizer, scheduler = build_optimizer(model, args)

    bioformer = FrozenBioformer(
        model_path=args.bioformer_path,
        checkpoint_path=args.bioformer_checkpoint,
        device=device,
    )

    best_test_auroc = -np.inf
    best_test_accuracy = 0.0

    train_losses = []
    test_losses = []

    print("\nTraining begins.")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            bioformer=bioformer,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            vision_dset=args.vision_dset,
        )

        train_losses.append(train_loss)

        # Scheduler behavior follows the original experiment.
        scheduler.step()

        print(
            f"\nEpoch [{epoch}/{args.epochs}] "
            f"- Train Loss: {train_loss:.4f}"
        )

        if epoch % args.eval_every == 0:

            metrics = evaluate(
                model=model,
                bioformer=bioformer,
                loader=test_loader,
                criterion=criterion,
                device=device,
                vision_dset=args.vision_dset,
            )

            test_losses.append(metrics["loss"])

            print_evaluation(metrics, epoch)

            '''# Save the model when test AUCROC improves.
            #
            # NOTE: This is for experiment purpose only
            if metrics["auroc"] > best_test_auroc:
                best_test_auroc = metrics["auroc"]
                best_test_accuracy = metrics["accuracy"]

                checkpoint_path = os.path.join(
                    output_dir,
                    "best_model.pth",
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "auroc": best_test_auroc,
                        "accuracy": best_test_accuracy,
                    },
                    checkpoint_path,
                )

                print(
                    f">>> New best test AUROC: "
                    f"{best_test_auroc:.4f}"
                )
                print(f"Saved: {checkpoint_path}")

                save_evaluation_plots(
                    metrics,
                    output_dir,
                    epoch,
                )
                '''
                # -------------------------------------------------------------
                # Validation
                # -------------------------------------------------------------

                valid_metrics = evaluate(
                    model=model,
                    bioformer=bioformer,
                    loader=valid_loader,
                    criterion=criterion,
                    device=device,
                    vision_dset=args.vision_dset,
                )

                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"- Validation Loss: {valid_metrics['loss']:.4f} "
                    f"- Validation AUROC: {valid_metrics['auroc']:.4f} "
                    f"- Validation Accuracy: {valid_metrics['accuracy']:.4f}"
                )

                # Select the best model using VALIDATION AUROC.
                if valid_metrics["auroc"] > best_valid_auroc:

                    best_valid_auroc = valid_metrics["auroc"]
                    best_valid_accuracy = valid_metrics["accuracy"]

                    checkpoint_path = os.path.join(
                        output_dir,
                        "best_model.pth",
                    )

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "valid_auroc": best_valid_auroc,
                            "valid_accuracy": best_valid_accuracy,
                        },
                        checkpoint_path,
                    )

                    print(
                        f">>> New best validation AUROC: "
                        f"{best_valid_auroc:.4f}"
                    )
                    print(f"Saved: {checkpoint_path}")


    if len(test_losses) == len(train_losses):
        loss_path = save_loss_curve(
            train_losses,
            test_losses,
            output_dir,
        )
        print(f"\nLoss curve saved to: {loss_path}")

    total_parameters = count_parameters(model)

    print("\nTraining complete.")
    print(f"Total trainable/model parameters: {total_parameters:,}")
    print(f"Best test AUROC: {best_test_auroc:.4f}")
    print(f"Accuracy at best test AUROC: {best_test_accuracy:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
