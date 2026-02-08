#!/usr/bin/env python3
"""Phase 2: Train ICR Probe on extracted ICR scores.

Loads ICR scores from Phase 1 output, pools per-token scores into
per-sample feature vectors [L], and trains the ICRProbe MLP classifier
to predict hallucination.

Output:
  - probe/model.pth: trained probe weights
  - probe/config.json: training config
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.icr_probe import ICRProbeTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ICR Probe")
    parser.add_argument("--data_dir", type=str, default="saves/halueval")
    parser.add_argument("--save_dir", type=str, default="saves/halueval/probe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Optional: separate directory with test split ICR scores")
    return parser.parse_args()


def load_icr_data(data_dir: str):
    """Load ICR scores and labels from Phase 1 output."""
    data_dir = Path(data_dir)

    logger.info(f"Loading ICR scores from {data_dir / 'icr_score.pt'}")
    icr_scores = torch.load(data_dir / "icr_score.pt", weights_only=False)

    logger.info(f"Loading labels from {data_dir / 'output_judge.jsonl'}")
    labels = {}
    with open(data_dir / "output_judge.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                labels[entry["id"]] = entry["result_type"]

    return icr_scores, labels


def pool_icr_scores(icr_scores, labels):
    """Pool per-token ICR scores into per-sample feature vectors.

    Each sample has ICR scores [L_attn][N_answer_tokens].
    Pool by taking the mean across answer tokens → [L_attn] vector.

    Returns:
        X: np.ndarray [N_samples, L_attn]
        y: np.ndarray [N_samples]
    """
    X_list = []
    y_list = []

    for sample_id in sorted(icr_scores.keys()):
        if sample_id not in labels:
            continue

        scores = icr_scores[sample_id]  # list[L][N_tokens]
        num_layers = len(scores)

        # Mean pool across tokens for each layer
        features = []
        valid = True
        for layer_scores in scores:
            if len(layer_scores) == 0:
                valid = False
                break
            layer_mean = np.mean(layer_scores)
            if not np.isfinite(layer_mean):
                valid = False
                break
            features.append(layer_mean)

        if not valid:
            logger.warning(f"Sample {sample_id}: invalid ICR scores, skipping")
            continue

        X_list.append(features)
        y_list.append(labels[sample_id])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Pooled data: {X.shape[0]} samples, {X.shape[1]} features (layers)")
    logger.info(f"Label distribution: {(y == 1).sum()} correct, {(y == 0).sum()} hallucinated")

    return X, y


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size, dataset_weight=True):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    if dataset_weight:
        # Compute class weights for balanced sampling
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / class_counts
        sample_weights = weights[y_train.astype(int)]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    icr_scores, labels = load_icr_data(args.data_dir)
    X, y = pool_icr_scores(icr_scores, labels)

    # Split or use pre-split data
    if args.test_data_dir:
        logger.info(f"Loading test split from {args.test_data_dir}")
        test_icr_scores, test_labels = load_icr_data(args.test_data_dir)
        X_train, y_train = X, y
        X_val, y_val = pool_icr_scores(test_icr_scores, test_labels)
    else:
        logger.info("Splitting data 80/20")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, args.batch_size
    )

    # Setup config
    config = Config(
        input_dim=X_train.shape[1],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=str(save_dir),
        data_dir=args.data_dir,
    )

    # Train
    trainer = ICRProbeTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.setup_model()

    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("=== Final Evaluation ===")
    final_metrics = trainer._validate_epoch()
    for name, value in final_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
