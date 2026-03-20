"""
Experiment: Is Copying More Calibrated if It's Specialized?

Compares calibration of copy decisions between:
1. Joint pointer-generator model (learns generation + copy decisions together)
2. Specialized copy-decision model (trained only for copy decisions)
3. Temperature-scaled joint model (post-hoc calibration baseline)

Uses a synthetic copy task with known ground-truth copy labels.
"""

import random
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data
    vocab_size: int = 500
    src_len: int = 40
    tgt_len: int = 20
    copy_ratio: float = 0.5  # ~50% of target tokens are copies
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 2000

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 30
    warmup_steps: int = 500
    seed: int = 42

    # Calibration
    n_bins: int = 15

    # Paths
    results_dir: str = "results"
    plots_dir: str = "results/plots"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Synthetic Copy Dataset
# ============================================================================

class SyntheticCopyDataset(Dataset):
    """
    Synthetic seq2seq task with known copy/generate ground truth.

    Input: sequence of tokens from vocab
    Output: sequence where each position either:
      - COPIES a token from a specific input position (copy_label=1)
      - GENERATES a new token from vocabulary (copy_label=0)

    Copy decision depends on:
      - Token "importance" (tokens 0-249 are "entity-like", more likely to be copied)
      - Positional pattern (tokens in first/last quarter of input more likely copied)
      - Random noise for realistic difficulty
    """

    def __init__(self, size, config, split='train'):
        self.size = size
        self.config = config
        self.data = self._generate(size, config, split)

    def _generate(self, size, cfg, split):
        data = []
        for i in range(size):
            # Generate source sequence
            src = np.random.randint(0, cfg.vocab_size, size=cfg.src_len)

            tgt = np.zeros(cfg.tgt_len, dtype=np.int64)
            copy_labels = np.zeros(cfg.tgt_len, dtype=np.float32)
            copy_positions = np.zeros(cfg.tgt_len, dtype=np.int64)

            for j in range(cfg.tgt_len):
                # Decide copy probability based on context
                # Map output position to rough input position
                aligned_pos = int(j * cfg.src_len / cfg.tgt_len)
                aligned_pos = min(aligned_pos, cfg.src_len - 1)

                src_token = src[aligned_pos]

                # Copy probability depends on multiple factors:
                # 1. Token type: "entity" tokens (0-249) are more copy-worthy
                token_factor = 0.7 if src_token < cfg.vocab_size // 2 else 0.3

                # 2. Position: edges of input have higher copy probability
                pos_factor = 0.6 if (aligned_pos < cfg.src_len // 4 or
                                      aligned_pos > 3 * cfg.src_len // 4) else 0.4

                # 3. Local repetition: if nearby tokens repeat, more likely to copy
                window = src[max(0, aligned_pos-2):min(cfg.src_len, aligned_pos+3)]
                repeat_factor = 0.6 if len(set(window)) < len(window) else 0.4

                # Combined probability
                p_copy = (token_factor + pos_factor + repeat_factor) / 3.0

                # Add noise
                p_copy = np.clip(p_copy + np.random.normal(0, 0.1), 0.05, 0.95)

                if np.random.random() < p_copy:
                    # Copy from input
                    copy_labels[j] = 1.0
                    # Copy from aligned position (with some variation)
                    offset = np.random.choice([-2, -1, 0, 1, 2],
                                              p=[0.05, 0.15, 0.6, 0.15, 0.05])
                    copy_pos = np.clip(aligned_pos + offset, 0, cfg.src_len - 1)
                    copy_positions[j] = copy_pos
                    tgt[j] = src[copy_pos]
                else:
                    # Generate from vocabulary
                    copy_labels[j] = 0.0
                    copy_positions[j] = 0  # ignored
                    tgt[j] = np.random.randint(0, cfg.vocab_size)

            data.append({
                'src': torch.tensor(src, dtype=torch.long),
                'tgt': torch.tensor(tgt, dtype=torch.long),
                'copy_labels': torch.tensor(copy_labels, dtype=torch.float32),
                'copy_positions': torch.tensor(copy_positions, dtype=torch.long),
            })
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Models
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class JointPointerGenerator(nn.Module):
    """
    Joint model: Transformer encoder-decoder with pointer-generator mechanism.
    Learns to generate tokens AND make copy decisions simultaneously.
    The p_gen value serves as the copy probability estimate.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Shared embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        # Generation head
        self.gen_head = nn.Linear(config.d_model, config.vocab_size)

        # Copy gate (p_gen): probability of generating vs copying
        # Follows See et al. (2017): p_gen from decoder state + context + input
        self.p_gen_linear = nn.Linear(config.d_model * 3, 1)

        # Pointer attention for copy location
        self.ptr_query = nn.Linear(config.d_model, config.d_model)
        self.ptr_key = nn.Linear(config.d_model, config.d_model)

    def forward(self, src, tgt):
        # Encode source
        src_emb = self.pos_enc(self.embedding(src))
        enc_out = self.encoder(src_emb)  # (B, S, D)

        # Decode target
        tgt_emb = self.pos_enc(self.embedding(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)  # (B, T, D)

        # Generation distribution
        gen_logits = self.gen_head(dec_out)  # (B, T, V)

        # Pointer attention
        ptr_q = self.ptr_query(dec_out)  # (B, T, D)
        ptr_k = self.ptr_key(enc_out)    # (B, S, D)
        ptr_scores = torch.bmm(ptr_q, ptr_k.transpose(1, 2)) / (self.config.d_model ** 0.5)  # (B, T, S)
        ptr_dist = F.softmax(ptr_scores, dim=-1)  # (B, T, S)

        # Context vector from pointer attention
        context = torch.bmm(ptr_dist, enc_out)  # (B, T, D)

        # p_gen: probability of generating (1 - p_copy)
        p_gen_input = torch.cat([dec_out, context, tgt_emb], dim=-1)  # (B, T, 3D)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)).squeeze(-1)  # (B, T)

        # p_copy = 1 - p_gen
        p_copy = 1.0 - p_gen

        return {
            'gen_logits': gen_logits,
            'p_copy': p_copy,
            'ptr_dist': ptr_dist,
            'enc_out': enc_out.detach(),  # for specialized model
            'dec_out': dec_out.detach(),
        }


class SpecializedCopyDecider(nn.Module):
    """
    Specialized model: Small Transformer that ONLY decides when to copy and from where.
    Takes encoder/decoder representations as input (no generation capability).
    Trained exclusively on copy decision loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Project encoder and decoder representations
        self.enc_proj = nn.Linear(config.d_model, config.d_model)
        self.dec_proj = nn.Linear(config.d_model, config.d_model)

        # Small Transformer layers for copy decision
        self.copy_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model, nhead=config.n_heads,
                dim_feedforward=config.d_ff, dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])

        # Copy decision head (binary: copy or generate)
        self.copy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1)
        )

        # Pointer head for copy location
        self.ptr_query = nn.Linear(config.d_model, config.d_model)
        self.ptr_key = nn.Linear(config.d_model, config.d_model)

    def forward(self, enc_out, dec_out):
        """
        Args:
            enc_out: encoder representations (B, S, D) - from joint model
            dec_out: decoder representations (B, T, D) - from joint model
        """
        enc_proj = self.enc_proj(enc_out)  # (B, S, D)
        dec_proj = self.dec_proj(dec_out)  # (B, T, D)

        # Process through copy-specialized Transformer layers
        h = dec_proj
        for layer in self.copy_layers:
            h = layer(h, enc_proj)  # (B, T, D)

        # Copy decision: sigmoid gives p(copy)
        copy_logits = self.copy_head(h).squeeze(-1)  # (B, T)
        p_copy = torch.sigmoid(copy_logits)

        # Pointer distribution for copy location
        ptr_q = self.ptr_query(h)       # (B, T, D)
        ptr_k = self.ptr_key(enc_proj)  # (B, S, D)
        ptr_scores = torch.bmm(ptr_q, ptr_k.transpose(1, 2)) / (self.config.d_model ** 0.5)
        ptr_dist = F.softmax(ptr_scores, dim=-1)  # (B, T, S)

        return {
            'p_copy': p_copy,
            'ptr_dist': ptr_dist,
            'copy_logits': copy_logits,
        }


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_data.append({'avg_conf': (lo + hi) / 2, 'avg_acc': 0, 'count': 0})
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]
        avg_confidence = bin_probs.mean()
        avg_accuracy = bin_labels.mean()
        bin_size = mask.sum()

        ece += (bin_size / len(probs)) * abs(avg_accuracy - avg_confidence)
        bin_data.append({
            'avg_conf': float(avg_confidence),
            'avg_acc': float(avg_accuracy),
            'count': int(bin_size)
        })

    return float(ece), bin_data


def compute_mce(probs, labels, n_bins=15):
    """Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))
    return float(mce)


def compute_brier_score(probs, labels):
    """Brier score (lower is better)."""
    return float(np.mean((probs - labels) ** 2))


def find_optimal_temperature(logits, labels, lr=0.01, max_iter=500):
    """Find optimal temperature for temperature scaling via gradient descent."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    def eval_fn():
        optimizer.zero_grad()
        scaled = torch.sigmoid(logits_t / temperature)
        loss = F.binary_cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return temperature.item()


# ============================================================================
# Training
# ============================================================================

def train_joint_model(model, train_loader, val_loader, config, device):
    """Train the joint pointer-generator model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            copy_labels = batch['copy_labels'].to(device)
            copy_positions = batch['copy_positions'].to(device)

            out = model(src, tgt)

            # Loss 1: Generation loss (cross-entropy on target tokens)
            gen_loss = F.cross_entropy(
                out['gen_logits'].reshape(-1, config.vocab_size),
                tgt.reshape(-1)
            )

            # Loss 2: Copy decision loss (binary cross-entropy)
            copy_loss = F.binary_cross_entropy(out['p_copy'], copy_labels)

            # Loss 3: Pointer loss (cross-entropy on copy positions, masked to copy tokens)
            copy_mask = copy_labels > 0.5
            if copy_mask.any():
                ptr_loss = F.cross_entropy(
                    out['ptr_dist'][copy_mask],
                    copy_positions[copy_mask]
                )
            else:
                ptr_loss = torch.tensor(0.0, device=device)

            loss = gen_loss + copy_loss + ptr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                copy_labels = batch['copy_labels'].to(device)
                copy_positions = batch['copy_positions'].to(device)

                out = model(src, tgt)
                copy_loss = F.binary_cross_entropy(out['p_copy'], copy_labels)
                val_loss += copy_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{config.results_dir}/models/joint_best.pt")

        if (epoch + 1) % 5 == 0:
            print(f"  Joint Epoch {epoch+1}/{config.epochs}: train_loss={avg_train_loss:.4f}, val_copy_loss={avg_val_loss:.4f}")

    model.load_state_dict(torch.load(f"{config.results_dir}/models/joint_best.pt", weights_only=True))
    return train_losses, val_losses


def extract_representations(model, data_loader, device):
    """Extract encoder/decoder representations from trained joint model."""
    all_enc = []
    all_dec = []
    all_copy_labels = []
    all_copy_positions = []
    all_p_copy_joint = []
    all_copy_logits = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            out = model(src, tgt)

            all_enc.append(out['enc_out'].cpu())
            all_dec.append(out['dec_out'].cpu())
            all_copy_labels.append(batch['copy_labels'])
            all_copy_positions.append(batch['copy_positions'])
            all_p_copy_joint.append(out['p_copy'].cpu())

            # Get logits for temperature scaling
            # Reconstruct logits from p_copy: logit = log(p/(1-p))
            p = out['p_copy'].cpu().clamp(1e-7, 1 - 1e-7)
            logits = torch.log(p / (1 - p))
            all_copy_logits.append(logits)

    return {
        'enc_out': torch.cat(all_enc),
        'dec_out': torch.cat(all_dec),
        'copy_labels': torch.cat(all_copy_labels),
        'copy_positions': torch.cat(all_copy_positions),
        'p_copy_joint': torch.cat(all_p_copy_joint),
        'copy_logits': torch.cat(all_copy_logits),
    }


class RepresentationDataset(Dataset):
    """Dataset of pre-extracted representations for training specialized model."""
    def __init__(self, reps):
        self.enc_out = reps['enc_out']
        self.dec_out = reps['dec_out']
        self.copy_labels = reps['copy_labels']
        self.copy_positions = reps['copy_positions']

    def __len__(self):
        return len(self.enc_out)

    def __getitem__(self, idx):
        return {
            'enc_out': self.enc_out[idx],
            'dec_out': self.dec_out[idx],
            'copy_labels': self.copy_labels[idx],
            'copy_positions': self.copy_positions[idx],
        }


def train_specialized_model(model, train_loader, val_loader, config, device):
    """Train the specialized copy-decision model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            enc_out = batch['enc_out'].to(device)
            dec_out = batch['dec_out'].to(device)
            copy_labels = batch['copy_labels'].to(device)
            copy_positions = batch['copy_positions'].to(device)

            out = model(enc_out, dec_out)

            # Loss 1: Copy decision (binary cross-entropy)
            copy_loss = F.binary_cross_entropy(out['p_copy'], copy_labels)

            # Loss 2: Pointer loss (for copy positions)
            copy_mask = copy_labels > 0.5
            if copy_mask.any():
                ptr_loss = F.cross_entropy(
                    out['ptr_dist'][copy_mask],
                    copy_positions[copy_mask]
                )
            else:
                ptr_loss = torch.tensor(0.0, device=device)

            loss = copy_loss + ptr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                enc_out = batch['enc_out'].to(device)
                dec_out = batch['dec_out'].to(device)
                copy_labels = batch['copy_labels'].to(device)

                out = model(enc_out, dec_out)
                copy_loss = F.binary_cross_entropy(out['p_copy'], copy_labels)
                val_loss += copy_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{config.results_dir}/models/specialized_best.pt")

        if (epoch + 1) % 5 == 0:
            print(f"  Spec. Epoch {epoch+1}/{config.epochs}: train_loss={avg_train_loss:.4f}, val_copy_loss={avg_val_loss:.4f}")

    model.load_state_dict(torch.load(f"{config.results_dir}/models/specialized_best.pt", weights_only=True))
    return train_losses, val_losses


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_calibration(model_name, probs, labels, n_bins=15):
    """Compute all calibration metrics for a model."""
    probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    ece, bin_data = compute_ece(probs_np, labels_np, n_bins)
    mce = compute_mce(probs_np, labels_np, n_bins)
    brier = compute_brier_score(probs_np, labels_np)

    # Copy decision accuracy (threshold at 0.5)
    preds = (probs_np > 0.5).astype(float)
    accuracy = (preds == labels_np).mean()

    # Precision, recall, F1 for copy class
    tp = ((preds == 1) & (labels_np == 1)).sum()
    fp = ((preds == 1) & (labels_np == 0)).sum()
    fn = ((preds == 0) & (labels_np == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'model': model_name,
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'bin_data': bin_data,
        'mean_prob': float(probs_np.mean()),
        'mean_label': float(labels_np.mean()),
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_reliability_diagrams(results_list, save_path):
    """Plot reliability diagrams for all models."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(5 * len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    for ax, result in zip(axes, results_list):
        confs = [b['avg_conf'] for b in result['bin_data'] if b['count'] > 0]
        accs = [b['avg_acc'] for b in result['bin_data'] if b['count'] > 0]
        counts = [b['count'] for b in result['bin_data'] if b['count'] > 0]

        # Normalize counts for bar width
        max_count = max(counts) if counts else 1

        ax.bar(confs, accs, width=1.0/15, alpha=0.6, color='steelblue',
               edgecolor='navy', label='Model')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f"{result['model']}\nECE={result['ece']:.4f}, Brier={result['brier']:.4f}",
                     fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved reliability diagrams to {save_path}")


def plot_combined_reliability(results_list, save_path):
    """Plot all models on one reliability diagram for direct comparison."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    for i, result in enumerate(results_list):
        confs = [b['avg_conf'] for b in result['bin_data'] if b['count'] > 0]
        accs = [b['avg_acc'] for b in result['bin_data'] if b['count'] > 0]

        ax.plot(confs, accs, 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=6,
                label=f"{result['model']} (ECE={result['ece']:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability', fontsize=13)
    ax.set_ylabel('Fraction of Positives', fontsize=13)
    ax.set_title('Reliability Diagram Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined reliability diagram to {save_path}")


def plot_training_curves(joint_train, joint_val, spec_train, spec_val, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(joint_train) + 1)
    ax1.plot(epochs, joint_train, 'b-', label='Train')
    ax1.plot(epochs, joint_val, 'b--', label='Val')
    ax1.set_title('Joint Model Training', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs = range(1, len(spec_train) + 1)
    ax2.plot(epochs, spec_train, 'r-', label='Train')
    ax2.plot(epochs, spec_val, 'r--', label='Val')
    ax2.set_title('Specialized Model Training', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(all_seed_results, save_path):
    """Plot bar chart comparing metrics across seeds."""
    models = list(all_seed_results[0].keys())
    metrics = ['ece', 'mce', 'brier', 'f1']
    metric_labels = ['ECE ↓', 'MCE ↓', 'Brier ↓', 'F1 ↑']

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    colors = ['steelblue', 'coral', 'seagreen']

    for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        x = np.arange(len(models))

        means = []
        stds = []
        for model in models:
            vals = [r[model][metric] for r in all_seed_results]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(x, means, yerr=stds, color=colors[:len(models)],
                      capsize=5, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved metric comparison to {save_path}")


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_seed(config, seed, device):
    """Run complete experiment for one seed."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}")

    # Generate data
    print("Generating synthetic data...")
    train_data = SyntheticCopyDataset(config.train_size, config, split='train')
    val_data = SyntheticCopyDataset(config.val_size, config, split='val')
    test_data = SyntheticCopyDataset(config.test_size, config, split='test')

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    # Print data stats
    all_labels = torch.cat([d['copy_labels'] for d in test_data.data])
    print(f"  Test set copy ratio: {all_labels.mean():.3f}")

    # ---- Train Joint Model ----
    print("\nTraining Joint Pointer-Generator...")
    joint_model = JointPointerGenerator(config).to(device)
    joint_train_losses, joint_val_losses = train_joint_model(
        joint_model, train_loader, val_loader, config, device
    )

    # ---- Extract Representations ----
    print("\nExtracting representations...")
    train_reps = extract_representations(joint_model, train_loader, device)
    val_reps = extract_representations(joint_model, val_loader, device)
    test_reps = extract_representations(joint_model, test_loader, device)

    rep_train_loader = DataLoader(
        RepresentationDataset(train_reps), batch_size=config.batch_size, shuffle=True
    )
    rep_val_loader = DataLoader(
        RepresentationDataset(val_reps), batch_size=config.batch_size
    )

    # ---- Train Specialized Model ----
    print("\nTraining Specialized Copy-Decision Model...")
    spec_model = SpecializedCopyDecider(config).to(device)
    spec_train_losses, spec_val_losses = train_specialized_model(
        spec_model, rep_train_loader, rep_val_loader, config, device
    )

    # ---- Evaluate on Test Set ----
    print("\nEvaluating on test set...")

    # Joint model predictions
    test_labels = test_reps['copy_labels'].reshape(-1).numpy()
    joint_probs = test_reps['p_copy_joint'].reshape(-1).numpy()
    joint_logits = test_reps['copy_logits'].reshape(-1).numpy()

    # Specialized model predictions
    spec_model.eval()
    all_spec_probs = []
    with torch.no_grad():
        for i in range(0, len(test_reps['enc_out']), config.batch_size):
            enc = test_reps['enc_out'][i:i+config.batch_size].to(device)
            dec = test_reps['dec_out'][i:i+config.batch_size].to(device)
            out = spec_model(enc, dec)
            all_spec_probs.append(out['p_copy'].cpu())
    spec_probs = torch.cat(all_spec_probs).reshape(-1).numpy()

    # Temperature scaling
    print("  Finding optimal temperature...")
    opt_temp = find_optimal_temperature(joint_logits, test_labels)
    print(f"  Optimal temperature: {opt_temp:.4f}")
    temp_scaled_probs = 1.0 / (1.0 + np.exp(-joint_logits / opt_temp))

    # Compute calibration metrics
    joint_results = evaluate_calibration("Joint", joint_probs, test_labels, config.n_bins)
    spec_results = evaluate_calibration("Specialized", spec_probs, test_labels, config.n_bins)
    temp_results = evaluate_calibration("Temp-Scaled", temp_scaled_probs, test_labels, config.n_bins)

    print(f"\n  {'Model':<15} {'ECE':>8} {'MCE':>8} {'Brier':>8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*55}")
    for r in [joint_results, spec_results, temp_results]:
        print(f"  {r['model']:<15} {r['ece']:>8.4f} {r['mce']:>8.4f} {r['brier']:>8.4f} "
              f"{r['accuracy']:>8.4f} {r['f1']:>8.4f}")

    return {
        'Joint': joint_results,
        'Specialized': spec_results,
        'Temp-Scaled': temp_results,
        'joint_train_losses': joint_train_losses,
        'joint_val_losses': joint_val_losses,
        'spec_train_losses': spec_train_losses,
        'spec_val_losses': spec_val_losses,
        'opt_temp': opt_temp,
    }


def run_size_ablation(config, device):
    """Ablation: vary model size and compare calibration."""
    print("\n" + "="*60)
    print("MODEL SIZE ABLATION")
    print("="*60)

    sizes = [
        (1, 64, "Tiny (1L, d=64)"),
        (2, 128, "Small (2L, d=128)"),
        (4, 256, "Medium (4L, d=256)"),
    ]

    results = {}
    set_seed(config.seed)

    # Generate data once
    train_data = SyntheticCopyDataset(config.train_size, config, split='train')
    val_data = SyntheticCopyDataset(config.val_size, config, split='val')
    test_data = SyntheticCopyDataset(config.test_size, config, split='test')

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    for n_layers, d_model, label in sizes:
        print(f"\n--- {label} ---")

        # Create config variant
        cfg = Config()
        cfg.n_layers = n_layers
        cfg.d_model = d_model
        cfg.d_ff = d_model * 2
        cfg.n_heads = max(2, d_model // 32)
        cfg.epochs = 20  # fewer epochs for ablation
        cfg.results_dir = config.results_dir

        set_seed(config.seed)

        # Train joint model
        joint_model = JointPointerGenerator(cfg).to(device)
        train_joint_model(joint_model, train_loader, val_loader, cfg, device)

        # Extract reps
        test_reps = extract_representations(joint_model, test_loader, device)

        # Train specialized model
        rep_train = extract_representations(joint_model, train_loader, device)
        rep_val = extract_representations(joint_model, val_loader, device)

        rep_train_loader = DataLoader(
            RepresentationDataset(rep_train), batch_size=cfg.batch_size, shuffle=True
        )
        rep_val_loader = DataLoader(
            RepresentationDataset(rep_val), batch_size=cfg.batch_size
        )

        spec_model = SpecializedCopyDecider(cfg).to(device)
        train_specialized_model(spec_model, rep_train_loader, rep_val_loader, cfg, device)

        # Evaluate
        test_labels = test_reps['copy_labels'].reshape(-1).numpy()
        joint_probs = test_reps['p_copy_joint'].reshape(-1).numpy()

        spec_model.eval()
        all_spec = []
        with torch.no_grad():
            for i in range(0, len(test_reps['enc_out']), cfg.batch_size):
                enc = test_reps['enc_out'][i:i+cfg.batch_size].to(device)
                dec = test_reps['dec_out'][i:i+cfg.batch_size].to(device)
                out = spec_model(enc, dec)
                all_spec.append(out['p_copy'].cpu())
        spec_probs = torch.cat(all_spec).reshape(-1).numpy()

        joint_r = evaluate_calibration(f"Joint-{label}", joint_probs, test_labels, cfg.n_bins)
        spec_r = evaluate_calibration(f"Spec-{label}", spec_probs, test_labels, cfg.n_bins)

        n_params_joint = sum(p.numel() for p in joint_model.parameters())
        n_params_spec = sum(p.numel() for p in spec_model.parameters())

        results[label] = {
            'joint': joint_r,
            'specialized': spec_r,
            'n_params_joint': n_params_joint,
            'n_params_spec': n_params_spec,
        }

        print(f"  Joint ECE={joint_r['ece']:.4f}, Spec ECE={spec_r['ece']:.4f}")
        print(f"  Joint params={n_params_joint:,}, Spec params={n_params_spec:,}")

    return results


def plot_ablation(ablation_results, save_path):
    """Plot model size ablation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = list(ablation_results.keys())
    joint_eces = [ablation_results[l]['joint']['ece'] for l in labels]
    spec_eces = [ablation_results[l]['specialized']['ece'] for l in labels]
    joint_params = [ablation_results[l]['n_params_joint'] for l in labels]
    spec_params = [ablation_results[l]['n_params_spec'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, joint_eces, width, label='Joint', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, spec_eces, width, label='Specialized', color='coral', alpha=0.8)
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('ECE ↓')
    ax1.set_title('ECE by Model Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x - width/2, [p/1000 for p in joint_params], width, label='Joint', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, [p/1000 for p in spec_params], width, label='Specialized', color='coral', alpha=0.8)
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Parameters (K)')
    ax2.set_title('Model Parameters')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    os.makedirs(f"{config.results_dir}/models", exist_ok=True)

    # ====================================================================
    # Experiment 1: Main comparison with multiple seeds
    # ====================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: MAIN COMPARISON (5 seeds)")
    print("="*60)

    seeds = [42, 123, 456, 789, 1024]
    all_seed_results = []

    for seed in seeds:
        result = run_single_seed(config, seed, device)
        all_seed_results.append({
            'Joint': result['Joint'],
            'Specialized': result['Specialized'],
            'Temp-Scaled': result['Temp-Scaled'],
        })

    # Aggregate results across seeds
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (5 seeds)")
    print("="*60)

    models = ['Joint', 'Specialized', 'Temp-Scaled']
    metrics = ['ece', 'mce', 'brier', 'accuracy', 'f1']

    aggregated = {}
    print(f"\n{'Model':<15} {'ECE':>12} {'MCE':>12} {'Brier':>12} {'Acc':>12} {'F1':>12}")
    print(f"{'-'*75}")

    for model in models:
        agg = {}
        for metric in metrics:
            vals = [r[model][metric] for r in all_seed_results]
            agg[metric] = {'mean': np.mean(vals), 'std': np.std(vals)}
        aggregated[model] = agg

        print(f"{model:<15} "
              f"{agg['ece']['mean']:.4f}±{agg['ece']['std']:.4f} "
              f"{agg['mce']['mean']:.4f}±{agg['mce']['std']:.4f} "
              f"{agg['brier']['mean']:.4f}±{agg['brier']['std']:.4f} "
              f"{agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f} "
              f"{agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")

    # Statistical tests
    print("\n--- Statistical Tests ---")
    joint_eces = [r['Joint']['ece'] for r in all_seed_results]
    spec_eces = [r['Specialized']['ece'] for r in all_seed_results]
    temp_eces = [r['Temp-Scaled']['ece'] for r in all_seed_results]

    t_stat, p_val = stats.ttest_rel(joint_eces, spec_eces)
    print(f"Joint vs Specialized ECE: t={t_stat:.4f}, p={p_val:.6f}")

    t_stat2, p_val2 = stats.ttest_rel(temp_eces, spec_eces)
    print(f"Temp-Scaled vs Specialized ECE: t={t_stat2:.4f}, p={p_val2:.6f}")

    # Effect size (Cohen's d)
    diff = np.array(joint_eces) - np.array(spec_eces)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else float('inf')
    print(f"Cohen's d (Joint vs Specialized): {cohens_d:.4f}")

    # Generate plots from last seed's results
    last_result = all_seed_results[-1]
    plot_reliability_diagrams(
        [last_result['Joint'], last_result['Specialized'], last_result['Temp-Scaled']],
        f"{config.plots_dir}/reliability_diagrams.png"
    )
    plot_combined_reliability(
        [last_result['Joint'], last_result['Specialized'], last_result['Temp-Scaled']],
        f"{config.plots_dir}/combined_reliability.png"
    )
    plot_metric_comparison(all_seed_results, f"{config.plots_dir}/metric_comparison.png")

    # ====================================================================
    # Experiment 2: Model Size Ablation
    # ====================================================================
    ablation_results = run_size_ablation(config, device)
    plot_ablation(ablation_results, f"{config.plots_dir}/size_ablation.png")

    # ====================================================================
    # Save all results
    # ====================================================================

    # Prepare serializable results
    save_results = {
        'config': {
            'vocab_size': config.vocab_size,
            'src_len': config.src_len,
            'tgt_len': config.tgt_len,
            'copy_ratio': config.copy_ratio,
            'train_size': config.train_size,
            'val_size': config.val_size,
            'test_size': config.test_size,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'epochs': config.epochs,
            'n_bins': config.n_bins,
        },
        'seeds': seeds,
        'aggregated': {
            model: {metric: {'mean': agg[metric]['mean'], 'std': agg[metric]['std']}
                    for metric in metrics}
            for model, agg in aggregated.items()
        },
        'per_seed': [
            {model: {k: v for k, v in r[model].items() if k != 'bin_data'}
             for model in models}
            for r in all_seed_results
        ],
        'statistical_tests': {
            'joint_vs_specialized': {'t_stat': float(t_stat), 'p_value': float(p_val)},
            'temp_vs_specialized': {'t_stat': float(t_stat2), 'p_value': float(p_val2)},
            'cohens_d_joint_vs_spec': float(cohens_d),
        },
        'ablation': {
            label: {
                'joint_ece': r['joint']['ece'],
                'spec_ece': r['specialized']['ece'],
                'joint_f1': r['joint']['f1'],
                'spec_f1': r['specialized']['f1'],
                'n_params_joint': r['n_params_joint'],
                'n_params_spec': r['n_params_spec'],
            }
            for label, r in ablation_results.items()
        },
    }

    with open(f"{config.results_dir}/all_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nAll results saved to {config.results_dir}/all_results.json")
    print(f"Plots saved to {config.plots_dir}/")
    print("\nExperiment complete!")

    return save_results


if __name__ == '__main__':
    results = main()
