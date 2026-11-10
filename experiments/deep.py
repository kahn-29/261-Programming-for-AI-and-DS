"""Shared PyTorch building blocks for the deep-learning notebooks (08 onward)."""

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.data import CATEGORICAL, NUMERIC

# Lower-level administrative names repeat across provinces ("Phường 1"),
# so each level is qualified by its parents before building vocabularies.
HIERARCHY = {
    "district_name": ["province_name"],
    "ward_name": ["province_name", "district_name"],
    "street_name": ["province_name", "district_name"],
    "project_name": ["province_name"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """CUDA when available (Colab), otherwise CPU. Apple MPS is skipped on purpose:
    for a model this small it is not faster, and its kernels are not deterministic."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_dim(levels):
    """fastai rule of thumb: grows sublinearly with cardinality, capped at 32."""
    return int(min(32, round(1.6 * levels ** 0.56)))


class TabularEncoder:
    """Train-only preprocessing: numeric -> log1p, clip, impute, flag, scale;
    categorical -> qualified key -> integer index (0 = rare/unseen)."""

    UNKNOWN = 0

    def __init__(self, min_frequency=5, clip_quantiles=(0.005, 0.995)):
        self.min_frequency = min_frequency
        self.clip_quantiles = clip_quantiles

    @staticmethod
    def _keys(frame, column):
        key = frame[column].astype(str)
        for parent in HIERARCHY.get(column, []):
            key = frame[parent].astype(str) + "|" + key
        return key

    def fit(self, frame):
        numeric = np.log1p(frame[NUMERIC].astype(float))
        low, high = self.clip_quantiles
        self.lower_ = numeric.quantile(low)
        self.upper_ = numeric.quantile(high)
        clipped = numeric.clip(self.lower_, self.upper_, axis=1)
        self.median_ = clipped.median()
        filled = clipped.fillna(self.median_)
        self.mean_ = filled.mean()
        self.std_ = filled.std().replace(0, 1.0)
        self.vocab_ = {}
        for column in CATEGORICAL:
            counts = self._keys(frame, column).value_counts()
            kept = sorted(counts.index[counts >= self.min_frequency])
            self.vocab_[column] = {value: i + 1 for i, value in enumerate(kept)}
        return self

    def transform(self, frame):
        numeric = np.log1p(frame[NUMERIC].astype(float))
        missing = numeric.isna().astype(np.float32).add_suffix("_missing")
        clipped = numeric.clip(self.lower_, self.upper_, axis=1).fillna(self.median_)
        scaled = (clipped - self.mean_) / self.std_
        dense = pd.concat([scaled, missing], axis=1).to_numpy(np.float32)
        codes = np.stack([
            self._keys(frame, column).map(self.vocab_[column]).fillna(self.UNKNOWN).to_numpy(np.int64)
            for column in CATEGORICAL], axis=1)
        return dense, codes

    @property
    def dense_names(self):
        return NUMERIC + [f"{column}_missing" for column in NUMERIC]

    @property
    def cardinalities(self):
        return [len(self.vocab_[column]) + 1 for column in CATEGORICAL]


class TabularDataset(Dataset):
    """Tensors are built once; __getitem__ only indexes them."""

    def __init__(self, dense, codes, target=None):
        self.dense = torch.as_tensor(dense, dtype=torch.float32)
        self.codes = torch.as_tensor(codes, dtype=torch.long)
        self.target = None if target is None else torch.as_tensor(
            np.asarray(target), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.dense)

    def __getitem__(self, index):
        items = (self.dense[index], self.codes[index])
        return items if self.target is None else (*items, self.target[index])


class ResidualBlock(nn.Module):
    """Pre-activation block h + F(h); F starts at zero so the block is identity."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.branch = nn.Sequential(
            nn.BatchNorm1d(dim), nn.SiLU(), nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, x):
        return x + self.branch(x)


class EmbeddingMLP(nn.Module):
    """Entity embeddings for categorical columns + residual MLP on log price."""

    def __init__(self, n_dense, cardinalities, hidden=256, blocks=2, dropout=0.2,
                 embedding_dropout=0.1, target_mean=0.0):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(levels, embedding_dim(levels)) for levels in cardinalities)
        width = n_dense + sum(e.embedding_dim for e in self.embeddings)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.input = nn.Sequential(nn.Linear(width, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.constant_(self.head[-1].bias, target_mean)

    def features(self, dense, codes):
        embedded = [embedding(codes[:, i]) for i, embedding in enumerate(self.embeddings)]
        x = torch.cat([dense, self.embedding_dropout(torch.cat(embedded, dim=1))], dim=1)
        return self.blocks(self.input(x))

    def forward(self, dense, codes):
        return self.head(self.features(dense, codes))


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, generator=generator)


@torch.no_grad()
def predict_log(model, loader, device):
    model.eval()
    outputs = []
    for batch in loader:
        dense, codes = batch[0].to(device), batch[1].to(device)
        outputs.append(model(dense, codes).float().cpu())
    return torch.cat(outputs).numpy().ravel()


def fit(model, train_loader, validation_loader, validation_log_price, device, *,
        loss="huber", huber_delta=1.0, lr=2e-3, weight_decay=1e-4, epochs=100,
        patience=12, scheduler_patience=4, max_grad_norm=1.0, log_every=5):
    """AdamW + ReduceLROnPlateau + clipping + early stopping on validation log-MAE.

    The best weights are kept in memory and restored before returning.
    """
    criterion = nn.HuberLoss(delta=huber_delta) if loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=scheduler_patience, min_lr=1e-5)
    best, best_state, best_epoch, waited, history = np.inf, None, 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for dense, codes, target in train_loader:
            dense, codes, target = dense.to(device), codes.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = criterion(model(dense, codes), target)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += batch_loss.item() * len(target)
            seen += len(target)
        prediction = predict_log(model, validation_loader, device)
        validation_mae = float(np.mean(np.abs(prediction - validation_log_price)))
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_mae)
        improved = validation_mae < best - 1e-4
        if improved:
            best, best_epoch, waited = validation_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "validation_log_mae": validation_mae, "lr": lr_now, "best": improved})
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"epoch {epoch:3d} train {total / seen:.4f} val log-MAE {validation_mae:.4f} lr {lr_now:.1e}")
        if waited >= patience:
            break
    model.load_state_dict(best_state)
    print(f"best epoch {best_epoch}, val log-MAE {best:.4f}, {time.perf_counter() - start:.0f}s")
    return pd.DataFrame(history).set_index("epoch"), best_epoch
