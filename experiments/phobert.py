"""Fine-tuning PhoBERT for log-price regression (notebook 09; reused by fusion)."""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.data import PROCESSED

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256          # PhoBERT has 258 positions: 256 tokens + 2 reserved


def load_text(parts):
    """Attach the segmented accented text to each partition, aligned by row_id."""
    text = pd.read_parquet(PROCESSED / "text_vi.parquet", columns=["row_id", "text_vi_seg"])
    lookup = text.set_index("row_id").text_vi_seg
    out = {}
    for name, frame in parts.items():
        values = frame.row_id.map(lookup)
        if values.isna().any():
            raise ValueError(f"{name}: {int(values.isna().sum())} rows lack text_vi")
        out[name] = values.tolist()
    return out


def tokenize(tokenizer, texts, max_length=MAX_LENGTH):
    """Tokenize once, without padding; batches are padded dynamically."""
    encoded = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)
    return encoded["input_ids"]


class TextDataset(Dataset):
    def __init__(self, input_ids, target=None):
        self.input_ids = input_ids
        self.target = None if target is None else np.asarray(target, dtype=np.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        item = {"input_ids": self.input_ids[index]}
        if self.target is not None:
            item["target"] = self.target[index]
        return item


def make_collate(pad_id):
    """Pad each batch to its own longest sequence (dynamic padding)."""
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        for row, item in enumerate(batch):
            ids[row, :len(item["input_ids"])] = torch.as_tensor(item["input_ids"])
        out = {"input_ids": ids, "attention_mask": (ids != pad_id).long()}
        for key in batch[0]:  # extra fixed-size inputs, e.g. tabular features for fusion
            if key not in ("input_ids", "target"):
                out[key] = torch.as_tensor(np.stack([item[key] for item in batch]))
        if "target" in batch[0]:
            out["target"] = torch.tensor([item["target"] for item in batch]).reshape(-1, 1)
        return out
    return collate


def make_loader(dataset, batch_size, shuffle, pad_id, seed=0):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      generator=generator, collate_fn=make_collate(pad_id))


def mean_pool(hidden, mask):
    """Average token states, ignoring padding."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class PhoBERTRegressor(nn.Module):
    """PhoBERT encoder -> mean pooling -> small MLP head -> log1p(price)."""

    def __init__(self, encoder, head_hidden=256, dropout=0.1, target_mean=0.0):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, head_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.constant_(self.head[-1].bias, target_mean)

    def features(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, attention_mask)

    def forward(self, input_ids, attention_mask):
        return self.head(self.features(input_ids, attention_mask))


def layerwise_groups(model, base_lr, head_lr, decay, weight_decay):
    """Layer-wise learning-rate decay: top layer gets base_lr, each lower layer x decay,
    embeddings lowest; the new head gets head_lr. Bias/LayerNorm: no weight decay."""
    layers = model.encoder.config.num_hidden_layers
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or ".pooler." in f".{name}":
            continue
        if name.startswith("head."):
            depth, lr = "head", head_lr
        elif name.startswith("encoder.embeddings."):
            depth, lr = "embeddings", base_lr * decay ** layers
        elif name.startswith("encoder.encoder.layer."):
            index = int(name.split(".")[3])
            depth, lr = f"layer{index}", base_lr * decay ** (layers - 1 - index)
        else:
            depth, lr = "other", base_lr
        no_decay = name.endswith("bias") or "LayerNorm" in name
        key = (depth, no_decay)
        groups.setdefault(key, {"params": [], "lr": lr, "is_head": depth == "head",
                                "weight_decay": 0.0 if no_decay else weight_decay})
        groups[key]["params"].append(parameter)
    return list(groups.values())


def warmup_linear(optimizer, warmup_steps, total_steps, encoder_start=0):
    """Per-group schedule: linear warm-up to the group's lr, then linear decay to zero.
    Encoder groups stay at zero until ``encoder_start`` (end of the frozen head warm-up)
    and only then begin their own warm-up, so unfreezing never meets a full lr."""
    def make(start):
        def factor(step):
            if step < start:
                return 0.0
            if step < start + warmup_steps:
                return (step - start + 1) / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - start - warmup_steps))
        return factor
    starts = [0 if group.get("is_head") else encoder_start for group in optimizer.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [make(start) for start in starts])


def set_encoder_trainable(model, trainable):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


@torch.no_grad()
def predict_log(model, loader, device, amp=True):
    model.eval()
    outputs = []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            output = model(**{k: v.to(device) for k, v in batch.items() if k != "target"})
        outputs.append(output.float().cpu())
    return torch.cat(outputs).numpy().ravel()


def fit(model, train_loader, validation_loader, validation_log_price, device, *,
        base_lr=3e-5, head_lr=1e-3, layer_decay=0.9, weight_decay=0.01, epochs=4,
        warmup_ratio=0.06, freeze_steps=300, huber_delta=1.0, accumulation=1,
        max_grad_norm=1.0, patience=2, evaluations_per_epoch=2, amp=True,
        max_steps=None, log_every=100):
    """Fine-tune with LLRD, warm-up + linear decay, a frozen-encoder head warm-up,
    fp16 autocast, gradient accumulation and clipping, and early stopping on
    validation log-MAE (checked several times per epoch). Best weights restored."""
    use_amp = amp and device.type == "cuda"
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(layerwise_groups(model, base_lr, head_lr, layer_decay, weight_decay))
    steps_per_epoch = len(train_loader) // accumulation
    total_steps = steps_per_epoch * epochs if max_steps is None else max_steps
    scheduler = warmup_linear(optimizer, int(warmup_ratio * total_steps), total_steps,
                              encoder_start=freeze_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    check_every = max(1, steps_per_epoch // evaluations_per_epoch)
    set_encoder_trainable(model, freeze_steps == 0)

    best, best_state, best_step, waited, history = np.inf, None, 0, 0, []
    step, running, seen, start = 0, 0.0, 0, time.perf_counter()
    done = False
    for epoch in range(1, epochs + 1):
        model.train()
        for index, batch in enumerate(train_loader):
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = inputs.pop("target")
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**inputs)
            loss = criterion(output.float(), target) / accumulation
            scaler.scale(loss).backward()
            running += loss.item() * accumulation * len(target)
            seen += len(target)
            if (index + 1) % accumulation:
                continue
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step == freeze_steps:
                set_encoder_trainable(model, True)
            if log_every and step % log_every == 0:
                print(f"step {step:5d}/{total_steps} train Huber {running / seen:.4f} "
                      f"{time.perf_counter() - start:.0f}s", flush=True)
            if step % check_every == 0 or step == total_steps:
                prediction = predict_log(model, validation_loader, device, amp)
                mae = float(np.mean(np.abs(prediction - validation_log_price)))
                improved = mae < best - 1e-4
                if improved:
                    best, best_step, waited = mae, step, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    waited += 1
                rates = {g["is_head"]: g["lr"] for g in optimizer.param_groups}
                history.append({"step": step, "epoch": epoch, "train_huber": running / seen,
                                "validation_log_mae": mae, "head_lr": rates[True],
                                "top_encoder_lr": max(g["lr"] for g in optimizer.param_groups
                                                      if not g["is_head"]),
                                "best": improved})
                print(f"  eval step {step}: val log-MAE {mae:.4f}{' *' if improved else ''}", flush=True)
                running, seen = 0.0, 0
                model.train()
                if waited >= patience:
                    done = True
            if done or step >= total_steps:
                done = True
                break
        if done:
            break
    model.load_state_dict(best_state)
    print(f"best step {best_step}, val log-MAE {best:.4f}, {time.perf_counter() - start:.0f}s")
    return pd.DataFrame(history).set_index("step"), best_step
