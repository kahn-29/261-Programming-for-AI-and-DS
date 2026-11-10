"""End-to-end fusion of the tabular MLP (notebook 08) and PhoBERT (notebook 09), notebook 10."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.phobert import mean_pool


class FusionDataset(Dataset):
    """Token ids plus the encoded tabular row; batched by phobert.make_collate."""

    def __init__(self, input_ids, dense, codes, target=None):
        self.input_ids, self.dense, self.codes = input_ids, dense, codes
        self.target = None if target is None else np.asarray(target, dtype=np.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        item = {"input_ids": self.input_ids[index], "dense": self.dense[index],
                "codes": self.codes[index]}
        if self.target is not None:
            item["target"] = self.target[index]
        return item


class FusionModel(nn.Module):
    """[MLP features | projected PhoBERT mean-pool] -> joint head -> log1p(price).

    Everything outside ``encoder`` lives under ``head`` so that phobert.layerwise_groups
    gives the new (tabular + fusion) layers the head learning rate and PhoBERT the
    layer-wise decayed rates; phobert.fit trains it unchanged.
    """

    def __init__(self, encoder, tabular, text_dim=64, hidden=256, dropout=0.1, target_mean=0.0):
        super().__init__()
        self.encoder = encoder
        tab_dim = tabular.head[-1].in_features
        out = nn.Sequential(nn.LayerNorm(tab_dim + text_dim), nn.Linear(tab_dim + text_dim, hidden),
                            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.normal_(out[-1].weight, std=0.01)
        nn.init.constant_(out[-1].bias, target_mean)
        self.head = nn.ModuleDict({
            "tabular": tabular,
            "text": nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(encoder.config.hidden_size, text_dim), nn.GELU()),
            "out": out,
        })

    def forward(self, input_ids, attention_mask, dense, codes):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text = self.head["text"](mean_pool(hidden, attention_mask))
        tab = self.head["tabular"].features(dense, codes)
        return self.head["out"](torch.cat([tab, text.to(tab.dtype)], dim=1))
