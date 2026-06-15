"""Fine-tune a Turkish BERT regression head on mentor_feedback_text.

Frozen-embedding Ridge probes plateau at text-only MSE ~143; the feedback is
rigidly templated, so a fine-tuned encoder can recover (possibly denoised)
attribute information far better. Trains per canonical CV fold (same folds
as train.py, so the OOF column stacks honestly), saves:

    artifacts/text_cache/bert_oof.npy    (10k OOF predictions)
    artifacts/text_cache/bert_test.npy   (10k test predictions, fold-avg)
    artifacts/models/bert_f{k}/          (fold checkpoints for inference)

Run AFTER the tabular pipeline (16GB machine: jobs must be sequential):
    .venv/bin/python finetune_text.py
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from utils import load_data, log, make_folds, seed_everything, timer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
LR = 2e-5
WARMUP_FRAC = 0.1


class BertRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(self.dropout(pooled)).squeeze(-1)


def tokenize(tok, texts):
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def run_fold(texts, y, tr_idx, va_idx, ids_all, mask_all, ids_te, mask_te,
             device, seed):
    from transformers import AutoModel
    torch.manual_seed(seed)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    model = BertRegressor(encoder).to(device)

    y_mu, y_sd = float(y[tr_idx].mean()), float(y[tr_idx].std())
    yt = torch.tensor((y[tr_idx] - y_mu) / y_sd, dtype=torch.float32)
    ds = TensorDataset(ids_all[tr_idx], mask_all[tr_idx], yt)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)

    steps_total = len(dl) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps_total, pct_start=WARMUP_FRAC,
        anneal_strategy="linear")
    lossf = nn.MSELoss()

    def predict(ids, mask, bs=128):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ids), bs):
                p = model(ids[i:i + bs].to(device), mask[i:i + bs].to(device))
                preds.append(p.cpu().numpy())
        return np.concatenate(preds) * y_sd + y_mu

    import copy
    best_mse, best_val, best_state = np.inf, None, None
    for ep in range(EPOCHS):
        model.train()
        for bids, bmask, by in dl:
            opt.zero_grad()
            loss = lossf(model(bids.to(device), bmask.to(device)),
                         by.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = predict(ids_all[va_idx], mask_all[va_idx])
        mse = float(np.mean((pv - y[va_idx]) ** 2))
        log.info("  epoch %d val MSE %.3f", ep + 1, mse)
        if mse < best_mse:
            best_mse, best_val = mse, pv
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)          # test predicted once per fold
    best_test = predict(ids_te, mask_te)
    return best_val, best_test, best_mse


def main():
    seed_everything()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_df, test_df = load_data()
    y = train_df[config.TARGET].to_numpy()
    years = train_df[config.YEAR_COL].to_numpy()
    folds = make_folds(y, seed=config.SEED,
                       years=years if config.STRATIFY_BY_YEAR else None)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    with timer("tokenize"):
        ids_all, mask_all = tokenize(tok, train_df[config.TEXT_COL])
        ids_te, mask_te = tokenize(tok, test_df[config.TEXT_COL])

    oof = np.full(len(y), np.nan)
    test_preds = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        with timer(f"fine-tune fold {k} on {device}"):
            val, test, mse = run_fold(train_df[config.TEXT_COL], y, tr_idx,
                                      va_idx, ids_all, mask_all, ids_te,
                                      mask_te, device, config.SEED + k)
        oof[va_idx] = val
        test_preds.append(test)
        log.info("fold %d best val MSE %.3f", k, mse)

    np.save(config.TEXT_CACHE / "bert_oof.npy", oof)
    np.save(config.TEXT_CACHE / "bert_test.npy",
            np.mean(test_preds, axis=0))
    log.info("TEXT-ONLY fine-tuned OOF MSE %.3f (frozen-emb Ridge was ~143)",
             np.mean((oof - y) ** 2))


if __name__ == "__main__":
    main()
