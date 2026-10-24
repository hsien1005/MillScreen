"""Tier 3: supervised SciBERT fine-tuned on title + abstract (following Scancar et al., BMJ 2026).

Trained on train-era normals + train-era paper mills (class-weighted cross-entropy).
Usage: python 06b_scibert.py <temporal|random> <seed> [lr]
Writes results/scores/scibert_<kind>_seed<s>.npz with val/test scores.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RESULTS
from src.methods import load_docs, load_split

kind, seed = sys.argv[1], int(sys.argv[2])
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
TUNE = "--tune" in sys.argv  # validation only: the test set is not scored while tuning
EPOCHS, BS, MAXLEN, NAME = 2, 32, 256, "allenai/scibert_scivocab_uncased"
torch.manual_seed(seed)
np.random.seed(seed)

docs = load_docs()
sp = load_split(docs, kind, seed)
text = (docs.title.fillna("") + " [SEP] " + docs.abstract.fillna("")).to_numpy()
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2).cuda()

tr_idx = np.concatenate([sp["train_normal"], sp["train_anomaly"]])
tr_y = np.concatenate([np.zeros(len(sp["train_normal"])), np.ones(len(sp["train_anomaly"]))]).astype(int)
w = torch.tensor([1.0, len(sp["train_normal"]) / len(sp["train_anomaly"])], device="cuda")


def collate(batch):
    idx, y = zip(*batch)
    enc = tok(list(text[list(idx)]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return enc, torch.tensor(y, dtype=torch.long)


loader = DataLoader(list(zip(tr_idx, tr_y)), batch_size=BS, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
steps = EPOCHS * len(loader)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.CrossEntropyLoss(weight=w)

t0 = time.time()
model.train()
for ep in range(EPOCHS):
    for enc, y in loader:
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**enc).logits.float(), y.cuda())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
train_sec = time.time() - t0


@torch.no_grad()
def predict(idx):
    model.eval()
    out = []
    dl = DataLoader(list(zip(idx, np.zeros(len(idx), int))), batch_size=128, collate_fn=collate)
    for enc, _ in dl:
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


val_s = predict(sp["val"][0])
from sklearn.metrics import average_precision_score
if TUNE:
    print("TUNE", kind, seed, lr, "val AP", round(average_precision_score(sp["val"][1], val_s), 4), f"train {train_sec:.0f}s")
    sys.exit()
t1 = time.time()
test_full = predict(sp["test_full"][0])
ms = 1000 * (time.time() - t1) / len(test_full)
test_s = test_full[: len(sp["test"][0])]
out = RESULTS / "scores"
out.mkdir(exist_ok=True)
np.savez(out / f"scibert_{kind}_seed{seed}_lr{lr:g}.npz", val=val_s, test=test_s, test_full=test_full,
         train_sec=train_sec, ms_per_doc=ms)
print(kind, seed, lr, "val AP", round(average_precision_score(sp["val"][1], val_s), 4),
      "test AP", round(average_precision_score(sp["test"][1], test_s), 4), f"train {train_sec:.0f}s", f"{ms:.3f} ms/doc")
