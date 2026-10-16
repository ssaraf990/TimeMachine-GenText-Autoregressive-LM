#!/usr/bin/env python3
"""
=============================================================================
  Autoregressive Text Generation Benchmark System
  Using RNN, BiRNN, LSTM, and GRU — PyTorch Implementation
  Student : Satyam Saraf
  Program : B.Tech CCE, Manipal Institute of Technology
=============================================================================
"""

import os
import re
import json
import math
import time
import random
import requests
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN    = 20
EMBED_DIM  = 128
HIDDEN_DIM = 256
DROPOUT    = 0.20
LR         = 1e-3
CLIP_NORM  = 5.0
EPOCHS     = 8
BATCH_SIZE = 128
VOCAB_SIZE = 5_000        # top-N tokens (+ <pad>=0, <unk>=1)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
GUTENBERG_URL = "https://www.gutenberg.org/ebooks/35.txt.utf-8"
LOCAL_CACHE   = "time_machine_raw.txt"
RESULTS_FILE  = "results_summary.json"


def download_text(url: str, cache_path: str) -> str:
    if os.path.exists(cache_path):
        print(f"[INFO] Using cached file: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[INFO] Downloading: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    text = resp.text
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Saved to {cache_path}")
    return text


def strip_gutenberg(text: str) -> str:
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]
    start_idx = 0
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break
    end_idx = len(text)
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx]


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"([.,!?;:\"\'()\[\]{}<>])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(tokens: list, max_size: int = VOCAB_SIZE):
    freq = Counter(tokens)
    most_common = [w for w, _ in freq.most_common(max_size - 2)]
    word2idx = {"<pad>": 0, "<unk>": 1}
    for w in most_common:
        word2idx[w] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def encode(tokens: list, word2idx: dict) -> list:
    unk = word2idx["<unk>"]
    return [word2idx.get(t, unk) for t in tokens]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    def __init__(self, ids: list, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────────────────────────────────────
class AutoregressiveLM(nn.Module):
    """
    Unified language model supporting:
        'rnn'   – Simple (Vanilla) RNN
        'birnn' – Bidirectional RNN
        'lstm'  – Long Short-Term Memory
        'gru'   – Gated Recurrent Unit
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT, model_type="lstm"):
        super().__init__()
        self.model_type = model_type.lower()
        bidirectional   = self.model_type == "birnn"

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop      = nn.Dropout(dropout)

        rnn_kwargs = dict(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=1,
            batch_first=True, dropout=0, bidirectional=bidirectional,
        )
        if   self.model_type in ("rnn", "birnn"): self.rnn = nn.RNN(**rnn_kwargs)
        elif self.model_type == "lstm":            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.model_type == "gru":             self.rnn = nn.GRU(**rnn_kwargs)
        else: raise ValueError(f"Unknown model_type: {model_type}")

        rnn_out = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_out, vocab_size)

    def forward(self, x, hidden=None):
        emb    = self.drop(self.embedding(x))
        out, h = self.rnn(emb, hidden)
        logits = self.fc(self.drop(out[:, -1, :]))
        return logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
            total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def evaluate_test(model, loader, criterion):
    model.eval()
    total_loss, c1, c5, total = 0.0, 0, 0, 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        logits, _= model(x)
        total_loss += criterion(logits, y).item()
        c1     += (logits.argmax(-1) == y).sum().item()
        top5    = logits.topk(5, dim=-1).indices
        c5     += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total  += y.size(0)
    avg = total_loss / len(loader)
    return avg, math.exp(avg), 100*c1/total, 100*c5/total


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GENERATION + DIVERSITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_text(model, prompt, word2idx, idx2word, n=50, temperature=1.0):
    model.eval()
    tokens = tokenize(prompt)
    ids    = encode(tokens, word2idx)
    if len(ids) < SEQ_LEN:
        ids = [word2idx["<pad>"]] * (SEQ_LEN - len(ids)) + ids
    else:
        ids = ids[-SEQ_LEN:]

    ctx  = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    out  = list(tokens)
    for _ in range(n):
        logits, _ = model(ctx)
        probs     = torch.softmax(logits.squeeze(0) / temperature, -1)
        nid       = torch.multinomial(probs, 1).item()
        out.append(idx2word.get(nid, "<unk>"))
        ctx = torch.cat([ctx[:, 1:], torch.tensor([[nid]], device=DEVICE)], 1)
    return " ".join(out)


def distinct_n(text, n):
    toks = text.split()
    if len(toks) < n: return 0.0
    ng = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(ng)) / len(ng)

def rep_rate(text):
    toks = text.split()
    return 1.0 - len(set(toks))/len(toks) if toks else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"SimpleRNN":"#EF5350","BiRNN":"#AB47BC","LSTM":"#42A5F5","GRU":"#26A69A"}

def plot_loss_curves(history, save_path="loss_comparison_curves.png"):
    fig = plt.figure(figsize=(18, 5), facecolor="#0D1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_tr = fig.add_subplot(gs[0])
    ax_vl = fig.add_subplot(gs[1])
    for name, h in history.items():
        c  = PALETTE.get(name, "white")
        ep = range(1, len(h["train"])+1)
        ax_tr.plot(ep, h["train"], color=c, lw=2.5, marker="o", ms=5, label=name)
        ax_vl.plot(ep, h["val"],   color=c, lw=2.5, marker="s", ms=5, label=name)
    for ax, title in [(ax_tr,"Training Loss"),(ax_vl,"Validation Loss")]:
        ax.set_facecolor("#161B22")
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", color="#8B949E", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", color="#8B949E", fontsize=11)
        ax.tick_params(colors="#8B949E")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#30363D")
        ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="white", fontsize=10)
        ax.grid(True, color="#21262D", ls="--", lw=0.7, alpha=0.8)
        ax.set_xticks(range(1, EPOCHS+1))
    fig.suptitle("Autoregressive LM — Training vs Validation Loss\n"
                 "(SimpleRNN | BiRNN | LSTM | GRU)",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Loss curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    raw   = download_text(GUTENBERG_URL, LOCAL_CACHE)
    body  = strip_gutenberg(raw)
    toks  = tokenize(body)
    print(f"[INFO] Total tokens : {len(toks):,}")

    word2idx, idx2word = build_vocab(toks)
    V = len(word2idx)
    print(f"[INFO] Vocab size   : {V:,}")

    ids = encode(toks, word2idx)
    n   = len(ids) - SEQ_LEN
    n_tr, n_vl = int(0.80*n), int(0.10*n)

    train_ids = ids[:n_tr + SEQ_LEN]
    val_ids   = ids[n_tr : n_tr + n_vl + SEQ_LEN]
    test_ids  = ids[n_tr + n_vl:]

    g = torch.Generator(); g.manual_seed(SEED)
    train_dl = DataLoader(SlidingWindowDataset(train_ids), BATCH_SIZE,
                          shuffle=True,  generator=g, num_workers=0)
    val_dl   = DataLoader(SlidingWindowDataset(val_ids),   BATCH_SIZE,
                          shuffle=False, num_workers=0)
    test_dl  = DataLoader(SlidingWindowDataset(test_ids),  BATCH_SIZE,
                          shuffle=False, num_workers=0)
    print(f"[INFO] Train/Val/Test : {len(train_dl.dataset):,} / "
          f"{len(val_dl.dataset):,} / {len(test_dl.dataset):,}")

    seed_prompts = [
        "the time traveller held in his hand",
        "we all sat in the dim light",
        "it was a delicate piece of work",
    ]
    temperatures = [0.5, 1.0, 1.5]

    # ── Load partial results if present ──────────────────────────────────────
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[INFO] Resuming from {RESULTS_FILE} — "
              f"completed: {[k for k in results if k != 'gen_samples']}")

    history   = {k: results[k]["loss_history"] for k in results if k != "gen_samples"}
    gen_data  = results.get("gen_samples", {})

    model_types  = ["rnn", "birnn", "lstm", "gru"]
    display_name = {"rnn":"SimpleRNN","birnn":"BiRNN","lstm":"LSTM","gru":"GRU"}
    criterion    = nn.CrossEntropyLoss()

    for mtype in model_types:
        dname = display_name[mtype]
        if dname in results:
            print(f"[SKIP] {dname} already done.")
            continue

        print(f"\n{'═'*62}\n  Training  ▶  {dname}\n{'═'*62}")
        model = AutoregressiveLM(V, EMBED_DIM, HIDDEN_DIM, DROPOUT, mtype).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        nparams = model.count_parameters()
        print(f"  Parameters: {nparams:,}")

        tr_losses, vl_losses = [], []
        t0 = time.time()
        for ep in range(1, EPOCHS+1):
            tl = run_epoch(model, train_dl, criterion, opt)
            vl = run_epoch(model, val_dl,   criterion)
            tr_losses.append(tl); vl_losses.append(vl)
            print(f"  Epoch [{ep:02d}/{EPOCHS}]  Train: {tl:.4f}  Val: {vl:.4f}")
        rt = time.time() - t0

        test_loss, ppl, top1, top5 = evaluate_test(model, test_dl, criterion)
        print(f"\n  ── Test Results ──")
        print(f"     Loss:       {test_loss:.4f}")
        print(f"     PPL:        {ppl:.2f}")
        print(f"     Top-1 Acc:  {top1:.2f}%")
        print(f"     Top-5 Acc:  {top5:.2f}%")
        print(f"     Runtime:    {rt:.1f}s  |  Params: {nparams:,}")

        # ── Generation ────────────────────────────────────────────────────────
        gen_data[dname] = {}
        for prompt in seed_prompts:
            gen_data[dname][prompt] = {}
            for temp in temperatures:
                txt = generate_text(model, prompt, word2idx, idx2word,
                                    n=50, temperature=temp)
                d1  = distinct_n(txt, 1)
                d2  = distinct_n(txt, 2)
                rr  = rep_rate(txt)
                gen_data[dname][prompt][str(temp)] = {
                    "text": txt, "D1": round(d1,4),
                    "D2": round(d2,4), "RepRate": round(rr,4)
                }
                print(f"  T={temp}  D1={d1:.3f}  D2={d2:.3f}  RepRate={rr:.3f}")
                print(f"    \"{txt[:110]}...\"")

        # ── Checkpoint after EACH model ───────────────────────────────────────
        results[dname] = {
            "params": nparams, "runtime_s": round(rt, 1),
            "test_loss": round(test_loss, 4),
            "ppl":       round(ppl, 2),
            "top1_acc":  round(top1, 2),
            "top5_acc":  round(top5, 2),
            "loss_history": {"train": tr_losses, "val": vl_losses},
        }
        results["gen_samples"] = gen_data

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [✓] Checkpoint saved → {RESULTS_FILE}")

        history[dname] = {"train": tr_losses, "val": vl_losses}

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_loss_curves(history)

    # ── Benchmark table ───────────────────────────────────────────────────────
    print(f"\n\n{'═'*90}")
    print("  FINAL BENCHMARK TABLE")
    print(f"{'═'*90}")
    print(f"{'Model':<12} {'Params':>10} {'Runtime(s)':>12} "
          f"{'Test Loss':>10} {'PPL':>8} {'Top-1%':>8} {'Top-5%':>8}")
    print("─"*90)
    for dname in ["SimpleRNN","BiRNN","LSTM","GRU"]:
        r = results[dname]
        print(f"{dname:<12} {r['params']:>10,} {r['runtime_s']:>12.1f} "
              f"{r['test_loss']:>10.4f} {r['ppl']:>8.2f} "
              f"{r['top1_acc']:>8.2f} {r['top5_acc']:>8.2f}")
    print("═"*90)
    print("[INFO] Pipeline complete ✓")


if __name__ == "__main__":
    main()
