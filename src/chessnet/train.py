from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from .data import read_jsonl
from .graph import fen_sequence_to_graphs
from .models import GATClassifier, HCGNN

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_games(games, seed=42):
    rng = random.Random(seed)
    ids = list(range(len(games))); rng.shuffle(ids)
    n = len(ids)
    a, b = int(.7*n), int(.85*n)
    return [games[i] for i in ids[:a]], [games[i] for i in ids[a:b]], [games[i] for i in ids[b:]]

def make_model(name):
    if name == "gat":
        return GATClassifier()
    if name == "hcgnn":
        return HCGNN()
    raise ValueError(name)

def game_graphs(game, fraction=1.0):
    n = max(1, int(len(game["states"]) * fraction))
    return fen_sequence_to_graphs(game["states"][:n])

def train(data_path, model_name="gat", epochs=10, lr=1e-3, seed=42, out="results/checkpoints"):
    seed_everything(seed)
    games = read_jsonl(data_path)
    train_g, val_g, test_g = split_games(games, seed)
    labels = torch.tensor([g["label"] for g in train_g])
    counts = torch.bincount(labels, minlength=3).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights = weights / weights.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    best = -1

    for epoch in range(1, epochs + 1):
        model.train(); losses=[]; pred=[]; true=[]
        for g in train_g:
            graphs = [x.to(device) for x in game_graphs(g)]
            y = torch.tensor([g["label"]], device=device)
            opt.zero_grad()
            logits = model(graphs).unsqueeze(0)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(loss.item())
            pred.append(int(logits.argmax(-1))); true.append(g["label"])
        model.eval(); vp=[]; vt=[]
        with torch.no_grad():
            for g in val_g:
                graphs=[x.to(device) for x in game_graphs(g)]
                vp.append(int(model(graphs).argmax())); vt.append(g["label"])
        acc=accuracy_score(vt,vp) if vt else 0.0
        f1=f1_score(vt,vp,average="macro",zero_division=0) if vt else 0.0
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_macro_f1={f1:.4f}")
        if acc > best:
            best=acc
            Path(out).mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "model_name": model_name, "seed": seed}, Path(out)/f"best_{model_name}.pt")
    return test_g
