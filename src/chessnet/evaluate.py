import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .data import read_jsonl
from .graph import fen_sequence_to_graphs
from .models import GATClassifier, HCGNN

def evaluate(data_path, checkpoint, model_name=None, fraction=1.0):
    games=read_jsonl(data_path)
    ckpt=torch.load(checkpoint,map_location="cpu")
    model_name=model_name or ckpt["model_name"]
    model=GATClassifier() if model_name=="gat" else HCGNN()
    model.load_state_dict(ckpt["model"])
    model.eval()
    pred=[]; true=[]
    with torch.no_grad():
        for g in games:
            n=max(1,int(len(g["states"])*fraction))
            graphs=fen_sequence_to_graphs(g["states"][:n])
            pred.append(int(model(graphs).argmax()))
            true.append(g["label"])
    return {
        "accuracy": accuracy_score(true,pred),
        "macro_f1": f1_score(true,pred,average="macro",zero_division=0),
        "confusion_matrix": confusion_matrix(true,pred).tolist(),
        "n_games": len(games),
        "fraction": fraction,
    }
