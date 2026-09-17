import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessnet.train import train
p=argparse.ArgumentParser()
p.add_argument("--data",required=True); p.add_argument("--model",choices=["gat","hcgnn"],default="gat")
p.add_argument("--epochs",type=int,default=10); p.add_argument("--lr",type=float,default=1e-3)
a=p.parse_args()
train(a.data,a.model,a.epochs,a.lr)
