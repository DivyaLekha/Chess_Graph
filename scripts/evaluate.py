import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessnet.evaluate import evaluate
p=argparse.ArgumentParser()
p.add_argument("--data",required=True); p.add_argument("--checkpoint",required=True)
p.add_argument("--fraction",type=float,default=1.0)
a=p.parse_args()
print(json.dumps(evaluate(a.data,a.checkpoint,fraction=a.fraction),indent=2))
