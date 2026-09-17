import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessnet.data import read_pgn, write_jsonl

p=argparse.ArgumentParser()
p.add_argument("--pgn",required=True)
p.add_argument("--output",required=True)
p.add_argument("--max-plies",type=int,default=None)
a=p.parse_args()
games=read_pgn(a.pgn,a.max_plies)
write_jsonl(games,a.output)
print(f"wrote {len(games)} games to {a.output}")
