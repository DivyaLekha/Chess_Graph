import json
from pathlib import Path
import chess
import chess.pgn

RESULT_TO_LABEL = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

def read_pgn(path, max_plies=None):
    path = Path(path)
    games = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in RESULT_TO_LABEL:
                continue
            board = game.board()
            states = []
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if max_plies is None or ply <= max_plies:
                    states.append(board.copy(stack=False))
                if max_plies is not None and ply >= max_plies:
                    break
            if states:
                games.append({
                    "id": len(games),
                    "result": result,
                    "label": RESULT_TO_LABEL[result],
                    "plies": len(states),
                    "states": [b.fen() for b in states],
                })
    return games

def write_jsonl(games, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for game in games:
            f.write(json.dumps(game) + "\n")

def read_jsonl(path):
    games = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                games.append(json.loads(line))
    return games
