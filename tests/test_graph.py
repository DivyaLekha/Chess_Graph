import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import chess
from chessnet.graph import board_to_graph

def test_initial_board_graph():
    d=board_to_graph(chess.Board())
    assert d.num_nodes == 64 + 32
    assert d.edge_index.shape[0] == 2
    assert d.x.shape[1] == 12

def test_after_move():
    b=chess.Board(); b.push_san("e4")
    d=board_to_graph(b,1)
    assert d.num_nodes == 96
    assert d.x.shape == (96,12)
