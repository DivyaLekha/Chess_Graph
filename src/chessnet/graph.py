import chess
import numpy as np
import torch
from torch_geometric.data import Data

PIECE_VALUE = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}

# Nodes 0-63 are board squares. Piece nodes are 64 onward.
# This keeps the graph heterogeneous conceptually while allowing a
# homogeneous GAT implementation as a first baseline.
def board_to_graph(board: chess.Board, ply: int = 0):
    piece_squares = list(board.piece_map().items())
    n_piece = len(piece_squares)
    n = 64 + n_piece

    x = np.zeros((n, 12), dtype=np.float32)
    node_type = np.zeros(n, dtype=np.int64)  # 0=square, 1=piece

    # Square features: x/y, occupancy, attacked-by-white, attacked-by-black.
    for sq in chess.SQUARES:
        rank, file_ = divmod(sq, 8)
        x[sq, 0] = file_ / 7.0
        x[sq, 1] = rank / 7.0
        x[sq, 2] = float(board.piece_at(sq) is not None)
        x[sq, 3] = float(board.is_attacked_by(chess.WHITE, sq))
        x[sq, 4] = float(board.is_attacked_by(chess.BLACK, sq))
        x[sq, 5] = float(sq in board.legal_moves)

    square_to_piece_node = {}
    for i, (sq, piece) in enumerate(piece_squares):
        idx = 64 + i
        square_to_piece_node[sq] = idx
        node_type[idx] = 1
        x[idx, 0] = piece.piece_type / 6.0
        x[idx, 1] = float(piece.color)
        x[idx, 2] = PIECE_VALUE[piece.piece_type] / 9.0
        rank, file_ = divmod(sq, 8)
        x[idx, 3] = file_ / 7.0
        x[idx, 4] = rank / 7.0
        x[idx, 5] = len(list(board.attacks(sq))) / 27.0
        x[idx, 6] = float(board.is_check())
        x[idx, 7] = float(piece.piece_type == chess.KING)

    # Dynamic features.
    x[:, 8] = ply / 200.0
    x[:, 9] = float(board.turn)
    x[:, 10] = float(board.has_castling_rights(chess.WHITE))
    x[:, 11] = float(board.has_castling_rights(chess.BLACK))

    edges = set()

    # Square-grid adjacency.
    for sq in chess.SQUARES:
        for to in chess.SquareSet(chess.BB_KING_ATTACKS[sq]):
            edges.add((sq, to))

    # Piece-square occupancy and piece attack relations.
    for sq, piece in piece_squares:
        pnode = square_to_piece_node[sq]
        edges.add((pnode, sq))
        edges.add((sq, pnode))
        for to in board.attacks(sq):
            edges.add((pnode, to))
            edges.add((to, pnode))

    # Piece-piece interactions: attack/defend.
    for sq, piece in piece_squares:
        a = square_to_piece_node[sq]
        for to in board.attacks(sq):
            if to in square_to_piece_node:
                b = square_to_piece_node[to]
                edges.add((a, b))
                edges.add((b, a))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    data = Data(
        x=torch.tensor(x),
        edge_index=edge_index,
        node_type=torch.tensor(node_type, dtype=torch.long),
    )
    data.ply = torch.tensor([ply], dtype=torch.long)
    data.material = torch.tensor([material_balance(board)], dtype=torch.float32)
    return data

def material_balance(board):
    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUE[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score

def fen_sequence_to_graphs(fens):
    return [board_to_graph(chess.Board(fen), ply=i + 1) for i, fen in enumerate(fens)]
