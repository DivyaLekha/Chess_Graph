# Chess as a Dynamic Network — HCGNN

A reproducible implementation of the **Hybrid Chess Graph Neural Network (HCGNN)** described in the accompanying M.Tech thesis.

The project models each chess game as a sequence of heterogeneous graph states containing **piece nodes** and **square nodes**, with intra-state relations and temporal transitions. It then provides:

- a simple material baseline,
- a GAT graph baseline,
- an HCGNN-style hybrid model with HGT-inspired heterogeneous attention, temporal attention, and GAT,
- full-game outcome prediction,
- partial-game outcome prediction,
- reproducible train/validation/test splits,
- graph visualisation and evaluation utilities.

## Important reproducibility note

The thesis reports experiments on approximately 10,000 PGN games, but the original PGN dataset and original implementation are not included here. Therefore this repository is an **independent, thesis-specification-based implementation**, not a claim of exact reproduction of the reported numerical results.

The implementation follows the thesis description of:
1. sequential board-state graph construction;
2. piece and square node categories;
3. spatial/tactical and piece-square relations;
4. inter-state temporal transitions;
5. categorical, numeric and dynamic features;
6. HGT + TGAT + GAT branches;
7. graph pooling, fusion and three-class outcome prediction.

## Quick start

### 1. Environment

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare PGN

Place a PGN file at:

```text
data/raw/games.pgn
```

A tiny example is included at `data/raw/sample.pgn`.

Then:

```powershell
python scripts/prepare_data.py --pgn data/raw/sample.pgn --output data/processed/sample.jsonl
```

For a real experiment:

```powershell
python scripts/prepare_data.py --pgn data/raw/games.pgn --output data/processed/games.jsonl --max-plies 120
```

### 3. Train

```powershell
python scripts/train.py --data data/processed/games.jsonl --model gat --epochs 10
```

and:

```powershell
python scripts/train.py --data data/processed/games.jsonl --model hcgnn --epochs 10
```

### 4. Evaluate

```powershell
python scripts/evaluate.py --data data/processed/games.jsonl --checkpoint results/checkpoints/best_hcgnn.pt
```

## Recommended scientific experiments

The most interesting experiment for the Edge Cases submission is **partial-game prediction**:

> How early can an evolving network predict the eventual winner?

Evaluate prefixes such as 10%, 20%, ..., 100% of each game, while keeping the train/validation/test split at the **game level** to prevent leakage.

Recommended comparison:

| Model | Input |
|---|---|
| Majority baseline | outcome frequency |
| Material baseline | board material at prefix |
| GAT | graph structure |
| HCGNN | heterogeneous + temporal + local attention |
| HCGNN ablations | remove one branch at a time |

Do not use future moves or final-board information when predicting from a prefix.

## Repository structure

```text
chess-network-science/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── sample.pgn
│   │   └── README.md
│   └── processed/
├── src/
│   └── chessnet/
│       ├── __init__.py
│       ├── data.py
│       ├── graph.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       └── visualize.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── experiments/
│   └── baseline.yaml
├── notebooks/
│   └── README.md
├── results/
│   └── .gitkeep
└── tests/
    └── test_graph.py
```

## Thesis alignment

The thesis specifies a dynamic graph sequence `G = {G1, ..., GT}` and a supra-adjacency construction linking intra-state and inter-state relations. It describes piece and square nodes, relations including attack/defense, adjacency, mobility and piece-square association, and features such as piece type, color, position, material, mobility and temporal state information.

The HCGNN architecture is implemented as three complementary branches followed by pooling and fusion.

## License

MIT for the implementation in this repository. Dataset licensing remains the responsibility of the dataset provider.
