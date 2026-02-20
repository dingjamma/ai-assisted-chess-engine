# ai-assisted-chess-engine

A complete, console-based chess game written in pure Python 3 — no external libraries required. Built with AI assistance (Claude).

## Features

- **Full chess rules** — all piece movements, castling (both sides), en passant, pawn promotion (auto-queen)
- **Win/draw detection** — checkmate, stalemate, threefold repetition, insufficient material
- **Legal-move validation** — no move can leave your king in check
- **Multiple game modes** — Human vs Human, Human vs AI, AI vs Human, AI vs AI (demo)
- **Simple AI opponent** — prefers captures, otherwise plays a random legal move
- **Move notation** — accepts coordinate (`e2e4`), SAN (`Nf3`, `exd5`), short pawn push (`e4`), and castling (`O-O`, `O-O-O`)
- **Console UI** — Unicode chess pieces (♔♕♖♗♘♙), labeled ranks/files, check warnings
- **Utility commands** — `legal` (list all legal moves), `undo` (take back last move), `quit`

## Requirements

- Python 3.7+
- No external packages

## Running the Game

```bash
python3 chess_game.py
```

You will be prompted to choose a game mode, then enter moves at the prompt.

## Move Input Formats

| Input | Meaning |
|---|---|
| `e2e4` | Coordinate: move piece from e2 to e4 |
| `e4` | Short pawn push to e4 |
| `Nf3` | Knight to f3 |
| `exd5` | Pawn on e-file captures on d5 |
| `O-O` | Kingside castling |
| `O-O-O` | Queenside castling |
| `legal` | List all legal moves |
| `undo` | Take back the last move |
| `quit` | Exit the game |

## Project Structure

```
chess_game.py   — entire game in one self-contained file
```

### Main Classes

| Class | Role |
|---|---|
| `Piece` + subclasses | Per-piece pseudo-legal move generation |
| `Board` | 8×8 grid, en-passant/castling state, apply/undo moves |
| `Move` | Move data + all metadata needed for full undo |
| `MoveGenerator` | Filters pseudo-legal → fully legal (no king left in check) |
| `NotationParser` | Parses user text into `Move` objects |
| `Player` | Human or greedy AI player |
| `Game` | Main game loop, terminal-state detection, console UI |

## Special Rules Handled

1. **Castling** — rights revoked when king/rook moves or rook is captured; king cannot castle through or into check
2. **En passant** — target square tracked after every double pawn push; captured pawn removed correctly on both apply and undo
3. **Pawn promotion** — auto-promotes to Queen on reaching the back rank
4. **Threefold repetition** — board fingerprint (grid + en-passant + castling rights) tracked; draw declared on third occurrence
5. **Insufficient material** — immediate draw for K vs K, K+minor vs K, and K+B vs K+B (same-color bishops)
