#!/usr/bin/env python3
"""
Console Chess Game
==================
A complete, two-player (or human vs. AI) chess game playable in the terminal.
Supports full chess rules: all piece movements, castling, en passant, pawn
promotion, check/checkmate/stalemate detection, and draws by insufficient
material or threefold repetition.

Usage:
    python chess_game.py

Move input formats accepted:
    e2e4        — coordinate notation (from-square to-square)
    e4          — short pawn push
    Nf3         — piece move (N=Knight, B=Bishop, R=Rook, Q=Queen, K=King)
    exd5        — pawn capture
    O-O         — kingside castling
    O-O-O       — queenside castling
    legal / moves — list all legal moves for the current player
    quit / exit   — exit the game
"""

import sys
import random
import copy
from typing import Optional, List, Tuple, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHITE = "white"
BLACK = "black"

PIECE_SYMBOLS = {
    ("King",   WHITE): "♔",
    ("Queen",  WHITE): "♕",
    ("Rook",   WHITE): "♖",
    ("Bishop", WHITE): "♗",
    ("Knight", WHITE): "♘",
    ("Pawn",   WHITE): "♙",
    ("King",   BLACK): "♚",
    ("Queen",  BLACK): "♛",
    ("Rook",   BLACK): "♜",
    ("Bishop", BLACK): "♝",
    ("Knight", BLACK): "♞",
    ("Pawn",   BLACK): "♟",
}

FILE_LETTERS = "abcdefgh"
RANK_NUMBERS  = "12345678"


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def file_to_col(f: str) -> int:
    """'a' → 0, 'h' → 7"""
    return FILE_LETTERS.index(f)


def rank_to_row(r: str) -> int:
    """'1' → 0, '8' → 7"""
    return int(r) - 1


def sq_to_coords(sq: str) -> Tuple[int, int]:
    """'e4' → (3, 4)  i.e. (row, col)"""
    col = file_to_col(sq[0])
    row = rank_to_row(sq[1])
    return (row, col)


def coords_to_sq(row: int, col: int) -> str:
    return FILE_LETTERS[col] + str(row + 1)


# ---------------------------------------------------------------------------
# Piece classes
# ---------------------------------------------------------------------------

class Piece:
    """Abstract base for all chess pieces."""

    def __init__(self, color: str):
        self.color = color
        self.has_moved = False          # used for castling / pawn double-push

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def symbol(self) -> str:
        return PIECE_SYMBOLS[(self.name, self.color)]

    def __repr__(self) -> str:
        return f"{self.color[0].upper()}{self.name[0]}"

    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        """Return list of (row, col) squares this piece can move to,
        ignoring check constraints (pseudo-legal)."""
        raise NotImplementedError

    def _slide(self, board: "Board", row: int, col: int,
               directions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Helper: sliding moves along given direction vectors."""
        moves = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                target = board.grid[r][c]
                if target is None:
                    moves.append((r, c))
                elif target.color != self.color:
                    moves.append((r, c))   # capture
                    break
                else:
                    break                  # blocked by own piece
                r += dr
                c += dc
        return moves


class Pawn(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        direction = 1 if self.color == WHITE else -1
        start_row  = 1 if self.color == WHITE else 6

        # One square forward
        r = row + direction
        if 0 <= r < 8 and board.grid[r][col] is None:
            moves.append((r, col))
            # Two squares from starting rank
            if row == start_row:
                r2 = row + 2 * direction
                if board.grid[r2][col] is None:
                    moves.append((r2, col))

        # Diagonal captures
        for dc in (-1, 1):
            c = col + dc
            r = row + direction
            if 0 <= r < 8 and 0 <= c < 8:
                target = board.grid[r][c]
                if target is not None and target.color != self.color:
                    moves.append((r, c))
                # En passant
                if (r, c) == board.en_passant_target:
                    moves.append((r, c))

        return moves


class Rook(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        return self._slide(board, row, col, [(1,0),(-1,0),(0,1),(0,-1)])


class Knight(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                target = board.grid[r][c]
                if target is None or target.color != self.color:
                    moves.append((r, c))
        return moves


class Bishop(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        return self._slide(board, row, col, [(1,1),(1,-1),(-1,1),(-1,-1)])


class Queen(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        return self._slide(board, row, col,
                           [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)])


class King(Piece):
    def pseudo_legal_moves(self, board: "Board", row: int, col: int) -> List[Tuple[int, int]]:
        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    target = board.grid[r][c]
                    if target is None or target.color != self.color:
                        moves.append((r, c))

        # Castling (pseudo-legal; legality checked later)
        if not self.has_moved:
            moves += board._castling_squares(self.color)

        return moves


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class Board:
    """
    Holds the 8×8 grid and all board-level state.

    grid[0][0] = a1 (White's queen-rook starting square)
    grid[7][7] = h8 (Black's king-rook starting square)
    Rows increase from White's side (row 0) to Black's (row 7).
    """

    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None]*8 for _ in range(8)]
        self.en_passant_target: Optional[Tuple[int,int]] = None
        # Castling rights: True = right still available
        self.castling_rights: Dict[str, Dict[str, bool]] = {
            WHITE: {"kingside": True, "queenside": True},
            BLACK: {"kingside": True, "queenside": True},
        }
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        """Place pieces in starting positions."""
        back_rank = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for col, PieceClass in enumerate(back_rank):
            self.grid[0][col] = PieceClass(WHITE)
            self.grid[7][col] = PieceClass(BLACK)
        for col in range(8):
            self.grid[1][col] = Pawn(WHITE)
            self.grid[6][col] = Pawn(BLACK)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, perspective: str = WHITE):
        """Print the board with Unicode pieces and rank/file labels."""
        print()
        rows = range(7, -1, -1) if perspective == WHITE else range(8)
        cols = range(8)         if perspective == WHITE else range(7, -1, -1)

        # Column header
        col_labels = "  " + "  ".join(FILE_LETTERS[c] for c in cols)
        print(col_labels)
        print("  " + "─" * 25)

        for row in rows:
            rank_label = str(row + 1)
            row_str = f"{rank_label} │"
            for col in cols:
                piece = self.grid[row][col]
                if piece:
                    row_str += f" {piece.symbol} "
                else:
                    # Checkerboard shading via center-dot
                    shade = "·" if (row + col) % 2 == 0 else " "
                    row_str += f" {shade} "
            row_str += f"│ {rank_label}"
            print(row_str)

        print("  " + "─" * 25)
        print(col_labels)
        print()

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def apply_move(self, move: "Move") -> "Move":
        """
        Apply a Move to the board (mutates in place).
        Returns the move (with captured piece recorded for undo).
        """
        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece = self.grid[fr][fc]

        # Record state for potential undo
        move.captured = self.grid[tr][tc]
        move.prev_en_passant = self.en_passant_target
        move.prev_castling_rights = copy.deepcopy(self.castling_rights)
        move.piece_had_moved = piece.has_moved

        # En passant capture: remove the captured pawn
        if isinstance(piece, Pawn) and (tr, tc) == self.en_passant_target:
            ep_pawn_row = fr           # captured pawn sits on same rank as moving pawn
            self.grid[ep_pawn_row][tc] = None
            move.ep_captured_sq = (ep_pawn_row, tc)
        else:
            move.ep_captured_sq = None

        # Reset or set en-passant target
        self.en_passant_target = None
        if isinstance(piece, Pawn) and abs(tr - fr) == 2:
            ep_row = (fr + tr) // 2
            self.en_passant_target = (ep_row, fc)

        # Move the piece
        self.grid[tr][tc] = piece
        self.grid[fr][fc] = None
        piece.has_moved = True

        # Castling: move the rook as well
        if isinstance(piece, King) and abs(tc - fc) == 2:
            if tc > fc:   # kingside
                rook_from_col, rook_to_col = 7, 5
            else:         # queenside
                rook_from_col, rook_to_col = 0, 3
            rook = self.grid[tr][rook_from_col]
            self.grid[tr][rook_to_col] = rook
            self.grid[tr][rook_from_col] = None
            if rook:
                rook.has_moved = True

        # Pawn promotion (auto-promote to Queen)
        promotion_row = 7 if piece.color == WHITE else 0
        if isinstance(piece, Pawn) and tr == promotion_row:
            promoted = Queen(piece.color)
            promoted.has_moved = True
            self.grid[tr][tc] = promoted
            move.promoted_to = promoted
        else:
            move.promoted_to = None

        # Update castling rights
        self._update_castling_rights(piece, fr, fc, tr, tc)

        return move

    def undo_move(self, move: "Move"):
        """Reverse a previously applied move."""
        fr, fc = move.from_sq
        tr, tc = move.to_sq

        # Restore en-passant and castling rights
        self.en_passant_target = move.prev_en_passant
        self.castling_rights   = move.prev_castling_rights

        # Get moving piece (might be promoted queen, revert to pawn)
        piece = self.grid[tr][tc]
        if move.promoted_to is not None:
            piece = Pawn(piece.color)
            piece.has_moved = move.piece_had_moved

        piece.has_moved = move.piece_had_moved
        self.grid[fr][fc] = piece
        self.grid[tr][tc] = move.captured

        # Undo en passant capture
        if move.ep_captured_sq is not None:
            ep_r, ep_c = move.ep_captured_sq
            color = opponent(piece.color)
            self.grid[ep_r][ep_c] = Pawn(color)
            self.grid[ep_r][ep_c].has_moved = True

        # Undo castling rook move
        if isinstance(piece, King) and abs(tc - fc) == 2:
            if tc > fc:   # kingside
                rook_from_col, rook_to_col = 7, 5
            else:
                rook_from_col, rook_to_col = 0, 3
            rook = self.grid[tr][rook_to_col]
            self.grid[tr][rook_from_col] = rook
            self.grid[tr][rook_to_col]   = None
            if rook:
                rook.has_moved = False

    def _update_castling_rights(self, piece: Piece, fr: int, fc: int, tr: int, tc: int):
        """Revoke castling rights when relevant pieces move or are captured."""
        # King moved
        if isinstance(piece, King):
            self.castling_rights[piece.color]["kingside"]  = False
            self.castling_rights[piece.color]["queenside"] = False
        # Rook moved
        if isinstance(piece, Rook):
            if piece.color == WHITE:
                if fc == 0: self.castling_rights[WHITE]["queenside"] = False
                if fc == 7: self.castling_rights[WHITE]["kingside"]  = False
            else:
                if fc == 0: self.castling_rights[BLACK]["queenside"] = False
                if fc == 7: self.castling_rights[BLACK]["kingside"]  = False
        # Rook captured on its starting square
        for color, row in [(WHITE, 0), (BLACK, 7)]:
            if tr == row and tc == 0:
                self.castling_rights[color]["queenside"] = False
            if tr == row and tc == 7:
                self.castling_rights[color]["kingside"]  = False

    # ------------------------------------------------------------------
    # Square attack / king safety
    # ------------------------------------------------------------------

    def is_square_attacked(self, row: int, col: int, by_color: str) -> bool:
        """Return True if (row, col) is attacked by any piece of by_color."""
        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                if piece and piece.color == by_color:
                    if (row, col) in piece.pseudo_legal_moves(self, r, c):
                        return True
        return False

    def king_position(self, color: str) -> Tuple[int, int]:
        """Return (row, col) of the king of given color."""
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and isinstance(p, King) and p.color == color:
                    return (r, c)
        raise ValueError(f"No {color} king found!")

    def in_check(self, color: str) -> bool:
        kr, kc = self.king_position(color)
        return self.is_square_attacked(kr, kc, opponent(color))

    # ------------------------------------------------------------------
    # Castling helpers
    # ------------------------------------------------------------------

    def _castling_squares(self, color: str) -> List[Tuple[int, int]]:
        """Return king destination squares for legal castling (path-clear only;
        check validation done in legal_moves)."""
        row   = 0 if color == WHITE else 7
        squares = []
        rights = self.castling_rights[color]

        if rights["kingside"]:
            # Squares between king (e-file=4) and rook (h-file=7) must be empty
            if self.grid[row][5] is None and self.grid[row][6] is None:
                rook = self.grid[row][7]
                if isinstance(rook, Rook) and not rook.has_moved:
                    squares.append((row, 6))

        if rights["queenside"]:
            if (self.grid[row][3] is None and self.grid[row][2] is None
                    and self.grid[row][1] is None):
                rook = self.grid[row][0]
                if isinstance(rook, Rook) and not rook.has_moved:
                    squares.append((row, 2))

        return squares

    # ------------------------------------------------------------------
    # Deep copy for move simulation
    # ------------------------------------------------------------------

    def copy(self) -> "Board":
        new = Board.__new__(Board)
        new.grid = [[None]*8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p:
                    np = p.__class__(p.color)
                    np.has_moved = p.has_moved
                    new.grid[r][c] = np
        new.en_passant_target  = self.en_passant_target
        new.castling_rights    = copy.deepcopy(self.castling_rights)
        return new

    # ------------------------------------------------------------------
    # Board state fingerprint (for repetition detection)
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Compact string uniquely describing the board position."""
        parts = []
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                parts.append(repr(p) if p else ".")
        parts.append(str(self.en_passant_target))
        parts.append(str(self.castling_rights))
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Piece lists
    # ------------------------------------------------------------------

    def pieces(self, color: str) -> List[Tuple[Piece, int, int]]:
        """Return list of (piece, row, col) for all pieces of given color."""
        result = []
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and p.color == color:
                    result.append((p, r, c))
        return result


# ---------------------------------------------------------------------------
# Move dataclass
# ---------------------------------------------------------------------------

class Move:
    """
    Represents a single chess move including all metadata needed for
    apply/undo and for notation generation.
    """

    def __init__(self, from_sq: Tuple[int,int], to_sq: Tuple[int,int]):
        self.from_sq = from_sq
        self.to_sq   = to_sq

        # Filled in by Board.apply_move()
        self.captured: Optional[Piece]                       = None
        self.prev_en_passant: Optional[Tuple[int,int]]       = None
        self.prev_castling_rights: Optional[Dict]            = None
        self.piece_had_moved: bool                           = False
        self.ep_captured_sq: Optional[Tuple[int,int]]        = None
        self.promoted_to: Optional[Piece]                    = None

    def uci(self) -> str:
        """e.g. 'e2e4'"""
        return coords_to_sq(*self.from_sq) + coords_to_sq(*self.to_sq)

    def __repr__(self) -> str:
        return self.uci()


# ---------------------------------------------------------------------------
# Move generator / validator
# ---------------------------------------------------------------------------

class MoveGenerator:
    """Generates fully legal moves (no leaving king in check)."""

    @staticmethod
    def legal_moves(board: Board, color: str) -> List[Move]:
        """Return all legal moves for color on board."""
        moves = []
        for piece, row, col in board.pieces(color):
            for tr, tc in piece.pseudo_legal_moves(board, row, col):
                m = Move((row, col), (tr, tc))
                # Validate castling: squares the king passes through cannot be attacked
                if isinstance(piece, King) and abs(tc - col) == 2:
                    if not MoveGenerator._castling_legal(board, color, col, tc):
                        continue
                # Simulate move and check own king
                sim = board.copy()
                sim_move = Move((row, col), (tr, tc))
                sim.apply_move(sim_move)
                if not sim.in_check(color):
                    moves.append(m)
        return moves

    @staticmethod
    def _castling_legal(board: Board, color: str, king_col: int, dest_col: int) -> bool:
        """
        Castling is illegal if:
          - King is currently in check
          - King passes through an attacked square
          - King ends up in check
        """
        row = 0 if color == WHITE else 7
        opp = opponent(color)
        # King must not be in check
        if board.is_square_attacked(row, king_col, opp):
            return False
        # Determine pass-through column
        direction = 1 if dest_col > king_col else -1
        pass_col  = king_col + direction
        if board.is_square_attacked(row, pass_col, opp):
            return False
        # Destination checked by full simulation in legal_moves
        return True


# ---------------------------------------------------------------------------
# Notation parser
# ---------------------------------------------------------------------------

class NotationParser:
    """
    Parse user input (various chess notation styles) into Move objects.
    Supports:
      - Coordinate: e2e4, E2E4
      - Short pawn push: e4
      - Piece move: Nf3, Bxe5, Rxd1
      - Pawn capture: exd5, exd5
      - Castling: O-O, O-O-O, 0-0, 0-0-0
    """

    PIECE_CHARS = {"N": Knight, "B": Bishop, "R": Rook, "Q": Queen, "K": King}

    @classmethod
    def parse(cls, text: str, board: Board, color: str) -> Optional[Move]:
        text = text.strip()

        # Castling
        if text in ("O-O", "0-0"):
            return cls._find_castling(board, color, kingside=True)
        if text in ("O-O-O", "0-0-0"):
            return cls._find_castling(board, color, kingside=False)

        legal = MoveGenerator.legal_moves(board, color)

        # Coordinate notation: e2e4 (possibly with promotion suffix Q/R/B/N)
        clean = text.rstrip("QRBNqrbn")
        if len(clean) == 4 and all(
            clean[0] in FILE_LETTERS and clean[1] in RANK_NUMBERS and
            clean[2] in FILE_LETTERS and clean[3] in RANK_NUMBERS
            for _ in [0]
        ):
            try:
                fr, fc = sq_to_coords(clean[:2])
                tr, tc = sq_to_coords(clean[2:])
                for m in legal:
                    if m.from_sq == (fr, fc) and m.to_sq == (tr, tc):
                        return m
            except (ValueError, IndexError):
                pass

        # Piece move:  Nf3, Bxe5, Rxd1+, Rxd1#, Qxh7, Nbd2, R1a3  …
        # Strip check/mate symbols and 'x'
        stripped = text.rstrip("+#!?").replace("x", "")

        if stripped and stripped[0] in cls.PIECE_CHARS:
            PieceClass = cls.PIECE_CHARS[stripped[0]]
            dest_str   = stripped[-2:]
            disambig   = stripped[1:-2]   # e.g. 'b' (file) or '1' (rank) or ''
            try:
                tr, tc = sq_to_coords(dest_str)
            except (ValueError, IndexError):
                return None
            candidates = [
                m for m in legal
                if (isinstance(board.grid[m.from_sq[0]][m.from_sq[1]], PieceClass)
                    and m.to_sq == (tr, tc))
            ]
            # Apply disambiguation
            if disambig:
                if disambig in FILE_LETTERS:
                    candidates = [m for m in candidates if m.from_sq[1] == file_to_col(disambig)]
                elif disambig in RANK_NUMBERS:
                    candidates = [m for m in candidates if m.from_sq[0] == rank_to_row(disambig)]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                return candidates[0]   # ambiguous but best-effort
            return None

        # Pawn capture: exd5, cxb4
        if len(stripped) >= 3 and stripped[0] in FILE_LETTERS and stripped[1:3] == stripped[1:3]:
            # e.g. "exd5"
            cap = stripped.rstrip("+#")
            if len(cap) >= 3 and cap[0] in FILE_LETTERS:
                dest = cap[-2:]
                try:
                    tr, tc = sq_to_coords(dest)
                    from_col = file_to_col(cap[0])
                except (ValueError, IndexError):
                    return None
                candidates = [
                    m for m in legal
                    if (isinstance(board.grid[m.from_sq[0]][m.from_sq[1]], Pawn)
                        and m.from_sq[1] == from_col
                        and m.to_sq == (tr, tc))
                ]
                if candidates:
                    return candidates[0]

        # Short pawn push: e4, d5
        if len(stripped) == 2 and stripped[0] in FILE_LETTERS and stripped[1] in RANK_NUMBERS:
            try:
                tr, tc = sq_to_coords(stripped)
            except (ValueError, IndexError):
                return None
            candidates = [
                m for m in legal
                if (isinstance(board.grid[m.from_sq[0]][m.from_sq[1]], Pawn)
                    and m.to_sq == (tr, tc))
            ]
            if candidates:
                return candidates[0]

        return None

    @classmethod
    def _find_castling(cls, board: Board, color: str, kingside: bool) -> Optional[Move]:
        """Find the castling move in the legal move list."""
        row = 0 if color == WHITE else 7
        dest_col = 6 if kingside else 2
        legal = MoveGenerator.legal_moves(board, color)
        for m in legal:
            kr, kc = board.king_position(color)
            if m.from_sq == (row, 4) and m.to_sq == (row, dest_col):
                return m
        return None


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    """Represents a human or AI player."""

    def __init__(self, color: str, is_ai: bool = False):
        self.color  = color
        self.is_ai  = is_ai

    def get_move(self, board: Board, legal: List[Move]) -> Optional[Move]:
        if self.is_ai:
            return self._ai_move(board, legal)
        return None   # Human: handled in game loop

    def _ai_move(self, board: Board, legal: List[Move]) -> Move:
        """Simple AI: prefer captures, then push to centre, else random."""
        # Greedy: pick any capturing move first
        captures = [m for m in legal if board.grid[m.to_sq[0]][m.to_sq[1]] is not None]
        if captures:
            return random.choice(captures)
        return random.choice(legal)


# ---------------------------------------------------------------------------
# Game controller
# ---------------------------------------------------------------------------

class Game:
    """
    Orchestrates game flow: turn management, move validation, check/mate/draw
    detection, and the console interface.
    """

    def __init__(self, white_is_ai: bool = False, black_is_ai: bool = False):
        self.board      = Board()
        self.players    = {
            WHITE: Player(WHITE, white_is_ai),
            BLACK: Player(BLACK, black_is_ai),
        }
        self.turn       = WHITE
        self.move_history: List[Move]  = []
        self.position_history: List[str] = []   # for threefold repetition
        self.halfmove_clock = 0   # for 50-move rule (not yet enforced as draw)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start and run the game until conclusion."""
        print("\n" + "="*50)
        print("       CONSOLE CHESS  ♟  Pure Python")
        print("="*50)
        print("Move formats: e2e4 | Nf3 | exd5 | O-O | O-O-O")
        print("Commands:     legal | undo | quit")
        print("="*50)

        while True:
            self.board.display(perspective=self.turn)
            legal = MoveGenerator.legal_moves(self.board, self.turn)

            # --- Terminal conditions ---
            if not legal:
                if self.board.in_check(self.turn):
                    winner = opponent(self.turn)
                    print(f"  ♛  CHECKMATE! {winner.capitalize()} wins!\n")
                else:
                    print("  ½  STALEMATE! Draw.\n")
                break

            if self._is_draw_by_repetition():
                print("  ½  DRAW by threefold repetition.\n")
                break

            if self._is_insufficient_material():
                print("  ½  DRAW by insufficient material.\n")
                break

            # --- Check announcement ---
            if self.board.in_check(self.turn):
                print(f"  ⚠  {self.turn.capitalize()} is in CHECK!")

            print(f"  Turn: {self.turn.capitalize()}  "
                  f"({'AI' if self.players[self.turn].is_ai else 'Human'})")

            # --- Get move ---
            player = self.players[self.turn]
            if player.is_ai:
                move = player.get_move(self.board, legal)
                print(f"  AI plays: {move.uci()}")
            else:
                move = self._prompt_human(legal)
                if move is None:
                    break   # quit

            # --- Apply move ---
            self._apply_and_advance(move)

    # ------------------------------------------------------------------
    # Human input
    # ------------------------------------------------------------------

    def _prompt_human(self, legal: List[Move]) -> Optional[Move]:
        """Prompt human for a move; return Move or None (quit)."""
        while True:
            try:
                text = input("  Enter move: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                return None

            if not text:
                continue

            lower = text.lower()

            if lower in ("quit", "exit", "q"):
                print("  Goodbye!")
                return None

            if lower in ("legal", "moves", "l"):
                self._print_legal(legal)
                continue

            if lower == "undo":
                if len(self.move_history) >= 2:
                    self._undo_last_two()
                    return self._prompt_human(
                        MoveGenerator.legal_moves(self.board, self.turn)
                    )
                elif len(self.move_history) == 1:
                    self._undo_one()
                    return self._prompt_human(
                        MoveGenerator.legal_moves(self.board, self.turn)
                    )
                else:
                    print("  No moves to undo.")
                    continue

            move = NotationParser.parse(text, self.board, self.turn)
            if move is None:
                print(f"  Invalid move '{text}'. Try 'legal' to see valid moves.")
                continue

            return move

    def _print_legal(self, legal: List[Move]):
        uci_list = sorted(m.uci() for m in legal)
        print("  Legal moves (" + str(len(uci_list)) + "):")
        # Print in rows of 8
        for i in range(0, len(uci_list), 8):
            print("    " + "  ".join(uci_list[i:i+8]))

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def _apply_and_advance(self, move: Move):
        """Apply move, record history, switch turn."""
        self.position_history.append(self.board.fingerprint())
        self.board.apply_move(move)
        self.move_history.append(move)
        self.turn = opponent(self.turn)

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _undo_one(self):
        if not self.move_history:
            return
        move = self.move_history.pop()
        self.board.undo_move(move)
        if self.position_history:
            self.position_history.pop()
        self.turn = opponent(self.turn)
        print("  Move undone.")

    def _undo_last_two(self):
        """Undo two half-moves so the same human gets to move again."""
        self._undo_one()
        self._undo_one()

    # ------------------------------------------------------------------
    # Draw conditions
    # ------------------------------------------------------------------

    def _is_draw_by_repetition(self) -> bool:
        """Threefold repetition: same position 3+ times."""
        fp = self.board.fingerprint()
        count = self.position_history.count(fp)
        return count >= 2   # current position is the 3rd occurrence

    def _is_insufficient_material(self) -> bool:
        """
        Draws when neither side can possibly checkmate:
          K vs K, K+B vs K, K+N vs K, K+B vs K+B (same colour bishops)
        """
        pieces = {WHITE: [], BLACK: []}
        for color in (WHITE, BLACK):
            for piece, _, _ in self.board.pieces(color):
                pieces[color].append(type(piece).__name__)

        def _has_only(color, *allowed):
            return all(p in allowed for p in pieces[color])

        w = sorted(pieces[WHITE])
        b = sorted(pieces[BLACK])

        # K vs K
        if w == ["King"] and b == ["King"]:
            return True
        # K+minor vs K
        if set(w) <= {"King","Bishop","Knight"} and len(w) == 2 and b == ["King"]:
            return True
        if set(b) <= {"King","Bishop","Knight"} and len(b) == 2 and w == ["King"]:
            return True
        # K+B vs K+B same-colour bishops (simplified: just check 2 bishops total)
        if (w == ["Bishop","King"] and b == ["Bishop","King"]):
            # Check bishop square colors
            def bishop_sq_color(color):
                for piece, r, c in self.board.pieces(color):
                    if isinstance(piece, Bishop):
                        return (r + c) % 2
                return None
            if bishop_sq_color(WHITE) == bishop_sq_color(BLACK):
                return True

        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nWelcome to Console Chess!")
    print("─" * 30)
    print("Game modes:")
    print("  1. Human vs Human")
    print("  2. Human (White) vs AI (Black)")
    print("  3. AI (White) vs Human (Black)")
    print("  4. AI vs AI  (demo)")
    while True:
        choice = input("Choose mode [1-4]: ").strip()
        if choice == "1":
            game = Game(white_is_ai=False, black_is_ai=False)
            break
        elif choice == "2":
            game = Game(white_is_ai=False, black_is_ai=True)
            break
        elif choice == "3":
            game = Game(white_is_ai=True,  black_is_ai=False)
            break
        elif choice == "4":
            game = Game(white_is_ai=True,  black_is_ai=True)
            break
        elif choice.lower() in ("quit","exit","q"):
            sys.exit(0)
        else:
            print("Please enter 1, 2, 3, or 4.")

    try:
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")


if __name__ == "__main__":
    main()
