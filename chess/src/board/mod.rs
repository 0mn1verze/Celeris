//! Represents the chess board, its state, and associated logic.
//!
//! This module is central to the chess engine, defining how a chess position
//! is stored and manipulated. It includes:
//! - The `Board` struct: the main representation of a chess game's state.
//! - The `BoardState` struct: stores per-move state information for undoing moves and checking rules.
//! - FEN parsing and generation.
//! - Move generation and application (delegated to submodules).
//! - Zobrist hashing for position identification.
pub mod fen;
pub mod mask;
pub mod movegen;
pub mod movement;
pub mod zobrist;

pub use fen::{KILLER_FEN, START_FEN, TRICKY_FEN};
pub use movegen::{
    CaptureGen, LegalGen, MoveList, QuietGen, attacks, bishop_attacks, king_attack, knight_attack,
    pawn_attack, queen_attacks, rook_attacks, sq_dist,
};
pub use zobrist::{Key, KeyBundle};

use crate::core::*;
use mask::CastlingMask;

/******************************************\
|==========================================|
|                Constants                 |
|==========================================|
\******************************************/

/// Maximum number of moves expected in a typical game, used for pre-allocating history.
pub const MAX_MOVES: usize = 256;

/******************************************\
|==========================================|
|               Board State                |
|==========================================|
\******************************************/

/// Stores the state of the board that changes with each move and needs to be restored upon undoing a move.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BoardState {
    /// Repetition counter.
    /// - `0`: This position has not been seen before in a way that counts for repetition, or it's the first time.
    /// - `> 0`: This position is a 2nd occurrence. The value is an index related to how far back the 1st occurrence was.
    /// - `< 0`: This position is a 3rd (or more) occurrence. The absolute value is an index.
    pub repetitions: i8,

    /// Counter for the fifty-move rule. Incremented for each half-move, reset on pawn moves or captures.
    fifty_move: u8,

    /// The piece that was captured on the last move, if any. `None` if the last move was not a capture.
    captured: Option<Piece>,

    /// The en passant target square, if one exists.
    /// This is the square *behind* a pawn that has just made a two-square advance.
    /// It's the square an opposing pawn would move *to* in order to capture en passant.
    enpassant: Option<Square>,

    /// Current castling rights for both players.
    castle: Castling,

    /// Zobrist keys for the current position.
    keys: KeyBundle,

    /// Bitboard mask indicating squares that, if occupied by the king, would mean the king is in check.
    /// Or, for non-king moves, squares a piece must move to or block to resolve a check.
    /// `Bitboard::FULL` if not in check.
    check_mask: Bitboard,

    /// Bitboard of pieces that are pinned diagonally to their king.
    diag_pin: Bitboard,

    /// Bitboard of pieces that are pinned horizontally or vertically to their king.
    hv_pin: Bitboard,

    /// Bitboard of all squares attacked by the opponent.
    attacked: Bitboard,

    /// Flag indicating if an en passant capture would result in discovering a check on the current player's king.
    /// `true` if the en passant capture is pinned (illegal), `false` otherwise.
    ep_pin: bool,
}

impl BoardState {
    /// Creates a snapshot of the current state, primarily for storing in history.
    /// Note: Some fields like `repetitions` and pin/check masks are reset to default
    /// in the *new* current state after this snapshot is taken and the old state is pushed to history.
    /// The pushed historical state retains its original values for these fields.
    pub(super) fn snapshot(&self) -> Self {
        Self {
            fifty_move: self.fifty_move,
            castle: self.castle,
            keys: self.keys,
            enpassant: self.enpassant,
            ..Default::default()
        }
    }
}

/******************************************\
|==========================================|
|                  Board                   |
|==========================================|
\******************************************/

/// Represents the entire state of a chess game.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// An array representing the 64 squares of the board, storing `Some(Piece)` or `None`.
    board: [Option<Piece>; Square::NUM],

    /// Bitboards for each piece type (Pawn, Knight, etc.), irrespective of color.
    /// `pieces[PieceType::Pawn.index()]` gives a bitboard of all pawns.
    pub pieces: [Bitboard; PieceType::NUM],

    /// Bitboards for all pieces of a specific color.
    /// `occupied[Colour::White.index()]` gives a bitboard of all white pieces.
    pub occupied: [Bitboard; Colour::NUM],

    /// Mask used for determining castling rights, especially useful for Chess960.
    /// Stores initial rook squares and squares that, if moved from/to, revoke rights.
    castling_mask: CastlingMask,

    /// Total number of half-moves (plies) played since the start of the game.
    half_moves: u16,

    /// The color of the player whose turn it is to move.
    stm: Colour,

    /// Flag indicating if the game is Chess960. Affects castling rules.
    chess960: bool,

    /// The current, active state of the board (see `BoardState`).
    pub state: BoardState,

    /// A history of `BoardState` objects, used for undoing moves.
    history: Vec<BoardState>,
}

/******************************************\
|==========================================|
|           Basic Implementation           |
|==========================================|
\******************************************/

impl Default for Board {
    /// Creates a new `Board` initialized to the standard chess starting position.
    fn default() -> Board {
        let mut board = Board::new();
        board.set(START_FEN).unwrap();
        board
    }
}

impl Board {
    /// Creates a new, empty `Board` structure.
    pub(crate) fn new() -> Board {
        Board {
            board: [None; Square::NUM],
            pieces: [Bitboard::EMPTY; PieceType::NUM],
            occupied: [Bitboard::EMPTY; Colour::NUM],
            castling_mask: CastlingMask::default(),
            stm: Colour::White,
            half_moves: 0,
            state: BoardState::default(),
            history: Vec::with_capacity(MAX_MOVES),
            chess960: false,
        }
    }

    /// Sets whether the board should follow Chess960 rules.
    pub fn set_chess960(&mut self, chess960: bool) {
        self.chess960 = chess960;
    }

    /// Returns `true` if the board is set to Chess960 mode, `false` otherwise.
    pub fn chess960(&self) -> bool {
        self.chess960
    }

    /// Returns the piece on the given square, if any.
    #[inline]
    pub fn on(&self, square: Square) -> Option<Piece> {
        unsafe { *self.board.get_unchecked(square.index()) }
    }

    /// Returns the piece on the given square, without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `square` is a valid square index.
    /// Panics if the square is empty.
    #[inline]
    pub unsafe fn on_unchecked(&self, square: Square) -> Piece {
        unsafe { self.board[square.index()].unwrap_unchecked() }
    }

    /// Returns a bitboard of all pieces of the given `PieceType`, regardless of color.
    #[inline]
    pub fn piecetype_bb(&self, piecetype: PieceType) -> Bitboard {
        unsafe { *self.pieces.get_unchecked(piecetype.index()) }
    }

    /// Returns a bitboard of all pieces of the given `Colour`.
    #[inline]
    pub fn occupied_bb(&self, colour: Colour) -> Bitboard {
        unsafe { *self.occupied.get_unchecked(colour.index()) }
    }

    /// Returns a bitboard of all occupied squares on the board.
    #[inline]
    pub fn all_occupied_bb(&self) -> Bitboard {
        self.occupied_bb(Colour::White) | self.occupied_bb(Colour::Black)
    }

    /// Returns a bitboard of pieces of a specific `Colour` and `PieceType`.
    #[inline]
    pub fn piece_bb(&self, col: Colour, pt: PieceType) -> Bitboard {
        self.piecetype_bb(pt) & self.occupied_bb(col)
    }

    /// Returns the `Colour` of the player whose turn it is to move.
    #[inline]
    pub fn stm(&self) -> Colour {
        self.stm
    }

    /// Returns the total number of half-moves (plies) played in the game.
    #[inline]
    pub fn half_moves(&self) -> u16 {
        self.half_moves
    }

    /// Returns the castling rights associated with a piece moving from or to a specific square.
    /// Used to determine if a move (e.g., king or rook move) affects castling availability.
    #[inline]
    pub(crate) fn castling_rights(&self, square: Square) -> Castling {
        unsafe { *self.castling_mask.castling.get_unchecked(square.index()) }
    }

    /// Returns the en passant target square, if one exists.
    /// This is the square *behind* a pawn that has just made a two-square advance.
    /// It's the square an opposing pawn would move *to* for an en passant capture (as per FEN standard).
    #[inline]
    pub fn ep(&self) -> Option<Square> {
        self.state.enpassant
    }

    /// Returns the square of the pawn that *would be captured* by an en passant move.
    /// This is different from `ep()`, which returns the square the capturing pawn moves *to*.
    /// For example, if White plays e2-e4, `ep()` returns `e3`. `stm` becomes Black.
    /// `ep_target()` would then return `e4` (e3 + North, as Black's `stm.forward()` is South).
    #[inline]
    pub fn ep_target(&self) -> Option<Square> {
        self.state
            .enpassant
            .map(|sq| unsafe { sq.add_unchecked(-self.stm.forward()) })
    }

    /// Given a castling right (e.g., `Castling::WK`), returns the initial square of the corresponding rook.
    /// Essential for Chess960 where rook positions can vary.
    #[inline]
    pub(crate) fn rook_sq(&self, rights: Castling) -> Square {
        unsafe { self.castling_mask.rook_sq[rights.0.trailing_zeros() as usize].unwrap_unchecked() }
    }

    /// Returns the current castling availability for both players.
    #[inline]
    pub fn castling(&self) -> Castling {
        self.state.castle
    }

    /// Stores the current `BoardState` into history and prepares the active `BoardState` for the next move.
    fn store_state(&mut self) {
        let state = self.state.snapshot();
        let old = std::mem::replace(&mut self.state, state);
        self.history.push(old);
    }

    /// Restores the previous state
    fn restore_state(&mut self) {
        // Restore the last saved state from history.
        self.state = self.history.pop().unwrap();
    }

    /// Returns the Zobrist key for the current board position.
    #[inline]
    pub fn key(&self) -> u64 {
        self.state.keys.key
    }

    /// Returns the Zobrist key component related to pawn structure.
    #[inline]
    pub fn pawn_key(&self) -> u64 {
        self.state.keys.pawn_key
    }

    /// Returns the Zobrist key components related to non-pawn material for both colors.
    #[inline]
    pub fn non_pawn_keys(&self) -> [u64; Colour::NUM] {
        self.state.keys.non_pawn_key
    }

    /// Returns the Zobrist key component related to non-pawn material for the specified color.
    #[inline]
    pub fn non_pawn_key(&self, col: Colour) -> u64 {
        self.state.keys.non_pawn_key[col.index()]
    }

    /// Returns `true` if the current side to move is in check.
    #[inline]
    pub fn in_check(&self) -> bool {
        self.state.check_mask != Bitboard::FULL
    }

    /// Checks if the current position is a draw by the fifty-move rule or threefold repetition.
    ///
    /// Check for fifty-move rule.
    /// According to FIDE rules, if the 50-move counter is >= 100, it's a draw,
    /// unless the move that reached this count was checkmate.
    #[inline]
    pub fn is_draw(&self, ply_from_null: u16) -> bool {
        // Avoid returning true when checkmate by ignoring positions that are in check, delegating them to later logic
        if self.state.fifty_move > 99 && !self.in_check() {
            return true;
        }
        // Check for threefold repetition.
        return self.state.repetitions != 0 && self.state.repetitions < ply_from_null as i8;
    }

    /// Checks if a file is semi-open for the given color (no pawns of that color on the file).
    #[inline]
    pub fn is_semi_open_file(&self, col: Colour, sq: Square) -> bool {
        (sq.file().bb() & self.piece_bb(col, PieceType::Pawn)).is_empty()
    }

    /// Checks if a file is open (no pawns of any color on the file).
    #[inline]
    pub fn is_open_file(&self, sq: Square) -> bool {
        (sq.file().bb() & self.piecetype_bb(PieceType::Pawn)).is_empty()
    }

    /// Returns a bitboard of all pieces attacking a given square `to`.
    pub fn attackers_to(&self, to: Square, occ: Bitboard) -> Bitboard {
        use crate::core::{Colour::*, PieceType::*};
        pawn_attack(White, to) & self.piece_bb(Black, Pawn)
            | pawn_attack(Black, to) & self.piece_bb(White, Pawn)
            | knight_attack(to) & self.piecetype_bb(Knight)
            | bishop_attacks(to, occ) & (self.piecetype_bb(Bishop) | self.piecetype_bb(Queen))
            | rook_attacks(to, occ) & (self.piecetype_bb(Rook) | self.piecetype_bb(Queen))
            | king_attack(to) & self.piecetype_bb(King)
    }

    /// Checks if the given color has any non-pawn material (excluding the king).
    pub fn has_non_pawn_material(&self, col: Colour) -> bool {
        (self.occupied_bb(col)
            & !(self.piecetype_bb(PieceType::Pawn) | self.piecetype_bb(PieceType::King)))
        .is_occupied()
    }
}

impl std::fmt::Display for Board {
    /// Formats the board for display, including board position, FEN string, and other state information.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SEPARATOR: &str = "\n     +---+---+---+---+---+---+---+---+";

        writeln!(f, "{}", SEPARATOR)?;

        for rank in Rank::iter().rev() {
            write!(f, " {}   |", rank as u8 + 1)?;

            for file in File::iter() {
                let square = Square::from_parts(file, rank);
                let cell = match self.on(square) {
                    Some(piece) => piece.to_string(),
                    None => " ".to_string(),
                };
                write!(f, " {} |", cell)?;
            }

            writeln!(f, "{}", SEPARATOR)?;
        }

        writeln!(f)?;
        writeln!(f, "       A   B   C   D   E   F   G   H")?;
        writeln!(f)?;
        writeln!(f, "Current Side: {:?}", self.stm())?;
        writeln!(f, "Castling: {}", self.state.castle)?;
        writeln!(
            f,
            "En Passant Square: {}",
            match self.state.enpassant {
                Some(square) => square.to_string(),
                None => "None".to_string(),
            }
        )?;
        writeln!(f, "Half Move Clock: {}", self.state.fifty_move)?;
        writeln!(f, "Full Move: {}", self.half_moves / 2 + 1)?;
        writeln!(f, "Fen: {}", self.fen())?;
        writeln!(f, "Key: {:#X}", self.state.keys.key)?;
        writeln!(f, "Pawn Key: {:#X}", self.state.keys.pawn_key)?;
        writeln!(
            f,
            "White Non Pawn Key: {:#X}",
            self.state.keys.non_pawn_key[Colour::White.index()]
        )?;
        writeln!(
            f,
            "Black Non Pawn Key: {:#X}",
            self.state.keys.non_pawn_key[Colour::Black.index()]
        )?;

        Ok(())
    }
}
