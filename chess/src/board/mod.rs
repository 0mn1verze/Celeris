//! # Module: `board`
//!
//! This module defines the core data structures and functionalities for representing a chess board
//! and managing its state. It includes the `Board` struct, which encapsulates the entire state of
//! a chess game, and the `BoardState` struct, which stores the state of the board at a specific
//! point in time.
//!
//! ## Overview
//!
//! The `board` module is a fundamental part of the chess engine, providing the basic building
//! blocks for representing the game state, managing piece positions, handling move history, and
//! enforcing game rules. It uses bitboards for efficient storage and manipulation of piece
//! locations and provides methods for accessing and modifying the board state.
//!
//! ## Key Components
//!
//! - **`Board`**: A struct representing the complete state of a chess game.
//!   - Contains bitboards for piece positions, occupied squares, and side to move.
//!   - Manages move counters, castling rights, en passant square, and game history.
//!   - Provides methods for accessing piece positions, occupied squares, and other board state information.
//!   - Implements `Display` for printing the board to the console.
//! - **`BoardState`**: A struct representing the state of the board at a specific point in time.
//!   - Stores information about castling rights, en passant square, captured piece, and move counters.
//!   - Includes Zobrist keys for position hashing and repetition detection.
//!   - Contains bitboard masks for move generation and legality checks.
//!
//! ## Functionality
//!
//! - **Board Creation**: `Board::new()` creates an empty board, while `Board::default()` creates a board with the standard starting position.
//! - **State Management**: `Board::store_state()` and `Board::restore_state()` manage the move history, allowing for undoing moves.
//! - **Piece Access**: `Board::on()` retrieves the piece on a given square.
//! - **Bitboard Access**: `Board::piecetype_bb()`, `Board::occupied_bb()`, `Board::all_occupied_bb()
mod fen;
mod mask;
mod movegen;
mod movement;
mod zobrist;

pub use fen::{KILLER_FEN, START_FEN, TRICKY_FEN};
pub use movegen::{
    CaptureGen, LegalGen, MoveList, QuietGen, attacks, bishop_attacks, king_attack, knight_attack,
    pawn_attack, queen_attacks, rook_attacks, sq_dist,
};
pub use zobrist::KeyBundle;

use crate::core::*;

/******************************************\
|==========================================|
|                Constants                 |
|==========================================|
\******************************************/

pub const MAX_MOVES: usize = 256;

/******************************************\
|==========================================|
|               Board State                |
|==========================================|
\******************************************/

/// # Board State (Used for move generation and history)
///
/// Stores the state of the board at a given point in time.
/// This is used to restore the board to a previous state after a move has been made (unmake move)
/// and to check for game rules like repetitions. It also caches information useful for move generation.
///
/// Caches the variables that cannot be rolled back in a predictable pattern, for example, the previously captured piece has to be stored or else it will be lost to history
///
/// ## Fields
/// - `enpassant`: The en passant square, if any, available on this turn. `None` if no en passant capture is possible.
/// - `castle`: The castling rights (`Castling`) available *before* the move that led to this state.
/// - `key`: The Zobrist key representing the board position (excluding pawn structure and move counters). Used for repetition detection.
/// - `pawn_key`: The Zobrist key specifically for the pawn structure. Potentially useful for pawn structure evaluation.
/// - `captured`: The `Piece` captured on the move leading to this state. `Piece::None` if no capture occurred. Essential for `unmake_move`.
/// - `repetitions`: The number of times this exact position (`key`) has been repeated in the game history up to this point.
/// - `fifty_move`: Counter for the fifty-move rule (number of plies since the last pawn move or capture).
/// - `check_mask`: A bitboard indicating squares that block or capture checking pieces. If not in check, this is typically all squares (`!0`).
/// - `diag_pin`: A bitboard representing squares occupied by pieces pinned diagonally to their king.
/// - `hv_pin`: A bitboard representing squares occupied by pieces pinned horizontally or vertically to their king.
/// - `king_ban`: A bitboard representing squares the king cannot move to (attacked squares or squares occupied by friendly pieces).
/// - `king_attacks`: A bitboard representing the squares attacked by the friendly king.
/// - `available`: A bitboard representing all squares not occupied by the side to move (potential destinations, excluding captures).
/// - `enpassant_pin`: A boolean indicating if the pawn performing an en passant capture is pinned to the king, making the en passant illegal.
#[derive(Debug, Default, Clone)]
pub struct BoardState {
    // --- Board State Variables -- //
    /// Stores gaps between last repeat, if negative it means three fold repetition
    pub repetitions: i8,
    /// Counter for the fifty-move rule (plies since last pawn move or capture).
    fifty_move: u8,
    /// The piece that was captured on the move leading to this state. `Piece::None` if no capture.
    captured: Option<Piece>, // Assuming Piece::None exists
    /// The square where an en passant capture is possible, if any. `None` otherwise.
    enpassant: Option<Square>,
    /// Castling rights (`Castling`) available *before* the move leading to this state.
    castle: Castling,
    /// Zobrist keys for the current position
    keys: KeyBundle,

    // --- Move generation masks ---
    /// Bitboard mask: squares that block check or capture the checking piece(s). If not in check, all squares (`!0`).
    check_mask: Bitboard,
    /// Bitboard mask: squares occupied by pieces pinned diagonally to their king.
    diag_pin: Bitboard,
    /// Bitboard mask: squares occupied by pieces pinned horizontally or vertically to their king.
    hv_pin: Bitboard,
    /// Bitboard mask: squares the king cannot move to (attacked or occupied by friendly pieces).
    attacked: Bitboard,
    /// Enpassant pin: whether enpassant pawn is pinned and cannot be taken without revealing a check
    ep_pin: bool,
}

impl BoardState {
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

/// # Chess Board Representation
///
/// Represents the complete state of a chess game at a specific point in time.
/// It encapsulates piece positions, castling rights, the side to move, move counters,
/// and a history of previous states for undoing moves and rule checking (like repetitions).
///
/// This struct is central to the chess engine's logic, providing the foundation for
/// move generation, move execution, and game state evaluation.
///
/// ## Representation
/// The board primarily uses `Bitboard`s for efficient storage and manipulation of piece locations.
///
/// ## Fields
/// - `pieces`: An array of `Bitboard`s, indexed by `PieceType`. Each bitboard (`pieces[PieceType::Pawn.index()]`)
///   stores the locations of all pieces of that specific type, regardless of colour.
/// - `occupied`: An array of `Bitboard`s, indexed by `Colour`. Each bitboard (`occupied[Colour::White.index()]`)
///   stores the locations of all pieces belonging to that specific colour. `occupied[0] | occupied[1]` gives all occupied squares.
/// - `side_to_move`: The `Colour` (White or Black) whose turn it is to move.
/// - `half_moves`: Counts the number of half-moves (plies) since the last capture or pawn advance.
///   Used for the fifty-move rule. It's incremented after each move and reset to 0 upon a capture or pawn move.
/// - `state`: The current `BoardState` of the board. This includes en passant square, castling rights,
///   captured piece, Zobrist keys, and move counters.
/// - `history`: A `Vec<BoardState>` acting as a stack. Each time a move is made, the current `BoardState`
///   (containing castling rights, en passant square, captured piece, keys, etc., *before* the move)
///   is pushed onto this stack. This allows for efficient `unmake_move` operations and tracking game history
///   for rules like three-fold repetition.
#[derive(Debug, Clone)]
pub struct Board {
    /// Array representing the board, where each element corresponds to a square.
    /// `board[Square::A1.index()]` holds the `Piece` on A1.
    /// `Piece::None` indicates an empty square.
    board: [Option<Piece>; Square::NUM],

    /// Bitboards for each piece type (Pawn, Knight, Bishop, Rook, Queen, King).
    /// `pieces[PieceType::Pawn.index()]` holds a bitboard of all pawns (both colours).
    pub pieces: [Bitboard; PieceType::NUM],

    /// Bitboards for all pieces of each colour.
    /// `occupied[Colour::White.index()]` holds all white pieces.
    /// `occupied[Colour::Black.index()]` holds all black pieces.
    pub occupied: [Bitboard; Colour::NUM],

    /// Castling Masks for each square, stores the original rook squares for chess960,
    castling_mask: CastlingMask,

    /// Counts the number of half-moves (plies) since the last capture or pawn advance.
    /// Used for the fifty-move rule. Reset to 0 on capture or pawn move.
    half_moves: u16, // Renamed from fifty_move in BoardState for clarity, common practice

    /// Indicates which player's turn it is (White or Black).
    stm: Colour,

    /// Current board state
    pub state: BoardState,
    // /// A stack-like structure storing previous board states (`BoardState`).
    // /// Used to undo moves (`unmake_move`) and track game history (e.g., for repetition checks).
    // history: UndoHistory<MAX_MOVES>,
    history: Vec<BoardState>,
}

/******************************************\
|==========================================|
|           Basic Implementation           |
|==========================================|
\******************************************/

/// # Default Board
///
/// - Creates a new board with the starting position
impl Default for Board {
    fn default() -> Board {
        let mut board = Board::new();
        board.set(START_FEN).unwrap();
        board
    }
}

impl Board {
    /// # New Board
    ///
    /// - Creates a new empty board, initialising the history stack
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
        }
    }

    /// # Get Piece on Square
    ///
    /// ## Arguments
    ///
    // / * `square` - The square to get the piece from    ///
    /// ## Returns
    ///
    /// * `Option<Piece>` - The piece on the square, or `None` if the square is empty
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.on(Square::A1), Some(Piece::WhiteRook));
    /// assert_eq!(board.on(Square::A2), Some(Piece::WhitePawn));
    /// assert_eq!(board.on(Square::A3), None);
    /// ```
    #[inline]
    pub fn on(&self, square: Square) -> Option<Piece> {
        unsafe { *self.board.get_unchecked(square.index()) }
    }

    /// Unsafe helper to get the piece on a square, assuming the square is occupied.
    ///
    /// # Safety
    /// Calling this on an empty square results in undefined behavior.
    #[inline]
    pub unsafe fn on_unchecked(&self, square: Square) -> Piece {
        unsafe { self.board[square.index()].unwrap_unchecked() }
    }

    /// # Get Piece Type Bitboard
    ///
    /// ## Argument
    ///
    /// * `piecetype` - The piece type to get the bitboard of
    ///
    /// ## Returns
    ///
    /// * `Bitboard` - The bitboard of the piece type
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.piecetype_bb(PieceType::Pawn), Bitboard::from([
    ///     Square::A2, Square::B2, Square::C2, Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    ///     Square::A7, Square::B7, Square::C7, Square::D7, Square::E7, Square::F7, Square::G7, Square::H7,
    /// ]));
    /// assert_eq!(board.piecetype_bb(PieceType::Rook), Bitboard::from([
    ///     Square::A1, Square::H1, Square::A8, Square::H8,
    /// ]));
    #[inline]
    pub fn piecetype_bb(&self, piecetype: PieceType) -> Bitboard {
        unsafe { *self.pieces.get_unchecked(piecetype.index()) }
    }

    /// # Get Occupied Bitboard
    ///
    /// ## Argument
    ///
    /// * `colour` - The colour to get the occupied bitboard of
    ///
    /// ## Returns
    ///
    /// * `Bitboard` - The bitboard of the occupied squares of the colour
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.occupied_bb(Colour::White), Bitboard::from([
    ///     Square::A1, Square::B1, Square::C1, Square::D1, Square::E1, Square::F1, Square::G1, Square::H1,
    ///     Square::A2, Square::B2, Square::C2, Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    /// ]));
    /// assert_eq!(board.occupied_bb(Colour::Black), Bitboard::from([
    ///     Square::A8, Square::B8, Square::C8, Square::D8, Square::E8, Square::F8, Square::G8, Square::H8,
    ///     Square::A7, Square::B7, Square::C7, Square::D7, Square::E7, Square::F7, Square::G7, Square::H7,
    /// ]));
    #[inline]
    pub fn occupied_bb(&self, colour: Colour) -> Bitboard {
        unsafe { *self.occupied.get_unchecked(colour.index()) }
    }

    /// # Get Total Occupied Bitboard
    ///
    /// ## Returns
    ///
    /// * `Bitboard` - The bitboard of the occupied squares
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.occupied_bb(Colour::White), Bitboard::from([
    ///     Square::A1, Square::B1, Square::C1, Square::D1, Square::E1, Square::F1, Square::G1, Square::H1,
    ///     Square::A2, Square::B2, Square::C2, Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    /// ] | Bitboard::from([
    ///     Square::A8, Square::B8, Square::C8, Square::D8, Square::E8, Square::F8, Square::G8, Square::H8,
    ///     Square::A7, Square::B7, Square::C7, Square::D7, Square::E7, Square::F7, Square::G7, Square::H7,
    /// ]));
    /// ```
    #[inline]
    pub fn all_occupied_bb(&self) -> Bitboard {
        self.occupied_bb(Colour::White) | self.occupied_bb(Colour::Black)
    }

    /// # Get Piece Bitboard
    ///
    /// ## Argument
    ///
    /// * `piece` - The piece to get the bitboard of
    /// ## Returns
    ///
    /// * `Bitboard` - The bitboard of the piece
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.piece_bb(Piece::WhitePawn), Bitboard::from([
    ///     Square::A2, Square::B2, Square::C2, Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    /// ]));
    /// assert_eq!(board.piece_bb(Piece::BlackRook), Bitboard::from([
    ///     Square::A8, Square::H8,
    /// ]));
    #[inline]
    pub fn piece_bb(&self, col: Colour, pt: PieceType) -> Bitboard {
        self.piecetype_bb(pt) & self.occupied_bb(col)
    }

    /// # Get Side To Move
    ///
    /// ## Returns
    ///
    /// * `Colour` - The colour of the side to move
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.side_to_move(), Colour::White);
    /// ```
    #[inline]
    pub fn stm(&self) -> Colour {
        self.stm
    }

    /// # Get half moves
    ///
    /// ## Returns
    ///
    /// * `u16` - The number of half moves since the last capture or pawn move
    ///
    /// ## Example
    ///
    /// ```
    /// use chess::core::*;
    /// let board = Board::default();
    /// assert_eq!(board.half_moves(), 0);
    /// ```
    #[inline]
    pub fn half_moves(&self) -> u16 {
        self.half_moves
    }

    /// # Get Castling Rights
    ///
    /// ## Arguments
    ///
    /// * `Square` - The square to get the castling rights from
    ///
    /// ## Returns
    ///
    /// * `Castling` - The castling rights masks that denoted the remaining possible castle rights after a piece has moved from the square.
    #[inline]
    pub(crate) fn castling_rights(&self, square: Square) -> Castling {
        unsafe { *self.castling_mask.castling.get_unchecked(square.index()) }
    }

    /// # Get Enpassant Square
    ///
    /// ## Returns
    ///
    /// * `Square` - The square where enpassant capture is possible
    #[inline]
    pub fn ep(&self) -> Option<Square> {
        self.state.enpassant
    }

    /// # Get Enpassant Target Square
    ///
    /// ## Returns
    ///
    /// * `Square` - The square of the pawn that can be enpassant captured
    #[inline]
    pub fn ep_target(&self) -> Option<Square> {
        self.state
            .enpassant
            .map(|sq| unsafe { sq.add_unchecked(-self.stm.forward()) })
    }

    /// # Get Rook Squares
    ///
    /// ## Returns
    ///
    /// * `[Square; 4]` - The squares where the rooks start from, in order of white king side, white queen side, black king side, black queen side
    #[inline]
    pub(crate) unsafe fn rook_sq(&self, index: usize) -> Square {
        unsafe {
            self.castling_mask
                .rook_sq
                .get_unchecked(index)
                .unwrap_unchecked()
        }
    }

    /// # Get Castling Rights
    ///
    /// ## Returns
    ///
    /// * `Castling` - The castling rights for the current position
    #[inline]
    pub fn castling(&self) -> Castling {
        self.state.castle
    }

    /// # Get Castling Rights for one side
    ///
    /// ## Returns
    ///
    /// * `Castling` - The castling rights for the current position for the side
    #[inline]
    pub fn castling_side(&self, side: Colour) -> Castling {
        self.state.castle
            & match side {
                Colour::White => Castling::WHITE_CASTLING,
                Colour::Black => Castling::BLACK_CASTLING,
            }
    }

    /// # Get Hash Key
    ///
    /// ## Returns
    ///
    /// * `Key` - The zobrist key for this board
    #[inline]
    pub fn key(&self) -> u64 {
        self.state.keys.key
    }

    /// # Get Pawn Hash Key
    ///
    /// ## Returns
    ///
    /// * `Key` - The zobrist key for the pawns on this board
    #[inline]
    pub fn pawn_key(&self) -> u64 {
        self.state.keys.pawn_key
    }

    /// # Get Non Pawn Hash Key
    ///
    /// ## Returns
    ///
    /// * `[Key; Colour::NUM]` - The zobrist key for the non pawns on this board
    #[inline]
    pub fn non_pawn_keys(&self) -> [u64; Colour::NUM] {
        self.state.keys.non_pawn_key
    }

    /// # Get Non Pawn Hash Key
    ///
    /// ## Arguments
    ///
    /// * `col` - Colour of the non pawn key
    ///
    /// ## Returns
    ///
    /// * `Key` - The zobrist key for the non pawns on this board for the colour in the argument
    #[inline]
    pub fn non_pawn_key(&self, col: Colour) -> u64 {
        self.state.keys.non_pawn_key[col.index()]
    }

    /// Returns whether the king is in check
    #[inline]
    pub fn in_check(&self) -> bool {
        self.state.check_mask != Bitboard::FULL
    }

    /// Returns whether the position is a draw
    #[inline]
    pub fn is_draw(&self, ply: u16) -> bool {
        let move_list = MoveList::new();

        if self.state.fifty_move > 99 && (!self.in_check() || move_list.len() == 0) {
            return true;
        }

        return self.state.repetitions != 0 && self.state.repetitions < (ply as i8);
    }

    /// Returns whether the file is semi open
    #[inline]
    pub fn is_semi_open_file(&self, col: Colour, sq: Square) -> bool {
        (sq.file().bb() & self.piece_bb(col, PieceType::Pawn)).is_empty()
    }

    /// Returns whether the file is open
    #[inline]
    pub fn is_open_file(&self, col: Colour, sq: Square) -> bool {
        (sq.file().bb() & self.piecetype_bb(PieceType::Pawn)).is_empty()
    }

    /// Helper to get the attackers to a square.
    pub fn attackers_to(&self, to: Square, occ: Bitboard) -> Bitboard {
        use crate::core::{Colour::*, PieceType::*};
        pawn_attack(White, to) & self.piece_bb(Black, Pawn)
            | pawn_attack(Black, to) & self.piece_bb(White, Pawn)
            | knight_attack(to) & self.piecetype_bb(Knight)
            | bishop_attacks(to, occ) & (self.piecetype_bb(Bishop) | self.piecetype_bb(Queen))
            | rook_attacks(to, occ) & (self.piecetype_bb(Rook) | self.piecetype_bb(Queen))
            | king_attack(to) & self.piecetype_bb(King)
    }
}

impl std::fmt::Display for Board {
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
