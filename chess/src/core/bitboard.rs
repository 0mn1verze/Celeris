//! # Module: `bitboard`
//!
//! This module defines the `Bitboard` struct and its associated methods for representing and
//! manipulating chess board states using bitwise operations. Bitboards are a highly efficient
//! way to represent sets of squares on a chess board, allowing for fast calculations and
//! operations.
//!
//! ## Overview
//!
//! A `Bitboard` is a 64-bit integer where each bit corresponds to a square on the chess board.
//! This representation enables the use of bitwise operations to perform set operations on
//! squares, such as union, intersection, and difference. It also allows for efficient
//! checking of whether a square is occupied or not.
//!
//! ## Key Components
//!
//! - **`Bitboard`**: A struct representing a bitboard.
//!   - Uses a `u64` to store the bitboard data.
//!   - Implements bitwise operations (AND, OR, XOR, NOT, SHL, SHR) for efficient manipulation.
//!   - Provides methods for setting, clearing, and toggling individual bits (squares).
//!   - Includes methods for finding the least significant bit (LSB) and most significant bit (MSB).
//!   - Supports iteration over set bits using `for_each()`.
//!   - Offers methods for counting set bits (`count_bits()`) and checking if the bitboard is empty (`is_empty()`).
//!   - Implements `Display` for printing the bitboard in a visual chess board format.
//!   - Provides constants for common bitboard patterns (e.g., `EMPTY`, `A1`, `RANK_1`, `FILE_A`).
//!   - Implements `From` for converting from `Square`, `Rank`, and `File` to `Bitboard`.
//!   - Implements `shift()` for moving bits in chess directions.
//!
//! ## Functionality
//!
//! - **Bitwise Operations**: `Bitboard` supports all standard bitwise operations, allowing for
//!   complex set manipulations.
//! - **Square Manipulation**: Methods like `set()`, `clear()`, and `toggle()` allow for easy
//!   modification of individual squares.
//! - **Bit Examination**: `lsb()` and `msb()` provide a way to find the first and last set bits,
//!   respectively. `pop_

use std::fmt;

use super::{Colour, Direction, File, Rank, Square};
use macros::{AriOps, BitManiOps, BitOps};

/// # Bitboard Representation
///
/// Represents a chess board using a 64-bit integer, where each bit corresponds to a square.
/// This allows for efficient board representation and operations using bitwise logic.
///
/// ## Visual Representation
///
/// ```text
///     +---+---+---+---+---+---+---+---+
/// 8   | * | * | * | * | * | * | * | * |    // * means empty square
///     +---+---+---+---+---+---+---+---+    // 1 means set bit
/// 7   | * | * | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 6   | * | * | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 5   | * | * | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 4   | * | * | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 3   | * | * | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 2   | 1 | 1 | * | * | * | * | * | * |
///     +---+---+---+---+---+---+---+---+
/// 1   | 1 | * | * | 1 | * | 1 | 1 | 1 |
///     +---+---+---+---+---+---+---+---+
///       A   B   C   D   E   F   G   H
///
/// Bitboard: 0x3e9
/// ```
///
/// ## Features
/// - Implements bitwise operations (AND, OR, XOR, NOT, SHL, SHR)
/// - Conversions from Square, Rank, and File
/// - Square manipulation: `set()`, `clear()`, `toggle()`, `get()`
/// - Bit examination: `lsb()`, `msb()`, `pop_lsb()`, `pop_msb()`
/// - Iteration: `for_each()`
/// - Counting: `count_bits()`, `is_empty()`
/// - Board movement: `shift()` for moving bits in chess directions
/// - Display: Pretty-printing of bitboards in a visual chess board format
///
/// ## Constants
/// - `EMPTY`: Empty bitboard (no bits set)
/// - `A1`: Bitboard with only the A1 square set
/// - `RANK_1`: Bitboard with all squares in rank 1 set
/// - `FILE_A`: Bitboard with all squares in file A set
///
/// ## Available Functions
/// ```rust,no_run
/// // Square manipulation
/// bitboard.set(square)    // Set a bit at the specified square
/// bitboard.clear(square)  // Clear a bit at the specified square
/// bitboard.toggle(square) // Toggle a bit at the specified square
/// bitboard.get(square)    // Check if a bit is set at the specified square
///
/// // Bit examination
/// bitboard.lsb()      // Get the least significant bit (lowest set square)
/// bitboard.msb()      // Get the most significant bit (highest set square)
/// bitboard.pop_lsb()  // Get and remove the least significant bit
/// bitboard.pop_msb()  // Get and remove the most significant bit
///
/// // Counting
/// bitboard.count_bits() // Count the number of bits set
/// bitboard.is_empty()   // Check if the bitboard is empty
///
/// // Iteration
/// bitboard.for_each(|square| {}) // Apply a function to each set square
///
/// // Movement
/// bitboard.shift(direction)   // Shift the bitboard in a chess direction (N, S, E, W, etc.)
/// ```
///
/// ## Implementation
/// Bitboards are manipulated using bitwise operations:
///
/// ```rust,no_run
/// // Create an empty bitboard
/// let empty = Bitboard::EMPTY;
///
/// // Create a bitboard with a single square set
/// let e4_bitboard = Square::E4.bb();
///
/// // Create a bitboard for a full rank
/// let rank_2 = Bitboard::from(Rank::Rank2);
///
/// // Check if a square is occupied
/// if e4_bitboard.get(Square::E4) {
///     println!("E4 is occupied");
/// }
///
/// // Set a square
/// let mut bitboard = Bitboard::EMPTY;
/// bitboard.set(Square::E4);
///
/// // Combine two bitboards
/// let combined = e4_bitboard | rank_2;
///
/// // Shift a bitboard in a chess direction
/// let moved_north = e4_bitboard.shift(Direction::N);  // Move one square north
/// let moved_knight = e4_bitboard.shift(Direction::NNE); // Knight-like move north-north-east
///
/// // Display the bitboard in a chess board format
/// println!("{}", bitboard);
///
/// // Iterate through all set squares
/// bitboard.for_each(|square| {
///     println!("Square: {:?}", square);
/// });
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, BitOps, BitManiOps, AriOps)]
pub struct Bitboard(pub u64);

/******************************************\
|==========================================|
|                Constants                 |
|==========================================|
\******************************************/

impl Bitboard {
    /// Empty bitboard (no bits set)
    pub const EMPTY: Bitboard = Bitboard(0);
    /// Full Bitboard (all bits set)
    pub const FULL: Bitboard = Self::EMPTY.not();
    /// Bitboard with only A1 square set
    pub const A1: Bitboard = Bitboard(1);
    /// Bitboard with all squares in rank 1 set
    pub const RANK_1: Bitboard = Bitboard(0x00000000000000ff);
    /// Bitboard with all squares in rank 8 set
    pub const RANK_8: Bitboard = Bitboard(0xff00000000000000);
    /// Bitboard with all squares in rank 2 set
    pub const RANK_12: Bitboard = Bitboard(0x000000000000ffff);
    /// Bitboard with all squares in rank 7 set
    pub const RANK_78: Bitboard = Bitboard(0xffff000000000000);
    /// Bitboard with all squares in file A set
    pub const FILE_A: Bitboard = Bitboard(0x0101010101010101);
    /// Bitboard with all squares in file H set
    pub const FILE_H: Bitboard = Bitboard(0x8080808080808080);
    /// Bitboard with all squares in file AB set
    const FILE_AB: Bitboard = Bitboard(0x303030303030303);
    /// Bitboard with all squares in file GH set
    const FILE_GH: Bitboard = Bitboard(0xC0C0C0C0C0C0C0C0);
}

/******************************************\
|==========================================|
|                Conversions               |
|==========================================|
\******************************************/

impl Square {
    pub const fn bb(&self) -> Bitboard {
        Bitboard(Bitboard::A1.0 << *self as u8)
    }
}

impl Rank {
    pub const fn bb(&self) -> Bitboard {
        Bitboard(Bitboard::RANK_1.0 << (8 * *self as u8))
    }
}

impl File {
    pub const fn bb(&self) -> Bitboard {
        Bitboard(Bitboard::FILE_A.0 << *self as u8)
    }
}

impl<const N: usize> From<[Square; N]> for Bitboard {
    /// Creates a `Bitboard` from a slice of `Square`s.
    ///
    /// This function sets the bits corresponding to each square in the input slice.
    ///    /// # Panics
    ///
    /// This function will panic if the input slice contains more than 64 squares.
    ///
    /// # Examples          
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let squares = [Square::E4, Square::D5, Square::A1];
    /// let bb = Bitboard::from(squares);
    ///
    /// assert!(bb.get(Square::E4));
    /// assert!(bb.get(Square::D5));
    /// assert!(bb.get(Square::A1));
    /// assert_eq!(bb.count_bits(), 3);
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `squares` - A slice of `Square`s to set in the bitboard.
    ///
    /// # Returns
    ///
    /// A `Bitboard` with the specified squares set.
    fn from(squares: [Square; N]) -> Bitboard {
        let mut bb = Bitboard::EMPTY;
        for square in squares {
            bb.set(square);
        }
        bb
    }
}

/******************************************\
|==========================================|
|         Bit Manipulation Functions       |
|==========================================|
\******************************************/

impl Bitboard {
    /// Returns the least significant bit (LSB) of the bitboard as a Square
    ///
    /// The LSB is the first (lowest) set bit in the bitboard's binary representation.
    /// This is useful for iterating through pieces or finding the "first" piece on a bitboard.
    ///
    /// # Returns
    ///
    /// * `Some(Square)` - The square corresponding to the least significant bit
    /// * `None` - If the bitboard is empty (no bits set)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    ///
    /// let e4_bitboard = Square::E4.bb();
    /// assert_eq!(e4_bitboard.lsb(), Some(Square::E4));
    /// ```
    #[inline]
    pub const fn lsb(&self) -> Option<Square> {
        match self.0 {
            0 => None,
            bits => Some(unsafe { Square::from_unchecked(bits.trailing_zeros() as u8) }),
        }
    }

    /// Returns the least significant bit (LSB) of the bitboard as a Square
    ///
    /// The LSB is the first (lowest) set bit in the bitboard's binary representation.
    /// This is useful for iterating through pieces or finding the "first" piece on a bitboard.
    ///
    /// **Warning**: Unsafe variation of `lsb()`
    ///
    /// # Returns
    ///
    /// * `Square` - The square corresponding to the least significant bit
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    ///
    /// let e4_bitboard = Square::E4.bb();
    /// assert_eq!(e4_bitboard.lsb_unchecked(), Square::E4);
    /// ```
    pub const unsafe fn lsb_unchecked(&self) -> Square {
        debug_assert!(self.0 != 0, "Bitboard is empty");
        unsafe { Square::from_unchecked(self.0.trailing_zeros() as u8) }
    }

    /// Returns the most significant bit (MSB) of the bitboard as a Square
    ///
    /// The MSB is the last (highest) set bit in the bitboard's binary representation.
    /// This is useful for finding the "last" or "highest" piece on a bitboard.
    ///
    /// # Returns
    ///
    /// * `Some(Square)` - The square corresponding to the most significant bit
    /// * `None` - If the bitboard is empty (no bits set)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    ///
    /// let multi_square = Square::E4.bb() | Square::G7.bb();
    /// assert_eq!(multi_square.msb(), Some(Square::G7)); // G7 is higher than E4
    ///
    /// let empty = Bitboard::EMPTY;
    /// assert_eq!(empty.msb(), None);
    /// ```
    #[inline]
    pub const fn msb(&self) -> Option<Square> {
        match self.0 {
            0 => None,
            bits => Some(unsafe { Square::from_unchecked(63 - bits.leading_zeros() as u8) }),
        }
    }

    /// Returns the least significant bit (LSB) and removes it from the bitboard
    ///
    /// This is an essential operation for efficiently iterating through pieces
    /// on a bitboard one at a time.
    ///
    /// # Returns
    ///
    /// * `Some(Square)` - The square corresponding to the removed least significant bit
    /// * `None` - If the bitboard is empty (no bits set)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut multi_square = Square::E4.bb() | Square::D2.bb();
    /// assert_eq!(multi_square.pop_lsb(), Some(Square::D2)); // D2 has lower index than E4
    /// assert_eq!(multi_square.pop_lsb(), Some(Square::E4));
    /// assert_eq!(multi_square.pop_lsb(), None); // Now empty
    /// ```
    #[inline]
    pub const fn pop_lsb(&mut self) -> Option<Square> {
        match self.0 {
            0 => None,
            _ => {
                let lsb_square = unsafe { self.lsb_unchecked() };
                self.0 &= self.0 - 1; // Clear the LSB
                Some(lsb_square)
            }
        }
    }

    /// Returns the least significant bit (LSB) and removes it from the bitboard
    ///
    /// This is an essential operation for efficiently iterating through pieces
    /// on a bitboard one at a time.
    ///
    /// **Warning**: Unsafe version of `pop_lsb()`
    ///
    /// # Returns
    ///
    /// * `Square` - The square corresponding to the removed least significant bit
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut multi_square = Square::E4.bb() | Square::D2.bb();
    /// assert_eq!(multi_square.pop_lsb_unchecked(), Some(Square::D2)); // D2 has lower index than E4
    /// assert_eq!(multi_square.pop_lsb_unchecked(), Some(Square::E4));
    /// assert_eq!(multi_square.pop_lsb_unchecked(), None); // Now empty
    /// ```
    #[inline]
    pub const unsafe fn pop_lsb_unchecked(&mut self) -> Square {
        debug_assert!(self.0 != 0, "Bitboard is empty");
        let lsb_square = unsafe { self.lsb_unchecked() };
        self.0 &= self.0 - 1; // Clear the LSB
        lsb_square
    }

    /// Returns the least significant bit (LSB) and removes it from the bitboard
    ///
    /// This is an essential operation for efficiently iterating through pieces
    /// on a bitboard one at a time.
    ///
    /// # Returns
    ///
    /// * `Some(Square)` - The square corresponding to the removed least significant bit
    /// * `None` - If the bitboard is empty (no bits set)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut multi_square = Square::E4.bb() | Square::D2.bb();
    /// assert_eq!(multi_square.pop_lsb(), Some(Square::D2)); // D2 has lower index than E4
    /// assert_eq!(multi_square.pop_lsb(), Some(Square::E4));
    /// assert_eq!(multi_square.pop_lsb(), None); // Now empty
    /// ```

    /// Returns the most significant bit (MSB) and removes it from the bitboard
    ///
    /// This allows you to process the "highest" pieces first.
    ///
    /// # Returns
    ///
    /// * `Some(Square)` - The square corresponding to the removed most significant bit
    /// * `None` - If the bitboard is empty (no bits set)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut multi_square = Square::E4.bb() | Square::G7.bb();
    /// assert_eq!(multi_square.pop_msb(), Some(Square::G7)); // G7 has higher index than E4
    /// assert_eq!(multi_square.pop_msb(), Some(Square::E4));
    /// assert_eq!(multi_square.pop_msb(), None); // Now empty
    /// ```
    #[inline]
    pub const fn pop_msb(&mut self) -> Option<Square> {
        match self.0 {
            0 => None,
            bits => {
                let msb_square = self.msb().unwrap();
                let msb_bit = 1u64 << (msb_square as u8 as u64);
                self.0 = bits & !msb_bit; // Clear the MSB
                Some(msb_square)
            }
        }
    }

    /// Returns the number of bits set in the bitboard (population count)
    ///
    /// This is useful for counting the number of pieces or squares.
    ///
    /// # Returns
    ///
    /// The number of bits set to 1 in the bitboard
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let bb = Square::E4.bb() | Square::D5.bb();
    /// assert_eq!(bb.count_bits(), 2);
    ///
    /// let empty = Bitboard::EMPTY;
    /// assert_eq!(empty.count_bits(), 0);
    /// ```
    #[inline]
    pub const fn count_bits(&self) -> u32 {
        self.0.count_ones()
    }

    /// Checks if the bitboard is empty (has no bits set)
    ///
    /// # Returns
    ///
    /// * `true` - If the bitboard has no bits set
    /// * `false` - If the bitboard has at least one bit set
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let bb = Square::E4.bb();
    /// assert!(!bb.is_empty());
    ///
    /// let empty = Bitboard::EMPTY;
    /// assert!(empty.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Checks if the bitboard is occupied (has some bits set)
    ///
    /// # Returns
    ///
    /// * `true` - If the bitboard has some bits set
    /// * `false` - If the bitboard has no bits set
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let bb = Square::E4.bb();
    /// assert!(bb.is_occupied());
    ///
    /// let empty = Bitboard::EMPTY;
    /// assert!(empty.is_occupied()));
    /// ```
    #[inline]
    pub const fn is_occupied(&self) -> bool {
        self.0 != 0
    }

    /// Checks if a specific square is set in the bitboard
    ///
    /// # Arguments
    ///
    /// * `square` - The square to check
    ///
    /// # Returns
    ///
    /// * `true` - If the bit corresponding to the square is set
    /// * `false` - If the bit corresponding to the square is not set
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let bb = Square::E4.bb();
    /// assert!(bb.get(Square::E4));
    /// assert!(!bb.get(Square::A1));
    /// ```
    #[inline]
    pub const fn contains(&self, square: Square) -> bool {
        (self.0 & (1u64 << (square as u8 as u64))) != 0
    }

    /// Sets a specific square in the bitboard
    ///
    /// # Arguments
    ///
    /// * `square` - The square to set
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut bb = Bitboard::EMPTY;
    /// bb.set(Square::E4);
    /// assert!(bb.get(Square::E4));
    /// ```
    #[inline]
    pub const fn set(&mut self, square: Square) {
        self.0 |= 1u64 << (square as u8 as u64);
    }

    /// Clears a specific square in the bitboard
    ///
    /// # Arguments
    ///
    /// * `square` - The square to clear
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut bb = Square::E4.bb();
    /// bb.clear(Square::E4);
    /// assert!(!bb.get(Square::E4));
    /// ```
    #[inline]
    pub const fn clear(&mut self, square: Square) {
        self.0 &= !(1u64 << (square as u8 as u64));
    }

    /// Toggles a specific square in the bitboard
    ///
    /// If the bit was set, it will be cleared.
    /// If the bit was clear, it will be set.
    ///
    /// # Arguments
    ///
    /// * `square` - The square to toggle
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut bb = Bitboard::EMPTY;
    /// bb.toggle(Square::E4);
    /// assert!(bb.get(Square::E4));
    /// bb.toggle(Square::E4);
    /// assert!(!bb.get(Square::E4));
    /// ```
    #[inline]
    pub const fn toggle(&mut self, square: Square) {
        self.0 ^= 1u64 << (square as u8 as u64);
    }

    /// Shifts the bitboard in the given direction, preventing edge wrapping.
    ///
    /// This function provides safe shifting operations that respect the chess board
    /// boundaries. This means bits that would wrap around the edge of the board are
    /// properly masked out before the shift operation.
    ///
    /// # Arguments
    /// * `dir` - The direction to shift the bitboard
    ///
    /// # Returns
    /// * `Bitboard` - The shifted bitboard. If the shift would move all bits off the board,
    ///                an empty bitboard is returned.
    #[inline]
    pub const fn shift(&self, dir: Direction) -> Bitboard {
        use Direction::*;

        let bb = *self;
        if bb.is_empty() {
            return Bitboard::EMPTY;
        }

        const NOT_RANK_1: Bitboard = Bitboard::RANK_1.not();
        const NOT_RANK_8: Bitboard = Bitboard::RANK_8.not();
        const NOT_RANK_12: Bitboard = Bitboard::RANK_12.not();
        const NOT_RANK_78: Bitboard = Bitboard::RANK_78.not();
        const NOT_FILE_A: Bitboard = Bitboard::FILE_A.not();
        const NOT_FILE_H: Bitboard = Bitboard::FILE_H.not();
        const NOT_FILE_AB: Bitboard = Bitboard::FILE_AB.not();
        const NOT_FILE_GH: Bitboard = Bitboard::FILE_GH.not();

        // Apply the appropriate mask and shift for each direction
        match dir {
            // Knight-like moves
            SSE => bb.bitand(NOT_RANK_12).bitand(NOT_FILE_H).shr(15),
            SEE => bb.bitand(NOT_RANK_1).bitand(NOT_FILE_GH).shr(6),
            SWW => bb.bitand(NOT_RANK_1).bitand(NOT_FILE_AB).shr(10),
            SSW => bb.bitand(NOT_RANK_12).bitand(NOT_FILE_A).shr(17),
            NNW => bb.bitand(NOT_RANK_78).bitand(NOT_FILE_A).shl(15),
            NNE => bb.bitand(NOT_RANK_78).bitand(NOT_FILE_H).shl(17),
            NWW => bb.bitand(NOT_RANK_8).bitand(NOT_FILE_AB).shl(6),
            NEE => bb.bitand(NOT_RANK_8).bitand(NOT_FILE_GH).shl(10),
            // King-like moves
            N => bb.bitand(NOT_RANK_8).shl(8),
            S => bb.bitand(NOT_RANK_1).shr(8),
            E => bb.bitand(NOT_FILE_H).shl(1),
            W => bb.bitand(NOT_FILE_A).shr(1),
            // Diagonal directions
            NE => bb.bitand(NOT_RANK_8).bitand(NOT_FILE_H).shl(9),
            NW => bb.bitand(NOT_RANK_8).bitand(NOT_FILE_A).shl(7),
            SE => bb.bitand(NOT_RANK_1).bitand(NOT_FILE_H).shr(7),
            SW => bb.bitand(NOT_RANK_1).bitand(NOT_FILE_A).shr(9),
            // Double moves
            NN => bb.bitand(NOT_RANK_78).shl(16),
            SS => bb.bitand(NOT_RANK_12).shr(16),
        }
    }

    /// Iterates through all set bits in the bitboard, applying the given function to each square
    ///
    /// This function processes squares from least significant bit to most significant bit.
    /// It uses pop_lsb() internally to efficiently iterate through the bitboard.
    ///
    /// # Arguments
    ///
    /// * `f` - A function or closure that takes a Square as its parameter
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use chess::{Bitboard, Square};
    ///
    /// let mut squares = Vec::new();
    /// let bb = Square::E4.bb() | Square::D5.bb();
    ///
    /// bb.for_each(|square| {
    ///     squares.push(square);
    /// });
    ///
    /// assert_eq!(squares.len(), 2);
    /// assert!(squares.contains(&Square::E4));
    /// assert!(squares.contains(&Square::D5));
    /// ```
    #[inline]
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Square),
    {
        let mut bb = *self;
        while bb.0 != 0 {
            f(unsafe { bb.pop_lsb_unchecked() });
        }
    }

    /// Returns if the bitboard is a singleton
    #[inline]
    pub const fn is_singleton(&self) -> bool {
        !self.is_empty() && !self.more_than_one()
    }

    /// Returns if the bitboard has more than one bit set
    #[inline]
    pub const fn more_than_one(&self) -> bool {
        self.0 & (self.0.wrapping_sub(1)) != 0
    }

    /// Return the extract bits using the intrinsic PEXT instruction
    #[cfg(target_feature = "bmi2")]
    #[inline]
    pub fn pext(&self, mask: u64) -> u64 {
        use core::arch::x86_64::_pext_u64;
        unsafe { _pext_u64(self.0, mask) }
    }

    #[inline]
    pub fn forward_ranks(col: Colour, sq: Square) -> Bitboard {
        match col {
            Colour::White => !Self::RANK_1 << 8 * (sq.relative(col).rank() as u8),
            Colour::Black => !Self::RANK_8 >> 8 * (sq.relative(col).rank() as u8),
        }
    }

    /// Return the forward file of the sq relative to a colour
    #[inline]
    pub fn forward_file(col: Colour, sq: Square) -> Bitboard {
        Self::forward_ranks(col, sq) & sq.file().bb()
    }

    /// Return the adjacent files of the sq
    #[inline]
    pub fn adjacent_files(sq: Square) -> Bitboard {
        let bb = sq.file().bb();
        bb.shift(Direction::E) | bb.shift(Direction::W)
    }

    /// Return the pawn attack span of the sq
    #[inline]
    pub fn pawn_attack_span(col: Colour, sq: Square) -> Bitboard {
        Self::forward_ranks(col, sq) & Self::adjacent_files(sq)
    }

    /// Return the passed pawn span of the sq
    #[inline]
    pub fn passed_pawn_span(col: Colour, sq: Square) -> Bitboard {
        Self::forward_file(col, sq) | Self::pawn_attack_span(col, sq)
    }

    /// Gets the precomputed attack span `Bitboard` for a bitboard of pawns
    ///
    /// This returns the squares a pawn of the given `Colour` on the given `Bitboard`
    /// would attack (i.e., the squares it could capture on). It does not consider
    /// whether pieces are present on the target squares.
    ///
    /// # Arguments
    /// * `col`: The `Colour` of the pawn.
    /// * `bb`: The `Bitboard` the pawns are on.
    ///
    /// # Returns
    /// A `Bitboard` representing the squares attacked by the pawns.
    ///
    /// # Example
    /// ```rust
    /// use chess::core::{Bitboard, Colour, Square};
    /// use chess::movegen::lookup::pawn_attack_span;
    ///
    /// let white_pawn_attacks = pawn_attack_span(Colour::White, Bitboard::from_square(Square::E4) | Bitboard::from_square(Square::D5));
    /// assert_eq!(white_pawn_attacks, Bitboard::from([Square::D5, Square::F5]) | Bitboard::from([Square::C6, Square::E6]));
    /// ```
    #[inline]
    pub fn pawn_attacks(col: Colour, bb: Bitboard) -> Bitboard {
        match col {
            Colour::White => bb.shift(Direction::NE) | bb.shift(Direction::NW),
            Colour::Black => bb.shift(Direction::SE) | bb.shift(Direction::SW),
        }
    }
}

/******************************************\
|==========================================|
|              Board Helpers               |
|==========================================|
\******************************************/

impl Bitboard {
    #[inline]
    pub const fn push_rank(col: Colour) -> Bitboard {
        match col {
            Colour::White => Rank::Rank2.bb(),
            Colour::Black => Rank::Rank7.bb(),
        }
    }

    #[inline]
    pub const fn promo_rank(col: Colour) -> Bitboard {
        match col {
            Colour::White => Rank::Rank7.bb(),
            Colour::Black => Rank::Rank2.bb(),
        }
    }

    #[inline]
    pub const fn ep_rank(col: Colour) -> Bitboard {
        match col {
            Colour::White => Rank::Rank3.bb(),
            Colour::Black => Rank::Rank6.bb(),
        }
    }
}

/******************************************\
|==========================================|
|                 Display                  |
|==========================================|
\******************************************/

/// Display function for bitboards
impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SEPARATOR: &str = "\n     +---+---+---+---+---+---+---+---+";

        writeln!(f, "{}", SEPARATOR)?;

        for rank in Rank::iter().rev() {
            write!(f, " {}   |", rank as u8 + 1)?;

            for file in File::iter() {
                let square = Square::from_parts(file, rank);
                let cell = if self.contains(square) { " 1 " } else { "   " };
                write!(f, "{}|", cell)?;
            }

            writeln!(f, "{}", SEPARATOR)?;
        }

        writeln!(f)?;
        writeln!(f, "       A   B   C   D   E   F   G   H")?;
        writeln!(f)?;
        writeln!(f, "Bitboard: {:#x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsb_msb() {
        // Test with single bits
        let a1 = Square::A1.bb();
        assert_eq!(a1.lsb(), Some(Square::A1));
        assert_eq!(a1.msb(), Some(Square::A1));

        let h8 = Square::H8.bb();
        assert_eq!(h8.lsb(), Some(Square::H8));
        assert_eq!(h8.msb(), Some(Square::H8));

        // Test with multiple bits
        let bb = Square::A1.bb() | Square::H8.bb();
        assert_eq!(bb.lsb(), Some(Square::A1)); // A1 has lower index than H8
        assert_eq!(bb.msb(), Some(Square::H8)); // H8 has higher index than A1

        // Test with empty bitboard
        let empty = Bitboard::EMPTY;
        assert_eq!(empty.lsb(), None);
        assert_eq!(empty.msb(), None);
    }

    #[test]
    fn test_pop_lsb() {
        // Test popping from a bitboard with multiple bits
        let mut bb = Square::E4.bb() | Square::A1.bb();
        assert_eq!(bb.pop_lsb(), Some(Square::A1)); // A1 has lowest index
        assert_eq!(bb.pop_lsb(), Some(Square::E4));
        assert_eq!(bb.pop_lsb(), None); // Now empty

        // Test popping from an already empty bitboard
        assert_eq!(bb.pop_lsb(), None);
    }

    #[test]
    fn test_pop_msb() {
        // Test popping from a bitboard with multiple bits
        let mut bb = Square::E4.bb() | Square::H8.bb();
        assert_eq!(bb.pop_msb(), Some(Square::H8)); // H8 has highest index
        assert_eq!(bb.pop_msb(), Some(Square::E4));
        assert_eq!(bb.pop_msb(), None); // Now empty
    }

    #[test]
    fn test_count_bits() {
        // Test empty bitboard
        let empty = Bitboard::EMPTY;
        assert_eq!(empty.count_bits(), 0);

        // Test single bit
        let single = Square::E4.bb();
        assert_eq!(single.count_bits(), 1);

        // Test multiple bits
        let multi = Square::E4.bb() | Square::D5.bb() | Square::A1.bb();
        assert_eq!(multi.count_bits(), 3);

        // Test all bits set (full board)
        let full = !Bitboard::EMPTY;
        assert_eq!(full.count_bits(), 64);
    }

    #[test]
    fn test_is_empty() {
        let empty = Bitboard::EMPTY;
        assert!(empty.is_empty());

        let non_empty = Square::E4.bb();
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_get_set_clear_toggle() {
        // Test set
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::E4);
        assert!(bb.contains(Square::E4));
        assert!(!bb.contains(Square::A1));

        // Test clear
        bb.clear(Square::E4);
        assert!(!bb.contains(Square::E4));

        // Test toggle
        bb.toggle(Square::D5);
        assert!(bb.contains(Square::D5));
        bb.toggle(Square::D5);
        assert!(!bb.contains(Square::D5));
    }

    #[test]
    fn test_bitloop_for_each() {
        let bb = Square::E4.bb() | Square::D5.bb();

        // Test bitloop
        let mut squares = Vec::new();
        bb.for_each(|sq| squares.push(sq));

        assert_eq!(squares.len(), 2);
        assert!(squares.contains(&Square::E4));
        assert!(squares.contains(&Square::D5));

        // Test for_each (should behave the same as bitloop)
        let mut squares2 = Vec::new();
        bb.for_each(|sq| squares2.push(sq));

        assert_eq!(squares2.len(), 2);
        assert!(squares2.contains(&Square::E4));
        assert!(squares2.contains(&Square::D5));
    }

    #[test]
    fn test_bitboard_operations() {
        // Test bitwise operations
        let a1 = Square::A1.bb();
        let h8 = Square::H8.bb();

        // OR operation
        let combined = a1 | h8;
        assert!(combined.contains(Square::A1));
        assert!(combined.contains(Square::H8));
        assert_eq!(combined.count_bits(), 2);

        // AND operation
        let intersection = a1 & h8;
        assert!(intersection.is_empty());

        // XOR operation
        let xor_result = a1 ^ a1;
        assert!(xor_result.is_empty());

        // NOT operation
        let inverted = !a1;
        assert!(!inverted.contains(Square::A1));
        assert_eq!(inverted.count_bits(), 63);
    }

    #[test]
    fn test_shift_basic_directions() {
        let bb = Square::E5.bb();

        // Cardinal directions
        assert_eq!(bb.shift(Direction::N), Square::E6.bb());
        assert_eq!(bb.shift(Direction::S), Square::E4.bb());
        assert_eq!(bb.shift(Direction::E), Square::F5.bb());
        assert_eq!(bb.shift(Direction::W), Square::D5.bb());

        // Diagonal directions
        assert_eq!(bb.shift(Direction::NE), Square::F6.bb());
        assert_eq!(bb.shift(Direction::NW), Square::D6.bb());
        assert_eq!(bb.shift(Direction::SE), Square::F4.bb());
        assert_eq!(bb.shift(Direction::SW), Square::D4.bb());

        // Double moves
        assert_eq!(bb.shift(Direction::NN), Square::E7.bb());
        assert_eq!(bb.shift(Direction::SS), Square::E3.bb());

        // Knight-like moves
        assert_eq!(bb.shift(Direction::NNE), Square::F7.bb());
        assert_eq!(bb.shift(Direction::NNW), Square::D7.bb());
        assert_eq!(bb.shift(Direction::NEE), Square::G6.bb());
        assert_eq!(bb.shift(Direction::NWW), Square::C6.bb());
        assert_eq!(bb.shift(Direction::SEE), Square::G4.bb());
        assert_eq!(bb.shift(Direction::SWW), Square::C4.bb());
        assert_eq!(bb.shift(Direction::SSE), Square::F3.bb());
        assert_eq!(bb.shift(Direction::SSW), Square::D3.bb());
    }

    #[test]
    fn test_shift_edge_cases() {
        // Test edge wrapping prevention

        // H-file (right edge) tests
        let h5 = Square::H5.bb();
        assert_eq!(h5.shift(Direction::E), Bitboard::EMPTY);
        assert_eq!(h5.shift(Direction::NE), Bitboard::EMPTY);
        assert_eq!(h5.shift(Direction::SE), Bitboard::EMPTY);
        assert_eq!(h5.shift(Direction::W), Square::G5.bb());

        // A-file (left edge) tests
        let a5 = Square::A5.bb();
        assert_eq!(a5.shift(Direction::W), Bitboard::EMPTY);
        assert_eq!(a5.shift(Direction::NW), Bitboard::EMPTY);
        assert_eq!(a5.shift(Direction::SW), Bitboard::EMPTY);
        assert_eq!(a5.shift(Direction::E), Square::B5.bb());

        // Rank 8 (top edge) tests
        let e8 = Square::E8.bb();
        assert_eq!(e8.shift(Direction::N), Bitboard::EMPTY);
        assert_eq!(e8.shift(Direction::NE), Bitboard::EMPTY);
        assert_eq!(e8.shift(Direction::NW), Bitboard::EMPTY);
        assert_eq!(e8.shift(Direction::S), Square::E7.bb());

        // Rank 1 (bottom edge) tests
        let e1 = Square::E1.bb();
        assert_eq!(e1.shift(Direction::S), Bitboard::EMPTY);
        assert_eq!(e1.shift(Direction::SE), Bitboard::EMPTY);
        assert_eq!(e1.shift(Direction::SW), Bitboard::EMPTY);
        assert_eq!(e1.shift(Direction::N), Square::E2.bb());

        // G-file edge cases for double-east shifts
        let g5 = Square::G5.bb();
        assert_eq!(g5.shift(Direction::NEE), Bitboard::EMPTY);
        assert_eq!(g5.shift(Direction::SEE), Bitboard::EMPTY);

        // B-file edge cases for double-west shifts
        let b5 = Square::B5.bb();
        assert_eq!(b5.shift(Direction::NWW), Bitboard::EMPTY);
        assert_eq!(b5.shift(Direction::SWW), Bitboard::EMPTY);
    }

    #[test]
    fn test_shift_multiple_bits() {
        // Test with multiple bits set
        let bb = Square::E4.bb() | Square::D4.bb();

        // Shift north
        assert_eq!(bb.shift(Direction::N), Square::E5.bb() | Square::D5.bb());

        // Shift east
        assert_eq!(bb.shift(Direction::E), Square::F4.bb() | Square::E4.bb());

        // Test with edge bits that would wrap
        let edge_case = Square::H1.bb() | Square::A1.bb();

        // East shift should only move the A1 bit to B1
        assert_eq!(edge_case.shift(Direction::E), Square::B1.bb());

        // West shift should only move the H1 bit to G1
        assert_eq!(edge_case.shift(Direction::W), Square::G1.bb());
    }
}
