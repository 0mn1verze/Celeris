use chess::{Colour, Move, Piece, Square, board::Board};

use crate::eval::Eval;

#[derive(Debug, Clone, Copy, Default)]
pub struct KillerEntry {
    killers: [Move; 2],
}

impl KillerEntry {
    pub fn get(&self) -> [Move; 2] {
        self.killers
    }

    pub fn clear(&mut self) {
        self.killers = [Move::NONE; 2];
    }

    pub fn update(&mut self, move_: Move) {
        if move_ != self.killers[0] {
            self.killers[1] = self.killers[0];
            self.killers[0] = move_;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct HistoryEntry<const MAX: i16> {
    entry: Eval,
}

impl<const MAX: i16> HistoryEntry<MAX> {
    /// # Returns a reference to the entry value
    pub fn get(&self) -> Eval {
        self.entry
    }

    /// # Updates history
    ///
    /// ### History gravity formula
    /// - https://www.chessprogramming.org/History_Heuristic
    ///
    /// - Scales up history when a beta cutoff is unexpected, and scales down history when a beta cutoff is expected
    /// - Keeps the history values clamped between MAX and -MAX
    pub fn update(&mut self, bonus: i16) {
        let bonus = bonus.clamp(-MAX, MAX);
        let product = self.entry.0 as i64 * bonus.abs() as i64 / MAX as i64;
        self.entry += Eval(bonus - product as i16);
    }
}

pub const MAX_MAIN_HISTORY: i16 = 16384;

pub trait History<const MAX: i16> {
    /// Returns an immutable reference to the history entry.
    /// Implementors must provide this method.
    fn get_entry_ref(&self, board: &Board, move_: Move) -> &HistoryEntry<MAX>;
    /// Returns a mutable reference to the history entry for updates.
    /// Implementors must provide this method.
    fn probe_mut(&mut self, board: &Board, move_: Move) -> &mut HistoryEntry<MAX>;

    /// Gets the immutable Eval score for a move.
    /// This is automatically implemented using `get_entry_ref`.
    fn get(&self, board: &Board, move_: Move) -> Eval {
        self.get_entry_ref(board, move_).get()
    }

    /// Updates the history score for a move.
    fn update(&mut self, board: &Board, move_: Move, bonus: i16) {
        let entry = self.probe_mut(board, move_);
        entry.update(bonus);
    }

    fn clear(&mut self)
    where
        Self: Sized + Default,
    {
        *self = Self::default();
    }
}

#[derive(Debug, Clone)]
pub struct MainHistory {
    history: [[[HistoryEntry<MAX_MAIN_HISTORY>; Square::NUM]; Piece::NUM]; Colour::NUM],
}

impl Default for MainHistory {
    fn default() -> Self {
        Self {
            history: [[[HistoryEntry::default(); Square::NUM]; Piece::NUM]; Colour::NUM],
        }
    }
}

impl MainHistory {
    /// Helper function to get the indices for the history table.
    #[inline(always)]
    fn get_indices(board: &Board, move_: Move) -> (usize, usize, usize) {
        // Safety: Assumes the 'from' square is occupied, which should be true for valid moves.
        let moved_piece = unsafe { board.on(move_.from()).unwrap_unchecked() };
        (
            board.stm() as usize,
            moved_piece as usize,
            move_.to() as usize,
        )
    }
}

impl History<MAX_MAIN_HISTORY> for MainHistory {
    fn get_entry_ref(&self, board: &Board, move_: Move) -> &HistoryEntry<MAX_MAIN_HISTORY> {
        let (c, p, sq) = Self::get_indices(board, move_);
        &self.history[c][p][sq]
    }
    fn probe_mut(&mut self, board: &Board, move_: Move) -> &mut HistoryEntry<MAX_MAIN_HISTORY> {
        let (c, p, sq) = Self::get_indices(board, move_);
        &mut self.history[c][p][sq]
    }
}
