use chess::{Colour, Move, Piece, Square, board::Board};

use crate::{constants::MAX_MAIN_HISTORY, eval::Eval};

/******************************************\
|==========================================|
|              Killer Moves                |
|==========================================|
\******************************************/

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
pub struct Entry<const MAX: i16> {
    entry: Eval,
}

pub trait History {
    fn get(&self) -> Eval;
    fn update(&mut self, bonus: i16);
}

impl<const MAX: i16> History for Entry<MAX> {
    /// # Returns a reference to the entry value
    fn get(&self) -> Eval {
        self.entry
    }

    /// # Updates history
    ///
    /// ### History gravity formula
    /// - https://www.chessprogramming.org/History_Heuristic
    ///
    /// - Scales up history when a beta cutoff is unexpected, and scales down history when a beta cutoff is expected
    /// - Keeps the history values clamped between MAX and -MAX
    fn update(&mut self, bonus: i16) {
        let bonus = bonus.clamp(-MAX, MAX);
        let product = self.entry.0 as i32 * bonus.abs() as i32 / MAX as i32;
        self.entry += Eval(bonus as i32 - product);
    }
}

/******************************************\
|==========================================|
|              History Trait               |
|==========================================|
\******************************************/

pub trait HistoryTable<T> {
    /// Returns an immutable reference to the history entry.
    /// Implementors must provide this method.
    fn get_entry_ref(&self, board: &Board, move_: Move) -> &T;
    /// Returns a mutable reference to the history entry for updates.
    /// Implementors must provide this method.
    fn probe_mut(&mut self, board: &Board, move_: Move) -> &mut T;

    fn clear(&mut self)
    where
        Self: Sized + Default,
    {
        *self = Self::default();
    }
}

pub trait Interface<T: History>: HistoryTable<T> {
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
}

/******************************************\
|==========================================|
|               History Macro              |
|==========================================|
\******************************************/

macro_rules! define_history {
    ($name:ident, $t:ident, [$($dims:expr),+]) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            history: define_nd_array_struct!(@array $t, [$($dims),+]),
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    history: define_nd_array_struct!(@default $t, [$($dims),+]),
                }
            }
        }
    };
}

macro_rules! impl_history_probe {
    ($name:ident, $t:ident, [$($idx:ident),+]) => {
        impl HistoryTable<$t> for $name {
            fn get_entry_ref(&self, board: &Board, move_: Move) -> &$t {
                let ($($idx),+)= Self::get_indices(board, move_);
                &self.history$([$idx])+
            }
            fn probe_mut(
                &mut self,
                board: &Board,
                move_: Move,
            ) -> &mut $t {
                let ($($idx),+) = Self::get_indices(board, move_);
                &mut self.history$([$idx])+
            }
        }
    }
}

macro_rules! define_nd_array_struct {
    // Recursive array type constructor
    (@array $t:ident, [$dim:expr]) => {
        [$t; $dim]
    };
    (@array $t:ident, [$dim:expr, $($rest:expr),+]) => {
        [define_nd_array_struct!(@array $t, [$($rest),+]); $dim]
    };

    // Recursive default initialiser
    (@default $t:ident, [$dim:expr]) => {
        [$t::default(); $dim]
    };
    (@default $t:ident, [$dim:expr, $($rest:expr),+]) => {
        [define_nd_array_struct!(@default $t, [$($rest),+]); $dim]
    };
}

/******************************************\
|==========================================|
|               Main History               |
|==========================================|
\******************************************/

type MainHistoryEntry = Entry<MAX_MAIN_HISTORY>;

define_history!(
    MainHistory,
    MainHistoryEntry,
    [Colour::NUM, Piece::NUM, Square::NUM]
);

impl_history_probe!(MainHistory, MainHistoryEntry, [colour, piece, square]);

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

impl Interface<MainHistoryEntry> for MainHistory {}

/******************************************\
|==========================================|
|              Capture History             |
|==========================================|
\******************************************/

type CaptureHistoryEntry = Entry<MAX_MAIN_HISTORY>;

define_history!(
    CaptureHistory,
    CaptureHistoryEntry,
    [Colour::NUM, Piece::NUM, Square::NUM, Piece::NUM]
);

impl_history_probe!(
    CaptureHistory,
    CaptureHistoryEntry,
    [colour, moved_piece, square, captured_piece]
);

impl CaptureHistory {
    /// Helper function to get the indices for the capture history table.
    #[inline(always)]
    fn get_indices(board: &Board, move_: Move) -> (usize, usize, usize, usize) {
        // Safety: Assumes the 'from' square is occupied, which should be true for valid moves.
        let moved_piece = unsafe { board.on_unchecked(move_.from()) };
        let captured_piece = unsafe { board.on_unchecked(move_.to()) };
        (
            board.stm().index(),
            moved_piece.index(),
            move_.to().index(),
            captured_piece.index(),
        )
    }
}

impl Interface<CaptureHistoryEntry> for CaptureHistory {}

/******************************************\
|==========================================|
|               Continuation               |
|==========================================|
\******************************************/

type ContinuationEntry = Entry<MAX_MAIN_HISTORY>;

define_history!(Continuation, ContinuationEntry, [Piece::NUM, Square::NUM]);

impl_history_probe!(Continuation, ContinuationEntry, [piece, to]);

impl Continuation {
    /// Helper function to get the indices for the continuation table.
    #[inline(always)]
    fn get_indices(board: &Board, move_: Move) -> (usize, usize) {
        // Safety: Assumes the 'from' square is occupied, which should be true for valid moves.
        let moved_piece = unsafe { board.on_unchecked(move_.from()) };
        (moved_piece.index(), move_.to().index())
    }
}

impl Interface<ContinuationEntry> for Continuation {}

/******************************************\
|==========================================|
|           Continuation History           |
|==========================================|
\******************************************/

define_history!(ContinuationTable, Continuation, [Square::NUM, Piece::NUM]);

impl ContinuationTable {
    pub fn get_entry_ref(&self, piece: Piece, to: Square) -> &Continuation {
        &self.history[to.index()][piece.index()]
    }

    pub fn probe_mut(&mut self, piece: Piece, to: Square) -> &mut Continuation {
        &mut self.history[to.index()][piece.index()]
    }

    pub fn clear(&mut self) {
        for piece in Piece::iter() {
            for to in Square::iter() {
                self.history[to.index()][piece.index()].clear();
            }
        }
    }
}
