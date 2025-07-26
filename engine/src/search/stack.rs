use chess::{Move, Piece, Square};

use crate::{
    Eval, KillerEntry,
    constants::{MAX_DEPTH, SEARCH_STACK_OFFSET},
};

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct SearchStackEntry {
    pub(crate) killers: KillerEntry,
    pub(crate) curr_move: Move,
    pub(crate) excl_move: Move,
    pub(crate) moved: Option<Piece>,
    pub(crate) eval: Eval,
    pub(crate) in_check: bool,
    pub(crate) ply_from_null: u16,
}

#[derive(Debug, Clone)]
pub(crate) struct SearchStack {
    // Search Stack
    stack: [SearchStackEntry; MAX_DEPTH as usize + SEARCH_STACK_OFFSET],
}

impl Default for SearchStack {
    fn default() -> Self {
        Self {
            stack: [SearchStackEntry::default(); MAX_DEPTH as usize + SEARCH_STACK_OFFSET],
        }
    }
}

impl SearchStackEntry {
    pub(crate) fn piece_to(&self) -> (Piece, Square) {
        unsafe { (self.moved.unwrap_unchecked(), self.curr_move.to()) }
    }
}

impl SearchStack {
    pub(crate) fn at(&self, ply: u16, offset: i8) -> SearchStackEntry {
        assert!(ply as i8 + SEARCH_STACK_OFFSET as i8 - offset >= 0);
        self.stack[(ply as i8 + SEARCH_STACK_OFFSET as i8 - offset) as usize]
    }

    pub(crate) fn at_mut(&mut self, ply: u16, offset: i8) -> &mut SearchStackEntry {
        assert!(ply as i8 + SEARCH_STACK_OFFSET as i8 - offset >= 0);
        &mut self.stack[(ply as i8 + SEARCH_STACK_OFFSET as i8 - offset) as usize]
    }
}
