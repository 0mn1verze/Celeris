use chess::Move;

use crate::constants::MOVE_BUFFER_MAX;

#[derive(Debug, Default)]
pub(crate) struct MoveBuffer {
    moves: [Move; MOVE_BUFFER_MAX],
    index: usize,
}

impl MoveBuffer {
    pub(crate) fn push(&mut self, move_: Move) {
        if self.index < MOVE_BUFFER_MAX {
            self.moves[self.index] = move_;
            self.index += 1;
        }
    }
}

impl<'a> IntoIterator for &'a MoveBuffer {
    type Item = &'a Move;
    type IntoIter = core::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves[..self.index].iter()
    }
}
