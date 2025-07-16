use chess::{
    Move,
    board::{Board, MAX_MOVES},
};

#[derive(Debug, Clone)]
pub struct PVLine {
    moves: [Move; MAX_MOVES],
    length: usize,
}

impl Default for PVLine {
    fn default() -> Self {
        Self {
            moves: [Move::NONE; MAX_MOVES],
            length: 0,
        }
    }
}

impl std::ops::Index<usize> for PVLine {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.moves[index]
    }
}

impl PVLine {
    pub fn update_line(&mut self, move_: Move, old: &Self) {
        self.length = old.length + 1;
        self.moves[0] = move_;
        self.moves[1..=old.length].copy_from_slice(&old.moves[..old.length]);
    }

    pub fn clear(&mut self) {
        self.length = 0;
    }

    pub fn to_str(&self, board: &Board) -> String {
        let mut s = String::from("pv");
        for m in &self.moves[0..self.length] {
            s.push_str(&format!(" {}", m.to_str(board)));
        }

        format!("{}", s)
    }
}
