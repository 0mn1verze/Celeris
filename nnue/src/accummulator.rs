use chess::{Bitboard, Board, Colour};

use crate::{
    params::{L1, MODEL, QA, QAB, SCALE},
    utils::Align64,
};

type SideAccumulator = Align64<[i16; L1]>;

#[derive(Clone, Copy, Debug)]
pub struct Accumulator {
    pub white: SideAccumulator,
    pub black: SideAccumulator,

    c_bb: [Bitboard; 2],
    p_bb: [Bitboard; 6],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            white: MODEL.feature_bias,
            black: MODEL.feature_bias,
            c_bb: [Bitboard::EMPTY; 2],
            p_bb: [Bitboard::EMPTY; 6],
        }
    }
}

pub const ON: bool = true;
pub const OFF: bool = false;

/// Get the index of the given feature given some color, piece and square of that feature
const fn index(c: usize, p: usize, s: usize) -> (usize, usize) {
    const C_BASE: usize = 384;
    const P_BASE: usize = 64;

    let w = c * C_BASE + p * P_BASE + s;
    let b = (1 ^ c) * C_BASE + p * P_BASE + (s ^ 56);

    (w * L1, b * L1)
}

/// Weight updates
impl Accumulator {
    fn update_weights<const ON: bool>(&mut self, ft: (usize, usize)) {
        let update = |acc: &mut SideAccumulator, idx: usize| {
            acc.iter_mut()
                .zip(&MODEL.feature_weights[idx..idx + L1])
                .for_each(|(acc_val, &weight)| {
                    *acc_val += if ON { weight } else { -weight };
                });
        };

        update(&mut self.white, ft.0);
        update(&mut self.black, ft.1);
    }

    /// Update accumulator to a given board.
    pub fn update(&mut self, board: &Board) {
        // Update features
        for c in 0..2 {
            let old_c = self.c_bb[c];
            let new_c = board.occupied[c];

            for p in 0..6 {
                let old_pc = old_c & self.p_bb[p];
                let new_pc = new_c & board.pieces[p];

                // Turn on new weights
                (new_pc & !old_pc).for_each(|s| {
                    let ft = index(c, p, s.index());
                    self.update_weights::<ON>(ft);
                });

                // Turn off old weights
                (old_pc & !new_pc).for_each(|s| {
                    let ft = index(c, p, s.index());
                    self.update_weights::<OFF>(ft);
                });
            }
        }

        // Update board memory
        self.c_bb = board.occupied;
        self.p_bb = board.pieces;
    }
}

// Squared Clipped ReLU
#[inline]
pub fn screlu(x: i16) -> i32 {
    (x.clamp(0, QA as i16) as i32).pow(2)
}

/// Propagate through layers
impl Accumulator {
    pub fn evaluate(&mut self, board: &Board) -> i32 {
        self.update(board);

        let (stm, opp) = match board.stm() {
            Colour::White => (self.white.iter(), self.black.iter()),
            Colour::Black => (self.black.iter(), self.white.iter()),
        };

        let mut out = 0;

        out += stm
            .zip(&MODEL.output_weights[..L1])
            .map(|(&value, &weight)| screlu(value) * (weight as i32))
            .sum::<i32>();

        out += opp
            .zip(&MODEL.output_weights[L1..])
            .map(|(&value, &weight)| screlu(value) * (weight as i32))
            .sum::<i32>();

        (out / QA + MODEL.output_bias as i32) * SCALE / QAB
    }
}
