use chess::{Bitboard, Colour, board::Board};

use crate::{
    flatten::flatten,
    params::{L1, MODEL, QA, QAB, SCALE},
    utils::Align64,
};

pub type SideAccumulator = Align64<[i16; L1]>;

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

const fn index(c: usize, p: usize, s: usize) -> (usize, usize) {
    const C_BASE: usize = 384;
    const P_BASE: usize = 64;

    let w = c * C_BASE + p * P_BASE + s;
    let b = (1 ^ c) * C_BASE + p * P_BASE + (s ^ 56);

    (w * L1, b * L1)
}

impl Accumulator {
    #[inline(never)]
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

    pub fn update(&mut self, board: &Board) {
        for c in 0..2 {
            let old_c = self.c_bb[c];
            let new_c = board.occupied[c];

            for p in 0..6 {
                let old_pc = old_c & self.p_bb[p];
                let new_pc = new_c & board.pieces[p];

                (new_pc & !old_pc).for_each(|s| {
                    let ft = index(c, p, s.index());
                    self.update_weights::<ON>(ft);
                });

                (old_pc & !new_pc).for_each(|s| {
                    let ft = index(c, p, s.index());
                    self.update_weights::<OFF>(ft);
                });
            }
        }

        self.c_bb = board.occupied;
        self.p_bb = board.pieces;
    }
}

#[inline]
pub fn screlu(x: i16) -> i32 {
    (x.clamp(0, QA as i16) as i32).pow(2)
}

impl Accumulator {
    pub fn evaluate(&mut self, board: &Board) -> i32 {
        self.update(board);

        let out = self.propagate(board.stm());

        (out / QA + MODEL.output_bias as i32) * SCALE / QAB
    }

    pub fn propagate(&self, c: Colour) -> i32 {
        let (stm, opp) = match c {
            Colour::White => (self.white, self.black),
            Colour::Black => (self.black, self.white),
        };

        let weights = MODEL.output_weights;
        return flatten(&stm, &weights[0]) + flatten(&opp, &weights[1]);
    }
}
