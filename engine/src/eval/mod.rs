mod eval;

pub use eval::evaluate_nnue;

use crate::constants::*;
use chess::impl_ari_ops;

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Eval(pub i32);

impl_ari_ops!(Eval);

impl Eval {
    pub const ZERO: Eval = Eval(0);
    pub const DRAW: Eval = Eval(0);

    pub const INFINITY: Eval = Eval(32001);

    pub const MATE: Eval = Eval(32000);

    pub const MATE_BOUND: Eval = Eval(Self::MATE.0 - MAX_DEPTH as i32);

    pub fn abs(&self) -> Eval {
        Eval(self.0.abs())
    }

    pub fn is_valid(&self) -> bool {
        self.abs() <= Eval::INFINITY
    }

    pub fn is_terminal(&self) -> bool {
        self.abs() >= Eval::MATE_BOUND && self.is_valid()
    }

    pub fn mated_in(ply: u16) -> Eval {
        -Self::MATE + Eval(ply as i32)
    }

    pub fn mate_in(ply: u16) -> Eval {
        Self::MATE - Eval(ply as i32)
    }

    pub fn from_tt(&self, ply: u16) -> Eval {
        let ply = Eval(ply as i32);

        if *self >= Self::MATE_BOUND {
            *self - ply
        } else if *self <= -Self::MATE_BOUND {
            *self + ply
        } else {
            *self
        }
    }

    pub fn to_tt(&self, ply: u16) -> Eval {
        let ply = Eval(ply as i32);

        if *self >= Self::MATE_BOUND {
            *self + ply
        } else if *self <= -Self::MATE_BOUND {
            *self - ply
        } else {
            *self
        }
    }
}

impl std::ops::Neg for Eval {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Eval(-self.0)
    }
}

impl std::fmt::Display for Eval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.abs() >= Self::MATE_BOUND.0 {
            let moves_to_mate = (Self::MATE.0 - self.0.abs() + 1) / 2;
            if self.0 > 0 {
                write!(f, "mate +{}", moves_to_mate)
            } else {
                write!(f, "mate -{}", moves_to_mate)
            }
        } else {
            write!(f, "cp {}", self.0)
        }
    }
}
