use crate::engine::tunables::*;

use super::Eval;
use chess::{PieceType, board::Board};
use nnue::accumulator::Accumulator;

#[rustfmt::skip]
 pub fn evaluate_nnue(board: &Board, nnue: &mut Accumulator) -> Eval {
     // nnue output
     let mut v = nnue.evaluate(board);
 
     let material_scale = (
         pawn_val()   * board.piecetype_bb(PieceType::Pawn).count_bits()   as i32 +
         knight_val() * board.piecetype_bb(PieceType::Knight).count_bits() as i32 +
         bishop_val() * board.piecetype_bb(PieceType::Bishop).count_bits() as i32 +
         rook_val()   * board.piecetype_bb(PieceType::Rook).count_bits()   as i32 +
         queen_val()  * board.piecetype_bb(PieceType::Queen).count_bits()  as i32
     ) / 32;
 
     v = (v * (nnue_base() + material_scale)) / 1024;
 
     Eval(v as i32)
 }
