use chess::{Board, Colour, PieceType};
use nnue::accummulator::Accumulator;

use super::psqt::{calc_game_phase, calc_psqt};
use super::{Eval, PawnTable};

pub fn evaluate(board: &Board, pawn_table: &mut PawnTable) -> Eval {
    let mut score = calc_psqt(board);

    let pawn_entry = &mut pawn_table.get(board);

    score += pawn_entry.pawn_score(Colour::White);
    score -= pawn_entry.pawn_score(Colour::Black);
    score += pawn_entry.king_safety(board, Colour::White);
    score -= pawn_entry.king_safety(board, Colour::Black);

    let (mg_phase, eg_phase) = calc_game_phase(board);

    let weighted_mg = (score.0.0 as i64) * (mg_phase as i64);
    let weighted_eg = (score.1.0 as i64) * (eg_phase as i64);

    let eval = (weighted_mg + weighted_eg) / 24;

    let v = Eval(eval.clamp(-Eval::MATE.0 as i64, Eval::MATE.0 as i64) as i16);

    if board.stm() == Colour::White { v } else { -v }
}

#[rustfmt::skip]
pub fn evaluate_nnue(board: &Board, nnue: &mut Accumulator) -> Eval {
    // nnue output
    let mut v = nnue.evaluate(board);

    let material_scale = (
        82   * board.piecetype_bb(PieceType::Pawn).count_bits()   as i32 +
        337  * board.piecetype_bb(PieceType::Knight).count_bits() as i32 +
        365  * board.piecetype_bb(PieceType::Bishop).count_bits() as i32 +
        477  * board.piecetype_bb(PieceType::Rook).count_bits()   as i32 +
        1025 * board.piecetype_bb(PieceType::Queen).count_bits()  as i32
    ) / 32;

    v = (v * (700 + material_scale)) / 1024;
    v = if board.stm() == Colour::White { v } else { -v };

    Eval(v as i16)
}
