use chess::board::attacks;
use chess::{Bitboard, Board, Colour, Direction, File, PieceType, Rank, Square};

use super::psqt::{calc_game_phase, calc_psqt};
use super::{Eval, PawnTable, S, Score};
use crate::MATE;

// --- Constants for King Safety Evaluation ---
// These values are just starting points and will likely need tuning.

// Attack scores (penalty per attacking piece type near the king)
// Values are higher in MG as king safety is generally more critical then.
const QUEEN_ATTACK_WEIGHT: Score = S!(9, 0); // Penalty per queen attack square in king zone
const ROOK_ATTACK_WEIGHT: Score = S!(5, 0); // Penalty per rook attack square
const BISHOP_ATTACK_WEIGHT: Score = S!(3, 0); // Penalty per bishop attack square
const KNIGHT_ATTACK_WEIGHT: Score = S!(3, 0); // Penalty per knight attack square

fn eval_king(board: &Board, us: Colour) -> Score {
    let them = !us;
    let ksq = board.ksq(us);
    let occupied = board.all_occupied_bb();

    let mut safety_score = Score::ZERO;

    // --- 1. Evaluate Attacks on the King Ring ---
    let king_attacker_mask = Bitboard::passed_pawn_span(us, ksq)
        & Bitboard::forward_ranks(them, Square::E5.relative(us));

    // Iterate through opponent's pieces and sum weighted attacks on the king ring
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let attackers_bb = board.piece_bb(them, pt);
        let weight = match pt {
            PieceType::Queen => QUEEN_ATTACK_WEIGHT,
            PieceType::Rook => ROOK_ATTACK_WEIGHT,
            PieceType::Bishop => BISHOP_ATTACK_WEIGHT,
            PieceType::Knight => KNIGHT_ATTACK_WEIGHT,
            _ => Score::ZERO, // Should not happen
        };

        attackers_bb.for_each(|sq| {
            let attacks_bb = attacks(them, pt, sq, occupied);
            let ring_attacks = attacks_bb & king_attacker_mask;
            // Add weight for each square attacked within the ring
            safety_score -= weight * ring_attacks.count_bits() as i16;
        });
    }

    safety_score
}

// pub fn evaluate(board: &Board, pawn_table: &mut PawnTable) -> Eval {
pub fn evaluate(board: &Board) -> Eval {
    let mut score = calc_psqt(board);

    // let pawn_entry = &mut pawn_table.get(board);

    // score += pawn_entry.scores[Colour::White as usize];
    // score -= pawn_entry.scores[Colour::Black as usize];
    // score += pawn_entry.king_safety(board, Colour::White);
    // score -= pawn_entry.king_safety(board, Colour::Black);

    // score += eval_king(board, Colour::White);
    // score -= eval_king(board, Colour::Black);

    let (mg_phase, eg_phase) = calc_game_phase(board);

    let weighted_mg = (score.0.0 as i64) * (mg_phase as i64);
    let weighted_eg = (score.1.0 as i64) * (eg_phase as i64);

    let eval = ((weighted_mg + weighted_eg) / 24);

    let v = Eval(eval.clamp(-MATE.0 as i64, MATE.0 as i64) as i16);

    if board.stm() == Colour::White { v } else { -v }
}
