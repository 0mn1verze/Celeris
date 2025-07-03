use chess::{
    Bitboard, Colour, Move, MoveFlag, Piece, PieceType, Square,
    board::{Board, bishop_attacks, rook_attacks},
};

use crate::Eval;

const VALUES: [Eval; PieceType::NUM] = [
    Eval(150),
    Eval(340),
    Eval(360),
    Eval(480),
    Eval(1000),
    Eval(0),
];

/// Calculates the immediate material value change associated with a move.
///
/// This function determines the value of the piece captured or the net value change
/// in case of a promotion. It doesn't account for subsequent recaptures (that's SEE's job).
///
/// # Arguments
/// * `move_` - The `Move` being evaluated.
///
/// # Returns
/// The material value gained by the move (e.g., value of captured piece,
/// value difference in promotion). Returns 0 for quiet moves without promotion.
fn move_value(board: &Board, move_: Move) -> Eval {
    let (from, to) = (move_.from(), move_.to());

    // Get the piece type of the piece we are moving
    let moved_pt = unsafe { board.on_unchecked(from).pt().index() };

    let mut value = Eval::ZERO;

    // --- Handle Captures ---
    if move_.is_capture() {
        // En Passant Capture: Value is always a pawn's value.
        if move_.flag() == MoveFlag::EPCapture {
            return VALUES[PieceType::Pawn.index()];
        } else {
            value += unsafe { VALUES[board.on_unchecked(to).pt().index()] }; // Value of captured piece
        }
    }

    // If the move is a promotion, we need to check the type of the piece we are promoting to
    // and the type of the piece we are moving
    if move_.is_promotion() {
        // Value is (Promoted Piece) - (Pawn).
        // Note: moved_pt is always Pawn for a promotion.
        value += -VALUES[moved_pt] + VALUES[unsafe { move_.promotion_pt().index() }];
    }

    // --- Handle Quiet Moves (no capture, no promotion) ---
    // The immediate material value change is zero.
    value
}

/// Returns the least valuable attacker of a given type on the specified square.
///
/// This function checks all attackers of the specified type and returns the first one found.
///
/// # Arguments
/// * `attackers` - The set of attackers to check against.
/// * `us` - The color of the pieces for which we are checking the attackers.
///
/// # Returns
/// A tuple containing the square of the attacker and the piece type.
///
/// # Panics
/// This function will panic if no attacker is found in the provided bitboard.
fn lva(board: &Board, attackers: Bitboard, us: Colour) -> (Square, Piece) {
    for pt in PieceType::iter() {
        let bb = attackers & board.piece_bb(us, pt);

        if bb.is_occupied() {
            // If we have a piece attacking the square, we can return it.
            let square = bb.lsb_unchecked();
            return (square, Piece::from_parts(us, pt));
        }
    }

    // If we reach here, it means we didn't find any attackers of the specified type.
    unreachable!()
}

/// Determines if a move is likely to be profitable using Static Exchange Evaluation (SEE).
///
/// SEE estimates the material gain or loss resulting from a series of captures on a
/// specific square. It helps prune moves that are obviously bad captures.
///
/// # Arguments
/// * `move_` - The potentially capturing `Move` to evaluate.
/// * `threshold` - The minimum gain required for the move to be considered potentially good.
///                 Typically, this is `Eval::ZERO` (0). If SEE predicts a gain >= threshold,
///                 it returns `true`.
///
/// # Returns
/// `true` if the estimated material exchange is greater than or equal to the `threshold`,
/// `false` otherwise.
pub fn see(board: &Board, move_: Move, threshold: Eval) -> bool {
    use PieceType::*;

    // Set up SEE
    let (from, to) = (move_.from(), move_.to());

    // If the move is a castling move then there is no point doing SEE since the to square is not attacked (the king will be moving there)
    if move_.is_castle() {
        return true;
    }

    // Calculate the initial material gain from making the move, adjusted by the threshold.
    // gain = (value of captured piece or promotion difference) - threshold
    let mut gain = move_value(board, move_) - threshold;

    // If the initial gain (e.g., capturing a piece) is already less than what we need
    // (threshold), then the move is definitely not profitable.
    if gain < Eval::ZERO {
        return false;
    }

    let victim = if move_.is_promotion() {
        unsafe { move_.promotion_pt().index() }
    } else {
        unsafe { board.on_unchecked(from).pt().index() }
    };

    // Subtract the value of the piece *making* the move. This represents the potential loss
    // if the opponent recaptures.
    // gain = (initial gain) - threshold - (value of moving piece)
    gain -= VALUES[victim];

    // If, even after considering the loss of our moving piece, the gain is still positive
    // (meaning the initial capture was valuable enough), the move is profitable.
    if gain >= Eval::ZERO {
        return true;
    }

    let diag_sliders = board.piecetype_bb(Bishop) | board.piecetype_bb(Queen);
    let hv_sliders = board.piecetype_bb(Rook) | board.piecetype_bb(Queen);

    // --- Simulate the exchange on the target square 'to' ---
    let mut occ = board.all_occupied_bb();

    // Simulate the exchange
    occ = (occ ^ from.bb()) | to.bb();
    if move_.flag() == MoveFlag::EPCapture {
        occ ^= board.ep_target().unwrap().bb();
    }

    // Get all the attackers to the 'to' square for both sides.
    let mut attackers = board.attackers_to(to, occ) & occ;

    let mut stm = !board.stm(); // Side to move

    // For each iteration we switch sides and get the least valuable attacker,
    // and that attacker's value is assumed to be lost (assuming the opponent has an attacker left,
    // if there isn't its handled at the start of the next loop),
    // so the value becomes the amount the current side to move is left with after the trade,
    // if the trade is good enough then the function returns.
    loop {
        // Get the attackers for the side to move
        let stm_attackers = attackers & board.occupied_bb(stm);

        // If there are no attacker for this side then this side then we have to see if the side has lost the trade
        if stm_attackers.is_empty() {
            break;
        }

        // Get least valuable attacker
        let (lva_sq, lva) = lva(board, stm_attackers, stm);
        // Remove attack from the occupancy board (simulate a move)
        occ.clear(lva_sq);

        let lva_pt = lva.pt();
        // Diagonal captures may uncover discover attacks
        if matches!(lva_pt, Pawn | Bishop | Queen) {
            attackers |= bishop_attacks(to, occ) & diag_sliders;
        }
        // Orthogonal captures may uncover discover attacks
        if matches!(lva_pt, Rook | Queen) {
            attackers |= rook_attacks(to, occ) & hv_sliders;
        }

        // Restrict the attackers to the occupancy (remove every square that does not contain a possible attacker)
        attackers &= occ;

        // Flip the side
        stm = !stm;

        // Negamax the values (Incentivise the engine to not go for drawing trades with no goal in mind)
        gain = -gain - Eval(1) - VALUES[lva_pt.index()];

        if gain >= Eval::ZERO {
            // If the recapturing piece is the king and the opponent has another attacker, then a
            if lva_pt == King && (attackers & board.occupied_bb(stm)).is_occupied() {
                stm = !stm;
            }
            break;
        }
    }

    stm != board.stm() // Since the loop breaks when the stm has no attackers left, if it's our side then the SEE check should return false (Not favourable exchange)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::{
        Move, MoveFlag, Square,
        board::Board,
        board::{LegalGen, MoveList},
    };

    #[test]
    fn test_see_simple_winning_capture() {
        // White Queen captures undefended Black Rook
        let board = Board::from_fen("r1k5/8/8/8/8/8/8/Q6K w - - 0 1").unwrap();
        let move_ = Move::new(Square::A1, Square::A8, MoveFlag::Capture);
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_simple_losing_capture() {
        // White Queen captures pawn defended by Queen
        let board = Board::from_fen("2q5/1k1p4/8/8/Q7/8/8/7K w - - 0 1").unwrap();
        let move_ = Move::new(Square::A4, Square::D7, MoveFlag::Capture);
        assert!(!see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_equal_exchange() {
        // White Rook captures Black Rook, defended by King only
        let board = Board::from_fen("r6k/8/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let move_ = Move::new(Square::A1, Square::A8, MoveFlag::Capture);
        // Should be false because equal trades are slightly penalized (-1)
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_complex_exchange_winning() {
        // PxN, QxP, RxQ - White starts, should be winning
        let board = Board::from_fen("1k1r4/1ppn2b1/p7/4pp2/P3P1q1/1P1N4/2P1QPPP/R2B1RK1 w - - 0 1")
            .unwrap();
        let move_ = Move::new(Square::E2, Square::G4, MoveFlag::Capture);
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_complex_exchange_losing() {
        // White N takes pawn, but loses N for P
        let board =
            Board::from_fen("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq - 0 1")
                .unwrap();
        let move_ = Move::new(Square::C3, Square::D5, MoveFlag::Capture);
        assert!(!see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_en_passant() {
        // White pawn captures en passant, defended by black pawn
        let board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let move_ = Move::new(Square::E5, Square::D6, MoveFlag::EPCapture); // exd6 e.p.
        // Pawn takes pawn, should be equal, so SEE returns false
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_promotion_capture() {
        // White pawn captures rook and promotes to Queen
        let board = Board::from_fen("r7/1P6/k7/8/8/8/8/K7 w - - 0 1").unwrap();
        let move_ = Move::new(Square::B7, Square::A8, MoveFlag::QueenPromo); // bxa8=Q
        // Gain Queen + Rook, Lose Pawn -> Winning
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_underpromotion_capture() {
        // White pawn captures rook and promotes to Knight
        let board = Board::from_fen("r7/1P6/k7/8/8/8/8/K7 w - - 0 1").unwrap();
        let move_ = Move::new(Square::B7, Square::A8, MoveFlag::KnightPromo); // bxa8=N
        // Gain Knight + Rook, Lose Pawn -> Winning
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_king_involved_losing() {
        // White Queen takes pawn, Black King recaptures. Q vs P -> losing.
        let board = Board::from_fen("8/8/8/4k3/3p4/8/8/3QK3 w - - 0 1").unwrap();
        let move_ = Move::new(Square::D1, Square::D4, MoveFlag::Capture);
        assert!(!see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_castling() {
        // Castling should always pass SEE as it's not a capture.
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let move_ = Move::new(Square::E1, Square::G1, MoveFlag::KingCastle);
        assert!(see(&board, move_, Eval::ZERO));
    }

    #[test]
    fn test_see_bulk() {
        const P: i16 = 150;
        const N: i16 = 340;
        const R: i16 = 480;
        const Q: i16 = 1000;

        #[rustfmt::skip]
        const SEE_TESTS: &[(&str, &str, i16, bool)] = &[
            ("2k5/8/8/4p3/8/8/2K1R3/8 w - - 0 1", "e2e5", 0, true),
            ("3k4/8/8/4p3/3P4/8/8/5K2 w - - 0 1", "d4e5", P, true),
            ("3k4/8/5p2/4p3/3P4/8/8/5K2 w - - 0 1", "d4e5", P, false),
            ("8/3k4/2n2b2/8/3P4/8/3KN3/8 b - - 0 1", "c6d4", P, true),
            ("8/3k4/2n2b2/8/3P4/8/3KN3/8 b - - 0 1", "c6d4", N, false),
            ("3kr3/8/4q3/8/4P3/5P2/8/3K4 b - - 0 1", "e6e4", 0, false),
            ("3kr3/8/4q3/8/4P3/5P2/8/3K4 b - - 0 1", "e6e4", -Q, true),
            ("8/3k4/2n2b2/8/3P4/3K4/4N3/8 b - - 0 1", "c6d4", P, false),
            ("5k2/2P5/4b3/8/8/8/8/2R2K2 w - - 0 1", "c7c8q", 0, true),
            ("5k2/2P5/4b3/8/8/8/8/3R1K2 w - - 0 1", "c7c8q", 0, false),
            ("8/3k2b1/2n2b2/8/3P4/3K4/4N3/8 b - - 0 1", "c6d4", 0, true),
            ("3k4/8/2q5/2b5/2r5/8/2P5/2R1K3 b - - 0 1", "c4c2", 0, false),
            ("3k4/8/2q5/2b5/2r5/8/2P5/2R1K3 b - - 0 1", "c4c2", P - R, true),
            ("2k5/3n2b1/2nq4/4R3/5P2/3N1N2/8/5K2 b - - 0 1", "d6e5", 0, false),
            ("2k5/3n2b1/2nq4/4R3/5P2/3N1N2/8/5K2 b - - 0 1", "d6e5", R - Q + P, true),
            ("5r1k/3b1q1p/1npb4/1p6/pPpP1N2/2P4B/2NBQ1P1/5R1K b - - 0 1", "d6f4", 0, false),
            ("5r1k/3b1q1p/1npb4/1p6/pPpP1N2/2P4B/2NBQ1P1/5R1K b - - 0 1", "d6f4", -P, true),
        ];

        for (fen, move_str, threshold, result) in SEE_TESTS {
            let board: Board = Board::from_fen(fen).unwrap();

            let mut move_list = MoveList::new();
            board.generate_moves::<LegalGen>(&mut move_list);

            let move_ = *move_list
                .iter()
                .find(|&move_| move_.to_str(&board) == *move_str)
                .unwrap();

            println!("{}", board.fen());
            assert_eq!(see(&board, move_, Eval(*threshold)), *result);
        }
    }
}
