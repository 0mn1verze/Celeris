use chess::{
    Move, MoveFlag, PieceType,
    board::{Board, CaptureGen, MoveList, QuietGen},
};

use crate::{
    SearchStackEntry,
    constants::{CONT_HIST_SIZE, MVV},
    eval::Eval,
    movepick::history::Interface,
    search::SearchStats,
};

use super::{MoveStage, see::see};

pub struct MovePicker<const TACTICAL: bool> {
    pub stage: MoveStage,
    skip_quiets: bool,

    tt_move: Move,
    killers: [Move; 2],

    // Scored move list items
    move_list: MoveList,
    scores: [i32; 256],
    index: usize,

    quiet_start: usize,
    bad_cap_start: usize,
}

fn captured_value(captured: PieceType) -> i32 {
    MVV[captured.index()]
}

impl<const TACTICAL: bool> MovePicker<TACTICAL> {
    pub fn new(board: &Board, mut tt_move: Move, mut killers: [Move; 2]) -> Self {
        let in_check = board.in_check();

        if TACTICAL && !in_check && board.is_capture(tt_move) {
            tt_move = Move::NONE;
        }

        // A valid killer must be legal and a non capture
        if !killers[0].is_valid() || !board.is_legal(killers[0]) {
            killers[0] = Move::NONE;
        }

        // A valid killer must be legal and a non capture
        if !killers[1].is_valid() || !board.is_legal(killers[1]) {
            killers[1] = Move::NONE;
        }

        Self {
            stage: MoveStage::TTMove,
            skip_quiets: !in_check && TACTICAL,
            tt_move,
            killers,
            move_list: MoveList::new(),
            scores: [0; 256],
            index: 0,
            quiet_start: 0,
            bad_cap_start: 0,
        }
    }

    pub fn skip_quiets(&mut self) {
        self.skip_quiets = true;
    }

    fn score_captures(&mut self, board: &Board, stats: &SearchStats) {
        let mut next_good_cap = 0;

        for i in 0..self.move_list.len() {
            let move_ = self.move_list[i];

            let captured = match move_.flag() {
                MoveFlag::EPCapture => PieceType::Pawn,
                _ => unsafe { board.on_unchecked(move_.to()).pt() },
            };

            let mut score = captured_value(captured) + stats.cht.get(board, move_).0 as i32;

            if move_.is_promotion() {
                score += captured_value(unsafe { move_.promotion_pt() });
            }

            self.scores[i] = score;

            if see(board, move_, Eval::ZERO) {
                self.scores.swap(i, next_good_cap);
                self.move_list.swap(i, next_good_cap);
                next_good_cap += 1;
            }
        }

        self.bad_cap_start = next_good_cap;
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        stats: &SearchStats,
        ss_buffer: &[SearchStackEntry; CONT_HIST_SIZE],
    ) {
        for i in self.quiet_start..self.move_list.len() {
            let move_ = self.move_list[i];

            self.scores[i] = stats.ht.get(board, move_).0 as i32;

            for i in 0..CONT_HIST_SIZE {
                let (piece, to) = ss_buffer[i].piece_to();
                self.scores[i] += stats.ct.get_entry_ref(piece, to).get(board, move_).0 as i32;
            }
        }
    }

    fn gen_quiets(
        &mut self,
        board: &Board,
        stats: &SearchStats,
        ss_buffer: &[SearchStackEntry; CONT_HIST_SIZE],
    ) {
        board.generate_moves::<QuietGen>(&mut self.move_list);

        self.score_quiets(board, stats, ss_buffer);
        self.partial_sort(self.quiet_start, self.move_list.len());
    }

    fn gen_captures(&mut self, board: &Board, stats: &SearchStats) {
        board.generate_moves::<CaptureGen>(&mut self.move_list);
        self.quiet_start = self.move_list.len();
        self.bad_cap_start = 0;

        self.score_captures(board, stats);
        self.partial_sort(0, self.bad_cap_start);
        self.partial_sort(self.bad_cap_start, self.move_list.len());
    }

    fn partial_sort(&mut self, start: usize, end: usize) {
        for i in start..end {
            let mut j = i;
            let move_ = self.move_list[j];
            let score = self.scores[j];

            while j > start && self.scores[j - 1] < self.scores[j] {
                self.move_list.swap(j, j - 1);
                self.scores.swap(j, j - 1);
                j -= 1;
            }

            self.scores[j] = score;
            self.move_list[j] = move_;
        }
    }

    fn next_best<F>(&mut self, end: usize, pred: F) -> Option<Move>
    where
        F: Fn(Move) -> bool,
    {
        let slice_len = end.saturating_sub(self.index); // How many items to potentially look at

        let found = self
            .move_list
            .iter()
            .skip(self.index)
            .take(slice_len)
            .enumerate()
            .find(|&(_i, &m)| pred(m) && m != self.tt_move); // Find the first suitable move

        // Update the main index based on whether a move was found
        self.index = found.map_or(end, |(i, _m)| self.index + i + 1);

        found.map(|(_, m)| *m) // Return the move itself if found
    }

    pub(crate) fn next(
        &mut self,
        board: &Board,
        stats: &SearchStats,
        ss_buffer: &[SearchStackEntry; CONT_HIST_SIZE],
    ) -> Option<Move> {
        let killers = self.killers;
        let cap_pred = |_| true;
        let quiet_pred = |move_: Move| move_ != killers[0] && move_ != killers[1];

        match self.stage {
            MoveStage::TTMove => {
                self.stage = MoveStage::GenCaptures;
                if self.tt_move.is_valid() {
                    Some(self.tt_move)
                } else {
                    self.next(board, stats, ss_buffer)
                }
            }
            MoveStage::GenCaptures => {
                self.stage = MoveStage::GoodCaptures;
                self.gen_captures(board, stats);
                self.next(board, stats, ss_buffer)
            }
            MoveStage::GoodCaptures => {
                if let Some(move_) = self.next_best(self.bad_cap_start, cap_pred) {
                    Some(move_)
                } else {
                    self.stage = if self.skip_quiets {
                        self.index = self.bad_cap_start;
                        MoveStage::BadCaptures
                    } else {
                        MoveStage::Killer1
                    };
                    self.next(board, stats, ss_buffer)
                }
            }
            MoveStage::Killer1 => {
                self.stage = MoveStage::Killer2;
                if !self.skip_quiets
                    && self.killers[0].is_valid()
                    && self.killers[0] != self.tt_move
                {
                    Some(self.killers[0])
                } else {
                    self.next(board, stats, ss_buffer)
                }
            }
            MoveStage::Killer2 => {
                self.stage = MoveStage::GenQuiets;
                if !self.skip_quiets
                    && self.killers[1].is_valid()
                    && self.killers[1] != self.tt_move
                {
                    Some(self.killers[1])
                } else {
                    self.next(board, stats, ss_buffer)
                }
            }
            MoveStage::GenQuiets => {
                if !self.skip_quiets {
                    self.gen_quiets(board, stats, ss_buffer);

                    self.stage = MoveStage::Quiets;
                    self.index = self.quiet_start;
                } else {
                    self.stage = MoveStage::BadCaptures;
                    self.index = self.bad_cap_start;
                }
                self.next(board, stats, ss_buffer)
            }
            MoveStage::Quiets => {
                let next_best = self.next_best(self.move_list.len(), quiet_pred);
                if !self.skip_quiets && next_best.is_some() {
                    next_best
                } else {
                    self.stage = MoveStage::BadCaptures;
                    self.index = self.bad_cap_start;
                    self.next(board, stats, ss_buffer)
                }
            }
            MoveStage::BadCaptures => {
                if let Some(move_) = self.next_best(self.quiet_start, cap_pred) {
                    Some(move_)
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use chess::board::*;

    #[rustfmt::skip]
    const BENCH_LIST: &[(&str, usize, usize)] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 7, 178633661),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033),
        ("1k6/1b6/8/8/7R/8/8/4K2R b K - 0 1", 5, 1063513), 
        ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888), 
        ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133), 
        ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467), 
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072), 
        ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711), 
        ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206), 
        ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476), 
        ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001), 
        ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1004658), 
        ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217342), 
        ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683), 
        ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217), 
        ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584), 
        ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527), 
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1 ", 6, 764643),
        ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1 ", 6, 846648),
        ("4k2r/8/8/8/8/8/8/4K3 w k - 0 1 ", 6, 899442),
        ("r3k3/8/8/8/8/8/8/4K3 w q - 0 1 ", 6, 1001523),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ", 5, 89941194),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ", 5, 164075551),
        ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001),
        ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217),
        ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467),
        ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888),
        ("rbbknnqr/pppppppp/8/8/8/8/PPPPPPPP/RBBKNNQR w KQkq - 0 1", 6, 124381396),
        ("bnrkrnqb/pppppppp/8/8/8/8/PPPPPPPP/BNRKRNQB w KQkq - 0 1", 6, 146858295),
        ("nrbbqknr/pppppppp/8/8/8/8/PPPPPPPP/NRBBQKNR w KQkq - 0 1", 6, 97939069),
        ("bnrbnkrq/pppppppp/8/8/8/8/PPPPPPPP/BNRBNKRQ w KQkq - 0 1", 6, 145999259),
        ("rbknqnbr/pppppppp/8/8/8/8/PPPPPPPP/RBKNQNBR w KQkq - 0 1", 6, 126480040),
        ("qbrnnkbr/pppppppp/8/8/8/8/PPPPPPPP/QBRNNKBR w KQkq - 0 1", 6, 121613156),
    ];

    fn perft(board: &mut Board, depth: usize) -> usize {
        let mut move_list = MoveList::new();

        board.generate_moves::<LegalGen>(&mut move_list);

        if depth == 1 {
            return move_list.len();
        }

        let mut move_picker =
            MovePicker::<false>::new(&board, Move::NONE, [Move::NONE, Move::NONE]);
        let search_stats = SearchStats::default();

        let ss_buffer = [SearchStackEntry::default(); CONT_HIST_SIZE];

        let mut nodes = 0;

        while let Some(move_) = move_picker.next(board, &search_stats, &ss_buffer) {
            board.make_move(move_);
            nodes += perft(board, depth - 1);
            board.undo_move(move_);
        }

        nodes
    }

    fn perft_bench() {
        use std::time::Instant;

        println!("=============  START BENCH  =============");

        for (fen, depth, expected_nodes) in BENCH_LIST.iter() {
            let start = Instant::now();
            let mut board = Board::from_fen(fen).unwrap();
            let nodes = perft(&mut board, *depth);

            let status: &str = if nodes == *expected_nodes {
                "PASSED"
            } else {
                "FAILED"
            };

            assert!(nodes == *expected_nodes);

            let time = start.elapsed().as_millis();

            println!(
                "status: {status}, time: {time:4}ms, Mnps: {:0.1}, Fen: {fen} )",
                (nodes as f64 / time as f64 / 1000.0)
            )
        }
    }

    #[test]
    fn test_perft() {
        perft_bench();
    }
}
