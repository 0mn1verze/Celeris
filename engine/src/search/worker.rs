use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Move, board::Board};
use nnue::accumulator::Accumulator;

use crate::{
    History, MoveBuffer, SearchStackEntry, SearchStats, SearchWorker,
    constants::{MAX_DEPTH, MIN_DEPTH, SEARCH_STACK_OFFSET},
    eval::{Eval, evaluate_nnue},
    search::PVLine,
};

use super::{Clock, TT, helper::*, tt::TTEntry};

impl SearchWorker {
    pub fn new(thread_id: usize, stop: Arc<AtomicBool>, nodes: Arc<AtomicU64>) -> Self {
        Self {
            clock: Clock::default(stop, nodes),
            thread_id,
            eval: -Eval::INFINITY,
            board: Board::default(),
            stack: [SearchStackEntry::default(); MAX_DEPTH + SEARCH_STACK_OFFSET],
            nodes: 0,
            seldepth: 0,
            depth: 0,
            ply: 0,
            ply_from_null: 0,
            pv: PVLine::default(),
            stop: false,
            stats: SearchStats::default(),
            nnue: Accumulator::default(),
        }
    }

    pub fn thread_id(&self) -> usize {
        self.thread_id
    }

    pub fn reset(&mut self) {
        self.stats.ht.clear();
        self.stats.cht.clear();
    }

    pub fn prepare_search(&mut self) {
        self.clock.last_nodes = 0;
        self.nodes = 0;
        self.seldepth = 0;
        self.ply = 0;
        self.ply_from_null = 0;
        self.pv = PVLine::default();
        self.eval = -Eval::INFINITY;
        self.stop = false;
    }

    pub fn setup(&mut self, board: Board) {
        self.board = board;
    }

    pub fn best_move(&self) -> Move {
        self.pv[0]
    }

    pub fn should_start_iteration(&mut self) -> bool {
        self.depth < MAX_DEPTH
            && self
                .clock
                .start_search(self.depth + 1, self.nodes, self.best_move())
    }

    pub fn should_stop_search(&mut self) -> bool {
        let should_stop =
            self.depth >= MIN_DEPTH && (self.stop || !self.clock.continue_search(self.nodes));
        if should_stop {
            self.stop = true;
        }
        should_stop
    }

    pub(super) fn ss(&self) -> SearchStackEntry {
        self.stack[self.ply as usize]
    }

    pub(super) fn ss_mut(&mut self) -> &mut SearchStackEntry {
        &mut self.stack[self.ply as usize]
    }

    pub(super) fn ss_at(&self, offset: usize) -> SearchStackEntry {
        self.stack[self.ply as usize - offset]
    }

    pub(super) fn ss_at_mut(&mut self, offset: usize) -> &mut SearchStackEntry {
        &mut self.stack[self.ply as usize - offset]
    }

    pub(super) fn print_info(&self, tt: &TT) {
        let time = self.clock.elapsed().as_millis();

        let nodes_per_second = (self.clock.global_nodes() * 1000) as u128 / time.max(1);

        println!(
            "info depth {} seldepth {} score {} time {} nodes {} nps {} hashfull {} {}",
            self.depth + 1,
            self.seldepth + 1,
            self.eval,
            time,
            self.clock.global_nodes(),
            nodes_per_second,
            tt.hashfull(),
            self.pv.to_str(&self.board)
        );
    }

    pub(super) fn make_move(&mut self, tt: &TT, move_: Move) {
        self.board.make_move(move_);

        tt.prefetch(self.board.key());

        self.ss_mut().curr_move = move_;
        self.ss_mut().moved = self.board.on(move_.to());
        self.ss_mut().ply_from_null = self.ply_from_null;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    pub(super) fn make_null_move(&mut self, tt: &TT) {
        self.board.make_null_move();

        tt.prefetch(self.board.key());

        self.ss_mut().curr_move = Move::NULL;
        self.ss_mut().moved = None;
        self.ss_mut().ply_from_null = 0;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    pub(super) fn undo_move(&mut self, move_: Move) {
        self.board.undo_move(move_);

        self.ply -= 1;

        self.ply_from_null = self.ss().ply_from_null;
    }

    pub(super) fn undo_null_move(&mut self) {
        self.board.undo_null_move();

        self.ply -= 1;

        self.ply_from_null = self.ss().ply_from_null;
    }

    pub(super) fn update_search_stats(
        &mut self,
        best_move: Move,
        depth: usize,
        caps_tried: &MoveBuffer,
        quiets_tried: &MoveBuffer,
    ) {
        let bonus = calculate_bonus(depth);

        if !best_move.is_capture() {
            self.ss_mut().killers.update(best_move);
            self.stats.ht.update(&self.board, best_move, bonus);

            for &move_ in quiets_tried {
                self.stats.ht.update(&self.board, move_, -bonus);
            }
        } else {
            self.stats.cht.update(&self.board, best_move, bonus);
        }

        for &move_ in caps_tried {
            self.stats.cht.update(&self.board, move_, -bonus);
        }
    }

    pub(super) fn static_eval(&mut self, in_check: bool, tt_entry: Option<TTEntry>) -> Eval {
        if in_check {
            self.ss_mut().eval = -Eval::INFINITY;
            self.ss().eval
        } else if let Some(tt_entry) = tt_entry {
            let tt_eval = tt_entry.eval;
            let tt_value = tt_entry.value.from_tt(self.ply);

            let eval = if tt_eval == -Eval::INFINITY {
                self.evaluate()
            } else {
                tt_eval
            };

            self.ss_mut().eval = eval;

            // If we probe the tt_entry and the tt_value is tighter than the eval, then we can use it
            if can_use_tt_value(tt_entry.bound, tt_value, eval, eval) {
                tt_value
            } else {
                eval
            }
        } else {
            // self.ss_mut().eval = evaluate_nnue(&self.board, &mut self.nnue);
            self.ss_mut().eval = self.evaluate();
            self.ss().eval
        }
    }

    pub(super) fn evaluate(&mut self) -> Eval {
        evaluate_nnue(&self.board, &mut self.nnue)
    }

    pub(super) fn improving(&self) -> bool {
        if self.ply >= 2 {
            self.ss().eval > self.ss_at(2).eval
        } else {
            false
        }
    }

    pub(super) fn opp_worsening(&self) -> bool {
        if self.ply >= 1 {
            self.ss().eval > -self.ss_at(1).eval
        } else {
            false
        }
    }

    pub(super) fn can_do_nmp(&self, depth: usize, eval: Eval, beta: Eval) -> bool {
        depth >= 2
            && self.ply_from_null > 0
            && eval >= beta
            && self.board.has_non_pawn_material(self.board.stm())
            && beta >= -Eval::MATE_BOUND
    }
}
