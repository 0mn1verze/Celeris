use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Move, Piece, board::Board};
use nnue::accummulator::Accumulator;

use crate::{
    History, KillerEntry, MainHistory,
    constants::MAX_DEPTH,
    eval::{Eval, evaluate_nnue},
    movepick::MovePicker,
    search::PVLine,
    tunables::*,
};

use super::{
    Clock, MIN_DEPTH, NodeType, NonPV, Root, TT,
    tt::{TTBound, TTEntry},
};

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct SearchStackEntry {
    killers: KillerEntry,
    curr_move: Move,
    excl_move: Move,
    moved: Option<Piece>,
    eval: Eval,
    move_count: u8,
    in_check: bool,
    ply_from_null: u16,
}

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub main_history: MainHistory,
}

const OFFSET: usize = 2;

#[derive(Debug, Clone)]
pub(crate) struct SearchWorker {
    pub clock: Clock,

    board: Board,

    thread_id: usize,

    stack: [SearchStackEntry; MAX_DEPTH + OFFSET],

    nodes: u64,
    pub depth: usize,
    seldepth: usize,
    ply: u16,

    ply_from_null: u16,

    stats: SearchStats,

    pv: PVLine,
    eval: Eval,

    stop: bool,

    pub nnue: Accumulator,
}

impl SearchWorker {
    pub fn new(thread_id: usize, stop: Arc<AtomicBool>, nodes: Arc<AtomicU64>) -> Self {
        Self {
            clock: Clock::default(stop, nodes),
            thread_id,
            eval: -Eval::INFINITY,
            board: Board::default(),
            stack: [SearchStackEntry::default(); MAX_DEPTH + OFFSET],
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
        self.stats.main_history.clear();
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

    fn ss(&self) -> SearchStackEntry {
        self.stack[self.ply as usize]
    }

    fn ss_mut(&mut self) -> &mut SearchStackEntry {
        &mut self.stack[self.ply as usize]
    }

    fn ss_at(&self, offset: usize) -> SearchStackEntry {
        self.stack[self.ply as usize - offset]
    }

    fn ss_at_mut(&mut self, offset: usize) -> &mut SearchStackEntry {
        &mut self.stack[self.ply as usize - offset]
    }

    pub fn iterative_deepening(&mut self, tt: &TT) {
        self.depth = 0;

        while self.should_start_iteration() {
            self.search_position(tt);

            if self.stop {
                break;
            }

            if self.thread_id == 0 {
                self.print_info(tt);
            }

            self.depth += 1;
        }
    }

    fn search_position(&mut self, tt: &TT) {
        let alpha = -Eval::INFINITY;
        let beta = Eval::INFINITY;

        let mut pv = PVLine::default();

        let eval = self.negamax::<Root>(tt, &mut pv, alpha, beta, self.depth + 1);

        if self.stop {
            return;
        }

        self.eval = eval;

        self.pv = pv;
    }

    fn print_info(&self, tt: &TT) {
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

    fn negamax<NT: NodeType>(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        mut alpha: Eval,
        mut beta: Eval,
        mut depth: usize,
    ) -> Eval {
        let us = self.board.stm();

        pv.clear();

        if self.should_stop_search() {
            return Eval::DRAW;
        }

        let in_check = self.board.in_check();

        if depth == 0 && !in_check {
            return self.quiescence::<NT::Next>(tt, pv, alpha, beta);
        }

        if !NT::ROOT {
            if self.ply >= MAX_DEPTH as u16 && !in_check {
                return self.evaluate();
            }

            if self.board.is_draw() {
                return Eval::DRAW;
            }

            alpha = alpha.max(Eval::mated_in(self.ply));
            beta = beta.min(Eval::mate_in(self.ply + 1));

            if alpha >= beta {
                return alpha;
            }
        }

        depth = depth.max(1);

        self.seldepth = if NT::ROOT {
            0
        } else {
            self.seldepth.max(self.ply as usize)
        };

        let mut child_pv = PVLine::default();

        let tt_entry = tt.get(self.board.key());
        let mut tt_move = Move::NONE;
        let mut tt_value = -Eval::INFINITY;
        let mut tt_bound = TTBound::None;
        let mut tt_capture = false;
        let tt_hit = tt_entry.is_some();

        if let Some(tt_entry) = tt_entry {
            tt_value = tt_entry.value.from_tt(self.ply);
            tt_bound = tt_entry.bound;

            if !NT::PV && tt_entry.depth >= depth as u8 {
                match tt_entry.bound {
                    TTBound::Exact => return tt_value,

                    TTBound::Upper if tt_value <= alpha => return tt_value,

                    TTBound::Lower if tt_value >= beta => return tt_value,
                    _ => {}
                }
            }

            tt_move = tt_entry.best_move;
            tt_capture = self.board.is_capture(tt_move);
        }

        let eval = self.static_eval(in_check, tt_entry);

        let improving = self.improving();

        let opp_worsening = self.opp_worsening();

        if !NT::PV
            && depth >= 2
            && !in_check
            && self.ply_from_null > 0
            && eval >= beta
            && self.board.has_non_pawn_material(us)
            && beta >= -Eval::MATE_BOUND
            && (!tt_hit || tt_bound == TTBound::Lower || tt_value >= beta)
        {
            let r = (nmp_min() + depth / nmp_div()).min(depth);

            self.make_null_move(tt);

            let value =
                -self.negamax::<NonPV>(tt, &mut child_pv, -beta, -beta + Eval(1), depth - r);

            self.undo_null_move();

            if value >= beta {
                return if value > -Eval::MATE_BOUND {
                    beta
                } else {
                    value
                };
            }
        }

        let mut best_value = -Eval::INFINITY;
        let mut best_move = Move::NONE;

        let mut move_count = 0;

        self.stack[(self.ply + 2) as usize].killers.clear();

        let killers = self.ss().killers.get();

        let mut move_picker = MovePicker::<false>::new(&self.board, tt_move, killers);

        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            move_count += 1;

            let start_nodes = self.nodes;

            let is_capture = move_.is_capture();

            let new_depth = depth.max(1) - 1;

            let mut value = alpha;

            self.make_move(tt, move_);

            let full_search = if depth >= lmr_min_depth()
                && move_count >= lmr_min_moves() + NT::PV as usize
            {
                // Late move reductions.
                let mut r = late_move_reduction(depth, move_count);

                // Increase reductions for moves we think might be bad
                r += !NT::PV as usize;

                // Decrease reductions for moves we think might be good
                r -= in_check as usize;

                // We don't want to extend or go into qsearch.
                // Since we have checked for qsearch, depth is guaranteed to be >= 1.
                r = r.clamp(1, depth - 1);

                value =
                    -self.negamax::<NonPV>(tt, &mut child_pv, -alpha - Eval(1), -alpha, depth - r);

                value > alpha
            } else {
                !NT::PV || move_count > 1 || is_capture
            };

            if full_search {
                value =
                    -self.negamax::<NonPV>(tt, &mut child_pv, -alpha - Eval(1), -alpha, new_depth);
            }

            if NT::PV && (move_count == 1 || value > alpha) {
                value = -self.negamax::<NT::Next>(tt, &mut child_pv, -beta, -alpha, new_depth);
            }

            self.undo_move(move_);

            if self.stop {
                return Eval::DRAW;
            }

            if NT::ROOT {
                self.clock
                    .update_node_counts(move_, self.nodes - start_nodes);
            }

            if value > best_value {
                best_value = value;

                if value > alpha {
                    best_move = move_;

                    if value >= beta {
                        break;
                    }

                    alpha = value;
                    if NT::PV {
                        pv.update_line(move_, &child_pv);
                    }
                }
            }
        }

        self.ss_mut().move_count = move_count as u8;

        self.ss_mut().in_check = in_check;

        if move_count == 0 {
            best_value = if in_check {
                Eval::mated_in(self.ply)
            } else {
                Eval::DRAW
            };
        } else if best_move.is_valid() && !best_move.is_capture() {
            self.update_search_stats(best_move, depth);
        }

        if !NT::ROOT {
            let bound = if best_value >= beta {
                TTBound::Lower
            } else if NT::PV && best_move.is_valid() {
                TTBound::Exact
            } else {
                TTBound::Upper
            };

            tt.write(
                self.board.key(),
                bound,
                self.ply,
                depth as u8,
                best_move,
                eval,
                best_value,
            );
        }

        best_value
    }

    fn quiescence<NT: NodeType>(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        mut alpha: Eval,
        beta: Eval,
    ) -> Eval {
        self.seldepth = self.seldepth.max(self.ply as usize);

        pv.clear();

        if self.should_stop_search() {
            return Eval::DRAW;
        }

        let in_check = self.board.in_check();

        if self.ply >= MAX_DEPTH as u16 {
            return if in_check {
                Eval::DRAW
            } else {
                self.evaluate()
            };
        }

        if self.board.is_draw() {
            return Eval::DRAW;
        }

        let tt_entry = tt.get(self.board.key());
        let mut tt_move = Move::NONE;

        if let Some(entry) = tt_entry {
            let tt_value = entry.value.from_tt(self.ply);

            if !NT::PV {
                match entry.bound {
                    TTBound::Exact => return tt_value,
                    TTBound::Upper if tt_value <= alpha => return tt_value,
                    TTBound::Lower if tt_value >= beta => return tt_value,
                    _ => {}
                }
            }

            tt_move = entry.best_move;
        }

        let eval = self.static_eval(in_check, tt_entry);

        if eval >= beta {
            return beta;
        }

        alpha = alpha.max(eval);

        let mut best_value = eval;
        let mut child_pv = PVLine::default();
        let mut best_move = Move::NONE;

        let mut move_picker = MovePicker::<true>::new(&self.board, tt_move, [Move::NONE; 2]);

        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            self.make_move(tt, move_);

            let value = -self.quiescence::<NT::Next>(tt, &mut child_pv, -beta, -alpha);

            self.undo_move(move_);

            if self.stop {
                return Eval::DRAW;
            }

            if value > best_value {
                best_value = value;

                if value > alpha {
                    best_move = move_;

                    if value >= beta {
                        break;
                    }

                    if NT::PV {
                        pv.update_line(move_, &child_pv);
                    }

                    alpha = value;
                }
            }
        }

        if in_check && best_value == -Eval::INFINITY {
            return Eval::mated_in(self.ply);
        }

        let bound = if best_value >= beta {
            TTBound::Lower
        } else {
            TTBound::Upper
        };

        tt.write(
            self.board.key(),
            bound,
            self.ply,
            0,
            best_move,
            eval,
            best_value,
        );

        best_value
    }

    fn make_move(&mut self, tt: &TT, move_: Move) {
        self.board.make_move(move_);

        tt.prefetch(self.board.key());

        self.ss_mut().curr_move = move_;
        self.ss_mut().moved = self.board.on(move_.to());
        self.ss_mut().ply_from_null = self.ply_from_null;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    fn make_null_move(&mut self, tt: &TT) {
        self.board.make_null_move();

        tt.prefetch(self.board.key());

        self.ss_mut().curr_move = Move::NULL;
        self.ss_mut().moved = None;
        self.ss_mut().ply_from_null = 0;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    fn undo_move(&mut self, move_: Move) {
        self.board.undo_move(move_);

        self.ply -= 1;

        self.ply_from_null = self.ss().ply_from_null;
    }

    fn undo_null_move(&mut self) {
        self.board.undo_null_move();

        self.ply -= 1;

        self.ply_from_null = self.ss().ply_from_null;
    }

    fn update_search_stats(&mut self, best_move: Move, depth: usize) {
        self.ss_mut().killers.update(best_move);

        let bonus = (depth * depth) as i16;

        self.stats
            .main_history
            .update(&self.board, best_move, bonus);
    }

    fn static_eval(&mut self, in_check: bool, tt_entry: Option<TTEntry>) -> Eval {
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

            match tt_entry.bound {
                TTBound::Exact => tt_value,

                TTBound::Upper if tt_value <= eval => tt_value,

                TTBound::Lower if tt_value >= eval => tt_value,

                _ => eval,
            }
        } else {
            self.ss_mut().eval = self.evaluate();
            self.ss().eval
        }
    }

    fn evaluate(&mut self) -> Eval {
        evaluate_nnue(&self.board, &mut self.nnue)
    }

    fn improving(&self) -> bool {
        if self.ply >= 2 && self.stack[self.ply as usize - 2].eval != -Eval::INFINITY {
            self.stack[self.ply as usize].eval > self.stack[self.ply as usize - 2].eval
        } else if self.ply >= 4 && self.stack[self.ply as usize - 4].eval != -Eval::INFINITY {
            self.stack[self.ply as usize].eval > self.stack[self.ply as usize - 4].eval
        } else {
            false
        }
    }

    fn opp_worsening(&self) -> bool {
        if self.ply >= 1 {
            self.stack[self.ply as usize].eval > -self.stack[self.ply as usize - 1].eval
        } else {
            false
        }
    }
}

fn late_move_reduction(depth: usize, move_count: usize) -> usize {
    let a = (lmr_a() as f32) / 100.0;
    let b = (lmr_b() as f32) / 100.0;
    (a + (depth as f32).ln() * (move_count as f32).ln() / b) as usize
}
