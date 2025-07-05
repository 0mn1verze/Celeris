use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Move, Piece, board::Board};
use nnue::accumulator::Accumulator;

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
    // Search Clock
    pub clock: Clock,
    // Main Board
    board: Board,

    thread_id: usize,

    // Search Stack
    stack: [SearchStackEntry; MAX_DEPTH + OFFSET],

    // Search Info
    nodes: u64,
    pub depth: usize,
    seldepth: usize,
    ply: u16,
    // Plies from previous null move
    ply_from_null: u16,

    // Move orderin statistics
    stats: SearchStats,

    // Search results
    pv: PVLine,
    eval: Eval,

    // Search thread internal stop flag
    stop: bool,

    // NNUE
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

        // --- Quiescence search in base case ---
        if depth == 0 && !in_check {
            return self.quiescence::<NT::Next>(tt, pv, alpha, beta);
        }

        if !NT::ROOT {
            // Check ply limit to prevent infinite recursion in rare cases
            if self.ply >= MAX_DEPTH as u16 && !in_check {
                // return evaluate_nnue(&self.board, &mut self.nnue); // Return static eval if too deep
                return self.evaluate();
            }
            // Check for draws (Repetition, 50-move rule)
            if self.board.is_draw(self.ply_from_null) {
                return Eval::DRAW;
            }

            alpha = alpha.max(Eval::mated_in(self.ply));
            beta = beta.min(Eval::mate_in(self.ply + 1));

            if alpha >= beta {
                return alpha;
            }
        }

        // --- Set up Search ---
        // Make sure the depth is not going to be negative
        depth = depth.max(1);
        // Update self depth
        self.seldepth = if NT::ROOT {
            0
        } else {
            self.seldepth.max(self.ply as usize)
        };

        // Create child pv to pass to the next recursive call to negamax
        let mut child_pv = PVLine::default();

        // --- Hash Table Lookup ---
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
                    // If we already evaluated this position at a higher depth, then we can prune this branch (Redundant).
                    TTBound::Exact => return tt_value,
                    // If we already established an upper bound for alpha, if we have some better moves then we can prune this branch.
                    TTBound::Upper if tt_value <= alpha => return tt_value,
                    // If we already established a lower bound for beta, if the opponent has some better moves then this node is too good.
                    TTBound::Lower if tt_value >= beta => return tt_value,
                    _ => {}
                }
            }

            // Update best move from hash table
            tt_move = tt_entry.best_move;
            tt_capture = self.board.is_capture(tt_move);
        }
        let eval = self.static_eval(in_check, tt_entry);

        let improving = self.improving();

        let opp_worsening = self.opp_worsening();

        // --- Null Move Pruning ---
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

        // --- Set up main loop ---
        let mut best_value = -Eval::INFINITY;
        let mut best_move = Move::NONE;

        let mut move_count = 0;

        // Clear child killer moves
        self.stack[(self.ply + 2) as usize].killers.clear();
        // Get killer moves
        let killers = self.ss().killers.get();
        // Initialise move picker
        let mut move_picker = MovePicker::<false>::new(&self.board, tt_move, killers);

        // --- Main Loop ---
        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            // Update number of moves searched in this node
            move_count += 1;
            // Make move and update ply, node counters, prefetch hash entry, etc...
            self.make_move(tt, move_);
            // Remember previous node count
            let start_nodes = self.nodes;
            // Move flags
            let is_capture = move_.is_capture();
            // let is_promotion = move_.is_promotion();
            // New depth
            let new_depth = depth.max(1) - 1;

            // Recursive search
            let mut value = alpha;

            // Late Move Reduction, search moves that are sufficiently far from the terminal nodes and are not tactical using a reduced depth zero window search to see if it is promising or not.
            let full_search = if depth >= 2 && move_count > 2 + NT::PV as usize {
                // Calculate dynamic depth reduction
                let mut r = self.lmr_base_reduction(depth, move_count);

                // Increase reductions for moves we think might be bad
                r += !NT::PV as usize;
                r += !improving as usize;
                r += tt_capture as usize;

                // Decrease reductions for moves we think might be good
                r -= in_check as usize;
                r -= self.board.in_check() as usize;

                // We don't want to extend or go into qsearch.
                // Since we have checked for qsearch, depth is guaranteed to be >= 1.
                r = r.clamp(1, depth - 1);

                value = -self.negamax::<NonPV>(
                    tt,
                    &mut child_pv,
                    -alpha - Eval(1),
                    -alpha,
                    new_depth - r,
                );

                value > alpha && r > 1
            } else {
                !NT::PV || move_count > 1
            };

            // We do a zero window full depth search if the reduced depth search revealed a potentially better move,
            // or if other conditions like being in a NonPV node, searching non-first moves, are met.
            if full_search {
                value =
                    -self.negamax::<NonPV>(tt, &mut child_pv, -alpha - Eval(1), -alpha, new_depth);
            }

            // If we are current searching in a PV node, the if its the first move or further searches revealed a potentially better move,
            // we do a full search to further investigate
            if NT::PV && (move_count == 1 || value > alpha) {
                value = -self.negamax::<NT::Next>(tt, &mut child_pv, -beta, -alpha, new_depth);
            }

            // Undo move and decrement ply counter
            self.undo_move(move_);

            // Check for engine stop flag
            if self.stop {
                return Eval::DRAW;
            }

            // If this is a root node, update the node counts.
            if NT::ROOT {
                self.clock
                    .update_node_counts(move_, self.nodes - start_nodes);
            }

            // If the move we just search is better than best_value (The best we can do in this subtree), we can update best_value to be alpha.
            if value > best_value {
                best_value = value; // Update best_value
                // Alpha Update: Check if this move's score (`value`) is better than the
                // best score we are *already guaranteed* (`alpha`) from other parts of the tree.
                if value > alpha {
                    // We found a new best move sequence overall.
                    best_move = move_; // Update the best move.

                    // Beta Cutoff (Fail-High): Check if our guaranteed score (`alpha`)
                    // meets or exceeds the opponent's limit (`beta`).
                    // This move is "too good". The opponent (at a higher node)
                    // would have already had a better alternative than allowing this position.
                    // Therefore, exploring further sibling moves at this node is unnecessary.
                    if value >= beta {
                        break;
                    }

                    alpha = value; // Update alpha: Raise the lower bound of our guaranteed score.
                    if NT::PV {
                        pv.update_line(move_, &child_pv); // Update the Principal Variation (best move sequence).
                    }
                }
            }
        }

        // Update search stack move count
        self.ss_mut().move_count = move_count as u8;
        // Update search stack in check flag
        self.ss_mut().in_check = in_check;

        // If move count is 0, it is either a stalemate or a mate in self.ply
        if move_count == 0 {
            best_value = if in_check {
                Eval::mated_in(self.ply)
            } else {
                Eval::DRAW
            };
            // If there is a new best move, and it is not a capture, update the killer move and history move table
        } else if best_move.is_valid() && !best_move.is_capture() {
            self.update_search_stats(best_move, depth);
        }

        // Write to TT, save static eval
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

        // Check ply limit to prevent infinite recursion in rare cases
        if self.ply >= MAX_DEPTH as u16 {
            // evaluate_nnue(&self.board, &mut self.nnue)
            return if in_check {
                Eval::DRAW
            } else {
                self.evaluate()
            };
        }

        // Check for draws (Repetition, 50-move rule)
        if self.board.is_draw(self.ply_from_null) {
            return Eval::DRAW;
        }

        // --- Hash table probe ---
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

        // --- Stand Pat Score ---
        // Get the static evaluation of the current position.
        // This score assumes no further captures are made (the "stand pat" score).
        // let eval = self.static_eval(in_check, tt_entry);
        let eval = self.static_eval(in_check, tt_entry);
        // --- Alpha-Beta Pruning based on Stand Pat ---
        // If the static evaluation is already >= beta, the opponent won't allow this position.
        // We can prune immediately, assuming the static eval is a reasonable lower bound.
        if eval >= beta {
            return beta; // Fail-High based on static eval
        }
        // If the static evaluation is better than alpha, update alpha.
        // This becomes the baseline score we need to beat with captures.
        alpha = alpha.max(eval);

        // Initialize best_value with stand_pat. We are looking for captures that improve on this.
        let mut best_value = eval;
        let mut child_pv = PVLine::default();
        let mut best_move = Move::NONE;

        // --- Generate and Explore Captures Only ---
        // The generic parameter 'true' tells MovePicker to skip quiet moves.
        let mut move_picker = MovePicker::<true>::new(&self.board, tt_move, [Move::NONE; 2]);

        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            // Make the capture
            self.make_move(tt, move_);
            // Recursive call
            let value = -self.quiescence::<NT::Next>(tt, &mut child_pv, -beta, -alpha);
            // Undo the capture
            self.undo_move(move_);

            // Check for stop signal after recursive call
            if self.stop {
                return Eval::DRAW;
            }

            // If the move we just search is better than best_value (The best we can do in this subtree), we can update best_value to be alpha.
            if value > best_value {
                // Update best_value
                best_value = value;
                // Alpha Update: Check if this move's score (`value`) is better than the
                // best score we are *already guaranteed* (`alpha`) from other parts of the tree.
                if value > alpha {
                    // We found a new best move sequence overall.
                    best_move = move_; // Update the best move.

                    // Beta Cutoff (Fail-High): Check if our guaranteed score (`alpha`)
                    // meets or exceeds the opponent's limit (`beta`).
                    // This move is "too good". The opponent (at a higher node)
                    // would have already had a better alternative than allowing this position.
                    // Therefore, exploring further sibling moves at this node is unnecessary.
                    if value >= beta {
                        break;
                    }

                    if NT::PV {
                        pv.update_line(move_, &child_pv); // Update the Principal Variation (best move sequence).
                    }

                    alpha = value; // Update alpha: Raise the lower bound of our guaranteed score.
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

            // If we probe the tt_entry and the tt_value is tighter than the eval, then we can use it
            match tt_entry.bound {
                // If the current node has already been searched to a higher depth, then the tt_value will be a better score.
                TTBound::Exact => tt_value,
                // If the current node has a upper bound, then if the current eval is greater than the tt_value, it means the current eval is optimistic so we should use tt_value instead
                TTBound::Upper if tt_value <= eval => tt_value,
                // If the current node has a lower bound, then if the current eval is lower than the tt_value, it means we are underestimating our opponent's response, so we should use tt_value instead
                TTBound::Lower if tt_value >= eval => tt_value,
                // If there is no stored value, or the above conditions are not met, we use eval, since it provides a tighter bound
                _ => eval,
            }
        } else {
            // self.ss_mut().eval = evaluate_nnue(&self.board, &mut self.nnue);
            self.ss_mut().eval = self.evaluate();
            self.ss().eval
        }
    }

    fn evaluate(&mut self) -> Eval {
        evaluate_nnue(&self.board, &mut self.nnue)
    }

    fn improving(&self) -> bool {
        if self.ply >= 2 {
            self.ss().eval > self.ss_at(2).eval
        } else {
            false
        }
    }

    fn opp_worsening(&self) -> bool {
        if self.ply >= 1 {
            self.ss().eval > -self.ss_at(1).eval
        } else {
            false
        }
    }

    fn lmr_base_reduction(&self, depth: usize, move_count: usize) -> usize {
        if depth == 0 || move_count == 0 {
            return 0;
        }

        let lmr_base = lmr_base() as f32 / 1024.0;
        let lmr_mult = lmr_mult() as f32 / 1024.0;

        (lmr_base + (depth as f32).ln() * (move_count as f32).ln() / lmr_mult) as usize
    }
}
