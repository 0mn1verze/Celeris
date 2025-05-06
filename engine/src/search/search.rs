use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Board, Move};

use crate::{
    History, KillerTable, MainHistory,
    engine::MAX_DEPTH,
    eval::{Eval, PawnTable, evaluate},
    movepick::MovePicker,
    search::{PVLine, SearchStack},
};

use super::{
    Clock, MIN_DEPTH, NodeTypeTrait, NonPV, PV, Root, TT,
    tt::{self, TTBound},
};

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub killers: KillerTable,
    pub main_history: MainHistory,
}

#[derive(Debug, Clone)]
pub(crate) struct SearchWorker {
    pub clock: Clock,
    board: Board,

    thread_id: usize,
    stack: SearchStack,

    nodes: u64,
    depth: usize,
    seldepth: usize,
    ply: u16,

    stats: SearchStats,

    pv: PVLine,
    eval: Eval,

    stop: bool,
    // Tables
    pub pawn_table: PawnTable,
}

impl SearchWorker {
    pub fn new(thread_id: usize, stop: Arc<AtomicBool>, nodes: Arc<AtomicU64>) -> Self {
        Self {
            clock: Clock::default(stop, nodes),
            thread_id,
            eval: -Eval::INFINITY,
            board: Board::default(),
            stack: SearchStack::default(),
            nodes: 0,
            seldepth: 0,
            depth: 0,
            ply: 0,
            pv: PVLine::default(),
            stop: false,
            pawn_table: PawnTable::new(),
            stats: SearchStats::default(),
        }
    }

    pub fn thread_id(&self) -> usize {
        self.thread_id
    }

    pub fn reset(&mut self) {
        self.stats.main_history.clear();
        self.stats.killers.clear();
    }

    pub fn prepare_search(&mut self) {
        self.stack.clear();
        self.clock.last_nodes = 0;
        self.nodes = 0;
        self.seldepth = 0;
        self.ply = 0;
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

    // Search control
    pub fn should_start_iteration(&mut self) -> bool {
        self.depth < MAX_DEPTH
            && self
                .clock
                .start_search(self.depth + 1, self.nodes, self.best_move())
    }

    pub fn should_stop_search(&mut self) -> bool {
        // Search up to at least depth 3
        let should_stop =
            self.depth >= MIN_DEPTH && (self.stop || !self.clock.continue_search(self.nodes));
        if should_stop {
            self.stop = true;
        }
        should_stop
    }

    pub fn start_search(&mut self, tt: &TT) {
        self.iterative_deepening(tt);
    }

    fn iterative_deepening(&mut self, tt: &TT) {
        self.depth = 0;

        while self.should_start_iteration() {
            self.search_position(tt);

            if self.stop {
                break;
            }

            self.depth += 1;
        }

        println!("bestmove {}", self.pv[0]);
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

        self.print_info(tt);
    }

    fn print_info(&self, tt: &TT) {
        let time = self.clock.elapsed().as_millis();

        let nodes_per_second = (self.nodes * 1000) as u128 / time.max(1);

        println!(
            "info depth {} seldepth {} score {} time {} nodes {} nps {} hashfull {} {}",
            self.depth + 1,
            self.seldepth + 1,
            self.eval,
            time,
            self.nodes,
            nodes_per_second,
            tt.hashfull(),
            self.pv
        );
    }

    /// Performs a Negamax search with Alpha-Beta pruning to find the best move and evaluation.
    ///
    /// This is the core recursive function of the chess engine's search algorithm. It explores
    /// the game tree from the current board position (`self.board`) up to a specified `depth`.
    /// It evaluates positions from the perspective of the player whose turn it is (`self.board.stm()`).
    ///
    /// # Algorithm
    ///
    /// Implements the Negamax framework, which simplifies Minimax by always maximizing the score
    /// relative to the current player. The score of the opponent's moves is negated (`-negamax(...)`).
    ///
    /// Alpha-Beta pruning is used to significantly reduce the search space by cutting off branches
    /// that cannot possibly influence the final score within the current `(alpha, beta)` window.
    ///
    /// # Parameters
    ///
    /// * `NT`: A generic type parameter implementing `NodeTypeTrait`. This distinguishes between
    ///   the `Root` (the initial call) and `PV` (recursive calls), allowing for
    ///   slightly different behavior (e.g., root move printing, draw checks).
    /// * `pv`: A mutable reference to a `PVLine` struct. This will be updated to store the
    ///   Principal Variation (the sequence of best moves found) for the current node.
    /// * `alpha`: The lower bound of the search window (`Eval`). Represents the minimum score
    ///   that the *current player* is already guaranteed to achieve based on previously explored
    ///   moves. The function aims to find a score greater than `alpha`.
    /// * `beta`: The upper bound of the search window (`Eval`). Represents the maximum score
    ///   that the *opponent* will allow the current player to achieve. If a move results in a
    ///   score `>= beta`, the opponent would have had a better alternative earlier, so this
    ///   branch can be pruned (Beta Cutoff).
    /// * `depth`: The remaining search depth (`usize`). The search stops when `depth` reaches 0.
    ///
    /// # Returns
    ///
    /// An `Eval` representing the score of the current position from the perspective of the
    /// player whose turn it is. The score is guaranteed to be within the `(alpha, beta)`
    /// window if a cutoff occurs.
    /// - If the exact score is found, it will be between `alpha` and `beta`.
    /// - If `score >= beta` (Beta Cutoff / Fail-High), the returned score is at least `beta`.
    /// - If `score <= alpha` (Fail-Low), the returned score is at most `alpha`.
    ///
    /// Returns `Eval::DRAW` immediately if the search is stopped (`self.stop == true`).
    ///
    /// # Pruning Logic
    ///
    /// 1.  **Alpha Update:** If a move yields a `value` better than the current `alpha`, `alpha` is updated
    ///     to `value`. This means we've found a better guaranteed score for the current player.
    ///     The `pv` line is also updated with this better move sequence.
    /// 2.  **Beta Cutoff (Fail-High):** If a move yields a `value >= beta`, it means this move is "too good".
    ///     The opponent (at a higher node) would have already had a better alternative than allowing
    ///     this position. Therefore, exploring further sibling moves at this node is unnecessary.
    ///     The function immediately returns the current `best_value` (which is `>= beta`).
    fn negamax<NT: NodeTypeTrait>(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        mut alpha: Eval,
        beta: Eval,
        mut depth: usize,
    ) -> Eval {
        // Clear pv for new search
        pv.clear();

        if self.should_stop_search() {
            return Eval::DRAW;
        }

        let in_check = self.board.in_check();

        // --- Quiescence search in base case ---
        if depth == 0 && !in_check {
            return self.quiescence::<NT>(tt, pv, alpha, beta);
        }

        if self.is_draw::<NT>() {
            return Eval::DRAW;
        }

        // --- Set up Search ---
        // Make sure the depth is not going to be negative
        depth = depth.max(1);
        // Update self depth
        self.update_seldepth::<NT>();
        // Create child pv to pass to the next recursive call to negamax
        let mut child_pv = PVLine::default();

        // Reset killer child nodes
        self.stats.killers.clear_child(self.ply);

        // --- Hash Table Lookup ---

        let tt_entry = tt.get(self.board.key());
        let mut tt_move = Move::NONE;

        if let Some(entry) = tt_entry {
            let tt_value = entry.value.from_tt(self.ply);

            if !NT::PV
                && entry.depth >= depth as u8
                && (entry.bound == TTBound::Exact
                    || entry.bound == TTBound::Lower && tt_value >= beta
                    || entry.bound == TTBound::Upper && tt_value <= alpha)
            {
                return tt_value;
            }

            // Update best move from hash table
            tt_move = entry.best_move;
        }

        // --- Set up main loop ---

        let mut best_value = -Eval::INFINITY;
        let mut best_move = Move::NONE;
        let mut move_count = 0;

        // Get killer moves
        let killers = self.stats.killers.get(self.ply);
        // Initialise move picker
        let mut move_picker = MovePicker::<false>::new(&self.board, tt_move, killers);

        // --- Main Loop ---
        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            // Update number of moves searched in this node
            move_count += 1;

            // Make move and update ply, node counters, prefetch hash entry, etc...
            self.make_move(tt, move_);

            // TODO: Principal Variation Search
            // Recursive search
            let value = -self.negamax::<PV>(tt, &mut child_pv, -beta, -alpha, depth - 1);
            // Undo move and decrement ply counter
            self.undo_move(move_);

            // Check for engine stop flag
            if self.stop {
                return Eval::DRAW;
            }

            // If the move we just search is better than best_value (The best we can do in this subtree), we can update best_value to be alpha
            best_value = best_value.max(value); // Update the best score found locally at this node.

            // Alpha Update: Check if this move's score (`value`) is better than the
            // best score we are *already guaranteed* (`alpha`) from other parts of the tree.
            if value > alpha {
                // We found a new best move sequence overall.
                best_move = move_; // Update the best move.
                pv.update_line(move_, &child_pv); // Update the Principal Variation (best move sequence).
                alpha = value; // Update alpha: Raise the lower bound of our guaranteed score.
            }

            // Beta Cutoff (Fail-High): Check if our guaranteed score (`alpha`)
            // meets or exceeds the opponent's limit (`beta`).
            // This move is "too good". The opponent (at a higher node)
            // would have already had a better alternative than allowing this position.
            // Therefore, exploring further sibling moves at this node is unnecessary.
            if value >= beta {
                best_value = beta;
                break;
            }
        }

        // If move count is 0, it is either a stalemate or a mate in self.ply
        if move_count == 0 {
            best_value = self.terminal_score(in_check);
        // If there is a new best move, and it is not a capture, update the killer move and history move table
        } else if best_move.is_valid() && !best_move.is_capture() {
            self.update_search_stats(best_move, depth);
        }

        // Write to TT, save static eval
        if !NT::ROOT {
            self.store_results(tt, best_value, best_move, beta, Eval::ZERO, depth);
        }

        best_value
    }

    /// Performs a Quiescence Search to stabilize the evaluation at tactical leaves.
    ///
    /// Extends the search beyond the nominal depth limit (`depth == 0` in `negamax`)
    /// by considering only non-quiet moves (primarily captures) until a position
    /// is reached where no immediate, significant tactical changes are likely.
    ///
    /// # Parameters
    ///
    /// * `NT`: Generic node type (usually `PV` when called from `negamax`).
    /// * `pv`: Mutable reference to a `PVLine` (less critical in qsearch but kept for consistency).
    /// * `alpha`: The lower bound of the search window.
    /// * `beta`: The upper bound of the search window.
    ///
    /// # Returns
    ///
    /// An `Eval` representing the stabilized score of the position, considering captures.
    fn quiescence<NT: NodeTypeTrait>(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        mut alpha: Eval,
        beta: Eval,
    ) -> Eval {
        self.update_seldepth::<NT>();

        pv.clear();

        if self.should_stop_search() {
            return Eval::DRAW;
        }

        // Check ply limit to prevent infinite recursion in rare cases
        if self.ply >= MAX_DEPTH as u16 {
            return evaluate(&self.board, &mut self.pawn_table); // Return static eval if too deep
        }

        // Check for draws (Repetition, 50-move rule)
        if self.is_draw::<NT>() {
            return Eval::DRAW;
        }

        let in_check = self.board.in_check();

        // --- Stand Pat Score ---
        // Get the static evaluation of the current position.
        // This score assumes no further captures are made (the "stand pat" score).
        let stand_pat = evaluate(&self.board, &mut self.pawn_table);

        // --- Alpha-Beta Pruning based on Stand Pat ---
        // If the static evaluation is already >= beta, the opponent won't allow this position.
        // We can prune immediately, assuming the static eval is a reasonable lower bound.
        if stand_pat >= beta {
            return beta; // Fail-High based on static eval
        }

        // If the static evaluation is better than alpha, update alpha.
        // This becomes the baseline score we need to beat with captures.
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let tt_entry = tt.get(self.board.key());
        let mut tt_move = Move::NONE;

        if let Some(entry) = tt_entry {
            let tt_value = entry.value.from_tt(self.ply);

            match entry.bound {
                TTBound::Exact => return tt_value,
                TTBound::Lower if !NT::PV && !in_check && tt_value >= beta => return beta,
                TTBound::Upper if !NT::PV && !in_check && tt_value <= alpha => return alpha,
                _ => tt_move = entry.best_move,
            }
        }

        // Initialize best_value with stand_pat. We are looking for captures that improve on this.
        let mut best_value = stand_pat;
        let mut child_pv = PVLine::default();
        let mut best_move = Move::NONE;

        // --- Generate and Explore Captures Only ---
        // The generic parameter 'true' tells MovePicker to skip quiet moves.
        let mut move_picker = MovePicker::<true>::new(&self.board, tt_move, [Move::NONE; 2]);

        while let Some(move_) = move_picker.next(&self.board, &self.stats) {
            self.make_move(tt, move_); // Make the capture
            let value = -self.quiescence::<PV>(tt, &mut child_pv, -beta, -alpha); // Recursive call
            self.undo_move(move_); // Undo the capture

            // Check for stop signal after recursive call
            if self.stop {
                return Eval::DRAW;
            }

            // --- Alpha-Beta Update (same as negamax) ---
            best_value = best_value.max(value); // Update the best score found locally at this node.

            if value > alpha {
                best_move = move_; // Update the best move.
                pv.update_line(move_, &child_pv); // Update PV line for quiescence
                alpha = value; // Update alpha
            }

            if value >= beta {
                // Beta Cutoff (Fail-High)
                best_value = beta;
                break;
            }
        }

        // No moves left, but still in check: checkmate
        if in_check && best_value == -Eval::INFINITY {
            return Eval::mated_in(self.ply);
        }

        self.store_results(tt, best_value, best_move, beta, stand_pat, 0);

        best_value
    }

    fn make_move(&mut self, tt: &TT, move_: Move) {
        self.board.make_move(move_);
        tt.prefetch(self.board.key());
        self.ply += 1;
        self.nodes += 1;
    }

    fn undo_move(&mut self, move_: Move) {
        self.board.undo_move(move_);
        self.ply -= 1;
    }

    fn update_seldepth<NT: NodeTypeTrait>(&mut self) {
        self.seldepth = if NT::ROOT {
            0
        } else {
            self.seldepth.max(self.ply as usize)
        };
    }

    fn update_search_stats(&mut self, best_move: Move, depth: usize) {
        self.stats.killers.update(self.ply, best_move);

        let bonus = (depth * depth) as i16;

        self.stats
            .main_history
            .update(&self.board, best_move, bonus);
    }

    fn store_results(
        &self,
        tt: &TT,
        best_value: Eval,
        best_move: Move,
        beta: Eval,
        eval: Eval,
        depth: usize,
    ) {
        let bounds = if best_value >= beta {
            TTBound::Lower // Means we are storing the lower bound for beta
        // If the best we can do (alpha in some later search) is better than the best the opponent can do in this line (beta here),
        // we can prune the tree for that branch
        } else if best_move.is_valid() {
            TTBound::Exact // Means we are storing the exact score for this position
        } else {
            TTBound::Upper // Means we are storing the upper bound for alpha
            // If the best the opponent can do (beta in a later search) is better than the best we can do in this line (alpha here),
            // we can prune the tree for that branch
        };

        tt.write(
            self.board.key(),
            bounds,
            self.ply,
            depth as u8,
            best_move,
            eval,
            best_value,
        );
    }

    fn is_draw<NT: NodeTypeTrait>(&mut self) -> bool {
        !NT::ROOT && self.board.is_draw(self.ply)
    }

    fn terminal_score(&self, in_check: bool) -> Eval {
        if in_check {
            Eval::mated_in(self.ply)
        } else {
            Eval::DRAW
        }
    }
}
