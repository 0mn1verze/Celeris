use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Board, Move};

use crate::{
    INFINITY,
    engine::MAX_DEPTH,
    eval::{self, Eval, PawnTable, evaluate},
    movepick::{MovePicker, MoveStage},
    search::{PVLine, SearchStack},
    types::TT,
};

use super::{Clock, MIN_DEPTH, NodeType, NodeTypeTrait, PVNode, RootNode};

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

    pv: PVLine,
    completed_depth: usize,
    eval: Eval,

    stop: bool,

    // Tables
    // pub pawn_table: PawnTable,

    // Search statistics
    cut_offs: usize,
    immediate_cut_offs: usize,
}

impl SearchWorker {
    pub fn new(thread_id: usize, stop: Arc<AtomicBool>, nodes: Arc<AtomicU64>) -> Self {
        Self {
            clock: Clock::default(stop, nodes),
            thread_id,
            eval: -INFINITY,
            board: Board::default(),
            stack: SearchStack::default(),
            nodes: 0,
            seldepth: 0,
            depth: 0,
            ply: 0,
            pv: PVLine::default(),
            completed_depth: 0,
            stop: false,
            cut_offs: 0,
            immediate_cut_offs: 0,
            // pawn_table: PawnTable::new(),
        }
    }

    pub fn thread_id(&self) -> usize {
        self.thread_id
    }

    pub fn reset(&mut self) {
        self.stack.clear();
        self.clock.last_nodes = 0;
        self.nodes = 0;
        self.seldepth = 0;
        self.ply = 0;
        self.pv = PVLine::default();
        self.completed_depth = 0;
        self.eval = -INFINITY;
        self.stop = false;
        self.cut_offs = 0;
        self.immediate_cut_offs = 0;
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
        let alpha = -INFINITY;
        let beta = INFINITY;

        let mut pv = PVLine::default();

        let eval = self.negamax::<RootNode>(&mut pv, alpha, beta, self.depth + 1);

        if self.stop {
            return;
        }

        self.eval = eval;

        self.pv = pv;

        println!(
            "info depth {} seldepth {} score {} time {} nodes {} {}",
            self.depth + 1,
            self.seldepth + 1,
            self.eval,
            self.clock.elapsed().as_millis(),
            self.nodes,
            self.pv
        );

        if self.cut_offs > 0 {
            println!(
                "Move ordering statistics: {}% immediate cutoffs",
                self.immediate_cut_offs * 100 / self.cut_offs
            );
        }
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
    ///   the `RootNode` (the initial call) and `PVNode` (recursive calls), allowing for
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
    /// Returns `Eval::ZERO` immediately if the search is stopped (`self.stop == true`).
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
        pv: &mut PVLine,
        mut alpha: Eval,
        beta: Eval,
        mut depth: usize,
    ) -> Eval {
        if self.should_stop_search() {
            return Eval::ZERO;
        }

        let in_check = self.board.in_check();

        // --- Quiescence search in base case ---
        if depth == 0 && !in_check {
            return self.quiescence::<NT>(pv, alpha, beta);
        }

        // --- Set up Search ---
        depth = depth.max(1);
        self.update_seldepth::<NT>();
        let mut child_pv = PVLine::default();
        pv.clear();

        if self.is_draw::<NT>() {
            return Eval::ZERO;
        }

        let mut best_value = -INFINITY;
        let mut move_count = 0;

        let mut move_picker = MovePicker::<false>::new();

        while let Some(move_) = move_picker.next(&self.board) {
            // Update number of moves searched in this node
            move_count += 1;

            // Make move and update ply, node counters
            self.make_move(move_);
            // Recursive search
            let value = -self.negamax::<PVNode>(&mut child_pv, -beta, -alpha, depth - 1);
            // Undo move and decrement ply counter
            self.undo_move(move_);

            if self.stop {
                return Eval::ZERO;
            }

            // If the move we just search is better than best_value (The best we can do in this subtree), we can update best_value to be alpha
            if value > best_value {
                best_value = value; // Update the best score found locally at this node.
            }

            // Beta Cutoff (Fail-High): Check if our guaranteed score (`alpha`)
            // meets or exceeds the opponent's limit (`beta`).
            // This move is "too good". The opponent (at a higher node)
            // would have already had a better alternative than allowing this position.
            // Therefore, exploring further sibling moves at this node is unnecessary.
            if value >= beta {
                if move_count == 1 {
                    self.immediate_cut_offs += 1;
                }
                self.cut_offs += 1;
                best_value = beta;
                break;
            }

            // Alpha Update: Check if this move's score (`value`) is better than the
            // best score we are *already guaranteed* (`alpha`) from other parts of the tree.
            if value > alpha {
                // We found a new best move sequence overall.
                pv.update_line(move_, &child_pv); // Update the Principal Variation (best move sequence).

                alpha = value; // Update alpha: Raise the lower bound of our guaranteed score.
            }
        }

        if move_count == 0 {
            best_value = if in_check {
                Eval::mated_in(self.ply)
            } else {
                Eval::ZERO
            };
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
    /// * `NT`: Generic node type (usually `PVNode` when called from `negamax`).
    /// * `pv`: Mutable reference to a `PVLine` (less critical in qsearch but kept for consistency).
    /// * `alpha`: The lower bound of the search window.
    /// * `beta`: The upper bound of the search window.
    ///
    /// # Returns
    ///
    /// An `Eval` representing the stabilized score of the position, considering captures.
    fn quiescence<NT: NodeTypeTrait>(
        &mut self,
        pv: &mut PVLine,
        mut alpha: Eval,
        beta: Eval,
    ) -> Eval {
        if self.should_stop_search() {
            return Eval::ZERO;
        }

        self.update_seldepth::<NT>();

        pv.clear();

        // Check ply limit to prevent infinite recursion in rare cases
        if self.ply >= MAX_DEPTH as u16 {
            // return evaluate(&self.board, &mut self.pawn_table); // Return static eval if too deep
            return evaluate(&self.board);
        }

        // Check for draws (Repetition, 50-move rule)
        if self.is_draw::<NT>() {
            return Eval::ZERO;
        }

        let in_check = self.board.in_check();

        // --- Stand Pat Score ---
        // Get the static evaluation of the current position.
        // This score assumes no further captures are made (the "stand pat" score).
        // let stand_pat = evaluate(&self.board, &mut self.pawn_table);
        let stand_pat = evaluate(&self.board);

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

        // Initialize best_value with stand_pat. We are looking for captures that improve on this.
        let mut best_value = stand_pat;
        let mut child_pv = PVLine::default();

        // --- Generate and Explore Captures Only ---
        // The generic parameter 'true' tells MovePicker to skip quiet moves.
        let mut move_picker = MovePicker::<true>::new();

        let mut move_count = 0;

        while let Some(move_) = move_picker.next(&self.board) {
            move_count += 1;

            self.make_move(move_); // Make the capture
            let value = -self.quiescence::<PVNode>(&mut child_pv, -beta, -alpha); // Recursive call
            self.undo_move(move_); // Undo the capture

            // Check for stop signal after recursive call
            if self.stop {
                return Eval::ZERO;
            }

            // --- Alpha-Beta Update (same as negamax) ---
            if value > best_value {
                best_value = value; // Found a better capture sequence
            }

            if value > alpha {
                alpha = value; // Update alpha
                pv.load(move_, &child_pv); // Update PV line for quiescence
            }

            if value >= beta {
                if move_count == 1 {
                    self.immediate_cut_offs += 1;
                }
                self.cut_offs += 1;
                // Beta Cutoff (Fail-High)
                best_value = beta;
                break;
            }
        }

        // No moves left, but still in check: checkmate
        if in_check && best_value == -INFINITY {
            return Eval::mated_in(self.ply);
        }

        best_value
    }

    fn make_move(&mut self, move_: Move) {
        self.board.make_move(move_);
        self.ply += 1;
        self.nodes += 1;
    }

    fn undo_move(&mut self, move_: Move) {
        self.board.undo_move(move_);
        self.ply -= 1;
    }

    fn update_seldepth<NT: NodeTypeTrait>(&mut self) {
        self.seldepth = if NT::node_type() == NodeType::Root {
            0
        } else {
            self.seldepth.max(self.ply as usize)
        };
    }

    fn is_draw<NT: NodeTypeTrait>(&mut self) -> bool {
        NT::node_type() != NodeType::Root && self.board.is_draw(self.ply)
    }
}
