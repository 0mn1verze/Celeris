use chess::Move;

use crate::{
    MoveBuffer, MoveStage, SearchStackEntry, SearchWorker,
    constants::{CONT_HIST_SIZE, MAX_DEPTH},
    eval::Eval,
    movepick::MovePicker,
    search::PVLine,
    see,
};

use super::{NodeType, NonPV, Root, TT, helper::*, tt::TTBound};

impl SearchWorker {
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
        let mut alpha = -Eval::INFINITY;
        let mut beta = Eval::INFINITY;
        let mut delta = Eval(20);

        let mut search_depth = self.depth + 1;
        let full_depth = self.depth + 1;

        if search_depth >= 4 {
            alpha = (self.eval - delta).max(-Eval::INFINITY);
            beta = (self.eval + delta).min(Eval::INFINITY);
        }

        loop {
            let mut pv = PVLine::default();

            let eval = self.negamax::<Root>(tt, &mut pv, alpha, beta, search_depth, false);

            if self.stop {
                return;
            }

            if eval <= alpha {
                beta = (alpha + beta) / Eval(2);
                alpha = (eval - delta).max(-Eval::INFINITY);
                search_depth = full_depth;
            } else if eval >= beta {
                beta = (eval + delta).min(Eval::INFINITY);
                self.pv = pv.clone();

                if search_depth > 1 && eval.abs() <= Eval::MATE_BOUND {
                    search_depth -= 1;
                }
            } else {
                self.eval = eval;
                self.pv = pv;
                break;
            }

            delta += delta / Eval(2);
        }
    }

    fn nw_search(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        window: Eval,
        depth: usize,
        cutnode: bool,
    ) -> Eval {
        self.negamax::<NonPV>(tt, pv, window, window + Eval(1), depth, cutnode)
    }

    fn negamax<NT: NodeType>(
        &mut self,
        tt: &TT,
        pv: &mut PVLine,
        mut alpha: Eval,
        mut beta: Eval,
        mut depth: usize,
        cutnode: bool,
    ) -> Eval {
        let us = self.board.stm();

        pv.clear();

        if self.should_stop_search() {
            return Eval::DRAW;
        }

        let in_check = self.board.in_check();
        let excl_move = self.ss().excl_move;
        let singular = excl_move.is_valid();

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
        let mut tt_capture = false;
        let tt_hit = tt_entry.is_some();
        let mut tt_bound = TTBound::None;
        let mut tt_depth = 0;
        let mut tt_value = Eval::ZERO;
        // --- Hash Table Cut ---
        // If a previously stored value can be trusted (higher depth),
        // then it would be safe to cut the branch and return the stored value
        if let Some(tt_entry) = tt_entry {
            tt_value = tt_entry.value.from_tt(self.ply);

            if !NT::PV
                && !singular
                && tt_entry.depth >= depth as u8
                && can_use_tt_value(tt_entry.bound, tt_value, alpha, beta)
            {
                return tt_value;
            }

            // Update best move from hash table
            tt_move = tt_entry.best_move;
            tt_capture = self.board.is_capture(tt_move);
            tt_bound = tt_entry.bound;
            tt_depth = tt_entry.depth as usize;
        }

        // Get the best static evaluation of the position
        let eval = self.static_eval(in_check, tt_entry);
        // Set up flags to record trends of the game
        let improving = self.improving();
        let opp_worsening = self.opp_worsening();

        // --- Pruning ---
        if !NT::PV && !in_check && !singular {
            // --- Futility Pruning ---
            // If the eval is well above beta, then we assume the eval will hold above beta
            if self.can_do_fp(depth, eval, beta, improving) {
                return beta;
            }

            // --- Null Move Pruning ---
            // If the position is so strong that giving our opponent a
            // double move still allows us to maintain our advantage,
            // then we can prune early with some safety
            if self.can_do_nmp(depth, eval, beta) {
                let r = nmp_reduction(depth);

                self.make_null_move(tt);
                let value = -self.nw_search(tt, &mut child_pv, -beta, depth - r, false);
                self.undo_null_move();

                if value >= beta {
                    return beta;
                }
            }
        }
        // --- Set up main loop ---
        let mut best_value = -Eval::INFINITY;
        let mut best_move = Move::NONE;
        // Set up capture and quiet move list
        let mut caps_tried = MoveBuffer::default();
        let mut quiets_tried = MoveBuffer::default();
        let mut move_count = 0;
        // Clear child killer moves
        self.ss_look_ahead(2).killers.clear();
        // Get killer moves
        let killers = self.ss().killers.get();
        // Create search stack buffer for continuation history lookup
        let ss_buffer = [self.ss_at(1), self.ss_at(2)];
        // Initialise move picker
        let mut mp = MovePicker::<false>::new(&self.board, tt_move, killers);
        // --- Main Loop ---
        while let Some(move_) = mp.next(&self.board, &self.stats, &ss_buffer) {
            // Skip excluded move
            if move_ == excl_move {
                continue;
            }

            // Update number of moves searched in this node
            move_count += 1;
            // Move flags
            let is_capture = move_.is_capture();
            let is_promotion = move_.is_promotion();
            // New depth
            let mut new_depth = depth.max(1) - 1;

            // --- Quiet Move Pruning ---
            if !NT::ROOT && !in_check && self.can_do_pruning(best_value) {
                // --- Late Move Pruning ---
                // Near the leafs, trust that the move ordering is sound
                // and ignore the quiet moves after a certain threshold
                if self.can_do_lmp(depth, move_count, improving) {
                    mp.skip_quiets();
                }
            }

            // --- SEE Pruning ---
            // Near the leafs we can ignore the bad noisy moves that fail
            // Static Exchange Evaluation by a threshold
            if !in_check && self.can_do_see_prune(depth, best_value, mp.stage, move_) {
                continue;
            }

            // --- Singular Extension Search ---
            if !NT::ROOT
                && !singular
                && self.can_do_singular(depth, tt_depth, move_, tt_move, tt_value, tt_bound)
            {
                // Calculate a value
                let singular_beta = (tt_value - Eval(depth as i32)).max(-Eval::MATE);

                // Tell child nodes that we are in a singular search
                self.ss_mut().excl_move = move_;
                // Search the rest of the moves at a reduced depth.
                let value = self.nw_search(
                    tt,
                    &mut child_pv,
                    singular_beta - Eval(1),
                    new_depth / 2,
                    cutnode,
                );
                self.ss_mut().excl_move = Move::NONE;

                // --- Multi Cut Pruning ---
                // if the only move is even better than beta then we can prune it
                if value >= beta {
                    return beta;
                }

                // If no other move can reach the value of the tt_move (best_move),
                // then extend this move to check if it is really the only move
                let extension =
                    // If a null window search indicated that all the other moves are a lot worse,
                    // then this move is very promising
                    if value < singular_beta - Eval(50) {
                        3
                    } else if value < singular_beta - Eval(25) {
                        2
                    } else if value < singular_beta {
                        1
                    // Opposite Effect: Move is too good to be true, so we try to ignore it safely
                    } else if tt_value >= beta {
                        -3
                    // We are in node after aggressive pruning, so if the tt_move is not the best only move,
                    // then take it with a grain of salt
                    } else if cutnode {
                        -2
                    // Deeper search suggests that the tt_move is not so promising after all
                    } else if tt_value <= value {
                        -1
                    } else {
                        0
                    };

                // Update new depth with new extensions
                new_depth = if extension >= 0 {
                    new_depth.saturating_add(extension as usize)
                } else {
                    new_depth.saturating_sub((-extension) as usize)
                };
            }

            // Make move and update ply, node counters, prefetch hash entry, etc...
            self.make_move(tt, move_);
            // Remember previous node count
            let start_nodes = self.nodes;
            // Recursive search
            let mut value = alpha;

            // --- Late Move Reduction ---
            // Search moves that are sufficiently far from the terminal nodes
            // and are not tactical using a reduced depth zero window search
            // to see if it is promising or not.
            let full_search = if self.can_do_lmr(depth, move_count, NT::PV) {
                // Calculate dynamic depth reduction
                let mut r = lmr_base_reduction(depth, move_count);
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

                value = -self.nw_search(tt, &mut child_pv, -alpha - Eval(1), new_depth - r, false);

                value > alpha && r > 1
            } else {
                !NT::PV || move_count > 1
            };

            // We do a zero window full depth search if the reduced depth search revealed a potentially better move,
            // or if other conditions like being in a NonPV node, searching non-first moves, are met.
            if full_search {
                value = -self.nw_search(tt, &mut child_pv, -alpha - Eval(1), new_depth, !cutnode);
            }

            // If we are current searching in a PV node, the if its the first move or further searches revealed a potentially better move,
            // we do a full search to further investigate
            if NT::PV && (move_count == 1 || value > alpha) {
                value = -self.negamax::<NT::Next>(
                    tt,
                    &mut child_pv,
                    -beta,
                    -alpha,
                    new_depth,
                    !cutnode,
                );
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

                    if NT::PV {
                        pv.update_line(move_, &child_pv); // Update the Principal Variation (best move sequence).
                    }

                    // Beta Cutoff (Fail-High): Check if our guaranteed score (`alpha`)
                    // meets or exceeds the opponent's limit (`beta`).
                    // This move is "too good". The opponent (at a higher node)
                    // would have already had a better alternative than allowing this position.
                    // Therefore, exploring further sibling moves at this node is unnecessary.
                    if value >= beta {
                        break;
                    }

                    alpha = value; // Update alpha: Raise the lower bound of our guaranteed score.
                }
            }

            // Add moves tried to the buffer
            if move_.is_capture() {
                caps_tried.push(move_);
            } else {
                quiets_tried.push(move_);
            }
        }

        // Update search stack move count
        self.ss_mut().move_count = move_count as u8;
        // Update search stack in check flag
        self.ss_mut().in_check = in_check;

        // If move count is 0, it is either a stalemate or a mate in self.ply
        if move_count == 0 {
            best_value = if excl_move.is_valid() {
                alpha // Return the alpha gathered from other branches, since singular search does not explore all legal moves
            } else if in_check {
                Eval::mated_in(self.ply)
            } else {
                Eval::DRAW
            };
            // If there is a new best move, and it is not a capture, update the killer move and history move table
        } else if best_move.is_valid() {
            self.update_search_stats(best_move, depth, &caps_tried, &quiets_tried);
        }

        if !singular {
            // Write to TT, save static eval
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
}
