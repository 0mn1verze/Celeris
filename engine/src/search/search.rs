use chess::Move;

use crate::{
    MoveBuffer, SearchStackEntry, SearchWorker,
    constants::{CONT_HIST_SIZE, MAX_DEPTH},
    eval::Eval,
    movepick::MovePicker,
    search::PVLine,
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

            let eval = self.negamax::<Root>(tt, &mut pv, alpha, beta, search_depth);

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
        let mut tt_capture = false;
        let tt_hit = tt_entry.is_some();

        if let Some(tt_entry) = tt_entry {
            let tt_value = tt_entry.value.from_tt(self.ply);

            if !NT::PV
                && tt_entry.depth >= depth as u8
                && can_use_tt_value(tt_entry.bound, tt_value, alpha, beta)
            {
                return tt_value;
            }

            // Update best move from hash table
            tt_move = tt_entry.best_move;
            tt_capture = self.board.is_capture(tt_move);
        }

        let eval = self.static_eval(in_check, tt_entry);

        let improving = self.improving();

        let opp_worsening = self.opp_worsening();

        // --- Null Move Pruning ---
        if !NT::PV && !in_check && self.can_do_nmp(depth, eval, beta) {
            let r = nmp_reduction(depth);

            self.make_null_move(tt);

            let value =
                -self.negamax::<NonPV>(tt, &mut child_pv, -beta, -beta + Eval(1), depth - r);

            self.undo_null_move();

            if value >= beta {
                return beta;
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
            // Update number of moves searched in this node
            move_count += 1;

            if !NT::ROOT && best_value.is_valid() && self.board.has_non_pawn_material(us) {
                if depth <= 8 && move_count >= (5 + 2 * depth * depth) / (2 - improving as usize) {
                    mp.skip_quiets();
                }
            }

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
            let full_search = if depth >= 2 && move_count > 3 + NT::PV as usize {
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
            best_value = if in_check {
                Eval::mated_in(self.ply)
            } else {
                Eval::DRAW
            };
            // If there is a new best move, and it is not a capture, update the killer move and history move table
        } else if best_move.is_valid() {
            self.update_search_stats(best_move, depth, &caps_tried, &quiets_tried);
        }

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

        best_value
    }
}
