use chess::Move;

use super::tt::TTBound;
use crate::{
    Depth, Eval, Interface, MoveStage, SearchWorker, constants::CONT_HIST_SIZE, evaluate_nnue,
    search::tt::TTEntry, see, tunables::*, utils::MoveBuffer,
};

pub(crate) fn lmr_base_reduction(depth: Depth, move_count: usize) -> Depth {
    if depth == 0 || move_count == 0 {
        return 0;
    }

    let lmr_base = lmr_base() as f32 / 1024.0;
    let lmr_mult = lmr_mult() as f32 / 1024.0;

    (lmr_base + (depth as f32).ln() * (move_count as f32).ln() / lmr_mult) as i16
}

pub(crate) fn nmp_reduction(depth: Depth) -> Depth {
    (nmp_min() + depth / nmp_div()).min(depth)
}

pub(crate) fn can_use_tt_value(tt_bound: TTBound, tt_value: Eval, alpha: Eval, beta: Eval) -> bool {
    tt_value.is_valid()
        && match tt_bound {
            // If the current node has already been searched to a higher depth, then the tt_value will be a better score.
            TTBound::Exact => true,
            // If the current node has a upper bound, then if the current eval is greater than the tt_value, it means the current eval is optimistic so we should use tt_value instead
            TTBound::Upper if tt_value <= alpha => true,
            // If the current node has a lower bound, then if the current eval is lower than the tt_value, it means we are underestimating our opponent's response, so we should use tt_value instead
            TTBound::Lower if tt_value >= beta => true,
            // If there is no stored value, or the above conditions are not met, we use eval, since it provides a tighter bound
            _ => false,
        }
}

pub(crate) fn calculate_bonus(depth: Depth) -> i16 {
    (350 * (depth.saturating_sub(1))).min(1600) as i16
}

impl SearchWorker {
    fn update_continuations(&mut self, move_: Move, bonus: i16) {
        for offset in 0..CONT_HIST_SIZE as i8 {
            if self.ss_at(offset).curr_move.is_valid() {
                let (piece, to) = self.piece_to_at(offset);

                self.stats
                    .ct
                    .probe_mut(piece, to)
                    .update(&self.board, move_, bonus);
            }
        }
    }

    pub(super) fn update_search_stats(
        &mut self,
        best_move: Move,
        depth: Depth,
        caps_tried: &MoveBuffer,
        quiets_tried: &MoveBuffer,
    ) {
        let bonus = calculate_bonus(depth);

        if !best_move.is_capture() {
            self.ss_at_mut(0).killers.update(best_move);
            self.stats.ht.update(&self.board, best_move, bonus);

            self.update_continuations(best_move, bonus);

            for &move_ in quiets_tried {
                self.stats.ht.update(&self.board, move_, -bonus);

                self.update_continuations(move_, -bonus);
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
            self.ss_at_mut(0).eval = -Eval::INFINITY;
            self.ss_at(0).eval
        } else if self.ss_at(0).excl_move.is_valid() {
            self.ss_at(0).eval
        } else if let Some(tt_entry) = tt_entry {
            let tt_eval = tt_entry.eval;
            let tt_value = tt_entry.value.from_tt(self.ply);

            let eval = if tt_eval.abs() >= Eval::INFINITY {
                self.evaluate()
            } else {
                tt_eval
            };

            self.ss_at_mut(0).eval = eval;

            // If we probe the tt_entry and the tt_value is tighter than the eval, then we can use it
            if can_use_tt_value(tt_entry.bound, tt_value, eval, eval) {
                tt_value
            } else {
                eval
            }
        } else {
            // self.ss_at_mut(0).eval = evaluate_nnue(&self.board, &mut self.nnue);
            self.ss_at_mut(0).eval = self.evaluate();

            self.ss_at(0).eval
        }
    }

    pub(super) fn evaluate(&mut self) -> Eval {
        evaluate_nnue(&self.board, &mut self.nnue)
    }

    pub(super) fn improving(&self) -> bool {
        if self.ply >= 2 {
            self.ss_at(0).eval > self.ss_at(2).eval
        } else {
            false
        }
    }

    pub(super) fn opp_worsening(&self) -> bool {
        if self.ply >= 1 {
            self.ss_at(0).eval > -self.ss_at(1).eval
        } else {
            false
        }
    }

    pub(super) fn terminal_score(&self, in_check: bool) -> Eval {
        if in_check {
            Eval::mated_in(self.ply)
        } else {
            Eval::DRAW
        }
    }

    pub(super) fn can_do_pruning(&self, best_value: Eval) -> bool {
        best_value.is_valid() && self.board.has_non_pawn_material(self.board.stm())
    }

    pub(super) fn can_do_nmp(&self, depth: Depth, eval: Eval, beta: Eval) -> bool {
        depth >= 2
            && self.ply_from_null > 0
            && eval >= beta
            && self.board.has_non_pawn_material(self.board.stm())
            && beta >= -Eval::MATE_BOUND
    }

    pub(super) fn can_do_fp(&self, depth: Depth, eval: Eval, beta: Eval, improving: bool) -> bool {
        let fp_margin = Eval(80 * depth as i32 - 60 * improving as i32);
        depth <= 8 && eval - fp_margin >= beta
    }

    pub(super) fn can_do_lmp(&self, depth: Depth, move_count: usize, improving: bool) -> bool {
        depth <= 8 && move_count >= ((5 + 2 * depth * depth) / (2 - improving as i16)) as usize
    }

    pub(super) fn can_do_lmr(&self, depth: Depth, move_count: usize, is_pv: bool) -> bool {
        depth >= 2 && move_count > 3 + is_pv as usize
    }

    pub(super) fn can_do_see_prune(
        &self,
        depth: Depth,
        best_value: Eval,
        stage: MoveStage,
        move_: Move,
    ) -> bool {
        // Set up SEE margins
        let d = depth as i32;
        let see_margins = [Eval(-70 * d), Eval(-20 * d * d)];
        let is_capture = move_.is_capture();

        !best_value.is_terminal()
            && depth <= 10
            && stage > MoveStage::GoodCaptures
            && !see(&self.board, move_, see_margins[is_capture as usize])
    }

    pub(super) fn can_do_singular(
        &self,
        depth: Depth,
        tt_depth: Depth,
        move_: Move,
        tt_move: Move,
        tt_value: Eval,
        tt_bound: TTBound,
    ) -> bool {
        depth >= 8
            && move_ == tt_move
            && tt_value.is_valid()
            && !tt_value.is_terminal()
            && tt_depth >= depth - 3
            && matches!(tt_bound, TTBound::Lower | TTBound::Exact)
    }
}
