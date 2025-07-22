use super::tt::TTBound;
use crate::{Depth, Eval, tunables::*};

pub(crate) fn lmr_base_reduction(depth: Depth, move_count: usize) -> Depth {
    if depth == 0 || move_count == 0 {
        return 0;
    }

    let lmr_base = lmr_base() as f32 / 1024.0;
    let lmr_mult = lmr_mult() as f32 / 1024.0;

    (lmr_base + (depth as f32).ln() * (move_count as f32).ln() / lmr_mult) as i32
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
