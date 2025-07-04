mod history;
mod movepick;
mod see;

pub use history::{History, KillerEntry, MainHistory};
pub use movepick::MovePicker;
pub use see::see;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MoveStage {
    TTMove,
    GenCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    GenQuiets,
    Quiets,
    BadCaptures,
}
