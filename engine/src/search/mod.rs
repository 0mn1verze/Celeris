mod clock;
mod helper;
mod pv;
mod quiescence;
mod search;
mod tt;
mod worker;

pub use clock::Clock;
use nnue::accumulator::Accumulator;
pub use pv::PVLine;
pub use tt::TT;

use crate::{
    CaptureHistory, Eval, KillerEntry, MainHistory,
    constants::{MAX_DEPTH, MIN_DEPTH, SEARCH_STACK_OFFSET},
};
use chess::{Move, Piece, board::Board};

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
    pub ht: MainHistory,
    pub cht: CaptureHistory,
}

#[derive(Debug, Clone)]
pub(crate) struct SearchWorker {
    // Search Clock
    pub clock: Clock,
    // Main Board
    board: Board,

    thread_id: usize,

    // Search Stack
    stack: [SearchStackEntry; MAX_DEPTH + SEARCH_STACK_OFFSET],

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

pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
    type Next: NodeType;
}

pub struct PV;

pub struct Root;

pub struct NonPV;

impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
    type Next = PV;
}

impl NodeType for PV {
    const PV: bool = true;
    const ROOT: bool = false;
    type Next = Self;
}

impl NodeType for NonPV {
    const PV: bool = false;
    const ROOT: bool = false;
    type Next = Self;
}
