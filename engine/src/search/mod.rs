mod pv;
mod quiescence;
mod search;
mod stack;
mod tt;
mod utils;
mod worker;

pub use pv::PVLine;
pub(crate) use stack::{SearchStack, SearchStackEntry};
pub use tt::TT;

use crate::{CaptureHistory, ContinuationTable, Depth, Eval, MainHistory, time::Clock};
use chess::board::Board;
use nnue::accumulator::Accumulator;

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub ht: MainHistory,
    pub cht: CaptureHistory,
    pub ct: Box<ContinuationTable>,
}

#[derive(Debug, Clone)]
pub(crate) struct SearchWorker {
    // Search Clock
    pub clock: Clock,
    // Main Board
    board: Board,

    thread_id: usize,

    // Search Stack
    stack: SearchStack,

    // Search Info
    nodes: u64,
    pub depth: Depth,
    seldepth: Depth,
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
