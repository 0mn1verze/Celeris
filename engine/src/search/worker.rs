use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

use chess::{Move, Piece, Square, board::Board};
use nnue::accumulator::Accumulator;

use crate::{
    HistoryTable, SearchStats, SearchWorker,
    constants::{MAX_DEPTH, MIN_DEPTH},
    eval::Eval,
    search::{
        PVLine,
        stack::{SearchStack, SearchStackEntry},
    },
    time::Clock,
};

use super::TT;

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
        self.stats.ht.clear();
        self.stats.cht.clear();
        self.stats.ct.clear();
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

    pub(super) fn ss_at(&self, offset: i8) -> SearchStackEntry {
        self.stack.at(self.ply, offset)
    }

    pub(super) fn ss_at_mut(&mut self, offset: i8) -> &mut SearchStackEntry {
        self.stack.at_mut(self.ply, offset)
    }

    pub(super) fn piece_to_at(&self, offset: i8) -> (Piece, Square) {
        self.ss_at(offset).piece_to()
    }

    pub(super) fn print_info(&self, tt: &TT) {
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

    pub(super) fn make_move(&mut self, tt: &TT, move_: Move) {
        self.board.make_move(move_);

        tt.prefetch(self.board.key());

        self.ss_at_mut(0).curr_move = move_;
        self.ss_at_mut(0).moved = self.board.on(move_.to());
        self.ss_at_mut(0).ply_from_null = self.ply_from_null;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    pub(super) fn make_null_move(&mut self, tt: &TT) {
        self.board.make_null_move();

        tt.prefetch(self.board.key());

        self.ss_at_mut(0).curr_move = Move::NULL;
        self.ss_at_mut(0).moved = None;
        self.ss_at_mut(0).ply_from_null = 0;

        self.ply += 1;
        self.nodes += 1;
        self.ply_from_null += 1;
    }

    pub(super) fn undo_move(&mut self, move_: Move) {
        self.board.undo_move(move_);

        self.ply -= 1;

        self.ply_from_null = self.ss_at(0).ply_from_null;
    }

    pub(super) fn undo_null_move(&mut self) {
        self.board.undo_null_move();

        self.ply -= 1;

        self.ply_from_null = self.ss_at(0).ply_from_null;
    }
}
