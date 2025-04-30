use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
    mpsc::Receiver,
};

use chess::{
    Board,
    utils::{perft_bench, perft_test},
};

use crate::{
    eval::{calc_psqt, evaluate},
    thread::ThreadPool,
    types::TT,
};

use super::command::EngineOption;
pub use super::{Command, TimeControl};

pub mod constants {

    use crate::eval::Eval;
    use chess::board::MAX_MOVES;

    pub const NAME: &str = "Celeris";
    pub const VERSION: &str = "0.0.1";
    pub const AUTHORS: &str = "0mn1verze, TheGogy";

    pub const THREADS: usize = 1;
    pub const DEBUG: bool = true;
    pub const TT_SIZE: usize = 64;

    pub const MAX_DEPTH: usize = MAX_MOVES;
    pub const INFINITY: Eval = Eval(32001);
    pub const MATE: Eval = Eval(32000);
    pub const LONGEST_MATE: Eval = Eval(MAX_DEPTH as i16);
    pub const MATE_BOUND: Eval = MATE.sub(LONGEST_MATE);
}

pub(super) struct EngineController {
    is_debug: bool,
    board: Board,
    tt: TT,
    thread_pool: ThreadPool,
    stop: Arc<AtomicBool>,
}

impl EngineController {
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        let tt = TT::default();
        let nodes = Arc::new(AtomicU64::new(0));
        let mut thread_pool = ThreadPool::new(Arc::clone(&stop));

        thread_pool.resize(constants::THREADS);

        Self {
            is_debug: constants::DEBUG,
            board: Board::default(),
            tt,
            thread_pool,
            stop,
        }
    }

    pub fn run(&mut self, rx: Receiver<Command>) {
        for command in rx {
            self.handle_command(command);
        }
    }

    fn handle_command(&mut self, command: Command) {
        match command {
            Command::Uci => self.introduce(),
            Command::Debug(is_debug) => self.set_debug(is_debug),
            Command::IsReady => println!("readyok"),
            Command::NewGame => self.new_game(),
            Command::SetOption(option) => self.set_option(option),
            Command::Position(board) => self.set_position(board),
            Command::Go(time_control) => self.go(time_control),
            Command::Perft(depth) => self.perft(depth),
            Command::Bench => self.bench(),
            Command::Print => self.print_board(),
            Command::Eval => self.evaluate(),
            _ => unreachable!(), // UCICommand::Quit and UCICommand::Stop is already handled by the UCI struct
        }
    }

    fn introduce(&self) {
        println!("id name {} {}", constants::NAME, constants::VERSION);
        println!("id author {}", constants::AUTHORS);
        println!("uciok");
    }

    fn set_debug(&mut self, is_debug: bool) {
        self.is_debug = is_debug;
    }

    fn new_game(&mut self) {
        self.board = Board::default();
    }

    fn resize_hash(&mut self, size_mb: usize) {
        if self.is_debug {
            println!("info string Attempting to resize hash to {} MB...", size_mb);
        }

        self.tt.resize(size_mb);

        if self.is_debug {
            println!("info string Hash resized to {} MB.", size_mb);
            println!("info string Hash entries: {}.", self.tt.size());
        }
    }

    fn clear_hash(&mut self) {
        if self.is_debug {
            println!("info string Attempting to clear hash table...");
        }

        self.thread_pool.clear_hash_table(&self.tt);

        if self.is_debug {
            println!("info string Hash cleared.");
        }
    }

    fn resize_threads(&mut self, threads: usize) {
        if self.is_debug {
            println!(
                "info string Attempting to resize the number of threads to {} ",
                threads
            );
        }

        self.thread_pool.resize(threads);

        if self.is_debug {
            println!("info string Threads resized to {} ", threads);
        }
    }

    fn set_option(&mut self, option: EngineOption) {
        match option {
            EngineOption::ClearHash() => self.clear_hash(),
            EngineOption::ResizeHash(size_mb) => self.resize_hash(size_mb),
            EngineOption::ResizeThreads(threads) => self.resize_threads(threads),
        }
    }

    fn set_position(&mut self, board: Board) {
        self.board = board;
    }

    fn go(&mut self, time_control: TimeControl) {
        self.thread_pool
            .start_search(time_control, &self.tt, &self.board);
    }

    fn bench(&self) {
        perft_bench();
    }

    fn perft(&mut self, depth: usize) {
        perft_test(&mut self.board, depth);
    }

    fn print_board(&self) {
        println!("{}", self.board);
    }

    fn evaluate(&mut self) {
        println!(
            "PSQ: {} {}",
            calc_psqt(&self.board).0,
            calc_psqt(&self.board).1
        );
        println!(
            "{}",
            // evaluate(&self.board, &mut self.thread_pool.main_worker.pawn_table)
            evaluate(&self.board)
        )
    }
}
