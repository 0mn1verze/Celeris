use std::sync::{
    Arc,
    // Atomic types for thread-safe sharing of stop signals and node counts.
    atomic::{AtomicBool, AtomicU64},
    mpsc::Receiver,
};

use chess::{
    Board,
    utils::{perft_bench, perft_test},
};

// Import local modules (evaluation, threading, transposition table).
use crate::{
    eval::{calc_psqt, evaluate},
    evaluate_nnue,
    search::TT,
    thread::ThreadPool,
};

use super::command::EngineOption;
// Re-export Command and TimeControl for easier use by the interface module.
pub use super::{Command, TimeControl};

/// Defines constants used by the engine controller and potentially other parts.
pub mod constants {
    // Import necessary types for constants.
    use crate::eval::Eval;
    use chess::board::MAX_MOVES;

    pub const NAME: &str = "Celeris";
    pub const VERSION: &str = "0.0.1";
    pub const AUTHORS: &str = "0mn1verze, TheGogy";

    // Default configuration values.
    pub const THREADS: usize = 1;
    pub const DEBUG: bool = true;
    pub const TT_SIZE: usize = 64;

    // Search-related constants.
    pub const MAX_DEPTH: usize = MAX_MOVES;
}

/// The core engine controller that manages the board state, search threads, and handles commands.
///
/// This struct runs in a dedicated thread and receives commands via a channel.
pub(super) struct EngineController {
    /// Flag indicating whether debug information should be printed.
    is_debug: bool,
    /// The current state of the chess board.
    board: Board,
    /// The transposition table for storing search results.
    tt: TT,
    /// Manages the pool of search worker threads.
    thread_pool: ThreadPool,
}

impl EngineController {
    /// Creates a new `EngineController`.
    /// `stop`: A shared `AtomicBool` used to signal termination.
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        let tt = TT::default();
        let mut thread_pool = ThreadPool::new(stop);

        thread_pool.resize(constants::THREADS);

        Self {
            is_debug: constants::DEBUG,
            board: Board::default(),
            tt,
            thread_pool,
        }
    }

    /// Runs the main loop of the engine controller, listening for commands on the receiver channel.
    pub fn run(&mut self, rx: Receiver<Command>) {
        for command in rx {
            self.handle_command(command);
        }
    }

    /// Dispatches received commands to the appropriate handler methods.
    /// Note: `Stop` and `Quit` are handled by the `UCI` struct before reaching here.
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

    /// Handles the "uci" command: Prints engine identification and options.
    fn introduce(&self) {
        println!("id name {} {}", constants::NAME, constants::VERSION);
        println!("id author {}", constants::AUTHORS);
        println!("uciok");
    }

    fn set_debug(&mut self, is_debug: bool) {
        self.is_debug = is_debug;
    }

    /// Handles the "ucinewgame" command: Resets the board to the default starting position.
    fn new_game(&mut self) {
        self.board = Board::default();
        self.reset();
    }

    /// Resizes the transposition table.
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

    /// Clears all entries from the transposition table.
    fn clear_hash(&mut self) {
        if self.is_debug {
            println!("info string Attempting to clear hash table...");
        }

        self.tt.reset_age();
        self.thread_pool.clear_hash_table(&self.tt);

        if self.is_debug {
            println!("info string Hash cleared.");
        }
    }

    /// Resizes the number of search threads in the thread pool.
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

    /// Handles the "setoption" command by dispatching to specific option handlers.
    fn set_option(&mut self, option: EngineOption) {
        match option {
            EngineOption::ClearHash() => self.clear_hash(),
            EngineOption::ResizeHash(size_mb) => self.resize_hash(size_mb),
            EngineOption::ResizeThreads(threads) => self.resize_threads(threads),
        }
    }

    /// Handles the "position" command: Sets the internal board state.
    fn set_position(&mut self, board: Board) {
        self.board = board;
    }

    /// Handles the "go" command: Starts the search process with the given time control.
    fn go(&mut self, time_control: TimeControl) {
        self.tt.increment_age();
        self.thread_pool
            .start_search(time_control, &self.tt, &self.board);
    }

    /// Handles the "bench" command (custom): Runs a standard benchmark suite.
    fn bench(&self) {
        perft_bench();
    }

    /// Handles the "perft" command (custom): Runs a performance test for move generation.
    fn perft(&mut self, depth: usize) {
        perft_test(&mut self.board, depth);
    }

    /// Handles the "print" or "b" command (custom): Prints the current board to the console.
    fn print_board(&self) {
        println!("{}", self.board);
    }

    /// Handles the "eval" command (custom): Calculates and prints the static evaluation of the current position.
    fn evaluate(&mut self) {
        println!(
            "PSQ: {} {}",
            calc_psqt(&self.board).0,
            calc_psqt(&self.board).1
        );
        println!(
            "standard {}",
            evaluate(&self.board, &mut self.thread_pool.main_worker.pawn_table)
        );

        println!(
            "nnue {}",
            evaluate_nnue(&self.board, &mut self.thread_pool.main_worker.nnue)
        )
    }

    fn reset(&mut self) {
        self.thread_pool.reset();
        self.tt.reset_age();
        self.thread_pool.clear_hash_table(&self.tt);
    }
}
