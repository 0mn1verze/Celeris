use std::sync::{Arc, atomic::AtomicBool, mpsc::Receiver};

use chess::{
    board::Board,
    utils::{perft_bench, perft_test},
};

// Import local modules (evaluation, threading, transposition table).
use crate::{evaluate_nnue, search::TT, thread::ThreadPool};

use super::EngineOption;
use super::constants::*;
#[cfg(feature = "tune")]
use super::tunables::{set_tunable, spsa_output_opts};
use super::{Command, TimeControl};

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

        thread_pool.resize(THREADS);

        Self {
            is_debug: DEBUG,
            board: Board::default(),
            tt,
            thread_pool,
        }
    }

    /// Runs the main loop of the engine controller, listening for commands on the receiver channel.
    pub fn run(&mut self, rx: Receiver<Command>) {
        for command in rx {
            if command == Command::Quit {
                return;
            }
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
        println!("id name {} {}", NAME, VERSION);
        println!("id author {}", AUTHORS);

        // Print command options
        println!("option name UCI_Chess960 type check default false");
        println!("option name ClearHash type button");
        println!("option name Hash type spin default 32 min 1 max 128");
        println!("option name Threads type spin default 12 min 1 max 12");

        #[cfg(feature = "tune")]
        println!("{}", spsa_output_opts());

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
            println!("info string Attempting to resize hash to {size_mb} MB...");
        }

        if !(1..=128).contains(&size_mb) {
            println!("info string hash spin value out of bounds (1 to 128).");
        }

        self.tt.resize(size_mb);

        if self.is_debug {
            println!("info string Hash resized to {size_mb} MB.");
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
            println!("info string Attempting to resize the number of threads to {threads} ");
        }

        if !(1..=12).contains(&threads) {
            println!("info string threads spin value out of bounds (1 to 12).");
        }

        self.thread_pool.resize(threads);

        if self.is_debug {
            println!("info string Threads resized to {threads} ");
        }
    }

    #[cfg(feature = "tune")]
    fn set_tunable(&mut self, tunable_name: &str, val: &str) {
        if let Err(e) = set_tunable(&tunable_name, &val) {
            println!("info string {}", e);
        } else if self.is_debug {
            println!("info string value {tunable_name} set to {val}.");
        }
    }

    /// Handles the "setoption" command by dispatching to specific option handlers.
    fn set_option(&mut self, option: EngineOption) {
        match option {
            EngineOption::Chess960(option) => self.board.set_chess960(option),
            EngineOption::ClearHash => self.clear_hash(),
            EngineOption::ResizeHash(size_mb) => self.resize_hash(size_mb),
            EngineOption::ResizeThreads(threads) => self.resize_threads(threads),
            #[cfg(feature = "tune")]
            EngineOption::SetTunable(tunable_name, val) => self.set_tunable(&tunable_name, &val),
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
            "NNUE Eval:{}",
            evaluate_nnue(&self.board, &mut self.thread_pool.main_worker.nnue)
        )
    }

    fn reset(&mut self) {
        self.thread_pool.reset();
        self.tt.reset_age();
        self.thread_pool.clear_hash_table(&self.tt);
    }
}
