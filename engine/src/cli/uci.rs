//! # Engine Interface Module
//!
//! This module provides the primary interfaces for interacting with the Celeris chess engine.
//!
//! - [`Engine`]: A struct providing programmatic control over the engine (e.g., for testing or embedding).
//!   It runs the core engine logic in a separate thread and communicates via commands.
//! - [`UCI`]: A struct that handles the Universal Chess Interface (UCI) protocol, allowing interaction
//!   with standard chess GUIs via standard input/output.

use crate::EngineController;

// Import necessary modules from the parent 'engine' module and standard library.
use super::Command;
use chess::board::movegen::init_magic_tables;
use std::{
    io::BufRead,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
    },
    thread,
};

/// Represents the UCI (Universal Chess Interface) handler.
///
/// This struct is intended to be the main entry point for running the engine
/// in UCI mode, suitable for interaction with graphical user interfaces (GUIs) like
/// Arena, Cute Chess, etc.
///
/// It takes control of the main thread, reads UCI commands from standard input,
/// parses them, and interacts with the [`EngineController`] running in a background thread.
/// This struct manages the main loop for interacting with a UCI-compliant GUI.
/// It reads commands from standard input, parses them, and sends them to the engine controller.
pub struct UCI {
    command_tx: Sender<Command>,
    stop: Arc<AtomicBool>,
}

impl UCI {
    /// Initializes the UCI interface and starts the engine.
    ///
    /// This function should typically be called from your `main` function when
    /// the engine is run without specific command-line arguments (like "bench").
    ///
    /// It performs the following steps:
    /// 1. Creates the communication channel for engine commands.
    /// 2. Spawns the background thread running the [`EngineController`].
    /// 3. Creates the `UCI` handler instance.
    /// 4. Enters the main UCI command loop (`run`), reading from stdin.
    /// This function takes control of the main thread, running the UCI command loop.
    pub fn init() {
        // initialise magic tables
        init_magic_tables();

        let (tx, rx) = mpsc::channel();

        let stop = Arc::new(AtomicBool::new(false));

        let mut engine = EngineController::new(stop.clone());

        let handle = thread::spawn(move || engine.run(rx));

        let uci = Self {
            command_tx: tx,
            stop,
        };

        uci.run();

        handle.join().unwrap();
    }

    /// Runs the main UCI command loop.
    /// Reads lines from standard input, parses them as `Command`s, and handles them.
    pub fn run(&self) {
        let stdin = std::io::stdin().lock();
        for line in stdin.lines().map(Result::unwrap) {
            match line.parse::<Command>() {
                Ok(command) => self.handle_command(command),
                Err(e) => println!("{}", e),
            }
        }
    }

    /// Handles a parsed `Command`.
    /// Sends most commands to the engine thread, but handles `Quit` and `Stop` locally.
    fn handle_command(&self, command: Command) {
        match command {
            Command::Quit => self.quit(),
            Command::Stop => self.stop(),
            // Forward other commands to the engine controller thread.
            command => self.command_tx.send(command).unwrap(),
        }
    }

    /// Signals the engine thread to stop its current task (e.g., searching).
    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Handles the "quit" command. Stops the engine and exits the process.
    fn quit(&self) {
        self.stop();
        std::process::exit(0);
    }
}
