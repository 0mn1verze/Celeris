use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use chess::{Colour, Move, Square};

use crate::{Depth, constants::MIN_DEPTH, time::TimeControl};

#[derive(Clone, Debug)]
pub struct Clock {
    global_stop: Arc<AtomicBool>,
    global_nodes: Arc<AtomicU64>,
    time_control: TimeControl,
    start_time: Instant,
    opt_time: Duration,
    max_time: Duration,
    pub last_nodes: u64,
    node_count: [[u64; Square::NUM]; Square::NUM],
}

impl Clock {
    pub const FREQUENCY: u64 = 2048;
    pub const OVERHEAD: u64 = 50;

    pub fn new(
        global_stop: Arc<AtomicBool>,
        global_nodes: Arc<AtomicU64>,
        time_control: TimeControl,
        stm: Colour,
    ) -> Self {
        let (opt_time, max_time) = match time_control {
            TimeControl::FixedTime(time) => Self::calc_fixed_time(time),
            TimeControl::Variable {
                wtime,
                btime,
                winc,
                binc,
                movestogo,
            } => Self::calc_variable_time(stm, wtime, btime, winc, binc, movestogo),
            _ => (Duration::ZERO, Duration::ZERO),
        };

        Self {
            global_stop,
            global_nodes,
            time_control,
            start_time: Instant::now(),
            opt_time,
            max_time,
            last_nodes: 0,
            node_count: [[0; Square::NUM]; Square::NUM],
        }
    }

    pub fn default(global_stop: Arc<AtomicBool>, global_nodes: Arc<AtomicU64>) -> Self {
        Self::new(
            global_stop,
            global_nodes,
            TimeControl::Infinite,
            Colour::White,
        )
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
    }

    pub fn get_elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn update_node_counts(&mut self, m: Move, delta: u64) {
        self.node_count[m.from().index()][m.to().index()] += delta;
    }

    fn get_time_and_increment(
        stm: Colour,
        wtime: u64,
        btime: u64,
        winc: Option<u64>,
        binc: Option<u64>,
    ) -> (u64, u64) {
        let (time, inc) = if stm == Colour::White {
            (wtime, winc.unwrap_or(0))
        } else {
            (btime, binc.unwrap_or(0))
        };

        // When below overhead, set time to 0
        let adjusted_time = time - Self::OVERHEAD.min(time);
        let adjusted_inc = if adjusted_time < Self::OVERHEAD {
            0
        } else {
            inc
        };

        (adjusted_time, adjusted_inc)
    }

    fn calc_fixed_time(time: u64) -> (Duration, Duration) {
        let adj_time = time - Self::OVERHEAD.min(time);
        let duration = Duration::from_millis(adj_time);
        (duration, duration)
    }

    fn calc_variable_time(
        stm: Colour,
        wtime: u64,
        btime: u64,
        winc: Option<u64>,
        binc: Option<u64>,
        movestogo: Option<u16>,
    ) -> (Duration, Duration) {
        let (time, inc) = Self::get_time_and_increment(stm, wtime, btime, winc, binc);
        let (opt, max) = if let Some(moves) = movestogo {
            // Formula from Crippa by Svart
            // https://github.com/crippa1337/svart/blob/master/engine/src/uci/timeman.rs
            let scale = 0.7 / (moves.min(50) as f64);
            let five = 0.5 * time as f64;
            let opt_time = (scale * time as f64).min(five);

            (opt_time, (5.0 * opt_time).min(five))
        } else {
            let total = ((time / 20) + (inc * 3 / 4)) as f64;

            (0.6 * total, (2.0 * total).min(0.5 * time as f64))
        };

        (
            Duration::from_millis(opt as u64),
            Duration::from_millis(max as u64),
        )
    }

    pub fn global_nodes(&self) -> u64 {
        self.global_nodes.load(Ordering::Relaxed)
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn start_search(&mut self, depth: Depth, nodes: u64, best_move: Move) -> bool {
        if self.global_stop.load(Ordering::Relaxed) {
            return false;
        }

        // at least depth 1
        if depth <= MIN_DEPTH {
            return true;
        }

        let start = match self.time_control {
            TimeControl::FixedDepth(d) => depth <= d,
            TimeControl::FixedNodes(n) => self.global_nodes() <= n,
            TimeControl::FixedTime(_) | TimeControl::Variable { .. } => {
                let opt_scale = if best_move.is_valid() && nodes != 0 {
                    let bm_nodes =
                        self.node_count[best_move.from().index()][best_move.to().index()];
                    let bm_fraction = bm_nodes as f64 / nodes as f64;

                    (0.4 + (1.0 - bm_fraction) * 2.0 as f64).max(0.5)
                } else {
                    1.0
                };

                self.elapsed() < self.opt_time.mul_f64(opt_scale)
            }
            _ => true,
        };

        if !start {
            self.global_stop.store(true, Ordering::Relaxed);
        }

        start
    }

    pub fn continue_search(&mut self, nodes: u64) -> bool {
        let searched = nodes - self.last_nodes;

        if searched >= Self::FREQUENCY {
            self.global_nodes.fetch_add(searched, Ordering::Relaxed);
            self.last_nodes = nodes;

            if self.global_stop.load(Ordering::Relaxed) {
                return false;
            }
        }

        let proceed = match self.time_control {
            TimeControl::FixedTime(_) | TimeControl::Variable { .. } => {
                searched < Self::FREQUENCY || self.elapsed() < self.max_time
            }
            _ => true,
        };

        if !proceed {
            self.global_stop.store(true, Ordering::Relaxed);
        }

        proceed
    }
}
