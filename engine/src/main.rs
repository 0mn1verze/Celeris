use chess::utils::perft_bench;
#[cfg(feature = "tune")]
use engine::tunables::spsa_output_txt;

use engine::{Depth, UCI, run_bench};
use std::env::args;

const DEFAULT_CMD_BENCH_DEPTH: Depth = 12;

fn main() {
    let mut cli_args = args();
    cli_args.next();

    match cli_args.next().as_deref() {
        #[cfg(feature = "tune")]
        Some("spsa") => println!("{}", spsa_output_txt()),

        Some("bench") => {
            let depth = cli_args
                .next()
                .and_then(|s| s.parse::<Depth>().ok())
                .unwrap_or_else(|| {
                    println!(
                        "info string invalid format, using default depth {DEFAULT_CMD_BENCH_DEPTH}."
                    );
                    DEFAULT_CMD_BENCH_DEPTH
                });
            if depth == 0 {
                println!(
                    "info string Invalid depth 0. Using default depth {DEFAULT_CMD_BENCH_DEPTH}."
                );
                run_bench(DEFAULT_CMD_BENCH_DEPTH);
            } else {
                run_bench(depth);
            }
        }

        Some("test") => {
            perft_bench();
        }
        _ => UCI::init(),
    }
}
