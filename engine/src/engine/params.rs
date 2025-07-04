use crate::init_tunables;

/// Defines constants used by the engine controller and potentially other parts.
pub mod constants {
    // Import necessary types for constants.
    use chess::board::MAX_MOVES;

    pub const NAME: &str = "Celeris";
    pub const VERSION: &str = "0.0.1";
    pub const AUTHORS: &str = "0mn1verze, TheGogy";

    // Default configuration values.
    pub const THREADS: usize = 1;
    pub const DEBUG: bool = true;
    pub const TT_SIZE: usize = 32;

    // Search-related constants.
    pub const MAX_DEPTH: usize = MAX_MOVES;
}

init_tunables! {
    // Format:
    // value: type = val, min, max, step;

    // Late Move Reduction
    lmr_base: i32 = 768, 500, 2000, 100;
    lmr_mult: i32 = 2048, 1500, 4000, 100;

    // Null Move Pruning
    nmp_min: usize = 4, 2, 6, 1;
    nmp_div: usize = 4, 2, 6, 1;

    // Piece values
    pawn_val:   i32 = 82, 60, 140, 5;
    knight_val: i32 = 337, 250, 370, 5;
    bishop_val: i32 = 365, 300, 400, 5;
    rook_val:   i32 = 477, 450, 550, 5;
    queen_val:  i32 = 1025, 950, 1100, 5;

    // NNUE scaling value
    nnue_base: i32 = 700, 600, 800, 10;
}
