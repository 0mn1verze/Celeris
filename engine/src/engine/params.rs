use crate::init_tunables;

pub mod constants {

    use chess::board::MAX_MOVES;

    pub const NAME: &str = "Celeris";
    pub const VERSION: &str = "0.0.1";
    pub const AUTHORS: &str = "0mn1verze, TheGogy";

    pub const THREADS: usize = 1;
    pub const DEBUG: bool = true;
    pub const TT_SIZE: usize = 64;

    pub const MAX_DEPTH: usize = MAX_MOVES;
}

init_tunables! {

    pawn_val:   i32 = 82, 60, 140, 5;
    knight_val: i32 = 337, 250, 370, 5;
    bishop_val: i32 = 365, 300, 400, 5;
    rook_val:   i32 = 477, 450, 550, 5;
    queen_val:  i32 = 1025, 950, 1100, 5;

    nnue_base: i32 = 700, 600, 800, 10;

    nmp_min: usize = 4, 2, 6, 1;
    nmp_div: usize = 4, 2, 6, 1;

    lmr_min_depth: usize = 2, 2, 4, 1;
    lmr_min_moves: usize = 2, 2, 4, 1;

    // These values follow the LMR formula from Ethereal:
    // (lmr_a + depth.ln() * moves.ln() / lmr_b)
    // Both of these values are floating point and tunables work with integers, so they are scaled
    // up x100.
    lmr_a: f32 = 75, 50, 100, 5;
    lmr_b: f32 = 200, 160, 240, 10;
}
