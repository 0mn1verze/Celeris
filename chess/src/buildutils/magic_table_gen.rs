use super::PRNG;
// use super::magic_numbers::*;
use crate::core::*;
const MAX_PERM: usize = 0x1000; // 2^16

/******************************************\
|==========================================|
|             Magics Definition            |
|==========================================|
\******************************************/

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Magic {
    #[cfg(not(target_feature = "bmi2"))]
    pub(crate) magic: u64,

    mask: Bitboard,
    shift: u8,
    offset: usize,
}

impl Magic {
    /// Calculates the index into the precomputed attack table.
    ///
    /// `occ` should represent the occupied squares on the board.
    /// The formula is `((occ & mask) * magic) >> shift + offset`.
    #[cfg(not(target_feature = "bmi2"))]
    #[inline]
    pub(crate) const fn index(self, occ: Bitboard) -> usize {
        // Note: Using wrapping_mul for the multiplication as standard practice in magic bitboards.
        ((occ.bitand(self.mask).0.wrapping_mul(self.magic)) >> self.shift) as usize + self.offset
    }

    /// Calculates the index into the precomputed attack table. (Using PEXT instruction)
    ///
    /// `occ` should represent the occupied squares on the board.
    /// The formula is `pext(occ, mask) + offset`.
    #[cfg(target_feature = "bmi2")]
    pub(crate) fn index(self, occ: Bitboard) -> usize {
        occ.pext(self.mask.0) as usize + self.offset
    }
}

/******************************************\
|==========================================|
|              Attack Tables               |
|==========================================|
\******************************************/

/// # Attack Table
///
/// ## Elements
/// - table - Hash table for bishop/rook moves
/// - magic - Contains information about table key generation
///
/// ## Function
/// - get_entry(sq, occ) - Gets the corresponding attack pattern for a given square and occupancy
pub(crate) struct SliderAttackTable<const N: usize> {
    pub(crate) table: Box<[Bitboard; N]>,
    pub(crate) magic: [Magic; Square::NUM],
}

///
impl<const N: usize> SliderAttackTable<N> {
    /// ### Gets the corresponding attack pattern for a given square and occupancy
    pub(crate) fn get_entry(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let magic = self.magic[sq.index()];
        let index = magic.index(occ);
        self.table[index as usize]
    }
}

/******************************************\
|==========================================|
|            Attacks on the fly            |
|==========================================|
\******************************************/

/// # Attacks on the fly
/// - Calculate the attacks of a piece on the fly (Slow approach used to populate
/// tables).
/// - This is used for rook and bishop attacks.
pub(crate) const fn attacks_on_the_fly(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
    use Direction::*;
    // Directions for rook and bishop
    let dirs: [Direction; 4] = match pt {
        PieceType::Rook => [N, E, W, S],
        PieceType::Bishop => [NE, NW, SE, SW],
        _ => unreachable!(),
    };

    let mut attacks = Bitboard::EMPTY;
    let mut i = 0;
    // Loop through the directions for the piece type
    while i < dirs.len() {
        let mut to = sq;
        // Shift in the direction if the current square is empty and not occupied
        while !occ.contains(to) {
            to = match to.add(dirs[i]) {
                Ok(to) => to,
                Err(_) => break,
            };
            attacks.bitor_assign(to.bb());
        }
        // The last square is either occupied or at the border
        i += 1;
    }
    attacks
}

/// # Populate reference and occupancy tables
fn populate_table(
    pt: PieceType,
    sq: Square,
    m: &Magic,
    reference: &mut [Bitboard; MAX_PERM],
    occupancy: &mut [Bitboard; MAX_PERM],
) {
    let perm = 1 << m.mask.count_bits();
    // Populate the reference and occupancy tables for the piece type on the square
    let mut occ = Bitboard::EMPTY;
    for i in 0..perm {
        // Generate the corresponding attack rays for the occupancy, piece type, and square
        reference[i] = attacks_on_the_fly(pt, sq, occ);
        // Store the occupancy bitboard at the current index
        occupancy[i] = occ;
        // Save occupancy bitboard at the current index
        occ = (occ - m.mask) & m.mask;
    }
}

/// # Find suitable magic numbers
#[cfg(not(target_feature = "bmi2"))]
fn find_magics<const N: usize>(
    seed: u64,
    m: &mut Magic,
    reference: &mut [Bitboard; MAX_PERM],
    occupancy: &mut [Bitboard; MAX_PERM],
    table: &mut [Bitboard; N],
) -> u32 {
    let mut trying = true;
    let mut attempt = 0;
    let mut rng = PRNG::new(seed);
    let mut epoch = [0u32; MAX_PERM];
    let perm = 1 << m.mask.count_bits();
    // Find a magic for a specific mask/square until we find one that passes the
    // verification test
    while trying {
        m.magic = 0;
        // Aim to find a magic number that is sparsely populated
        while ((m.magic.wrapping_mul(m.mask.0)) >> 56).count_ones() < 6 {
            m.magic = rng.random_sparse_u64();
            // New magic number
            attempt += 1;

            // Assume the new magic number is valid, until proven otherwise
            trying = false;
            // A valid magic must map every possible occupancy to an index that
            // looks up the correct sliding atttack in the attacks lookup table.

            for i in 0..perm {
                let idx = m.index(occupancy[i]);
                // Trick to speed up the search for magic numbers
                // The m.attacks table will be slowly replaced with the attacks
                // generated by the new magic number Avoid resetting m.attacks every
                // time
                if epoch[idx - m.offset] < attempt {
                    // Update the count
                    epoch[idx - m.offset] = attempt;
                    // Update the table with the attack pattern
                    table[idx] = reference[i];
                // If the index is already occupied, check if the attack pattern stored is the same as the one we are trying to insert
                } else if table[idx] != reference[i] {
                    // Keep trying new magic numbers until we find one that works
                    trying = true;
                    break;
                }
            }
        }
    }
    attempt
}

/******************************************\
|==========================================|
|              Generate Magic              |
|==========================================|
\******************************************/

/// Seeds for generating magic numbers that optimises for speed
const BISHOP_SEEDS: [u64; Rank::NUM] = [
    0xA4302F8257706E25, // 3444 attempts to generate magic number
    0xF5DFC414DBE20B81, // 1776 attempts to generate magic number
    0x7FA4E09723AAA919, // 7952 attempts to generate magic number
    0xC14C05BE2E3A10BF, // 14684 attempts to generate magic number
    0xC60BA0419F81238E, // 17837 attempts to generate magic number
    0x065AFE010904A1A4, // 10730 attempts to generate magic number
    0x5E040F6D2A6A6FC4, // 2346 attempts to generate magic number
    0xF26CD592A998EBFC, // 2715 attempts to generate magic number
];

const ROOK_SEEDS: [u64; Rank::NUM] = [
    0xA229B9598E24437C, // 474251 attempts to generate magic number
    0x563E0EF1ABAE19AD, // 87109 attempts to generate magic number
    0x81445AC4F7966038, // 153887 attempts to generate magic number
    0xC6D2FB81AC850941, // 111506 attempts to generate magic number
    0x283C0A606438D8B3, // 123736 attempts to generate magic number
    0xEE1D97DF3E4CB9D7, // 131030 attempts to generate magic number
    0xAC18FA699DF5F065, // 75650 attempts to generate magic number
    0x33481A03E05CFBED, // 103092 attempts to generate magic number
];

/// # Generate Magics (~240ms)
/// - Initialize the magic numbers for the rook and bishop pieces.
/// - This is used to generate the attacks for the pieces.
/// - The magic numbers are used to hash the occupancy of the squares to a unique index.
/// - The occupancy is the set of squares that are occupied by pieces.
pub(crate) fn gen_slider_attacks<const N: usize>(pt: PieceType) -> SliderAttackTable<N> {
    let mut offset = 0;
    let mut total_attempt = 0;
    let mut reference = [Bitboard::EMPTY; MAX_PERM];
    let mut occupancy = [Bitboard::EMPTY; MAX_PERM];
    let mut table = Box::new([Bitboard::EMPTY; N]);
    let mut magic = [Magic::default(); Square::NUM];

    let seeds = match pt {
        PieceType::Bishop => BISHOP_SEEDS,
        PieceType::Rook => ROOK_SEEDS,
        _ => unreachable!(),
    };

    // Loop through the squares and generate the magic numbers for the piece type
    for sq in Square::iter() {
        // Initialize the magic struct for the piece type on the square
        let mut m = init_magic_struct(pt, sq, &mut offset);

        populate_table(pt, sq, &m, &mut reference, &mut occupancy);

        total_attempt += find_magics::<N>(
            seeds[sq.rank().index()],
            &mut m,
            &mut reference,
            &mut occupancy,
            &mut table,
        );

        // Store the magic number for the square
        magic[sq.index()] = m;

        if sq.file() == File::FileH {
            println!("{}", total_attempt);
            total_attempt = 0;
        }
    }

    SliderAttackTable { table, magic }
}

// /******************************************\
// |==========================================|
// |            Helper Functions              |
// |==========================================|
// \******************************************/
/// # Edge Mask
/// - The edge mask is used to remove the edges of the board from the attack mask.
/// - The edges of the board are not included in the attack lookup because it doesn't matter if there is a blocker there or not.
/// - Except in the cases where the attacking piece is on the edge, if so the edge mask should not include the attack area.
pub(crate) const fn get_edge_mask(sq: Square) -> Bitboard {
    use File::*;
    use Rank::*;

    let rank_18bb: Bitboard = Rank1.bb().bitor(Rank8.bb());
    let file_ahbb: Bitboard = FileA.bb().bitor(FileH.bb());

    let sq_rank_bb = sq.rank().bb();
    let sq_file_bb = sq.file().bb();

    let rank_mask = rank_18bb.bitand(sq_rank_bb.not());

    let file_mask = file_ahbb.bitand(sq_file_bb.not());

    // Get the edges of the board
    rank_mask.bitor(file_mask)
}

pub(crate) const fn init_magic_struct(pt: PieceType, sq: Square, offset: &mut usize) -> Magic {
    let mask = attacks_on_the_fly(pt, sq, Bitboard::EMPTY).bitand(get_edge_mask(sq).not());

    let m = Magic {
        #[cfg(not(target_feature = "bmi2"))]
        magic: 0,

        mask,
        shift: 64 - mask.count_bits() as u8,
        offset: *offset,
    };

    *offset += 1 << m.mask.count_bits();

    m
}

/// # Find Best Magic Seeds
#[cfg(not(target_feature = "bmi2"))]
pub fn find_best_magic_seeds<const N: usize>(pt: PieceType) {
    let mut offset = 0;
    let mut magic = [Magic::default(); Square::NUM];

    for sq in Square::iter() {
        // Initialize the magic struct for the piece type on the square
        magic[sq.index()] = init_magic_struct(pt, sq, &mut offset);
    }

    let mut reference = [[Bitboard::EMPTY; MAX_PERM]; File::NUM];
    let mut occupancy = [[Bitboard::EMPTY; MAX_PERM]; File::NUM];
    let mut table = Box::new([Bitboard::EMPTY; N]);

    // Loop through the squares and generate the magic numbers for the piece type
    for rank in Rank::iter() {
        let mut seeder = PRNG::default();

        let mut best_attempt = 0;
        let mut best_seed = 0;
        for file in File::iter() {
            let sq = Square::from_parts(file, rank);
            let m = &magic[sq.index()];

            let perm = 1 << m.mask.count_bits();
            // Populate the reference and occupancy tables for the piece type on the square
            let mut occ = Bitboard::EMPTY;
            for i in 0..perm {
                // Generate the corresponding attack rays for the occupancy, piece type, and square
                reference[file.index()][i] = attacks_on_the_fly(pt, sq, occ);
                // Store the occupancy bitboard at the current index
                occupancy[file.index()][i] = occ;
                // Save occupancy bitboard at the current index
                occ = (occ - m.mask) & m.mask;
            }
        }

        for i in 0..10000 {
            let mut total_attempt = 0;
            let seed = seeder.random_u64();

            for file in File::iter() {
                let sq: Square = Square::from_parts(file, rank);
                let m = &mut magic[sq.index()];

                let mut trying = true;
                let mut attempt = 0;
                let mut rng = PRNG::new(seed);
                let mut epoch = [0u32; MAX_PERM];
                let perm = 1 << m.mask.count_bits();
                // Find a magic for a specific mask/square until we find one that passes the
                // verification test
                while trying {
                    m.magic = 0;

                    // Aim to find a magic number that is sparsely populated
                    while ((m.magic.wrapping_mul(m.mask.0)) >> 56).count_ones() < 6 {
                        m.magic = rng.random_sparse_u64();
                        // New magic number
                        attempt += 1;
                        // Assume the new magic number is valid, until proven otherwise
                        trying = false;
                        // A valid magic must map every possible occupancy to an index that
                        // looks up the correct sliding atttack in the attacks lookup table.

                        for i in 0..perm {
                            let idx = m.index(occupancy[file.index()][i]);
                            // Trick to speed up the search for magic numbers
                            // The m.attacks table will be slowly replaced with the attacks
                            // generated by the new magic number Avoid resetting m.attacks every
                            // time
                            if epoch[idx - m.offset] < attempt {
                                // Update the count
                                epoch[idx - m.offset] = attempt;
                                // Update the table with the attack pattern
                                table[idx] = reference[file.index()][i];
                            // If the index is already occupied, check if the attack pattern stored is the same as the one we are trying to insert
                            } else if table[idx] != reference[file.index()][i] {
                                // Keep trying new magic numbers until we find one that works
                                trying = true;
                                break;
                            }
                        }
                    }
                }

                total_attempt += attempt;

                if total_attempt > best_attempt && best_attempt != 0 {
                    break;
                }
            }

            if best_attempt > total_attempt || best_attempt == 0 {
                best_attempt = total_attempt;
                best_seed = seed;
            }

            if i % 100 == 0 {
                println!(
                    "Rank: {rank:?}, Iteration: {i}, Best: ({}, {:#X})",
                    best_attempt, best_seed
                );
            }
        }

        println!("============================================================");
        println!("Best Attempt: {best_attempt}, Best Seed: {best_seed:#X}");
        println!("============================================================");
    }
}
