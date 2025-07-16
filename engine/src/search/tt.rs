use std::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use chess::Move;

use crate::{eval::Eval, thread::ThreadPool};

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug, Default)]
pub enum TTBound {
    #[default]
    Upper, // Fail low nodes
    Lower, // Fail-High nodes (Beta cutoff)
    Exact, // PV nodes
    None,  // Just writing to tt
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct TTEntry {
    pub key: u64,        // 64 bits
    pub age: u8,         //  7 bits
    pub depth: u8,       //  7 bits
    pub bound: TTBound,  //  2 bits
    pub best_move: Move, // 16 bits
    pub eval: Eval,      // 16 bits
    pub value: Eval,     // 16 bits
}

impl TTEntry {
    pub const AGE_MASK: u64 = 0x7F;
    pub const DEPTH_MASK: u64 = 0x7F << 7;
    pub const BOUND_MASK: u64 = 0x3 << 14;
    pub const MOVE_MASK: u64 = 0xFFFF << 16;
    pub const EVAL_MASK: u64 = 0xFFFF << 32;
    pub const VALUE_MASK: u64 = 0xFFFF << 48;

    pub fn unpack_age(data: u64) -> u8 {
        (data & Self::AGE_MASK) as u8
    }

    pub fn unpack_depth(data: u64) -> u8 {
        ((data & Self::DEPTH_MASK) >> 7) as u8
    }

    pub fn unpack_bound(data: u64) -> TTBound {
        match (data & Self::BOUND_MASK) >> 14 {
            0 => TTBound::Upper,
            1 => TTBound::Lower,
            2 => TTBound::Exact,
            _ => TTBound::None,
        }
    }

    pub fn unpack_move(data: u64) -> Move {
        unsafe { Move::new_raw(((data & Self::MOVE_MASK) >> 16) as u16) }
    }

    pub fn unpack_eval(data: u64) -> Eval {
        Eval(((data & Self::EVAL_MASK) >> 32) as i32)
    }

    pub fn unpack_value(data: u64) -> Eval {
        Eval(((data & Self::VALUE_MASK) >> 48) as i32)
    }
}

impl From<TTEntry> for (u64, u64) {
    fn from(entry: TTEntry) -> Self {
        let mut data = 0;
        data |= entry.age as u64;
        data |= (entry.depth as u64) << 7;
        data |= (entry.bound as u64) << 14;
        data |= (entry.best_move.raw() as u64) << 16;
        data |= (entry.eval.0 as u16 as u64) << 32;
        data |= (entry.value.0 as u16 as u64) << 48;
        (entry.key ^ data, data)
    }
}

impl From<(u64, u64)> for TTEntry {
    fn from((key, data): (u64, u64)) -> Self {
        Self {
            key: key ^ data,
            age: Self::unpack_age(data),
            depth: Self::unpack_depth(data),
            bound: Self::unpack_bound(data),
            best_move: Self::unpack_move(data),
            eval: Self::unpack_eval(data),
            value: Self::unpack_value(data),
        }
    }
}

// The entry stored in the TT itself.
// This is the full compressed TT value, stored in 64 bits.
// The extra 64 bits are a checksum, where we can make sure that the key
// is valid and has not been modified by two threads simultaneously.

/// Packed TT key
///
/// ## Packing Scheme
///
/// | Field     | Bits | Offset |
/// | --------- | ---- | ------ |
/// | age       | 7    | 0      |
/// | depth     | 7    | 7      |
/// | bound     | 2    | 14     |
/// | best_move | 16   | 16     |
/// | eval      | 16   | 32     |
/// | value     | 16   | 48     |
///
/// ## Checksum
///
/// The checksum is calculated by XORing the key with the data.
///
/// ## Usage
///
/// The key is used to index into the TT.
/// The data is used to store the TT entry.
/// The checksum is used to verify that the key is valid.
///
/// ## Example
///
/// ```
/// use std::sync::atomic::{AtomicU64, Ordering};
/// use chess::Move;
/// use crate::types::tt::{TTBound, TTEntry};
///
/// let mut entry = TTEntry::default();
/// entry.age = 1;
/// entry.depth = 2;
/// entry.bound = TTBound::Exact;
/// entry.best_move = Move::default();
/// entry.eval = 100;
/// entry.value = 200;
/// entry.key = 12345;
///
// / let packed = entry.pack();
// / let unpacked = TTEntry::unpack(packed.key.load(Ordering::Relaxed), packed.data.load(Ordering::Relaxed));
///
/// assert_eq!(entry, unpacked);
///

#[derive(Debug, Default)]
struct PackedTTEntry {
    key: AtomicU64,
    data: AtomicU64,
}

impl PackedTTEntry {
    fn read(&self, pos_key: u64) -> Option<TTEntry> {
        let key = self.key.load(Ordering::Relaxed);
        let data = self.data.load(Ordering::Relaxed);
        if key ^ pos_key == data {
            Some((key, data).into())
        } else {
            None
        }
    }

    fn read_unchecked(&self) -> TTEntry {
        let key = self.key.load(Ordering::Relaxed);
        let data = self.data.load(Ordering::Relaxed);
        (key, data).into()
    }

    fn write(&self, entry: TTEntry) {
        let (key, data) = entry.into();
        self.key.store(key, Ordering::Relaxed);
        self.data.store(data, Ordering::Relaxed);
    }

    /// Clears the atomic values in this entry.
    #[inline]
    fn clear(&self) {
        // Use Relaxed ordering as we don't need synchronization between
        // clearing different entries, just atomicity for each store.
        self.key.store(0, Ordering::Relaxed);
        self.data.store(0, Ordering::Relaxed);
    }
}

/// Transposition table
pub struct TT {
    table: Vec<PackedTTEntry>,
    age: u8,
}

impl Default for TT {
    fn default() -> Self {
        let mut tt = Self {
            table: Vec::new(),
            age: 0,
        };

        tt.resize(Self::DEFAULT_SIZE);

        tt
    }
}

impl TT {
    pub const DEFAULT_SIZE: usize = 16;

    fn calc_no_of_entries(mb: usize) -> usize {
        (mb << 20) / size_of::<PackedTTEntry>()
    }

    /// Creates an empty hash table (Only used for tests)
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            age: 0,
        }
    }

    pub fn resize(&mut self, mb: usize) {
        self.table
            .resize_with(Self::calc_no_of_entries(mb), PackedTTEntry::default);
    }

    pub fn size(&self) -> usize {
        self.table.len()
    }

    fn index(&self, pos_key: u64) -> usize {
        let key = pos_key as u128;
        let len = self.table.len() as u128;
        ((key * len) >> 64) as usize
    }

    pub fn get(&self, pos_key: u64) -> Option<TTEntry> {
        self.table[self.index(pos_key)].read(pos_key)
    }

    fn get_unchecked(&self, pos_key: u64) -> &PackedTTEntry {
        unsafe { self.table.get_unchecked(self.index(pos_key)) }
    }

    /// Prefetches the hash table entry for the given position key.
    ///
    /// Note that prefetching is a hint to the processor and may not always result in
    /// a cache load, depending on various factors like cache size and memory
    /// controller behavior.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let entry = self.get_unchecked(key);
            _mm_prefetch((entry as *const PackedTTEntry).cast::<i8>(), _MM_HINT_T0);
        }
    }

    pub fn write(
        &self,
        key: u64,
        bound: TTBound,
        ply: u16,
        depth: u8,
        best_move: Move,
        eval: Eval,
        value: Eval,
    ) {
        let old_entry = self.get_unchecked(key);
        let old = old_entry.read_unchecked();

        // Always replace if the position is either older, different from the input, exact, or has a higher depth
        // Example: a Depth 10 search might reach the same position of a Depth 8 search but it would have a higher depth value,
        // so this replacement scheme prefers entries with higher depth

        if self.age != old.age || key != old.key || bound == TTBound::Exact || depth > old.depth {
            let new_best_move = if key == old.key && !best_move.is_valid() {
                old.best_move
            } else {
                best_move
            };

            old_entry.write(TTEntry {
                key,
                age: self.age,
                depth,
                bound,
                best_move: new_best_move,
                eval,
                value: value.to_tt(ply),
            });
        }
    }

    pub fn hashfull(&self) -> usize {
        self.table[..1000]
            .iter()
            .filter(|e| e.read_unchecked().key != 0 && e.read_unchecked().age == self.age)
            .count()
    }

    // --- Age methods need &mut self ---
    // These must be called when exclusive access is guaranteed (e.g., main thread, pool idle)
    pub fn increment_age(&mut self) {
        self.age = (self.age + 1) & TTEntry::AGE_MASK as u8;
    }

    pub fn reset_age(&mut self) {
        self.age = 0;
    }
}

impl ThreadPool {
    pub fn clear_hash_table(&mut self, tt: &TT) {
        let no_of_entries = tt.table.len();

        if no_of_entries == 0 {
            return;
        }

        let nums_threads = self.size();

        let chunk_size = no_of_entries / nums_threads; // Floor division

        std::thread::scope(|s| {
            for i in 0..nums_threads {
                let start = i * chunk_size;
                let end = if i == nums_threads - 1 {
                    no_of_entries
                } else {
                    (i + 1) * chunk_size
                };

                s.spawn(move || {
                    for j in start..end {
                        tt.table[j].clear();
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Eval; // Make sure Eval is in scope
    use chess::{Move, MoveFlag, Square};

    // Helper to create a default TTEntry for testing
    fn create_test_entry(
        key: u64,
        age: u8,
        depth: u8,
        bound: TTBound,
        best_move: Move,
        eval: Eval,
        value: Eval,
    ) -> TTEntry {
        TTEntry {
            key,
            age: age & TTEntry::AGE_MASK as u8, // Ensure age fits in 7 bits
            depth: depth & (TTEntry::DEPTH_MASK >> 7) as u8, // Ensure depth fits in 7 bits
            bound,
            best_move,
            eval,
            value,
        }
    }

    #[test]
    fn test_ttentry_pack_unpack() {
        let entry = create_test_entry(
            0x123456789ABCDEF0,
            5,
            10,
            TTBound::Lower,
            Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush),
            Eval(100),
            Eval(150),
        );

        let (packed_key, packed_data) = entry.into();
        let unpacked_entry: TTEntry = (packed_key, packed_data).into();

        assert_eq!(entry.key, unpacked_entry.key, "Key mismatch");
        assert_eq!(entry.age, unpacked_entry.age, "Age mismatch");
        assert_eq!(entry.depth, unpacked_entry.depth, "Depth mismatch");
        assert_eq!(entry.bound, unpacked_entry.bound, "Bound mismatch");
        assert_eq!(entry.best_move, unpacked_entry.best_move, "Move mismatch");
        assert_eq!(entry.eval, unpacked_entry.eval, "Eval mismatch");
        assert_eq!(entry.value, unpacked_entry.value, "Value mismatch");

        // Test edge cases for age/depth
        let entry_max = create_test_entry(
            0xAAAAAAAAAAAAAAAA,
            127, // Max 7-bit value
            127, // Max 7-bit value
            TTBound::Upper,
            Move::NULL,
            Eval(-30000),
            Eval(30000),
        );
        let (packed_key_max, packed_data_max) = entry_max.into();
        let unpacked_entry_max: TTEntry = (packed_key_max, packed_data_max).into();
        assert_eq!(entry_max.age, unpacked_entry_max.age, "Max Age mismatch");
        assert_eq!(
            entry_max.depth, unpacked_entry_max.depth,
            "Max Depth mismatch"
        );
    }

    #[test]
    fn test_packedttentry_read_write() {
        let packed_entry = PackedTTEntry::default();
        let entry1 = create_test_entry(
            0xABCDEF0123456789,
            5,
            10,
            TTBound::Exact,
            Move::new(Square::G1, Square::F3, MoveFlag::QuietMove),
            Eval(50),
            Eval(50),
        );

        // Write entry1
        packed_entry.write(entry1);

        // Read back with correct key
        let read_entry1 = packed_entry
            .read(entry1.key)
            .expect("Failed to read entry1");
        assert_eq!(entry1, read_entry1, "Read entry mismatch");

        // Read with incorrect key
        let read_wrong_key = packed_entry.read(0xDEADBEEFDEADBEEF);
        assert!(
            read_wrong_key.is_none(),
            "Reading with wrong key should return None"
        );

        // Read unchecked
        let read_unchecked_entry = packed_entry.read_unchecked();
        // Note: read_unchecked doesn't verify the key, so it might return garbage if the key doesn't match.
        // We only check if it *could* be the correct entry based on data.
        assert_eq!(entry1.age, read_unchecked_entry.age);
        assert_eq!(entry1.depth, read_unchecked_entry.depth);
        assert_eq!(entry1.bound, read_unchecked_entry.bound);
        assert_eq!(entry1.best_move, read_unchecked_entry.best_move);
        assert_eq!(entry1.eval, read_unchecked_entry.eval);
        assert_eq!(entry1.value, read_unchecked_entry.value);

        // Clear and read
        packed_entry.clear();
        let read_after_clear = packed_entry.read(entry1.key);
        assert!(
            read_after_clear.is_none(),
            "Reading after clear should return None"
        );
        let read_unchecked_after_clear = packed_entry.read_unchecked();
        assert_eq!(read_unchecked_after_clear.key, 0); // Key should be 0 after clear
        assert_eq!(read_unchecked_after_clear.age, 0); // Data fields should be 0
    }

    #[test]
    fn test_tt_get_write_basic() {
        let mut tt = TT::default();
        tt.resize(1); // Use a small table for easier testing
        tt.age = 10;

        let key = 0x1111;
        let entry = create_test_entry(
            key,
            tt.age,
            5,
            TTBound::Lower,
            Move::new(Square::A2, Square::A4, MoveFlag::QuietMove),
            Eval(20),
            Eval(30),
        );

        // Write and get
        tt.write(
            key,
            entry.bound,
            0,
            entry.depth,
            entry.best_move,
            entry.eval,
            entry.value,
        );
        let retrieved = tt.get(key).expect("Failed to retrieve written entry");

        assert_eq!(entry.key, retrieved.key);
        assert_eq!(entry.age, retrieved.age);
        assert_eq!(entry.depth, retrieved.depth);
        assert_eq!(entry.bound, retrieved.bound);
        assert_eq!(entry.best_move, retrieved.best_move);
        // Note: Eval is stored adjusted by ply, value is stored directly
        assert_eq!(entry.eval.to_tt(0), retrieved.eval);
        assert_eq!(entry.value, retrieved.value);

        // Get non-existent key
        assert!(tt.get(0x2222).is_none());
    }

    #[test]
    fn test_tt_replacement_scheme() {
        let mut tt = TT::default();
        tt.resize(1); // Force collisions
        tt.age = 5;

        let key1 = 0xAAAA;
        let key2 = 0xBBBB; // Different key, same index

        let entry_old = create_test_entry(
            key1,
            tt.age,
            4,
            TTBound::Upper,
            Move::new(Square::B1, Square::C3, MoveFlag::QuietMove),
            Eval(10),
            Eval(10),
        );
        tt.write(
            key1,
            entry_old.bound,
            0,
            entry_old.depth,
            entry_old.best_move,
            entry_old.eval,
            entry_old.value,
        );

        // 1. Replace due to different age
        tt.age = 6;
        let entry_new_age = create_test_entry(
            key1,
            tt.age,
            3,
            TTBound::Upper,
            Move::new(Square::G8, Square::F6, MoveFlag::QuietMove),
            Eval(5),
            Eval(5),
        ); // Shallower depth
        tt.write(
            key1,
            entry_new_age.bound,
            0,
            entry_new_age.depth,
            entry_new_age.best_move,
            entry_new_age.eval,
            entry_new_age.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().depth,
            entry_new_age.depth,
            "Should replace due to age"
        );
        assert_eq!(tt.get(key1).unwrap().age, tt.age, "Age should be updated");

        // Reset age for next tests
        tt.age = 5;
        tt.write(
            key1,
            entry_old.bound,
            0,
            entry_old.depth,
            entry_old.best_move,
            entry_old.eval,
            entry_old.value,
        ); // Put old entry back

        // 2. Replace due to different key (collision)
        let entry_collision = create_test_entry(
            key2,
            tt.age,
            2,
            TTBound::Upper,
            Move::new(Square::D7, Square::D5, MoveFlag::QuietMove),
            Eval(0),
            Eval(0),
        ); // Even shallower
        tt.write(
            key2,
            entry_collision.bound,
            0,
            entry_collision.depth,
            entry_collision.best_move,
            entry_collision.eval,
            entry_collision.value,
        );
        assert_eq!(
            tt.get(key2).unwrap().key,
            key2,
            "Should replace due to collision (key)"
        );
        assert!(
            tt.get(key1).is_none(),
            "Old key should be gone after collision replacement"
        );

        // Put old entry back
        tt.write(
            key1,
            entry_old.bound,
            0,
            entry_old.depth,
            entry_old.best_move,
            entry_old.eval,
            entry_old.value,
        );

        // 3. Replace due to new entry being deeper
        let entry_deeper = create_test_entry(
            key1,
            tt.age,
            6,
            TTBound::Upper,
            Move::new(Square::E7, Square::E5, MoveFlag::QuietMove),
            Eval(15),
            Eval(15),
        );
        tt.write(
            key1,
            entry_deeper.bound,
            0,
            entry_deeper.depth,
            entry_deeper.best_move,
            entry_deeper.eval,
            entry_deeper.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().depth,
            entry_deeper.depth,
            "Should replace due to depth"
        );

        // 4. Replace due to old entry being Exact (unusual case)
        let entry_exact = create_test_entry(
            key1,
            tt.age,
            4,
            TTBound::Exact,
            Move::new(Square::C1, Square::G5, MoveFlag::QuietMove),
            Eval(30),
            Eval(30),
        );
        tt.write(
            key1,
            entry_exact.bound,
            0,
            entry_exact.depth,
            entry_exact.best_move,
            entry_exact.eval,
            entry_exact.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().bound,
            TTBound::Exact,
            "Should store exact entry first"
        );

        let entry_replace_exact = create_test_entry(
            key1,
            tt.age,
            5,
            TTBound::Lower,
            Move::new(Square::H7, Square::H6, MoveFlag::QuietMove),
            Eval(28),
            Eval(28),
        ); // Shallower, not PV, but replacing Exact
        tt.write(
            key1,
            entry_replace_exact.bound,
            0,
            entry_replace_exact.depth,
            entry_replace_exact.best_move,
            entry_replace_exact.eval,
            entry_replace_exact.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().bound,
            TTBound::Lower,
            "Should replace Exact entry"
        );
        assert_eq!(
            tt.get(key1).unwrap().depth,
            entry_replace_exact.depth,
            "Depth should be updated after replacing Exact"
        );

        // 5. Do NOT replace if shallower and other conditions not met
        let entry_shallow = create_test_entry(
            key1,
            tt.age,
            2,
            TTBound::Upper,
            Move::new(Square::A7, Square::A6, MoveFlag::QuietMove),
            Eval(26),
            Eval(26),
        );
        tt.write(
            key1,
            entry_shallow.bound,
            0,
            entry_shallow.depth,
            entry_shallow.best_move,
            entry_shallow.eval,
            entry_shallow.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().depth,
            entry_replace_exact.depth,
            "Should NOT replace with shallower entry"
        );

        // 7. Preserve best move
        let good_move = Move::new(Square::E1, Square::G1, MoveFlag::QuietMove);
        let entry_with_move = create_test_entry(
            key1,
            tt.age,
            7,
            TTBound::Lower,
            good_move,
            Eval(50),
            Eval(50),
        );
        tt.write(
            key1,
            entry_with_move.bound,
            0,
            entry_with_move.depth,
            entry_with_move.best_move,
            entry_with_move.eval,
            entry_with_move.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().best_move,
            good_move,
            "Should have good move initially"
        );

        let entry_overwrite_no_move = create_test_entry(
            key1,
            tt.age,
            8,
            TTBound::Lower,
            Move::NULL,
            Eval(55),
            Eval(55),
        ); // Deeper, but no valid move
        tt.write(
            key1,
            entry_overwrite_no_move.bound,
            0,
            entry_overwrite_no_move.depth,
            entry_overwrite_no_move.best_move,
            entry_overwrite_no_move.eval,
            entry_overwrite_no_move.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().best_move,
            good_move,
            "Should preserve old best move"
        );

        let entry_overwrite_with_move = create_test_entry(
            key1,
            tt.age,
            9,
            TTBound::Lower,
            Move::new(Square::H1, Square::G1, MoveFlag::QuietMove),
            Eval(60),
            Eval(60),
        ); // Deeper, with a valid move
        tt.write(
            key1,
            entry_overwrite_with_move.bound,
            0,
            entry_overwrite_with_move.depth,
            entry_overwrite_with_move.best_move,
            entry_overwrite_with_move.eval,
            entry_overwrite_with_move.value,
        );
        assert_eq!(
            tt.get(key1).unwrap().best_move,
            entry_overwrite_with_move.best_move,
            "Should overwrite with new valid move"
        );
    }

    #[test]
    fn test_tt_age_increment() {
        let mut tt = TT::default();
        tt.age = 126; // Max 7-bit value - 1
        assert_eq!(tt.age, 126);

        tt.increment_age();
        assert_eq!(tt.age, 127);

        tt.increment_age(); // Should wrap around (127 + 1) & 0x7F = 128 & 0x7F = 0
        assert_eq!(tt.age, 0);

        tt.increment_age();
        assert_eq!(tt.age, 1);
    }

    #[test]
    fn test_eval_mate_score_adjustment() {
        let mate_in_3 = Eval::mate_in(3); // MATE - 3
        let mated_in_5 = Eval::mated_in(5); // -MATE + 5

        let ply = 10;

        // Storing mate scores
        let stored_mate_in_3 = mate_in_3.to_tt(ply); // Should add ply: (MATE - 3) + 10
        let stored_mated_in_5 = mated_in_5.to_tt(ply); // Should subtract ply: (-MATE + 5) - 10

        assert_eq!(stored_mate_in_3, Eval::MATE - Eval(3) + Eval(ply as i32));
        assert_eq!(stored_mated_in_5, -Eval::MATE + Eval(5) - Eval(ply as i32));

        // Retrieving mate scores
        let retrieved_mate_in_3 = stored_mate_in_3.from_tt(ply); // Should subtract ply
        let retrieved_mated_in_5 = stored_mated_in_5.from_tt(ply); // Should add ply

        assert_eq!(retrieved_mate_in_3, mate_in_3);
        assert_eq!(retrieved_mated_in_5, mated_in_5);

        // Test non-mate scores
        let normal_eval = Eval(100);
        assert_eq!(normal_eval.to_tt(ply), normal_eval);
        assert_eq!(normal_eval.from_tt(ply), normal_eval);
    }

    // Note: Testing ThreadPool::clear_hash_table requires a ThreadPool instance
    // and might be better suited for integration tests, but a basic check can be done.
    #[test]
    fn test_tt_clear_basic() {
        // This test doesn't use ThreadPool directly but simulates the clearing effect.
        let mut tt = TT::default();
        tt.resize(100); // Use a reasonable size
        tt.age = 1;

        // Write some entries
        tt.write(
            0x123456789ABCDEF0,
            TTBound::Exact,
            0,
            5,
            Move::NULL,
            Eval(10),
            Eval(10),
        );
        tt.write(
            0x1240283598731485,
            TTBound::Lower,
            0,
            6,
            Move::NULL,
            Eval(20),
            Eval(20),
        );
        // Calculate index for a key to check specific clearing
        let key3 = 0x10238404385783;
        let index3 = tt.index(key3);
        tt.write(key3, TTBound::Upper, 0, 7, Move::NULL, Eval(30), Eval(30));

        assert!(tt.get(0x123456789ABCDEF0).is_some());
        assert!(tt.get(0x1240283598731485).is_some());
        assert!(tt.get(key3).is_some());

        // Simulate clearing (manually clear the underlying Vec for simplicity)
        for entry in tt.table.iter() {
            entry.clear();
        }

        assert!(
            tt.get(0x123456789ABCDEF0).is_none(),
            "Entry 1 should be cleared"
        );
        assert!(
            tt.get(0x1240283598731485).is_none(),
            "Entry 2 should be cleared"
        );
        assert!(tt.get(key3).is_none(), "Entry 3 should be cleared");
        assert_eq!(
            tt.table[index3].key.load(Ordering::Relaxed),
            0,
            "Key at index 3 should be 0"
        );
        assert_eq!(
            tt.table[index3].data.load(Ordering::Relaxed),
            0,
            "Data at index 3 should be 0"
        );
    }
}
