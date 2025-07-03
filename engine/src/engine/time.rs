use std::str::{FromStr, SplitWhitespace};

/// Helper function to parse the next token in a `SplitWhitespace` iterator
/// into a specified type `T` that implements `FromStr`.
fn parse<T: FromStr>(tokens: &mut SplitWhitespace) -> Result<T, &'static str> {
    tokens
        .next()
        .ok_or("Missing value for time control!")?
        .parse::<T>()
        .map_err(|_| "Invalid value for time control!")
}

/// Represents the different time control modes for the engine's search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeControl {
    /// Search indefinitely until stopped.
    Infinite,
    /// Search up to a fixed depth.
    FixedDepth(usize),
    /// Search up to a fixed number of nodes.
    FixedNodes(u64),
    /// Search for a fixed amount of time (in milliseconds).
    FixedTime(u64),
    /// Search specifically for a checkmate within a certain number of moves.
    Mate(usize),
    /// Standard UCI time control with remaining time and optional increments/moves to go.
    Variable {
        wtime: u64,
        // Time remaining for white/black in milliseconds.
        btime: u64,
        winc: Option<u64>,
        binc: Option<u64>,
        movestogo: Option<u16>,
    },
}

// --- Builder for TimeControl::Variable ---

/// A builder pattern helper for constructing `TimeControl::Variable`.
#[derive(Debug, Default)] // Default initializes all Options to None
struct TimeControlBuilder {
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movestogo: Option<u16>,
}

impl TimeControlBuilder {
    /// Creates a new, empty builder.
    fn new() -> Self {
        Self::default()
    }

    /// Sets the white time.
    fn wtime(&mut self, val: u64) -> &mut Self {
        self.wtime = Some(val);
        self
    }

    /// Sets the black time.
    fn btime(&mut self, val: u64) -> &mut Self {
        self.btime = Some(val);
        self
    }

    /// Sets the white increment.
    fn winc(&mut self, val: u64) -> &mut Self {
        self.winc = Some(val);
        self
    }

    /// Sets the black increment.
    fn binc(&mut self, val: u64) -> &mut Self {
        self.binc = Some(val);
        self
    }

    /// Sets the moves to go.
    fn movestogo(&mut self, val: u16) -> &mut Self {
        self.movestogo = Some(val);
        self
    }

    /// Attempts to build the TimeControl::Variable variant.
    /// Requires wtime and btime to be set.
    fn build(self) -> Result<TimeControl, &'static str> {
        match (self.wtime, self.btime) {
            (Some(wt), Some(bt)) if wt > 0 && bt > 0 => Ok(TimeControl::Variable {
                wtime: wt,
                btime: bt,
                winc: self.winc,
                binc: self.binc,
                movestogo: self.movestogo,
            }),
            _ => Err("Unknown time control command or format"),
        }
    }
}

impl FromStr for TimeControl {
    type Err = &'static str;
    /// Parses a string slice (typically from the "go" command) into a `TimeControl` variant.

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokens = s.split_whitespace();

        match tokens.next() {
            Some("infinite") => Ok(TimeControl::Infinite),
            Some("depth") => Ok(TimeControl::FixedDepth(parse(&mut tokens)?)),
            Some("nodes") => Ok(TimeControl::FixedNodes(parse(&mut tokens)?)),
            Some("mate") => Ok(TimeControl::Mate(parse(&mut tokens)?)),
            Some("movetime") => Ok(TimeControl::FixedTime(parse(&mut tokens)?)),
            // If none of the specific keywords match, attempt to parse as Variable time control.
            _ => Self::parse_variable(&mut s.split_whitespace()),
        }
    }
}

/// Parses the parameters for `TimeControl::Variable` (wtime, btime, winc, binc, movestogo).
impl TimeControl {
    fn parse_variable(tokens: &mut SplitWhitespace) -> Result<TimeControl, &'static str> {
        // Initialize optional fields to None. These will be "built" up.
        let mut builder = TimeControlBuilder::new();

        while let Some(key) = tokens.next() {
            match key {
                "wtime" => builder.wtime(parse::<i64>(tokens)?.max(0) as u64),
                "btime" => builder.btime(parse::<i64>(tokens)?.max(0) as u64),
                "winc" => builder.winc(parse(tokens)?),
                "binc" => builder.binc(parse(tokens)?),
                "movestogo" => builder.movestogo(parse(tokens)?),
                _ => return Err("Unknown or unsupported parameter in go command"),
            };
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tests for simple commands (keep existing ones) ---
    #[test]
    fn test_parse_infinite() {
        assert_eq!("infinite".parse::<TimeControl>(), Ok(TimeControl::Infinite));
    }

    #[test]
    fn test_parse_depth() {
        assert_eq!(
            "depth 8".parse::<TimeControl>(),
            Ok(TimeControl::FixedDepth(8))
        );
        assert_eq!(
            "depth".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "depth eight".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
    }

    #[test]
    fn test_parse_nodes() {
        assert_eq!(
            "nodes 100000".parse::<TimeControl>(),
            Ok(TimeControl::FixedNodes(100000))
        );
        assert_eq!(
            "nodes".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "nodes الكثير".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
    }

    #[test]
    fn test_parse_mate() {
        assert_eq!("mate 5".parse::<TimeControl>(), Ok(TimeControl::Mate(5)));
        assert_eq!(
            "mate".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "mate -3".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
    }

    #[test]
    fn test_parse_movetime() {
        assert_eq!(
            "movetime 5000".parse::<TimeControl>(),
            Ok(TimeControl::FixedTime(5000))
        );
        assert_eq!(
            "movetime".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "movetime 1.5".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
    }

    // --- Tests for "go" command using the builder ---
    #[test]
    fn test_parse_go_variable_simple() {
        let expected = TimeControl::Variable {
            wtime: 10000,
            btime: 9000,
            winc: None,
            binc: None,
            movestogo: None,
        };
        assert_eq!(
            "wtime 10000 btime 9000".parse::<TimeControl>(),
            Ok(expected)
        );
    }

    #[test]
    fn test_parse_go_variable_all_params() {
        let expected = TimeControl::Variable {
            wtime: 300000,
            btime: 295000,
            winc: Some(2000),
            binc: Some(2000),
            movestogo: Some(40),
        };
        let input = "wtime 300000 btime 295000 winc 2000 binc 2000 movestogo 40";
        assert_eq!(input.parse::<TimeControl>(), Ok(expected));
    }

    #[test]
    fn test_parse_go_variable_reordered() {
        let expected = TimeControl::Variable {
            wtime: 300000,
            btime: 295000,
            winc: Some(2000),
            binc: Some(2000),
            movestogo: Some(40),
        };
        // Same values, different order
        let input = "movestogo 40 winc 2000 wtime 300000 binc 2000 btime 295000";
        assert_eq!(input.parse::<TimeControl>(), Ok(expected));
    }

    #[test]
    fn test_parse_go_variable_with_only_some_optional() {
        let expected = TimeControl::Variable {
            wtime: 120000,
            btime: 118000,
            winc: Some(1000),
            binc: None,      // binc is missing
            movestogo: None, // movestogo is missing
        };
        let input = "wtime 120000 btime 118000 winc 1000";
        assert_eq!(input.parse::<TimeControl>(), Ok(expected));

        let expected2 = TimeControl::Variable {
            wtime: 60000,
            btime: 59000,
            winc: None,
            binc: None,
            movestogo: Some(10), // Only movestogo present
        };
        let input2 = "wtime 60000 btime 59000 movestogo 10";
        assert_eq!(input2.parse::<TimeControl>(), Ok(expected2));
    }

    #[test]
    fn test_parse_go_variable_missing_required() {
        // These fail in the builder.build() step
        assert_eq!(
            "wtime 10000".parse::<TimeControl>(),
            Err("Unknown time control command or format")
        );
        assert_eq!(
            "btime 9000 winc 100".parse::<TimeControl>(),
            Err("Unknown time control command or format")
        );
        assert_eq!(
            "winc 100 binc 100".parse::<TimeControl>(),
            Err("Unknown time control command or format")
        );
        assert_eq!(
            "go".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        ); // Just "go"
    }

    #[test]
    fn test_parse_go_variable_invalid_value() {
        // These fail in parse_value_after_key
        assert_eq!(
            "wtime ten btime 9000".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
        assert_eq!(
            "wtime 10000 btime 9k".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
        assert_eq!(
            "wtime 10000 btime 9000 movestogo forty".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
        assert_eq!(
            "wtime 10000 btime 9000 winc -50".parse::<TimeControl>(),
            Err("Invalid value for time control!")
        );
    }

    #[test]
    fn test_parse_go_variable_missing_value() {
        // These fail in parse_value_after_key
        assert_eq!(
            "wtime 10000 btime".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "wtime".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
        assert_eq!(
            "wtime 10000 btime 9000 winc".parse::<TimeControl>(),
            Err("Missing value for time control!")
        );
    }

    #[test]
    fn test_parse_go_variable_unknown_param() {
        // This fails in the main loop of the 'go' arm
        assert_eq!(
            "wtime 10000 btime 9000 unknown 123".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        );
        assert_eq!(
            "go ponder wtime 10000 btime 9000".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        );
        // Test searchmoves is now treated as unknown
        assert_eq!(
            "wtime 10000 btime 9000 searchmoves e2e4".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        );
    }

    // --- General Error Tests ---
    #[test]
    fn test_parse_empty_string() {
        assert_eq!(
            "".parse::<TimeControl>(),
            Err("Unknown time control command or format")
        );
    }

    #[test]
    fn test_parse_unknown_command() {
        assert_eq!(
            "ponder".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        );
        assert_eq!(
            "startpos".parse::<TimeControl>(),
            Err("Unknown or unsupported parameter in go command")
        );
    }
}
