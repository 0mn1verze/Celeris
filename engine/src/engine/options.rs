use std::str::{FromStr, SplitWhitespace};

use super::command::UCICommandError;

#[derive(Debug, PartialEq, Eq)]
pub enum EngineOption {
    /// Command to enable chess960 option
    Chess960(bool),
    /// Command to clear the transposition table.
    ClearHash,
    /// Command to resize the transposition table (value in MB).
    ResizeHash(usize),
    /// Command to change the number of search threads.
    ResizeThreads(usize),
    /// Temporary option for tunables.
    #[cfg(feature = "tune")]
    SetTunable(String, String),
}

impl EngineOption {
    /// Helper to parse the value part of a "setoption" command into a specific type.
    fn parse_value<T: FromStr>(
        option_name: &str,
        tokens: SplitWhitespace,
    ) -> Result<T, UCICommandError> {
        // Collect the rest as value (values can also have spaces, though less common for standard UCI options)
        let value_str = tokens.collect::<Vec<&str>>().join(" ");

        if value_str.is_empty() {
            return Err(UCICommandError(format!(
                "Missing value for option '{}'",
                option_name
            )));
        }

        value_str
            .parse::<T>()
            .map_err(|_| UCICommandError(format!("Invalid value type")))
    }
}

impl TryFrom<(String, SplitWhitespace<'_>)> for EngineOption {
    type Error = UCICommandError;

    fn try_from((option_name, tokens): (String, SplitWhitespace)) -> Result<Self, UCICommandError> {
        let option = match option_name.to_ascii_lowercase().as_str() {
            "uci_chess960" => Self::Chess960(Self::parse_value(&option_name, tokens)?),
            "clear hash" => Self::ClearHash,
            "hash" => Self::ResizeHash(Self::parse_value(&option_name, tokens)?),
            "threads" => Self::ResizeThreads(Self::parse_value(&option_name, tokens)?),

            #[cfg(not(feature = "tune"))]
            _ => {
                return Err(UCICommandError(format!(
                    "Unknown option name: '{}'",
                    option_name
                )));
            }

            #[cfg(feature = "tune")]
            _ => Self::SetTunable(option_name, tokens.collect::<Vec<&str>>().join(" ")),
        };

        Ok(option)
    }
}
