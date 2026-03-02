//! ASCII character-level tokenizer.
//!
//! Maps characters to token IDs 0-127 (standard ASCII).

pub struct Tokenizer;

impl Tokenizer {
    pub fn encode_char(c: u8) -> u8 {
        if c < 128 { c } else { b'?' }
    }

    pub fn decode_char(token: u8) -> u8 {
        if token < 128 { token } else { b'?' }
    }

    pub fn encode_str(s: &[u8], out: &mut [u8]) -> usize {
        let len = s.len().min(out.len());
        for i in 0..len {
            out[i] = Self::encode_char(s[i]);
        }
        len
    }

    pub fn decode_token(token: u8) -> char {
        if token < 128 {
            token as char
        } else {
            '?'
        }
    }
}
