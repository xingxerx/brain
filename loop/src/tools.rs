//! Deterministic tools. Every result is checkable, which is what makes the eval honest.
//! File tools are sandboxed to one workspace directory.

use crate::decision::ToolName;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_FILE_CHARS: usize = 4000;

pub struct Tools {
    root: PathBuf,
}

impl Tools {
    pub fn new(workspace: &Path) -> anyhow::Result<Self> {
        let root = fs::canonicalize(workspace)
            .map_err(|e| anyhow::anyhow!("workspace {} not found: {e}", workspace.display()))?;
        Ok(Self { root })
    }

    /// Run a tool. Errors come back as text so the model can see and recover from them.
    pub fn run(&self, tool: ToolName, input: &str) -> String {
        let r = match tool {
            ToolName::Calculator => calculate(input).map(format_number),
            ToolName::ReadFile => self.read_file(input),
            ToolName::ListDir => self.list_dir(input),
            ToolName::None => Err("no tool named".into()),
        };
        match r {
            Ok(s) => s,
            Err(e) => format!("error: {e}"),
        }
    }

    pub fn describe() -> &'static str {
        "- calculator: evaluate arithmetic. Supports + - * / ^ and parentheses. Input: \"(2 + 3) * 4\"\n\
         - read_file: read a text file in the workspace. Input: a relative path like \"notes.txt\"\n\
         - list_dir: list a directory in the workspace. Input: a relative path, or \".\" for the root"
    }

    /// Resolve a user-supplied path and refuse anything outside the workspace.
    fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        let rel = rel.trim().trim_matches('"');
        let rel = if rel.is_empty() { "." } else { rel };
        let joined = self.root.join(rel);
        let canon = fs::canonicalize(&joined).map_err(|_| format!("path not found: {rel}"))?;
        if !canon.starts_with(&self.root) {
            return Err(format!("path is outside the workspace: {rel}"));
        }
        Ok(canon)
    }

    fn read_file(&self, rel: &str) -> Result<String, String> {
        let p = self.resolve(rel)?;
        if !p.is_file() {
            return Err(format!("not a file: {rel}"));
        }
        let text = fs::read_to_string(&p).map_err(|e| format!("could not read {rel}: {e}"))?;
        if text.chars().count() > MAX_FILE_CHARS {
            let cut: String = text.chars().take(MAX_FILE_CHARS).collect();
            Ok(format!("{cut}\n[truncated at {MAX_FILE_CHARS} chars]"))
        } else {
            Ok(text)
        }
    }

    fn list_dir(&self, rel: &str) -> Result<String, String> {
        let p = self.resolve(rel)?;
        if !p.is_dir() {
            return Err(format!("not a directory: {rel}"));
        }
        let mut names: Vec<String> = fs::read_dir(&p)
            .map_err(|e| format!("could not list {rel}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if e.path().is_dir() {
                    format!("{name}/")
                } else {
                    name
                }
            })
            .collect();
        names.sort();
        if names.is_empty() {
            Ok("(empty)".into())
        } else {
            Ok(names.join("\n"))
        }
    }
}

fn format_number(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        let s = format!("{x:.10}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Recursive-descent arithmetic parser. No eval, no code execution.
pub fn calculate(input: &str) -> Result<f64, String> {
    let tokens: Vec<char> = input.chars().filter(|c| !c.is_whitespace()).collect();
    if tokens.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser { t: &tokens, i: 0 };
    let v = p.expr()?;
    if p.i != tokens.len() {
        return Err(format!("unexpected '{}' at position {}", tokens[p.i], p.i));
    }
    if !v.is_finite() {
        return Err("result is not a finite number (division by zero?)".into());
    }
    Ok(v)
}

struct Parser<'a> {
    t: &'a [char],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.t.get(self.i).copied()
    }

    // expr := term (('+' | '-') term)*
    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        while let Some(c @ ('+' | '-')) = self.peek() {
            self.i += 1;
            let r = self.term()?;
            v = if c == '+' { v + r } else { v - r };
        }
        Ok(v)
    }

    // term := power (('*' | '/' | 'x') power)*
    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.power()?;
        while let Some(c @ ('*' | '/' | 'x' | '×' | '÷')) = self.peek() {
            self.i += 1;
            let r = self.power()?;
            v = if c == '/' || c == '÷' { v / r } else { v * r };
        }
        Ok(v)
    }

    // power := unary ('^' power)?   (right-associative)
    fn power(&mut self) -> Result<f64, String> {
        let base = self.unary()?;
        if self.peek() == Some('^') {
            self.i += 1;
            let exp = self.power()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    // unary := '-' unary | atom
    fn unary(&mut self) -> Result<f64, String> {
        if self.peek() == Some('-') {
            self.i += 1;
            return Ok(-self.unary()?);
        }
        if self.peek() == Some('+') {
            self.i += 1;
            return self.unary();
        }
        self.atom()
    }

    // atom := number | '(' expr ')'
    fn atom(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some('(') => {
                self.i += 1;
                let v = self.expr()?;
                if self.peek() != Some(')') {
                    return Err("missing ')'".into());
                }
                self.i += 1;
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.i;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.' || c == ',') {
                    self.i += 1;
                }
                let s: String = self.t[start..self.i]
                    .iter()
                    .filter(|&&c| c != ',')
                    .collect();
                s.parse::<f64>().map_err(|_| format!("bad number '{s}'"))
            }
            Some(c) => Err(format!("unexpected '{c}' at position {}", self.i)),
            None => Err("expression ended early".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(calculate("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(calculate("(2 + 3) * 4").unwrap(), 20.0);
        assert_eq!(calculate("2^3^2").unwrap(), 512.0);
        assert_eq!(calculate("-3 + 5").unwrap(), 2.0);
        assert_eq!(calculate("1,250 * 4").unwrap(), 5000.0);
        assert_eq!(format_number(calculate("10 / 4").unwrap()), "2.5");
        assert!(calculate("1 / 0").is_err());
        assert!(calculate("2 +").is_err());
        assert!(calculate("import os").is_err());
    }

    #[test]
    fn sandbox_blocks_escape() {
        let dir = std::env::temp_dir().join(format!("brain-tools-test-{}", std::process::id()));
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("a.txt"), "hello").unwrap();
        let t = Tools::new(&dir).unwrap();
        assert_eq!(t.run(ToolName::ReadFile, "a.txt"), "hello");
        assert_eq!(t.run(ToolName::ListDir, "."), "a.txt\nsub/");
        assert!(t
            .run(ToolName::ReadFile, "../../etc/passwd")
            .starts_with("error"));
        assert!(t
            .run(ToolName::ReadFile, "missing.txt")
            .starts_with("error"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
