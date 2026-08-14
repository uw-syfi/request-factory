//! The stub server's lifecycle, owned by the harness rather than the operator.
//!
//! Every check here compares what the client measured against what the server
//! was *told to do*, so the harness has to be the one telling it. A stub someone
//! started by hand with different flags would make every number here a
//! coincidence.

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// What the server was told to do — and therefore what the client's measurements
/// are checked against.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// Wall time before the first chunk. The ground truth for TTFT.
    pub prefill_delay_ms: f64,
    /// Wall time between chunks. The ground truth for TPOT.
    pub chunk_delay_ms: f64,
    /// Concurrent requests served; 0 is unlimited.
    pub capacity: usize,
}

impl Timing {
    /// What one request occupies the server for, by construction.
    ///
    /// `chunks - 1` gaps, not `chunks`: the delay is paid before each chunk and
    /// the first one's wait is the prefill.
    pub fn expected_total_ms(&self, chunks: usize) -> f64 {
        self.prefill_delay_ms + self.chunk_delay_ms * chunks.saturating_sub(1) as f64
    }
}

pub struct Stub {
    child: Child,
    pub port: u16,
}

impl Stub {
    /// Start a stub and wait until it is actually accepting connections.
    ///
    /// Waits on the socket rather than sleeping a fixed interval: a harness that
    /// races its own server produces a connection error that reads like a client
    /// defect.
    pub fn start(port: u16, timing: Timing) -> Result<Self> {
        let mut command = Command::new("uv");
        command
            .args([
                "run",
                "python",
                "tools/stub_server.py",
                "--port",
                &port.to_string(),
                "--prefill-delay-ms",
                &timing.prefill_delay_ms.to_string(),
                "--chunk-delay-ms",
                &timing.chunk_delay_ms.to_string(),
                "--capacity",
                &timing.capacity.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().context(
            "failed to start tools/stub_server.py through `uv`. Run selfcheck from the repo \
             root, with uv on PATH",
        )?;

        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Ok(Self { child, port });
            }
            if let Some(status) = child.try_wait()? {
                let mut reason = String::new();
                if let Some(stderr) = child.stderr.take() {
                    for line in BufReader::new(stderr)
                        .lines()
                        .map_while(Result::ok)
                        .take(20)
                    {
                        reason.push_str(&line);
                        reason.push('\n');
                    }
                }
                bail!("the stub server exited with {status} before listening:\n{reason}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        bail!("the stub server did not start listening on port {port} within 30s");
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for Stub {
    /// Killed here rather than at the end of `main`, so a check that returns
    /// early or panics does not leave a server holding the port. The next run of
    /// this harness would otherwise measure the previous run's server.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_counts_the_gaps_between_chunks_not_the_chunks() {
        // The off-by-one that would silently widen every timing check: sleeping
        // after the last chunk instead of before each one puts an extra gap in
        // the total that no client could attribute to anything.
        let timing = Timing {
            prefill_delay_ms: 50.0,
            chunk_delay_ms: 2.0,
            capacity: 0,
        };

        assert_eq!(timing.expected_total_ms(1), 50.0);
        assert_eq!(timing.expected_total_ms(16), 50.0 + 30.0);
    }
}
