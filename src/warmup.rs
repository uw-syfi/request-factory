use anyhow::{bail, Context, Result};
use std::io::Write;
use std::time::Duration;

use crate::{Args, BackendKind};

/// Keep all prompt shapes while warming only a short decode trajectory.
/// Session output lengths define later prefixes and cannot be capped this way.
pub(crate) fn limit_independent_decode(workload: &mut crate::workload::ReplayWorkload) {
    if let crate::workload::ReplayWorkload::IndependentRequests(requests) = workload {
        for request in requests {
            request.output_len = request.output_len.min(32);
        }
    }
}

pub(crate) async fn reset_after_drain(args: &Args) -> Result<()> {
    if matches!(args.backend, BackendKind::SglangTokens) {
        bail!("warmup prefix-cache reset is not supported for sglang-tokens");
    }
    let base = args.base_url.trim_end_matches('/');
    let root = base.strip_suffix("/v1").unwrap_or(base);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let response = client
            .get(format!("{root}/load"))
            .send()
            .await?
            .error_for_status()?;
        let load: serde_json::Value = response.json().await?;
        let count = load
            .get("server_load")
            .and_then(|value| value.as_u64())
            .context("warmup requires /load with integer server_load")?;
        if count == 0 {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            bail!("warmup drain timed out");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    client
        .post(format!("{root}/reset_prefix_cache"))
        .send()
        .await?
        .error_for_status()
        .context("warmup prefix-cache reset failed")?;
    Ok(())
}

pub(crate) async fn measurement_gate() -> Result<()> {
    tokio::task::spawn_blocking(|| -> Result<()> {
        println!("REQ_FRONTEND_MEASUREMENT_READY_V1");
        std::io::stdout().flush()?;
        let mut command = String::new();
        std::io::stdin().read_line(&mut command)?;
        if command.trim() != "continue" {
            bail!("measurement controller did not release the gate");
        }
        Ok(())
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn short_decode_preserves_prompt_shapes_and_original_requests() {
        use crate::schema::format::text_generation::independent::IndependentRequest;
        use crate::workload::ReplayWorkload;
        let original: Vec<_> = [1, 32, 4096]
            .into_iter()
            .enumerate()
            .map(|(i, output_len)| IndependentRequest {
                id: i.to_string(),
                input_len: 2048 * (i + 1),
                output_len,
                arrival_time: i as f64,
                slo: Default::default(),
                priority: Default::default(),
            })
            .collect();
        let mut warmup = ReplayWorkload::IndependentRequests(original.clone());
        limit_independent_decode(&mut warmup);
        let ReplayWorkload::IndependentRequests(requests) = warmup else {
            unreachable!()
        };
        for (actual, source) in requests.iter().zip(&original) {
            assert_eq!(actual.id, source.id);
            assert_eq!(actual.input_len, source.input_len);
            assert_eq!(actual.arrival_time, source.arrival_time);
        }
        assert_eq!(
            requests.iter().map(|r| r.output_len).collect::<Vec<_>>(),
            [1, 32, 32]
        );
        assert_eq!(original[2].output_len, 4096);
    }

    #[tokio::test]
    async fn drains_before_reset_and_propagates_reset_failure() {
        for status in ["200 OK", "500 Internal Server Error"] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/v1", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                for (method, path, body, code) in [
                    ("GET", "/load", "{\"server_load\":1}", "200 OK"),
                    ("GET", "/load", "{\"server_load\":0}", "200 OK"),
                    ("POST", "/reset_prefix_cache", "{}", status),
                ] {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    loop {
                        let byte = stream.read_u8().await.unwrap();
                        request.push(byte);
                        if request.ends_with(b"\r\n\r\n") {
                            break;
                        }
                    }
                    assert!(String::from_utf8(request)
                        .unwrap()
                        .starts_with(&format!("{method} {path} ")));
                    let response = format!(
                        "HTTP/1.1 {code}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                }
            });
            let args = Args::try_parse_from([
                "runner",
                "--trace",
                "trace.csv",
                "--text-file",
                "corpus.txt",
                "--tokenizer",
                "tokenizer",
                "--model",
                "model",
                "--base-url",
                &url,
            ])
            .unwrap();
            assert_eq!(reset_after_drain(&args).await.is_ok(), status == "200 OK");
            server.await.unwrap();
        }
    }
}
