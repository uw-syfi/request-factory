//! Minimal, deliberately fast vLLM token-protocol responder for load-generator profiling.
//!
//! This is not an inference-server mock. It implements only the wire behavior req-frontend
//! needs in order to measure its own ceiling: persistent HTTP/1.1, chunked SSE, exact token
//! counts, and the two-request prefix-cache preflight. Measured requests remain stateless so
//! server-side bookkeeping cannot become the benchmark's bottleneck.

#[cfg(feature = "runtime")]
mod runtime {
    use serde::Deserialize;
    use serde_json::json;
    use std::io;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    const DEFAULT_BIND: &str = "127.0.0.1:18080";
    const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
    const PREFLIGHT_ID: &str = "req-frontend-prefix-cache-preflight";

    #[derive(Debug, Clone, Copy)]
    struct Config {
        bind: SocketAddr,
        tokens_per_chunk: usize,
        max_body_bytes: usize,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct GenerateBody {
        #[serde(default)]
        rid: Option<String>,
        token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        #[serde(default)]
        stream: bool,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct SamplingParams {
        max_tokens: usize,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Request {
        path: String,
        request_id: Option<String>,
        close: bool,
        body: GenerateBody,
    }

    #[derive(Default)]
    struct State {
        preflight_requests: AtomicUsize,
    }

    pub(super) async fn main() -> anyhow::Result<()> {
        let config = parse_args()?;
        let listener = TcpListener::bind(config.bind).await?;
        let state = Arc::new(State::default());
        eprintln!(
            "loadgen-perf-server listening on http://{} ({} tokens/SSE chunk)",
            config.bind, config.tokens_per_chunk
        );

        loop {
            let (socket, _) = listener.accept().await?;
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                if let Err(error) = serve_connection(socket, config, state).await {
                    // One line per broken connection is useful; the successful hot path is silent.
                    eprintln!("loadgen-perf-server connection error: {error}");
                }
            });
        }
    }

    fn parse_args() -> anyhow::Result<Config> {
        let mut bind = DEFAULT_BIND.parse()?;
        let mut tokens_per_chunk = 1usize;
        let mut max_body_bytes = DEFAULT_MAX_BODY_BYTES;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let value = match arg.as_str() {
                "--bind" | "--tokens-per-chunk" | "--max-body-bytes" => args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} requires a value"))?,
                "--help" | "-h" => {
                    println!(
                        "Usage: loadgen_perf_server [--bind ADDR] [--tokens-per-chunk N] \
                         [--max-body-bytes N]"
                    );
                    std::process::exit(0);
                }
                _ => anyhow::bail!("unknown argument {arg}"),
            };
            match arg.as_str() {
                "--bind" => bind = value.parse()?,
                "--tokens-per-chunk" => tokens_per_chunk = value.parse()?,
                "--max-body-bytes" => max_body_bytes = value.parse()?,
                _ => unreachable!(),
            }
        }
        anyhow::ensure!(tokens_per_chunk > 0, "--tokens-per-chunk must be positive");
        anyhow::ensure!(max_body_bytes > 0, "--max-body-bytes must be positive");
        Ok(Config {
            bind,
            tokens_per_chunk,
            max_body_bytes,
        })
    }

    async fn serve_connection(
        mut socket: TcpStream,
        config: Config,
        state: Arc<State>,
    ) -> io::Result<()> {
        socket.set_nodelay(true)?;
        let mut input = Vec::with_capacity(16 * 1024);
        loop {
            let Some(frame_len) =
                read_http_request(&mut socket, &mut input, config.max_body_bytes).await?
            else {
                return Ok(());
            };
            let frame: Vec<u8> = input.drain(..frame_len).collect();
            match parse_request(&frame) {
                Ok(request) if request.path == "/inference/v1/generate" => {
                    let cached = if request.request_id.as_deref() == Some(PREFLIGHT_ID) {
                        let ordinal = state.preflight_requests.fetch_add(1, Ordering::Relaxed);
                        usize::from(ordinal > 0) * request.body.token_ids.len()
                    } else {
                        0
                    };
                    let response = response_bytes(&request.body, config.tokens_per_chunk, cached);
                    socket.write_all(&response).await?;
                    if request.close {
                        return Ok(());
                    }
                }
                Ok(_) => {
                    write_error(&mut socket, 404, "not found").await?;
                    return Ok(());
                }
                Err(error) => {
                    write_error(&mut socket, 400, &error).await?;
                    return Ok(());
                }
            }
        }
    }

    async fn read_http_request(
        socket: &mut TcpStream,
        input: &mut Vec<u8>,
        max_body_bytes: usize,
    ) -> io::Result<Option<usize>> {
        loop {
            if let Some(header_end) = find_bytes(input, b"\r\n\r\n") {
                let header_len = header_end + 4;
                let content_len = content_length(&input[..header_end]).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length")
                })?;
                if content_len > max_body_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "request body exceeds --max-body-bytes",
                    ));
                }
                let frame_len = header_len + content_len;
                if input.len() >= frame_len {
                    return Ok(Some(frame_len));
                }
            } else if input.len() > 64 * 1024 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HTTP headers exceed 64 KiB",
                ));
            }

            let read = socket.read_buf(input).await?;
            if read == 0 {
                return if input.is_empty() {
                    Ok(None)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed during request",
                    ))
                };
            }
        }
    }

    fn parse_request(frame: &[u8]) -> Result<Request, String> {
        let header_end =
            find_bytes(frame, b"\r\n\r\n").ok_or_else(|| "incomplete HTTP headers".to_string())?;
        let headers = std::str::from_utf8(&frame[..header_end])
            .map_err(|_| "HTTP headers are not UTF-8".to_string())?;
        let mut lines = headers.split("\r\n");
        let mut request_line = lines
            .next()
            .ok_or_else(|| "missing request line".to_string())?
            .split_ascii_whitespace();
        if request_line.next() != Some("POST") {
            return Err("only POST is supported".to_string());
        }
        let path = request_line
            .next()
            .ok_or_else(|| "request path is missing".to_string())?
            .to_string();
        if request_line.next() != Some("HTTP/1.1") {
            return Err("only HTTP/1.1 is supported".to_string());
        }

        let mut request_id = None;
        let mut close = false;
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                return Err("malformed HTTP header".to_string());
            };
            if name.eq_ignore_ascii_case("x-request-id") {
                request_id = Some(value.trim().to_string());
            } else if name.eq_ignore_ascii_case("connection")
                && value.trim().eq_ignore_ascii_case("close")
            {
                close = true;
            }
        }
        let body: GenerateBody = serde_json::from_slice(&frame[header_end + 4..])
            .map_err(|error| format!("invalid vLLM request JSON: {error}"))?;
        let request_id = request_id.or_else(|| body.rid.clone());
        Ok(Request {
            path,
            request_id,
            close,
            body,
        })
    }

    fn response_bytes(body: &GenerateBody, tokens_per_chunk: usize, cached: usize) -> Vec<u8> {
        let token_count = body.sampling_params.max_tokens;
        let prompt_tokens = body.token_ids.len();
        let mut response = Vec::with_capacity(512 + token_count * 12);
        response.extend_from_slice(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
              transfer-encoding: chunked\r\nconnection: keep-alive\r\n\r\n",
        );

        for (index, start) in (0..token_count).step_by(tokens_per_chunk).enumerate() {
            let end = (start + tokens_per_chunk).min(token_count);
            let token_ids: Vec<u32> = (start..end).map(|offset| 100_000 + offset as u32).collect();
            let finish_reason = (end == token_count).then_some("length");
            let event = json!({
                "request_id": format!("loadgen-perf-{index}"),
                "choices": [{
                    "index": 0,
                    "finish_reason": finish_reason,
                    "token_ids": token_ids,
                }],
                "usage": null,
            });
            push_chunk(&mut response, format!("data: {event}\n\n").as_bytes());
        }
        let usage = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
        });
        push_chunk(&mut response, format!("data: {usage}\n\n").as_bytes());
        push_chunk(&mut response, b"data: [DONE]\n\n");
        response.extend_from_slice(b"0\r\n\r\n");
        response
    }

    fn push_chunk(output: &mut Vec<u8>, bytes: &[u8]) {
        output.extend_from_slice(format!("{:x}\r\n", bytes.len()).as_bytes());
        output.extend_from_slice(bytes);
        output.extend_from_slice(b"\r\n");
    }

    async fn write_error(socket: &mut TcpStream, status: u16, message: &str) -> io::Result<()> {
        let body = json!({"error": message}).to_string();
        let response = format!(
            "HTTP/1.1 {status} Error\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await
    }

    fn content_length(headers: &[u8]) -> Option<usize> {
        let headers = std::str::from_utf8(headers).ok()?;
        headers.split("\r\n").skip(1).find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())
                .flatten()
        })
    }

    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn request(id: &str, max_tokens: usize) -> Vec<u8> {
            let body = json!({
                "rid": id,
                "token_ids": [11, 22, 33],
                "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
                "stream": true,
            })
            .to_string();
            format!(
                "POST /inference/v1/generate HTTP/1.1\r\nHost: localhost\r\n\
                 x-request-id: {id}\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .into_bytes()
        }

        #[test]
        fn parses_the_native_vllm_request_and_request_id() {
            let parsed = parse_request(&request("req-17", 9)).unwrap();
            assert_eq!(parsed.path, "/inference/v1/generate");
            assert_eq!(parsed.request_id.as_deref(), Some("req-17"));
            assert_eq!(parsed.body.token_ids, vec![11, 22, 33]);
            assert_eq!(parsed.body.sampling_params.max_tokens, 9);
            assert!(parsed.body.stream);
            assert!(!parsed.close);
        }

        #[test]
        fn emits_disjoint_token_chunks_terminal_usage_and_done() {
            let body = parse_request(&request("req-17", 5)).unwrap().body;
            let response = String::from_utf8(response_bytes(&body, 2, 0)).unwrap();
            assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(response.contains("transfer-encoding: chunked"));
            assert!(response.contains("\"token_ids\":[100000,100001]"));
            assert!(response.contains("\"token_ids\":[100002,100003]"));
            assert!(response.contains("\"token_ids\":[100004]"));
            assert!(response.contains("\"completion_tokens\":5"));
            assert!(response.contains("\"cached_tokens\":0"));
            assert!(response.contains("data: [DONE]\n\n"));
            assert!(response.ends_with("0\r\n\r\n"));
        }

        #[test]
        fn preflight_state_hits_only_after_the_warmup_and_never_tracks_measured_requests() {
            let state = State::default();
            let cached = |id: &str, prompt_tokens: usize| {
                if id == PREFLIGHT_ID {
                    let ordinal = state.preflight_requests.fetch_add(1, Ordering::Relaxed);
                    usize::from(ordinal > 0) * prompt_tokens
                } else {
                    0
                }
            };
            assert_eq!(cached(PREFLIGHT_ID, 512), 0);
            assert_eq!(cached(PREFLIGHT_ID, 512), 512);
            assert_eq!(cached("measured-1", 512), 0);
            assert_eq!(cached("measured-1", 512), 0);
        }
    }
}

#[cfg(feature = "runtime")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    runtime::main().await
}

#[cfg(not(feature = "runtime"))]
fn main() {
    eprintln!("loadgen_perf_server requires the `runtime` feature");
    std::process::exit(2);
}
