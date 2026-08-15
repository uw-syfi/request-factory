//! Component ceiling for shared-pool versus connection-owned HTTP/1 transport.
//!
//! See `benches/README.md` before interpreting the number this binary prints.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use hyper::body::HttpBody;
use hyper::{Body, Request, Uri};
use tokio::net::TcpStream;
use tokio::sync::{oneshot, watch};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Transport {
    ReqwestPool,
    HyperPool,
    DedicatedHttp1,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    transport: Transport,
    #[arg(long, default_value = "http://127.0.0.1:18080/inference/v1/generate")]
    endpoint: String,
    #[arg(long, default_value_t = 1_000_000)]
    requests: usize,
    #[arg(long, default_value_t = 32)]
    connections: usize,
    #[arg(long, default_value_t = 16)]
    runtime_worker_threads: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.requests == 0 || args.connections == 0 || args.runtime_worker_threads == 0 {
        return Err(anyhow!(
            "requests, connections, and runtime workers must be positive"
        ));
    }
    if args.requests < args.connections {
        return Err(anyhow!(
            "requests ({}) must be at least connections ({})",
            args.requests,
            args.connections
        ));
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.runtime_worker_threads)
        .enable_all()
        .build()?
        .block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    let body = Bytes::from_static(
        br#"{"model":"perf","rid":"transport-bench","token_ids":[1],"sampling_params":{"max_tokens":1,"temperature":0.0,"ignore_eos":true},"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    let (start_tx, start_rx) = watch::channel(false);
    let mut ready = Vec::with_capacity(args.connections);
    let mut tasks = tokio::task::JoinSet::new();
    match args.transport {
        Transport::ReqwestPool => {
            let client = Arc::new(
                reqwest::Client::builder()
                    .pool_max_idle_per_host(20_000)
                    .tcp_nodelay(true)
                    .timeout(Duration::from_secs(3600))
                    .build()?,
            );
            for worker in 0..args.connections {
                let client = client.clone();
                let endpoint = args.endpoint.clone();
                let body = body.clone();
                let start_rx = start_rx.clone();
                let count = worker_request_count(args.requests, args.connections, worker);
                let (ready_tx, ready_rx) = oneshot::channel();
                ready.push(ready_rx);
                tasks.spawn(async move {
                    if let Err(error) = reqwest_request(&client, &endpoint, body.clone()).await {
                        let _ = ready_tx.send(Err(error));
                        return Ok(());
                    }
                    if ready_tx.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    await_start(start_rx).await?;
                    for _ in 0..count {
                        reqwest_request(&client, &endpoint, body.clone()).await?;
                    }
                    Result::<()>::Ok(())
                });
            }
        }
        Transport::HyperPool => {
            let uri: Uri = args.endpoint.parse().context("invalid --endpoint")?;
            require_http(&uri)?;
            let mut connector = hyper::client::HttpConnector::new();
            connector.set_nodelay(true);
            let client = hyper::Client::builder()
                .pool_max_idle_per_host(20_000)
                .build::<_, Body>(connector);
            for worker in 0..args.connections {
                let client = client.clone();
                let uri = uri.clone();
                let body = body.clone();
                let start_rx = start_rx.clone();
                let count = worker_request_count(args.requests, args.connections, worker);
                let (ready_tx, ready_rx) = oneshot::channel();
                ready.push(ready_rx);
                tasks.spawn(async move {
                    if let Err(error) = hyper_pool_request(&client, &uri, body.clone()).await {
                        let _ = ready_tx.send(Err(error));
                        return Ok(());
                    }
                    if ready_tx.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    await_start(start_rx).await?;
                    for _ in 0..count {
                        hyper_pool_request(&client, &uri, body.clone()).await?;
                    }
                    Result::<()>::Ok(())
                });
            }
        }
        Transport::DedicatedHttp1 => {
            let uri: Uri = args.endpoint.parse().context("invalid --endpoint")?;
            require_http(&uri)?;
            for worker in 0..args.connections {
                let uri = uri.clone();
                let body = body.clone();
                let start_rx = start_rx.clone();
                let count = worker_request_count(args.requests, args.connections, worker);
                let (ready_tx, ready_rx) = oneshot::channel();
                ready.push(ready_rx);
                tasks.spawn(async move {
                    let mut sender = match connect(&uri).await {
                        Ok(sender) => sender,
                        Err(error) => {
                            let _ = ready_tx.send(Err(error));
                            return Ok(());
                        }
                    };
                    if let Err(error) = hyper_request(&mut sender, &uri, body.clone()).await {
                        let _ = ready_tx.send(Err(error));
                        return Ok(());
                    }
                    if ready_tx.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    await_start(start_rx).await?;
                    for _ in 0..count {
                        hyper_request(&mut sender, &uri, body.clone()).await?;
                    }
                    Result::<()>::Ok(())
                });
            }
        }
    }

    for (worker, ready) in ready.into_iter().enumerate() {
        ready
            .await
            .with_context(|| format!("transport worker {worker} exited during warm-up"))?
            .with_context(|| format!("transport worker {worker} warm-up failed"))?;
    }
    let start = Instant::now();
    start_tx
        .send(true)
        .map_err(|_| anyhow!("transport workers exited before the timed phase"))?;
    while let Some(result) = tasks.join_next().await {
        result.context("transport worker panicked")??;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "transport={:?} requests={} connections={} runtime_workers={} elapsed_s={:.6} requests_per_s={:.3}",
        args.transport,
        args.requests,
        args.connections,
        args.runtime_worker_threads,
        seconds,
        args.requests as f64 / seconds,
    );
    Ok(())
}

async fn await_start(mut start: watch::Receiver<bool>) -> Result<()> {
    while !*start.borrow() {
        start
            .changed()
            .await
            .context("benchmark start signal was dropped")?;
    }
    Ok(())
}

fn worker_request_count(requests: usize, workers: usize, worker: usize) -> usize {
    requests / workers + usize::from(worker < requests % workers)
}

async fn reqwest_request(client: &reqwest::Client, endpoint: &str, body: Bytes) -> Result<()> {
    let response = client
        .post(endpoint)
        .header("content-type", "application/json")
        .header("x-request-id", "transport-bench")
        .body(body)
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}", response.status()));
    }
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        chunk?;
    }
    Ok(())
}

fn require_http(uri: &Uri) -> Result<()> {
    if uri.scheme_str() != Some("http") {
        return Err(anyhow!("Hyper transport benchmarks support http:// only"));
    }
    Ok(())
}

async fn hyper_pool_request(
    client: &hyper::Client<hyper::client::HttpConnector>,
    uri: &Uri,
    body: Bytes,
) -> Result<()> {
    let request = Request::post(uri.clone())
        .header("content-type", "application/json")
        .header("x-request-id", "transport-bench")
        .header("content-length", body.len())
        .body(Body::from(body))?;
    let mut response = client.request(request).await?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}", response.status()));
    }
    while let Some(chunk) = response.body_mut().data().await {
        chunk?;
    }
    Ok(())
}

async fn connect(uri: &Uri) -> Result<hyper::client::conn::SendRequest<Body>> {
    let host = uri.host().context("endpoint has no host")?;
    let port = uri.port_u16().unwrap_or(80);
    let stream = TcpStream::connect((host, port)).await?;
    stream.set_nodelay(true)?;
    let (sender, connection) = hyper::client::conn::handshake(stream).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("HTTP/1 connection error: {error}");
        }
    });
    Ok(sender)
}

async fn hyper_request(
    sender: &mut hyper::client::conn::SendRequest<Body>,
    uri: &Uri,
    body: Bytes,
) -> Result<()> {
    futures::future::poll_fn(|context| sender.poll_ready(context)).await?;
    let target = uri
        .path_and_query()
        .map_or("/", hyper::http::uri::PathAndQuery::as_str);
    let authority = uri.authority().context("endpoint has no authority")?;
    let request = Request::post(target)
        .header("host", authority.as_str())
        .header("content-type", "application/json")
        .header("x-request-id", "transport-bench")
        .header("content-length", body.len())
        .body(Body::from(body))?;
    let mut response = sender.send_request(request).await?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}", response.status()));
    }
    while let Some(chunk) = response.body_mut().data().await {
        chunk?;
    }
    Ok(())
}
