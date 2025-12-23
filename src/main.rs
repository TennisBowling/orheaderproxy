use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::Response,
    routing::any,
    Router,
};
use clap::Parser;
use futures::TryStreamExt;
use reqwest::Client;
use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;
use tracing::{info, warn, Level};

const RATE_LIMIT_RPM: usize = 3; // requests per minute per model

const RATE_LIMITED_MODELS: &[&str] = &[
    "anthropic/claude-opus-4.5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
];

/// OpenRouter Header Proxy - Injects attribution headers for OpenRouter API requests
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8787)]
    port: u16,

    /// Upstream base URL to proxy requests to
    #[arg(short, long, default_value = "https://openrouter.ai")]
    upstream_base: String,

    /// HTTP-Referer header value for OpenRouter attribution
    #[arg(long, required = true)]
    http_referer: String,

    /// X-Title header value for OpenRouter attribution
    #[arg(long, required = true)]
    x_title: String,

    /// Enable verbose/debug logging
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Accept invalid/self-signed TLS certificates (for debugging with proxies like Proxyman or wtv)
    #[arg(long, default_value_t = false)]
    danger_accept_invalid_certs: bool,
}

/// Shared application state
struct AppState {
    client: Client,
    upstream_base: String,
    http_referer: String,
    x_title: String,
    // Maps model name -> list of request timestamps (for rate limiting)
    rate_limits: Mutex<HashMap<String, Vec<Instant>>>,
}

/// Hop-by-hop headers that must NOT be forwarded by proxies
const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Check if a header is a hop-by-hop header
fn is_hop_by_hop(name: &str) -> bool {
    let lower = name.to_lowercase();
    HOP_BY_HOP_HEADERS.contains(&lower.as_str())
}

/// Check if a header should be stripped (attribution headers we'll inject our own)
fn should_strip_request_header(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Strip: hop-by-hop headers, attribution headers we'll inject, host (reqwest sets it),
    // content-length (reqwest will set based on body), and accept-encoding (reqwest handles compression)
    matches!(
        lower.as_str(),
        "http-referer" | "referer" | "x-title" | "host" | "content-length" | "accept-encoding"
    ) || is_hop_by_hop(&lower)
}

/// Proxy handler - catches all requests and forwards them upstream
async fn proxy_handler(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> Result<Response, StatusCode> {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path();
    let query = uri.query();

    // Log the incoming request
    info!("Proxying {} {}", method, path);

    // Build the upstream URL
    let upstream_url = match query {
        Some(q) => format!("{}{}?{}", state.upstream_base, path, q),
        None => format!("{}{}", state.upstream_base, path),
    };

    tracing::debug!("Upstream URL: {}", upstream_url);

    // Build headers to forward
    let mut forward_headers = HeaderMap::new();
    for (name, value) in request.headers() {
        let name_str = name.as_str();

        // Skip headers we shouldn't forward
        if should_strip_request_header(name_str) {
            tracing::debug!("Stripping header: {}", name_str);
            continue;
        }

        tracing::debug!("Forwarding header: {} = {:?}", name_str, value);
        // Keep everything else (including Authorization)
        forward_headers.insert(name.clone(), value.clone());
    }

    // Inject attribution headers for OpenRouter
    if let Ok(referer_value) = HeaderValue::from_str(&state.http_referer) {
        forward_headers.insert(
            HeaderName::from_static("http-referer"),
            referer_value.clone(),
        );
        forward_headers.insert(HeaderName::from_static("referer"), referer_value);
    }
    if let Ok(title_value) = HeaderValue::from_str(&state.x_title) {
        forward_headers.insert(HeaderName::from_static("x-title"), title_value);
    }

    // Get the request body
    let body_bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| {
            tracing::error!("Failed to read request body: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    tracing::debug!("Request body size: {} bytes", body_bytes.len());

    // Extract model from request body and check rate limit
    if !body_bytes.is_empty() {
        if let Ok(body_str) = std::str::from_utf8(&body_bytes) {
            // Parse JSON to extract model field
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
                if let Some(model) = json.get("model").and_then(|m| m.as_str()) {
                    info!("Request for model: {}", model);

                    // Only rate limit if model is in the list
                    if !RATE_LIMITED_MODELS.is_empty() && RATE_LIMITED_MODELS.contains(&model) {
                        // Check rate limit and wait if needed
                        loop {
                            let wait_time = {
                                let now = Instant::now();
                                let mut limits = state.rate_limits.lock().await;
                                let timestamps =
                                    limits.entry(model.to_string()).or_insert_with(Vec::new);

                                // Remove timestamps older than 1 minute
                                timestamps.retain(|t| now.duration_since(*t).as_secs() < 60);

                                if timestamps.len() < RATE_LIMIT_RPM {
                                    // Record this request and proceed
                                    timestamps.push(now);
                                    info!(
                                        "Rate limit: {}/{} for model {}",
                                        timestamps.len(),
                                        RATE_LIMIT_RPM,
                                        model
                                    );
                                    None
                                } else {
                                    // Find oldest timestamp and calculate wait time
                                    let oldest = timestamps.iter().min().unwrap();
                                    let wait_secs = 60 - now.duration_since(*oldest).as_secs();
                                    Some(wait_secs)
                                }
                            };

                            match wait_time {
                                None => break,
                                Some(secs) => {
                                    warn!("Rate limit hit for model: {}, waiting {}s", model, secs);
                                    tokio::time::sleep(std::time::Duration::from_secs(secs + 2))
                                        .await;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Build the upstream request
    let upstream_request = state
        .client
        .request(method_to_reqwest(&method), &upstream_url)
        .headers(forward_headers.clone())
        .body(body_bytes);

    tracing::info!("Sending request to: {}", upstream_url);

    // Send the request and get a streaming response
    let upstream_response = upstream_request.send().await.map_err(|e| {
        tracing::error!("Upstream request failed: {}", e);
        // Log the full error chain
        let mut source = StdError::source(&e);
        let mut depth = 1;
        while let Some(err) = source {
            tracing::error!("  Caused by ({}): {}", depth, err);
            source = err.source();
            depth += 1;
        }
        // Log additional error details
        if e.is_connect() {
            tracing::error!("Error type: Connection error");
        }
        if e.is_timeout() {
            tracing::error!("Error type: Timeout");
        }
        if e.is_request() {
            tracing::error!("Error type: Request error");
        }
        if e.is_builder() {
            tracing::error!("Error type: Builder error");
        }
        if let Some(url) = e.url() {
            tracing::error!("Failed URL: {}", url);
        }
        StatusCode::BAD_GATEWAY
    })?;

    // Get the status code
    let status = upstream_response.status();

    // Build response headers (strip hop-by-hop)
    let mut response_headers = HeaderMap::new();
    for (name, value) in upstream_response.headers() {
        if !is_hop_by_hop(name.as_str()) {
            response_headers.insert(name.clone(), value.clone());
        }
    }

    // Stream the response body back to the client
    let stream = upstream_response
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
    *response.headers_mut() = response_headers;

    Ok(response)
}

/// Convert axum Method to reqwest Method
fn method_to_reqwest(method: &Method) -> reqwest::Method {
    match *method {
        Method::GET => reqwest::Method::GET,
        Method::POST => reqwest::Method::POST,
        Method::PUT => reqwest::Method::PUT,
        Method::DELETE => reqwest::Method::DELETE,
        Method::PATCH => reqwest::Method::PATCH,
        Method::HEAD => reqwest::Method::HEAD,
        Method::OPTIONS => reqwest::Method::OPTIONS,
        Method::CONNECT => reqwest::Method::CONNECT,
        Method::TRACE => reqwest::Method::TRACE,
        _ => reqwest::Method::GET,
    }
}

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let args = Args::parse();

    // Initialize tracing
    let log_level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(log_level).init();

    // Create HTTP client
    let mut client_builder = Client::builder();

    if args.danger_accept_invalid_certs {
        tracing::warn!("⚠️  TLS certificate verification is DISABLED - only use for debugging!");
        client_builder = client_builder.danger_accept_invalid_certs(true);
    }

    let client = client_builder
        .build()
        .expect("Failed to create HTTP client");

    // Create shared state
    let state = Arc::new(AppState {
        client,
        upstream_base: args.upstream_base.trim_end_matches('/').to_string(),
        http_referer: args.http_referer,
        x_title: args.x_title,
        rate_limits: Mutex::new(HashMap::new()),
    });

    // Build the router with catch-all route
    let app = Router::new()
        .route("/", any(proxy_handler))
        .route("/{*path}", any(proxy_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start the server
    let addr = format!("0.0.0.0:{}", args.port);
    info!("Starting proxy server on {}", addr);
    info!("Upstream: {}", args.upstream_base);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server error");
}
