set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --all-targets --all-features -- -D warnings

# nextest is faster and gives per-test isolation; fall back to cargo test.
test:
    @if command -v cargo-nextest >/dev/null; then cargo nextest run --locked; else cargo test --locked; fi

doc:
    cargo doc --no-deps --locked

check: fmt-check lint test doc
