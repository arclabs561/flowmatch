# flowmatch benchmarks & profiling

## Run benchmarks

```bash
# From the flowmatch repository root:

# Pairing microbench (isolates “coupling” costs)
cargo bench -p flowmatch --bench pairing

# USGS real-data pipeline (OT cost, sliced Wasserstein, short RFM train)
cargo bench -p flowmatch --bench realdata_usgs

# Protein torsions pipeline (Ramachandran JS + short RFM train)
cargo bench -p flowmatch --bench realdata_torsions
```

Criterion HTML reports are written under `target/criterion/`.

## Profiling (what’s hot)

If profiling CLIs such as `xctrace` are unavailable, or `pprof`/Criterion
profiling fails with SIGTRAP on your platform, use the timing examples below.

Instead, use the explicit timing breakdown examples:

```bash
# From the flowmatch repository root:
cargo run -p flowmatch --example profile_breakdown_usgs
cargo run -p flowmatch --example profile_breakdown_torsions
```

These print per-phase timings (sampling, Sinkhorn coupling, SGD) so you can see what dominates.

## Optional: enable pprof in Criterion config (edit benches)

You can wire pprof directly:

```rust,ignore
use pprof::criterion::{PProfProfiler, Output};

criterion_group! {
  name = benches;
  config = Criterion::default()
    .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
  targets = bench_realdata_usgs
}
criterion_main!(benches);
```

