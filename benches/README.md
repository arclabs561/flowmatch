# flowmatch benchmarks & profiling

## Run benchmarks

```bash
cd /Users/arc/Documents/dev

# Pairing microbench (isolates “coupling” costs)
cargo bench -p flowmatch --bench pairing

# USGS real-data pipeline (OT cost, sliced Wasserstein, short RFM train)
cargo bench -p flowmatch --bench realdata_usgs

# Protein torsions pipeline (Ramachandran JS + short RFM train)
cargo bench -p flowmatch --bench realdata_torsions
```

Criterion HTML reports are written under `target/criterion/`.

## Profiling (what’s hot)

This workspace’s macOS environment currently lacks the usual profiling CLIs (e.g. `xctrace`),
and `pprof`/Criterion profiling was unstable (SIGTRAP) in practice.

Instead, use the explicit timing breakdown examples:

```bash
cd /Users/arc/Documents/dev
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

