# Amalia — Hierarchical BSGS for Bitcoin Puzzles with Halving Tree

A private key search engine for the [Bitcoin Puzzle Transactions](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx), combining a **halving tree** (search space reduction via successive division by 2 on the elliptic curve) with a **hierarchical BSGS** (M/M2/M3 + Bloom filter cascade, in the style of [keyhunt](https://github.com/albertobsd/keyhunt)/[JLP BSGS](https://github.com/JeanLucPons/BSGS)), with an optional CUDA GPU-accelerated path.

Built and validated by solving **[Bitcoin Puzzle #64](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)** (private key `f7051f27b09112d4`) from scratch, with an optimization pipeline that delivered a large speedup over the first working version on CPU, plus a separate GPU implementation validated against the same known answer.

## ⚠️ Important notice on what this tool IS and IS NOT

This tool solves the subproblem *"find `A` in a 45-bit space, given `N` candidates"* optimally (`O(√(N·R))`, the theoretical Shoup bound for generic-group DLP). This makes it **practical at the scale of Puzzle #64** (`tree_steps=19`, total cost `~2^33`).

**It does not solve larger puzzles** (#135, #140, ...) even though the code supports arbitrary `tree_steps` and arbitrary bit ranges. For `tree_steps=95` (Puzzle #140, 140→45 bits), the theoretical minimum cost is `2^70` operations — even with cutting-edge GPU clusters, that still requires decades to millennia. This was extensively confirmed during this investigation (mathematically, empirically, and by comparison with real community GPU tools). See [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md) for the complete record.

`--attack-loop`, `--block-search`, and `--gpu-search` all **exist and work correctly** at any `tree_steps`/range size — but this does not change the underlying mathematical reality.

## Architecture

```
Master key (pubkey)
       │
       │  divide by 2 on the curve, "steps" times
       ▼
  N leaves (candidates for b)
       │
       │  for each leaf, giant-steps up to 2^45/M times
       ▼
hierarchical_lookup()
       │
   Bloom1 (M)  ──miss──> discard this giant-step
       │ pass
   i2-loop × Bloom2 (M2)
       │ pass
   i3-loop × Bloom3 (M3)
       │ pass
   exact table (M3, MSB-indexed)
       │
       ▼
    FOUND: K = A·2^steps + b
```

The same Bloom1/i2/i3/exact hierarchy is used by both the CPU search path and the GPU path — only the round loop that walks giant-steps and checks candidates against it differs (CPU threads vs. CUDA kernels).

### Implemented optimizations (all validated with self-tests and real measurement)

| Technique | Measured gain |
|---|---|
| MSB-indexed exact table (instead of binary search) | ~2× |
| Per-thread lockstep processing + batched modular inversion (giant-step) | included in overall gain |
| Bidirectional search (Meet-in-the-Middle from both ends of the range) | ~18× additional |
| Batched modular inversions in the `i2` loop (31 inversions → 1 batched) | ~18% additional |
| Non-atomic reads in the read-only search path (`bloom_check`) | ~2× per Bloom1 check |
| Manual Bloom1 hash-count tuning (`--bloom1-k 4`, see below) | ~10% additional |
| Isolated blocked-Bloom1 (`--bloom1-block-bits 8192`, 1KB blocks — see notes) | ~4% additional |
| **First-round memory pre-warming** (always on — see below) | **~35% additional, the single largest CPU-side win** |
| Fingerprint-loop unrolling (4-way, `-march=native` auto-vectorization) | ~2-8% additional |
| GPU: shared batch-inversion across a block's threads (vs. one inversion per thread per round) | ~2.2× |
| GPU: fixing `half_giant` to cover a disjoint half per front, not a redundant full range | ~1.8× on top |
| GPU: `--maxrregcount` register-pressure tuning (see below) | up to ~2.3× on top, hardware-specific |

See [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md) for the full analysis of each optimization, including those that were tried and **discarded** because they didn't pay off in practice.

## Build

### CPU-only (default)

```bash
make
```

Requires `g++` with OpenMP support and the secp256k1 library from [JeanLucPons/BSGS](https://github.com/JeanLucPons/BSGS) (included under `secp256k1/`, `hash/`, `util.c`). The `Makefile` compiles each source file incrementally (only rebuilds what changed) and also provides:

```bash
make test       # build, then run the core self-tests
make gen-test   # build, then validate leaf generation
make clean      # remove build artifacts
```

Note: `util.c` uses C++ headers (`<cstring>`, etc.) despite its `.c` extension — the `Makefile` compiles it with `g++`, not `gcc`. If you ever hand-roll a build command instead of using `make`, keep everything under a single `g++` invocation for this reason.

### GPU-accelerated build (`make GPU=1`)

Requires the CUDA toolkit (`nvcc`) and a CUDA-capable NVIDIA GPU.

```bash
make GPU=1
```

The GPU's compute capability (`sm_XX`) is **auto-detected** from the local machine via `nvidia-smi` — no need to know or pass it yourself when building on the same machine that will run the binary. If `nvidia-smi` can't find a GPU (headless build machine, driver issue), a warning is printed and `nvcc`'s own generic default is used instead — pass it explicitly in that case:

```bash
make GPU=1 GPU_ARCH=sm_75    # Turing (T4, RTX 20xx)
make GPU=1 GPU_ARCH=sm_86    # Ampere (RTX 30xx, A100 variants)
make GPU=1 GPU_ARCH=sm_90    # Hopper (H100, H200)
```

**Register-count tuning (`GPU_MAXREG`, optional, hardware-specific)**: the GPU kernel's default register allocation can leave a GPU under-occupied (too few thread blocks running concurrently per SM). Capping the register count via `nvcc -maxrregcount` can raise occupancy substantially at the cost of some register spilling — the net effect depends entirely on the specific GPU's SM resources, and swings performance by 2× or more in *either* direction depending on the exact value chosen, confirmed by sweeping values on a Tesla T4 during this project's own development (see [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md)). **Not auto-tuned** — re-run the sweep for your own GPU before trusting a value:

```bash
# 1. Build the isolated benchmark tool (not part of the main binary):
nvcc -O3 -arch=sm_75 -o gpu_multiround_bench gpu_multiround_bench.cu

# 2. Sweep --maxrregcount values, watching "Test B" throughput (Mchecks/sec):
for r in 32 48 64 72 80 88 96 112 128; do
  nvcc -O3 -arch=sm_75 -maxrregcount=$r -o /tmp/bench_$r gpu_multiround_bench.cu 2>/dev/null
  echo -n "maxrregcount=$r: "
  /tmp/bench_$r --n 2097152 --k 32 | grep "Test B"
done

# 3. Rebuild the real binary with whichever value won:
make clean
make GPU=1 GPU_ARCH=sm_75 GPU_MAXREG=80   # this project's own T4 result — re-tune for your GPU
```

The relationship is not smooth — nearby register-count values can swing throughput by 2× or more depending on exactly how many thread blocks per SM the value allows, so a coarse sweep (steps of 16) followed by a finer sweep around the best value found is the right approach, not a single guess.

Without `GPU_MAXREG` set, the build uses `nvcc`'s own default register allocation (still functional and correct, just not occupancy-tuned).

### Static linking (both builds)

Both the CPU-only and GPU builds link statically where possible (`-static` for the CPU build; `-Xcompiler -static` for the GPU build, on top of `nvcc`'s own default static linking of the CUDA runtime). For the GPU build, the **NVIDIA driver library (`libcuda.so`) can never be statically linked** — this is a hard constraint of the CUDA driver model on every CUDA program, not a limitation specific to this build: the driver must match whatever version is actually installed on the machine running the binary, the same way an OS kernel driver can't be baked into a portable static binary. A GPU-enabled binary therefore still needs a matching NVIDIA driver on the target machine (unavoidable — it needs an actual GPU to do anything useful anyway); everything else that can be static, is.

```bash
file amalia_hier_leaf && ldd amalia_hier_leaf
```

### Tuning Bloom1's hash count (`--bloom1-k`)

`fp-rate` and `--bloom1-k` are two **orthogonal** parameters — worth being precise about, since it's easy to conflate them. `fp-rate` controls Bloom1's **memory size** (bit count). `--bloom1-k` overrides the number of hash checks per query **without changing the memory size at all** — trading effective false-positive rate (and thus extra downstream ECC work in the `i2`/`i3` loops) for fewer memory accesses per check.

On this session's hardware (Puzzle #64 scale, `m-bits=31`), sweeping `--bloom1-k` found a real, validated **~10% additional speedup** at `k=4` (vs. the theoretically "optimal" auto-computed `k`, which turned out not to be optimal for *wall-clock time* — only for false-positive rate in isolation, ignoring the real cost of a memory access on this hardware):

| `--bloom1-k` | Search time | vs. baseline |
|---:|---:|---:|
| auto (baseline) | 127.99s | — |
| **4** | **115.89s** | **+10.4%** |
| 6 | 117.25s | +9.2% |
| 5 | 126.95s | +0.8% |
| 3 | 142.06s | −9.9% |
| 2 | 155.70s | −17.8% |

This is hardware- and workload-specific — re-run the sweep on your own machine before trusting a value. See [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md) for the full methodology.

### Memory pre-warming (always on, no flag needed)

Round-by-round instrumentation (`--dump-round-stats`, below) revealed the first ~64 rounds of any real CPU search cost far more than every subsequent round — a one-time anomaly caused by lazy physical-memory commit on first touch of Bloom1/2/3's several GB. The fix (now always active, no flag): touch every page of Bloom1/2/3 once, in parallel, right before starting the timed search — moving that one-time cost out of the measured search time. See [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md) for the full measurement.

### Round-by-round diagnostics (`--dump-round-stats FILE`)

Writes a CSV of per-stage cost deltas (Bloom1/i2/Bloom2/i3, both operation counts and time) every 64 rounds, for post-hoc analysis of where time actually goes over the course of a real search. Purely observational — does not change the search itself. Available on both `--search` and `--attack-loop` (CPU paths).

### NUMA-aware memory/thread placement (`--numa-aware`, multi-node servers only)

Interleaves Bloom1/2/3 across all NUMA nodes (`numa_interleave_memory`) and pins each OpenMP thread round-robin to a node (`numa_run_on_node`). Requires building with `make NUMA=1` (needs `libnuma-dev` installed) — the default build compiles it out entirely, so it's a zero-risk no-op on a single-node machine. Even when built in, it auto-disables itself if the machine only has one node.

## Getting started (CPU)

### 1. Always validate first (self-tests)

Before any real attack, run the self-tests — they include degenerate cases (`r2=0`, `r3=0`) that already caught real bugs during development. `make test` runs these for you, or manually:

```bash
./amalia_hier_leaf --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4
./amalia_hier_leaf --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 1
./amalia_hier_leaf --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 2
./amalia_hier_leaf --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 3
```

If you're going to use `--attack-loop`, also validate leaf generation (`make gen-test`, or manually):

```bash
./amalia_hier_leaf --gen-self-test --tree-steps 19
```

### 2. Build the M/M2/M3 hierarchy

```bash
./amalia_hier_leaf --build --m-bits 31 --hier-table hier_m31.bin -t 8 --gen-batch 64
```

`m-bits=31` (~3.9GB Bloom1) is the validated near-optimal point for `tree_steps=19` on a machine with ~13GB RAM. See below for other RAM tiers, and [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md) for the measured memory/compute trade-off at values below the optimum.

Add `--bloom1-k 4` (see above) for a further ~10% search speedup, validated on this session's hardware — re-sweep on your own machine before trusting it blindly.

### Choosing `m-bits` for your RAM and target

The optimal `M` (and thus `m-bits`) is `√(2^root_bits)` — it depends **only on the total bit-size of the puzzle you're targeting**, not on `tree_steps` or on how much RAM you happen to have. For Puzzle #64 (`root_bits=64`), the mathematical optimum is `m-bits=32`; having more RAM available does not help you go higher for *that* puzzle — it only matters if you're targeting a larger `root_bits`.

| `root_bits` (puzzle size) | Optimal `m-bits` |
|---:|---:|
| 64 | 32 |
| 80 | 40 |
| 96 | 48 |
| 112 | 56 |
| 128 | 64 |
| 140 | 70 |

Bloom1 size grows exponentially with `m-bits` (each `+1` roughly doubles it). Use the table below to pick a safe `m-bits` for your available RAM — capped at the mathematical optimum above, whichever is lower:

| RAM available | Safe `m-bits` (~30% of RAM) | Bloom1 size | Tight `m-bits` (~45% of RAM) | Bloom1 size |
|---:|---:|---:|---:|---:|
| 8 GB | 30 | 1.93 GB | 30 | 1.93 GB |
| 16 GB | 31 | 3.86 GB | 31 | 3.86 GB |
| 32 GB | 32 | 7.72 GB | 32 | 7.72 GB |
| 64 GB | 33 | 15.44 GB | 33 | 15.44 GB |
| 128 GB | 34 | 30.88 GB | 34 | 30.88 GB |

The "safe" column is a conservative target (Bloom1 ≤ ~30% of total RAM). The "tight" column (~45%) leaves less margin for the OS, the temporary build-time buffer, and Bloom2/Bloom3/the exact table (small relative to Bloom1, but not zero). Actual safe headroom depends on what else is running on your machine — treat these as starting points, not guarantees, and watch for OOM on your first `--build` at a new `m-bits`.

Note that for Puzzle #64 specifically, going past `m-bits=32` gives **no further benefit** regardless of available RAM — `M=2^32` is already the exact mathematical optimum for a 64-bit target (see the `N·R` conservation law in [`INVESTIGATION_NOTES.md`](INVESTIGATION_NOTES.md)). The RAM table above only matters if you're building the hierarchy for a larger `root_bits`.

### 3. Attack (CPU)

**Classic mode** (pre-generated leaves in a `leaves.bin`):
```bash
./amalia_hier_leaf --search --hier-table hier_m31.bin --target-bloom leaves.bin \
    --tree-steps 19 --root-bits 64 -t 8
```

**Integrated mode** (generates leaves in memory, sequential or random, in a loop):
```bash
./amalia_hier_leaf --attack-loop --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 \
    --batch-size 524288 --sequential --max-batches 1 -t 8
```

### Attacking with `tree_steps=95` (e.g. Puzzle #140) in a continuous batch loop

**Read the notice at the top of this README again before running this.** `tree_steps=95` (`root_bits=140`) still leaves `a_bits=140-95=45` — the *same* A-space size as Puzzle #64, so the same `hier_m31.bin`/`m-bits=31` hierarchy already built above works unchanged (no rebuild needed). What's different is `N` — the number of leaves — which grows from `2^19` (Puzzle #64) to `2^95`, and `N` sets *how many batches you need to run*, not how expensive each one is. Running the command below **works correctly** and **will not find the answer in any practical amount of time** for a real, unsolved target at this scale.

```bash
./amalia_hier_leaf --attack-loop --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 95 --root-bits 140 \
    --batch-size 524288 -t 8
```

Notes on this specific command:
- **No `--sequential`**: at `tree_steps=95` there are `2^95` possible leaves — enumerating them in order is as infeasible as the search itself. Omitting `--sequential` uses random leaf sampling (the default), drawing a fresh batch of `2^19` random candidates each round.
- **No `--max-batches`**: omitting it (or passing `0`) makes the loop run indefinitely, generating and searching a new batch as soon as the previous one is exhausted, until a match is found or you stop it (`Ctrl+C`).
- Add `--restrict-upper-half` if attacking a real puzzle-N target (see the flag's own description below) — never with synthetic/test values.
- Consider `--dump-round-stats` if you want to watch per-round cost live during a long run.

## Pruning (`--unsafe-prune`)

Excludes any `b` (or GPU lane starting point) whose bit sequence contains `N` consecutive identical bits, on the unproven theory that such runs are less likely for a real scalar.

**Proven unsafe by exhaustive scan.** `b=0` (all-zero) is excluded by this rule, yet is already generated and searched routinely with no special handling; exclusion rates range from 74.8% (`N=4`, Puzzle #64 scale) up toward 100% as the bit-width grows (see below). **No safety warnings are printed by the tool itself** — the flag name is the only signal left; the actual risk (exact exclusion rate for your parameters) must be checked with `--repeat-test` *before* trusting a real run:

```bash
./amalia_hier_leaf --repeat-test --prune-levels 19 --prune-repeat-n 4
```

**`--prune-repeat-n` must scale with the bit-width being pruned (`tree_steps`, or the bit-width of a GPU `--range-lo`/`--range-hi` search) — this is not optional.** A value that's a reasonable, modest exclusion at small bit-width becomes catastrophically aggressive at large bit-width, since a random run of `N` identical bits becomes near-certain to occur *somewhere* as the total bit count grows. Measured directly on random values:

| `--prune-repeat-n` | Exclusion rate at 19 bits | Exclusion rate at 95 bits |
|---:|---:|---:|
| 4 | ~75% | **~99.96%** (excludes nearly everything) |
| 6 | ~45% | ~79% |
| 8 | ~20% | ~30% |
| 10 | ~8% | ~8% |
| 12 | ~3% | ~2% |

At 95 bits, `--prune-repeat-n 4` leaves almost nothing to search — every block will report `0 leaves`, which is expected behavior for that parameter combination, **not a bug**. Always run `--repeat-test` at your actual `tree_steps`/range bit-width before picking a value, rather than reusing a value validated at a different scale.

```bash
# generate a batch and preview what a given rule keeps/excludes, without running a search
./amalia_hier_leaf --gen-pruned-leaves --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --batch-size 524288 \
    --unsafe-prune --prune-repeat-n 7 --preview-count 20

# apply pruning in a real attack
./amalia_hier_leaf --attack-loop --hier-table hier_m31.bin --start-pubkey <PUBKEY_HEX> \
    --tree-steps 19 --root-bits 64 --batch-size 524288 --sequential --max-batches 1 \
    --restrict-upper-half -t 8 --unsafe-prune --prune-repeat-n 7
```

## Other modes and flags

Self-test first for every one of these, exactly like the core self-tests above — see `--help` (or the source's `usage()`) for the full flag list and defaults.

- **`--radix4-test`**: experimental 5-level radix-4 alternative to the classic 2-level M/M2/M3 hierarchy. Validated correct after a real bug fix (exact-table sizing), measured **~5.1% faster** but at **~1.3GB extra memory** for zero additional gain-per-cost vs. every other optimization in this README — kept as a documented, validated experiment, not adopted into the default path.
- **`--block-search` (standalone mode, not `--attack-loop`)**: for `tree_steps` too large to enumerate as a single batch (Puzzle #140's `tree_steps=95`), divides `[0,2^tree_steps)` into fixed-size blocks and processes them one at a time (full BSGS search per block), advancing on a miss, indefinitely or until `--max-batches`. Uses `Int` (arbitrary precision) for the block offset — required correctness fix, since `tree_steps=95` exceeds `uint64_t`'s 64-bit range. Combines with `--unsafe-prune`: when active, entire blocks whose *fixed* high-bit prefix alone already satisfies the pruning rule are skipped without generating a single leaf. `--rounds-per-block N` optionally caps per-block search depth.
- **`--unsafe-prune --prune-repeat-n N`**: see the dedicated Pruning section above.
- **`--gen-pruned-leaves`**: generates a batch of leaves with whatever `--unsafe-prune` rule is active and reports the surviving set (count, percentage kept, a preview, optionally saved to a simple custom binary format) without running a search — useful for inspecting exactly what a given `--prune-repeat-n` will and won't exclude before committing to a real run.
- **`--numa-aware`**: see the pre-warming/NUMA section above.
- **`--dump-round-stats FILE`**: see the pre-warming section above.

## GPU-accelerated search (`--gpu-search`)

Requires a `make GPU=1` build (see above). Two distinct modes, sharing the same GPU-resident Bloom1/i2/i3/exact hierarchy:

### 1. Tree-based mode (same halving-tree leaves as the CPU `--attack-loop`/`--block-search` path)

```bash
./amalia_hier_leaf --gpu-search --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 \
    --batch-size 524288 --restrict-upper-half -t 2
```

`--batch-size` here is the number of tree leaves per GPU block and **must be ≤ `2^tree_steps`** (the total b-space for that `tree_steps`) — a larger value silently asks for leaves that cannot exist, and the build refuses to run rather than risk a wrong answer. `-t` controls only the small amount of CPU-side leaf generation between GPU rounds, not the GPU work itself.

**Multi-GPU (`--gpus N`)**: different GPUs process different blocks concurrently, each claiming the next available block as soon as it finishes its current one (natural load-balancing even across GPUs of different speeds — no fixed round-robin assignment):

```bash
./amalia_hier_leaf --gpu-search --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 \
    --batch-size 524288 --restrict-upper-half -t 2 --gpus 4
```

### 2. Single-target, arbitrary-range mode (`--target-pubkey`)

For searching one specific public key over an arbitrary `[range-lo, range-hi)` interval of `K` values, given directly in hex — **not** tied to the halving-tree/`tree_steps`/`root_bits` convention at all. Reports live throughput (Mk/s) on a continuously-updating line.

```bash
./amalia_hier_leaf --gpu-search --hier-table hier_m31.bin \
    --target-pubkey <PUBKEY_HEX> \
    --range-lo 1000000000000000 --range-hi 2000000000000000 \
    --lanes 2097152
```

- `--target-pubkey HEX`: the single target public key (66 hex chars, `02`/`03` prefix) — separate from `--start-pubkey`, which is only used by the tree-based modes.
- `--range-lo HEX` / `--range-hi HEX`: the `[range-lo, range-hi)` interval to search, as plain hex — any width, any starting point.
- `--lanes N`: how many parallel GPU "lanes" (candidate giant-step offsets) to advance together each round. Larger values give the GPU more concurrent work per kernel launch (generally faster up to a point, then diminishing/negative returns from launch and setup overhead) — a few hundred thousand to a few million is a reasonable starting range; benchmark on your own hardware.

**Multi-GPU (`--gpus N`)**: the total lane count (`--lanes`) is split into contiguous slices, one per GPU, each GPU running its own independent chunk loop over its slice — reported Mk/s is the sum across all active GPUs:

```bash
./amalia_hier_leaf --gpu-search --hier-table hier_m31.bin \
    --target-pubkey <PUBKEY_HEX> \
    --range-lo 1000000000000000 --range-hi 2000000000000000 \
    --lanes 8388608 --gpus 4
```

Pruning applies to this mode too, in the same way as the tree-based path — see the Pruning section above, and remember to scale `--prune-repeat-n` to the bit-width of `[range-lo, range-hi)`, not to whatever value you used for a different range size.

## Distributed computing: `--master` / `--worker`

For running the search across more than one machine. A single `--master` coordinates work; any number of `--worker` processes (same binary, or the standalone worker-only binary described below) connect to it, each pulling one b-space block at a time, running BSGS on it (CPU or GPU) with all available resources, and reporting back. The master never searches anything itself — it only hands out blocks and collects results.

**Protocol**: a deliberately simple line-based text protocol over TCP (not HTTP — no routing/headers/methods are needed here). Fully inspectable by hand with `nc HOST PORT`. Full detail in `INVESTIGATION_NOTES.md`.

- `WORK` → master replies `WORK <id> <offset_hex> <block_size> <prune> <prune_n>`, `WAIT <secs>`, or `STOP`
- `RESULT <id> NOTFOUND` / `RESULT <id> FOUND <K_hex>` → master replies `OK`
- `GETFILE hier` → master replies `FILE <size>` then streams the raw file bytes

**The found key is only ever printed by the master, never by a worker** — a worker's own log shows only `*** FOUND *** (result withheld — see master)`. Treat worker machines (especially ones run by others, across the internet) as untrusted for the *result*, even though they're fully trusted to do the *search* correctly (the master's own exact-verification step would reject a worker lying about a match anyway).

**Checkpointing** (`--master-checkpoint FILE`): the master's cursor (current b-space offset) is written to `FILE` every 20 blocks or 10 seconds, whichever comes first. On startup, if `FILE` exists, the master resumes from it instead of restarting at `0x0`.

**File serving**: if a worker's `--hier-table` path doesn't exist locally, it automatically downloads it from the master via `GETFILE` before proceeding.

```bash
# --- Puzzle #64 (tree_steps=19), CPU worker ---
# Master:
./amalia_hier_leaf --master --start-pubkey <PUBKEY_HEX> \
    --tree-steps 19 --root-bits 64 --batch-size 8192 --restrict-upper-half \
    --master-port 23941 --hier-table hier_m31.bin \
    --master-checkpoint master_64.chk
# Worker:
./amalia_hier_leaf --worker --master-host <MASTER_IP> --worker-master-port 23941 \
    --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 --restrict-upper-half -t 8

# --- Same, GPU worker (make GPU=1 build), multi-GPU on the worker side ---
./amalia_hier_leaf --worker --master-host <MASTER_IP> --worker-master-port 23941 \
    --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 --restrict-upper-half \
    --gpus 2

# --- Puzzle #140 (tree_steps=95), with pruning ---
# Master (prune parameters are decided here and sent to workers automatically):
./amalia_hier_leaf --master --start-pubkey <PUBKEY_HEX> \
    --tree-steps 95 --root-bits 140 --batch-size 524288 --restrict-upper-half \
    --unsafe-prune --prune-repeat-n 10 \
    --master-port 23941 --hier-table hier_m31.bin \
    --master-checkpoint master_140.chk
# Worker: identical command to the #64 case above, just --tree-steps 95 --root-bits 140 -
# it receives --unsafe-prune/--prune-repeat-n from the master's own WORK responses,
# not from its own command line.
./amalia_hier_leaf --worker --master-host <MASTER_IP> --worker-master-port 23941 \
    --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 95 --root-bits 140 --restrict-upper-half -t 8
```

### Standalone worker binary (`amalia_worker`) for distributing to machines across the internet

A separate, trimmed-down build (`amalia_worker.cpp`) containing *only* what `--worker` mode needs — no self-tests, no `--master`, no `--build`. Statically linked, so it runs on any Linux machine without needing matching glibc/libstdc++/libgomp versions installed — hand a worker machine this one binary and it just runs, no build step, no dependency installation.

```bash
make worker
file amalia_worker && ldd amalia_worker   # should say "not a dynamic executable"
```

**GPU-enabled worker binary** (`make worker GPU=1`): reuses the same GPU code as the main binary, statically linking everything except the NVIDIA driver library (see the static-linking note under Build, above — the same hard constraint applies here). Adds `--gpu` (use 1 GPU) and `--gpus N` (use N GPUs, each running its own independent work/result cycle against the master) to the worker's own command line:

```bash
make worker GPU=1
./amalia_worker --master-host <MASTER_IP> --master-port 23941 \
    --hier-table hier_m31.bin \
    --start-pubkey <PUBKEY_HEX> --tree-steps 19 --root-bits 64 --restrict-upper-half \
    --gpus 2
```

Usage is otherwise identical to `--worker` above, minus the `amalia_hier_leaf --worker` prefix.

**Known caveat of full static linking (CPU build)**: glibc's static NSS modules do not reliably resolve hostnames via `getaddrinfo()` in a statically-linked binary. **Always give `--master-host` as a raw IP address**, not a hostname, with this binary. See `WORKER_README.md` for a self-contained guide meant to be handed to someone running only this binary, without needing the rest of this repository's context.

## Why `a_bits=45`? The halving tradeoff, and why bigger is always better

`tree_steps` (the "halving" reduction from `root_bits` down to `a_bits`) is a **memory-management technique, not an algorithmic optimization**. For a fixed `root_bits`, the total cost of solving the whole problem via batched BSGS scales as:

```
total_cost ~ 2^(root_bits - a_bits/2)
```

Bigger `a_bits` → lower total cost, **monotonically**, all the way up to `a_bits = root_bits` (`tree_steps=0`, no tree reduction at all) — which recovers the theoretically optimal `O(√n)` BSGS/Pollard-rho complexity exactly. Every bit pushed into `tree_steps` instead loses BSGS's `√` advantage for that portion of the problem, since the b-space is enumerated linearly, one block at a time. Concretely, for Puzzle #140 (`root_bits=140`): `a_bits=20` costs `~2^130`, `a_bits=45` costs `~2^117.5`, `a_bits=100` would cost `~2^90` — each step up is a real, large win, not a wash.

**`a_bits=45` is a memory ceiling, not a mathematical one.** Bloom1 needs roughly 14.4 bits per element at this project's target false-positive rate, so `M=2^31` (this project's table size at `a_bits=45`) already costs ~3.86 GB; `a_bits=50` would need roughly 32× that (~124 GB). Going further scales the same way and quickly becomes impractical on any single machine. If more RAM ever becomes available, rebuilding the hierarchy for a larger `a_bits` is a substantially better lever than any other optimization in this README — the GPU `--target-pubkey` mode's arbitrary-range search is the natural way to exploit a larger `a_bits` once such a hierarchy exists, since it isn't tied to the halving-tree convention at all.

**A related question worth heading off**: could a smaller, cache-resident pre-filter checked *before* Bloom1 reduce Bloom1's own cost? No — Bloom1 already sits close to the information-theoretic minimum size for representing `M=2^31` elements at a useful false-positive rate; anything small enough to fit in cache would have a false-positive rate near 100%, filtering out almost nothing.

## Real bugs found and fixed during this investigation

Worth documenting, as these are generalizable for anyone reusing similar code:

1. **`Add(P,P)` instead of `Double(P)`** — adding an EC point to itself via the generic `Add()` gives a wrong result; needs `Double()`. Appeared in three independent places in the code.
2. **Memory leak in `GetBase16()`** inside a hot loop — allocates a new buffer on every call; never use inside loops running millions/billions of iterations.
3. **`ScalarMultiplication()` in the JeanLucPons/BSGS library** fails for any **even** scalar — it starts from an "identity" representation (`Clear()+z=1`) that doesn't work correctly in the subsequent additions. Affects ~50% of all possible scalars. Replaced with a safe version (`safe_scalar_mult`) that never needs to represent the identity.
4. **Lane-reload timing bug** — in a version of the engine that reloaded candidates mid-search, a candidate could be advanced one step before ever being tested at its own `cur_i=0`, silently missing exactly the degenerate cases. Fixed by eliminating mid-search reloads (each thread now loads its entire block upfront).
5. **Wall-clock instead of elapsed time** — missing capture of the start instant before measuring; the printed "search time" once showed the system's uptime instead of the actual search duration.
6. **Pruning checked the wrong bits for any batch/block after the first** — the leaf-generation pruning check tested the *local* index within a batch rather than the *global* `b` (`block_offset + local_index`), so every batch beyond the first (or any `--start-b > 0`) applied the rule to the wrong value entirely.
7. **`--attack-loop`'s sequential mode had no fully-excluded-block skip at all** — `--block-search` got this optimization, but plain `--attack-loop` advanced one `batch_size` block at a time with no jump, meaning `--unsafe-prune` at `tree_steps=95` would cycle through billions of individually-empty batches one by one, never visibly progressing. Fixed by applying the same `block_fully_excluded_jump` skip to `--attack-loop`'s own sequential cursor.
8. **`--master`'s block-dispatch loop was single-threaded and fully sequential** — a `GETFILE` transfer of a multi-GB table would block every other worker's `WORK`/`RESULT` request for the whole transfer duration. Fixed by moving to one thread per accepted connection.
9. **`run_hier_search`'s own internal `[+] FOUND` debug print computed `K` from the local, block-relative `b` only**, never adding the caller's `block_offset` — correct for the original single-full-search use case, but silently wrong whenever called from a block-based context. Confirmed directly: a real run's internal print showed a different low nibble than the correct, externally-verified, final answer — isolating the bug to that one print line, not to `A`, `b`, or the final reported result (all already correct). Fixed by suppressing this specific internal print in every block-based caller.
10. **GPU: fingerprint avalanche mix missing** — the GPU-side fingerprint function returned the raw high limb of `x` directly, while the CPU version applies a 3-step avalanche mix first; the two disagreed on every real (non-degenerate) input. Confirmed by computing the expected mixed value by hand and comparing exactly.
11. **GPU: `blockIdx.x` missing from a thread-index calculation** in the i2-loop kernel — every block computed the same small range of point indices regardless of which block it actually was, so only the first block's worth of points was ever checked, silently.
12. **GPU: round-counting off-by-one from check/shift ordering** — a multi-round kernel shifted a candidate point *before* checking it against Bloom1 on each internal iteration, meaning round `r`'s check actually tested the point for round `r+1`. Every reported round number was consistently one off from the point it actually corresponded to. Found by comparing an intermediate diagnostic dump (`i2`, `i3`, `idx`, `leaf_b` all exactly correct) against the final reconstructed key (off by exactly one unit of `M`) — isolating the bug to the round-number-to-candidate mapping specifically, not the hierarchy lookup itself. Fixed by checking the current point first, then shifting for the next round.
13. **GPU: bidirectional search's `half_giant` not divided by two** — both search fronts were each allowed to cover the *full* remaining range instead of splitting it in half and meeting in the middle, so roughly half of all GPU compute was spent redundantly re-covering territory the other front already covered. Found by noticing that a fix which eliminated 256× the kernel launches left measured throughput completely unchanged — the true bottleneck was unbatched per-thread modular inversion, not launch overhead, and was only found by building an isolated, `cudaEvent`-timed micro-benchmark to separate the two.
14. **`--batch-size` exceeding a tree-based GPU search's own b-space** — passing a `--batch-size` larger than `2^tree_steps` silently generated leaves with out-of-range `b` values, producing a plausible-looking but wrong final key (confirmed to be off by exactly one unit of `M` in the reconstructed `A`, the signature of the leaf index wrapping around). Fixed by validating `--batch-size ≤ 2^tree_steps` before starting the search and refusing to run otherwise, rather than silently producing a wrong answer.

## Credits

- Base secp256k1 library: [JeanLucPons/BSGS](https://github.com/JeanLucPons/BSGS)
- M/M2/M3 hierarchy architecture inspired by [albertobsd/keyhunt](https://github.com/albertobsd/keyhunt) (`amaliav19.cpp`)
- Original puzzle: [Bitcoin Puzzle Transaction](https://bitcointalk.org/index.php?topic=1306983.0)

## License

Inherited from the base library (GPLv3, JeanLucPons/BSGS).
