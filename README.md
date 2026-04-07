# Assignment 3 — GPU-Accelerated Image Processing Pipeline

## Quick Start

```bash
# 1. Set your GPU compute capability in the Makefile (ARCH flag).
#    Find it with: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
nano Makefile

# 2. Build
make build

# 3. Run on all 20 test images
make run

# 4. Check your outputs against the reference
make check

# 5. Profile with Nsight Systems (required for the report)
make profile
```

## Directory Structure

```
assignment3/
  Makefile                  — Build system
  src/
    main.cu                 — Entry point (batch loading, timing)
    pipeline.cu             — TODO: implement all three pipeline kernels
    pipeline.cuh            — Kernel declarations and TILE_W/TILE_H constants
    multigpu.cu             — TODO: multi-GPU batch distribution
    multigpu.cuh            — Multi-GPU interface
    pgm_io.cu               — PGM loader/saver (do not modify)
    pgm_io.cuh
  data/
    input/                  — 20 synthetic test images (512x512 PGM)
    expected_output/        — Reference outputs for all three pipeline stages
  scripts/
    check_outputs.py        — Automated correctness checker
  output/                   — Created by `make run` or `make run-single`; your pipeline outputs go here
```

## What You Need to Implement

All TODOs are marked with `// TODO [A3]:` in the source files.

| File          | What to implement                                              |
|---------------|----------------------------------------------------------------|
| pipeline.cu   | gaussianBlurKernel (shared memory tiling + halo cells)         |
| pipeline.cu   | sobelKernel (Gx + Gy in a single kernel)                       |
| pipeline.cu   | histogramKernel (atomic operations)                            |
| pipeline.cu   | equalizeKernel (CDF remapping)                                 |
| multigpu.cu   | process_batch_on_device (streams, async transfers, thrust CDF) |
| multigpu.cu   | run_pipeline_multigpu (cudaGetDeviceCount, cudaSetDevice, split)|

## Testing Incrementally

Test each stage in isolation before connecting them:

```bash
# Compare Stage 1 output against reference blurred images
python3 scripts/check_outputs.py --output output --reference data/expected_output/stage-1/
```

The reference directory contains three files per image:
- `stage-1/<name>_blurred.pgm`   — after Stage 1 (Gaussian blur)
- `stage-2/<name>_edges.pgm`     — after Stage 2 (Sobel)
- `stage-3/<name>_equalized.pgm` — after Stage 3 (histogram equalisation)

Your `make check` target compares your final output (`_out.pgm`) against
`stage-3/<name>_equalized.pgm`. For per-stage debugging, call `check_outputs.py` manually
with the appropriate reference directory

## Profiling

```bash
make profile
```

## Useful Commands

```bash
# Check GPU model and compute capability
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Check number of GPUs
nvidia-smi --list-gpus
```
