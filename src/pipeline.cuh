#pragma once
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Tile dimensions for shared-memory kernels.
// Change these to experiment with different occupancy / halo-overhead trade-offs.
// ─────────────────────────────────────────────────────────────────────────────
#define TILE_W 16
#define TILE_H 16

// Gaussian kernel radius (do not change — matches the 5x5 kernel in the spec).
#define GAUSS_RADIUS 2

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — Gaussian Blur
// Applies a 5x5 Gaussian blur to `in` and writes the result to `out`.
// Uses shared memory tiling with halo cells.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height);

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — Sobel Edge Detection
// Applies the Sobel operator to `in` and writes gradient magnitude to `out`.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height);

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3 — Histogram Equalisation
// Step A: accumulate per-intensity pixel counts into hist[256].
// hist must be zero-initialised with cudaMemset before this kernel is called.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height);

// Step C: remap pixel values using the pre-computed CDF.
// cdf[256] is the inclusive prefix sum of the histogram (float).
// cdf_min is the smallest non-zero CDF value.
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   cdf,
    float          cdf_min,
    int width, int height);
