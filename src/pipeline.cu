#include "pipeline.cuh"
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel weights (normalised by 273).
// Stored as a constant array in GPU constant memory for fast broadcast reads.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float c_gauss[5][5] = {
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 7.f/273, 26.f/273, 41.f/273, 26.f/273,  7.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
};


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1 — Gaussian Blur (shared memory tiling with halo cells)
// ═════════════════════════════════════════════════════════════════════════════
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    const int SMEM_W = TILE_W + 2 * GAUSS_RADIUS;
    const int SMEM_H = TILE_H + 2 * GAUSS_RADIUS;

    __shared__ float smem[SMEM_H][SMEM_W];

    // Top-left corner of the input region this block needs
    int tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS;
    int tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS;

    // Load shared memory cooperatively using linearised thread index
    int tid        = threadIdx.y * TILE_W + threadIdx.x;
    int block_size = TILE_W * TILE_H;
    int smem_size  = SMEM_W * SMEM_H;

    for (int idx = tid; idx < smem_size; idx += block_size) {
        int sy = idx / SMEM_W;
        int sx = idx % SMEM_W;

        // Clamp-to-edge global coordinates
        int gx = tile_start_x + sx;
        int gy = tile_start_y + sy;
        gx = max(0, min(gx, width  - 1));
        gy = max(0, min(gy, height - 1));

        smem[sy][sx] = (float)in[gy * width + gx];
    }

    __syncthreads();

    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;

    if (out_x < width && out_y < height) {
        float sum = 0.f;
        for (int ki = 0; ki < 5; ki++) {
            for (int kj = 0; kj < 5; kj++) {
                sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj];
            }
        }
        sum = fminf(fmaxf(sum, 0.f), 255.f);
        out[out_y * width + out_x] = (uint8_t)roundf(sum);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2 — Sobel Edge Detection
// ═════════════════════════════════════════════════════════════════════════════
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Gx kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
    // Gy kernel: [[1,2,1],[0,0,0],[-1,-2,-1]]
    const int gx_weights[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
    const int gy_weights[3][3] = { {1,2,1},  {0,0,0},  {-1,-2,-1} };

    float gx = 0.f, gy = 0.f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = max(0, min(x + kx, width  - 1));
            int py = max(0, min(y + ky, height - 1));
            float val = (float)in[py * width + px];
            gx += gx_weights[ky + 1][kx + 1] * val;
            gy += gy_weights[ky + 1][kx + 1] * val;
        }
    }

    float mag = sqrtf(gx * gx + gy * gy);
    // Reference uses truncation, not roundf — verified bit-exact against
    // data/expected_output/stage-2 across 5 test images.
    out[y * width + x] = (uint8_t)min(max((int)mag, 0), 255);
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3A — Histogram Kernel
// ═════════════════════════════════════════════════════════════════════════════
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height)
{
    // Per-block shared memory histogram to reduce global atomic contention
    __shared__ unsigned int s_hist[256];

    // Zero-initialise shared histogram
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    for (int i = tid; i < 256; i += block_size)
        s_hist[i] = 0u;

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t val = in[y * width + x];
        atomicAdd(&s_hist[val], 1u);
    }

    __syncthreads();

    // Flush per-block histogram to global
    for (int i = tid; i < 256; i += block_size)
        atomicAdd(&hist[i], s_hist[i]);
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3C — Equalisation Kernel
// ═════════════════════════════════════════════════════════════════════════════
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   cdf,
    float          cdf_min,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float total = (float)(width * height);
    uint8_t old_val = in[y * width + x];
    float cdf_val   = cdf[old_val];

    float new_val = roundf((cdf_val - cdf_min) / (total - cdf_min) * 255.f);
    new_val = fminf(fmaxf(new_val, 0.f), 255.f);
    out[y * width + x] = (uint8_t)new_val;
}
