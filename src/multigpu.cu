#include "multigpu.cuh"
#include "pipeline.cuh"
#include "pgm_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

// ─────────────────────────────────────────────────────────────────────────────
// process_batch_on_device
// ─────────────────────────────────────────────────────────────────────────────
static void process_batch_on_device(std::vector<ImageEntry>& sub_batch, int device_id)
{
    cudaSetDevice(device_id);

    int W = sub_batch[0].width;
    int H = sub_batch[0].height;
    size_t img_bytes  = (size_t)W * H * sizeof(uint8_t);
    size_t hist_bytes = 256 * sizeof(unsigned int);
    size_t cdf_bytes  = 256 * sizeof(float);
    int    n_images   = (int)sub_batch.size();

    // ── Per-image device buffers ──────────────────────────────────────────
    std::vector<uint8_t*>      d_in(n_images);
    std::vector<uint8_t*>      d_blur(n_images);
    std::vector<uint8_t*>      d_edges(n_images);
    std::vector<uint8_t*>      d_out(n_images);
    std::vector<unsigned int*> d_hist(n_images);
    std::vector<float*>        d_cdf(n_images);

    for (int i = 0; i < n_images; i++) {
        cudaMalloc(&d_in[i],   img_bytes);
        cudaMalloc(&d_blur[i], img_bytes);
        cudaMalloc(&d_edges[i],img_bytes);
        cudaMalloc(&d_out[i],  img_bytes);
        cudaMalloc(&d_hist[i], hist_bytes);
        cudaMalloc(&d_cdf[i],  cdf_bytes);
    }

    // ── Per-image CUDA streams ────────────────────────────────────────────
    std::vector<cudaStream_t> streams(n_images);
    for (int i = 0; i < n_images; i++)
        cudaStreamCreate(&streams[i]);

    // ── Kernel launch config ──────────────────────────────────────────────
    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    // ── Submit all images to the GPU ──────────────────────────────────────
    for (int i = 0; i < n_images; i++) {

        // H→D transfer
        cudaMemcpyAsync(d_in[i], sub_batch[i].host_in, img_bytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // Stage 1: Gaussian blur
        gaussianBlurKernel<<<grid, block, 0, streams[i]>>>(
            d_in[i], d_blur[i], W, H);

        // Stage 2: Sobel edge detection
        sobelKernel<<<grid, block, 0, streams[i]>>>(
            d_blur[i], d_edges[i], W, H);

        // Stage 3A: Histogram — zero-init then launch
        cudaMemsetAsync(d_hist[i], 0, hist_bytes, streams[i]);
        histogramKernel<<<grid, block, 0, streams[i]>>>(
            d_edges[i], d_hist[i], W, H);

        // Stage 3B: CDF via thrust (thrust uses default/internal stream;
        //   we must sync this image's stream before calling thrust so the
        //   histogram is fully computed).
        cudaStreamSynchronize(streams[i]);

        thrust::device_ptr<unsigned int> hist_ptr(d_hist[i]);
        thrust::device_ptr<float>        cdf_ptr(d_cdf[i]);
        thrust::exclusive_scan(hist_ptr, hist_ptr + 256, cdf_ptr);

        // Find cdf_min on host
        float h_cdf[256];
        cudaMemcpy(h_cdf, d_cdf[i], cdf_bytes, cudaMemcpyDeviceToHost);
        float cdf_min = 0.f;
        for (int b = 0; b < 256; b++) {
            if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }
        }

        // Stage 3C: Equalize
        equalizeKernel<<<grid, block, 0, streams[i]>>>(
            d_edges[i], d_out[i], d_cdf[i], cdf_min, W, H);

        // D→H transfer
        cudaMemcpyAsync(sub_batch[i].host_out, d_out[i], img_bytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // ── Wait for all images to finish ─────────────────────────────────────
    for (int i = 0; i < n_images; i++)
        cudaStreamSynchronize(streams[i]);

    // ── Save results ──────────────────────────────────────────────────────
    for (int i = 0; i < n_images; i++)
        pgm_save(sub_batch[i].output_path, sub_batch[i].host_out, W, H);

    // ── Clean up ──────────────────────────────────────────────────────────
    for (int i = 0; i < n_images; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_in[i]);
        cudaFree(d_blur[i]);
        cudaFree(d_edges[i]);
        cudaFree(d_out[i]);
        cudaFree(d_hist[i]);
        cudaFree(d_cdf[i]);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// run_pipeline_multigpu  (public entry point)
// ─────────────────────────────────────────────────────────────────────────────
void run_pipeline_multigpu(std::vector<ImageEntry>& batch)
{
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 2) {
        fprintf(stderr,
                "[multigpu] Warning: fewer than 2 GPUs detected (%d). "
                "Falling back to GPU 0.\n", num_gpus);
        process_batch_on_device(batch, 0);
        return;
    }

    // Split batch: GPU 0 gets first half, GPU 1 gets remainder
    int split = (int)batch.size() / 2;
    std::vector<ImageEntry> sub0(batch.begin(), batch.begin() + split);
    std::vector<ImageEntry> sub1(batch.begin() + split, batch.end());

    // Run both GPUs in parallel using host threads so we get ~2x speedup.
    std::thread t0([&sub0]() { process_batch_on_device(sub0, 0); });
    std::thread t1([&sub1]() { process_batch_on_device(sub1, 1); });
    t0.join();
    t1.join();
}

void run_pipeline_singlegpu(std::vector<ImageEntry>& batch)
{
    process_batch_on_device(batch, 0);
}
