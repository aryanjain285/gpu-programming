#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include "pgm_io.cuh"
#include "pipeline.cuh"
#include "multigpu.cuh"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Usage:
//   ./pipeline --input <input_dir> --output <output_dir>
//
// Processes all .pgm files found in <input_dir> through the full pipeline
// (Gaussian blur → Sobel → histogram equalisation) and writes results to
// <output_dir>.
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --input <dir> --output <dir> [--single-gpu]\n", prog);
    fprintf(stderr, "  --input      directory containing input .pgm files\n");
    fprintf(stderr, "  --output     directory to write output .pgm files\n");
    fprintf(stderr, "  --single-gpu force single-GPU mode (default: multi-GPU)\n");
}

int main(int argc, char** argv)
{
    std::string input_dir  = "data/input";
    std::string output_dir = "output";
    bool        single_gpu = false;

    // ── Parse arguments ───────────────────────────────────────────────────
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input")      == 0 && i+1 < argc) input_dir  = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "--single-gpu") == 0)           single_gpu = true;
        else { print_usage(argv[0]); return 1; }
    }

    // ── Create output directory ───────────────────────────────────────────
    fs::create_directories(output_dir);

    // ── Collect input files ───────────────────────────────────────────────
    std::vector<std::string> input_files;
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".pgm")
            input_files.push_back(entry.path().string());
    }
    std::sort(input_files.begin(), input_files.end());

    if (input_files.empty()) {
        fprintf(stderr, "[main] No .pgm files found in %s\n", input_dir.c_str());
        return 1;
    }
    printf("[main] Found %zu image(s) in %s\n", input_files.size(), input_dir.c_str());

    // ── Load all images and allocate pinned host buffers ──────────────────
    // We use cudaMallocHost (pinned / page-locked memory) so that
    // cudaMemcpyAsync can truly run asynchronously with kernel execution.
    // Using malloc here would silently cause transfers to be synchronous.

    std::vector<ImageEntry> batch;
    batch.reserve(input_files.size());

    for (auto& path : input_files) {
        ImageEntry e;
        e.input_path = path;

        // Derive output filename: same stem with "_out" suffix.
        fs::path p(path);
        e.output_path = output_dir + "/" + p.stem().string() + "_out.pgm";

        // Load input into a temporary heap buffer, then copy to pinned memory.
        uint8_t* tmp = nullptr;
        if (!pgm_load(path, &tmp, &e.width, &e.height)) {
            fprintf(stderr, "[main] Failed to load %s — skipping.\n", path.c_str());
            continue;
        }

        size_t n = (size_t)e.width * e.height;

        // Allocate pinned host buffers.
        cudaError_t err;
        err = cudaMallocHost(&e.host_in,  n);
        if (err != cudaSuccess) {
            fprintf(stderr, "[main] cudaMallocHost failed for %s: %s\n",
                    path.c_str(), cudaGetErrorString(err));
            delete[] tmp; continue;
        }
        err = cudaMallocHost(&e.host_out, n);
        if (err != cudaSuccess) {
            fprintf(stderr, "[main] cudaMallocHost failed for %s: %s\n",
                    path.c_str(), cudaGetErrorString(err));
            cudaFreeHost(e.host_in); delete[] tmp; continue;
        }

        memcpy(e.host_in, tmp, n);
        memset(e.host_out, 0, n);
        delete[] tmp;

        batch.push_back(e);
    }

    if (batch.empty()) {
        fprintf(stderr, "[main] No images loaded successfully. Exiting.\n");
        return 1;
    }

    printf("[main] Loaded %zu image(s) (%dx%d each).\n",
           batch.size(), batch[0].width, batch[0].height);

    // ── Run the pipeline ──────────────────────────────────────────────────
    printf("[main] Running pipeline (%s mode)...\n",
           single_gpu ? "single-GPU" : "multi-GPU");

    if (single_gpu) {
        run_pipeline_singlegpu(batch);
    } else {
        run_pipeline_multigpu(batch);
    }

    // ── Clean up pinned host memory ───────────────────────────────────────
    for (auto& e : batch) {
        cudaFreeHost(e.host_in);
        cudaFreeHost(e.host_out);
    }

    printf("[main] Output written to %s\n", output_dir.c_str());
    return 0;
}
