#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Describes one image in the batch.
struct ImageEntry {
    std::string  input_path;   // path to input PGM
    std::string  output_path;  // path to write final output PGM
    uint8_t*     host_in;      // pinned host input buffer  (allocated in main)
    uint8_t*     host_out;     // pinned host output buffer (allocated in main)
    int          width;
    int          height;
};

// Process a batch of images distributed across available GPUs.
// Each image is run through the full pipeline (blur → Sobel → equalise).
// Results are written to entry.output_path for each entry.
void run_pipeline_multigpu(std::vector<ImageEntry>& batch);

void run_pipeline_singlegpu(std::vector<ImageEntry>& batch);
