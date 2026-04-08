# ─────────────────────────────────────────────────────────────────────────────
# Assignment 3 — GPU Image Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := -std=c++17

# Set compute capability for your GPU.
#   Run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` to find it.
#   ASPIRE 2A (A100): sm_80  |  GTX/RTX alternatives listed below.
#   sm_70: V100, sm_75: RTX 20xx/T4, sm_80: A100, sm_86: RTX 30xx, sm_89: RTX 40xx
ARCH        := -gencode arch=compute_80,code=sm_80

# Optimisation and debug flags.
# For profiling: keep -lineinfo so Nsight can map back to source lines.
OPT_FLAGS   := -O2 -lineinfo

# Warning flags
WARN_FLAGS  := -Xcompiler -Wall,-Wextra

NVCC_FLAGS  := $(CXX_STD) $(ARCH) $(OPT_FLAGS) $(WARN_FLAGS)

# Source files
SRCS        := src/main.cu \
               src/pipeline.cu \
               src/multigpu.cu \
               src/pgm_io.cu

TARGET      := pipeline

INPUT_DIR   := data/input
OUTPUT_DIR  := output

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build run clean check

build: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread -lstdc++fs

# Run the pipeline on all test images.
run: build
	@mkdir -p $(OUTPUT_DIR)
	./$(TARGET) --input $(INPUT_DIR) --output $(OUTPUT_DIR)

# Run in single-GPU mode (for timing comparison).
run-single: build
	@mkdir -p $(OUTPUT_DIR)
	./$(TARGET) --input $(INPUT_DIR) --output $(OUTPUT_DIR) --single-gpu

# Profile with Nsight Systems.
profile: build
	@mkdir -p $(OUTPUT_DIR)
	nsys profile --trace=cuda,nvtx --output=pipeline_profile \
	    ./$(TARGET) --input $(INPUT_DIR) --output $(OUTPUT_DIR)
	@echo "Profile saved to pipeline_profile.nsys-rep"
	@echo "Open with: nsys-ui pipeline_profile.nsys-rep"

# Diff outputs against reference images.
# Allows ±2 intensity levels per pixel (accounts for floating-point rounding).
check: run
	@echo "Checking outputs against reference..."
	@python3 scripts/check_outputs.py \
	    --output $(OUTPUT_DIR) \
	    --reference data/expected_output/stage-3 \
	    --tolerance 2
	@echo "Check complete."

clean:
	rm -f $(TARGET) *.nsys-rep *.sqlite *.qdrep
	rm -rf $(OUTPUT_DIR)
