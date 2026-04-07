#pragma once
#include <cstdint>
#include <string>

// Loads a binary PGM (P5) file into a host buffer.
// Caller is responsible for freeing *data with delete[].
// Returns true on success, false on error.
bool pgm_load(const std::string& path, uint8_t** data, int* width, int* height);

// Saves a greyscale pixel buffer as a binary PGM (P5) file.
// Returns true on success, false on error.
bool pgm_save(const std::string& path, const uint8_t* data, int width, int height);
