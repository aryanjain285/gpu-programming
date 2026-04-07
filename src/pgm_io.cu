#include "pgm_io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

bool pgm_load(const std::string& path, uint8_t** data, int* width, int* height) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[pgm_load] Cannot open file: %s\n", path.c_str());
        return false;
    }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "[pgm_load] Not a binary PGM (P5) file: %s\n", path.c_str());
        fclose(f);
        return false;
    }

    // Skip comments
    int c = fgetc(f);
    while (c == '#') {
        while (fgetc(f) != '\n');
        c = fgetc(f);
    }
    ungetc(c, f);

    int maxval;
    if (fscanf(f, "%d %d %d", width, height, &maxval) != 3) {
        fprintf(stderr, "[pgm_load] Failed to read PGM header: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fgetc(f); // consume the single whitespace after maxval

    size_t n = (size_t)(*width) * (*height);
    *data = new uint8_t[n];
    if (fread(*data, 1, n, f) != n) {
        fprintf(stderr, "[pgm_load] Failed to read pixel data: %s\n", path.c_str());
        delete[] *data;
        *data = nullptr;
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

bool pgm_save(const std::string& path, const uint8_t* data, int width, int height) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[pgm_save] Cannot open file for writing: %s\n", path.c_str());
        return false;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    size_t n = (size_t)width * height;
    if (fwrite(data, 1, n, f) != n) {
        fprintf(stderr, "[pgm_save] Failed to write pixel data: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}
