#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <array>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <stdexcept>

//  Constants
static constexpr int BIT_DEPTH    = 8;
static constexpr int NUM_PLANES   = 8;
static constexpr int NUM_TIERS    = 3;
static constexpr int NUM_CHANNELS = 3;   // coded channels: G, R-G, B-G

// Tier importance thresholds (tunable)
static constexpr float TIER1_THRESHOLD = 0.65f; // high saliency
static constexpr float TIER2_THRESHOLD = 0.30f; // medium saliency

// Sanity limits used when reading untrusted files
static constexpr uint32_t MAX_DIM    = 65535;
static constexpr uint64_t MAX_PIXELS = 1ull << 28;

//  Core data structures

// A single RGB image loaded from disk
struct Image {
    int width = 0, height = 0, channels = 3;
    std::vector<uint8_t> data;  // row-major, interleaved RGB

    uint8_t& at(int y, int x, int c) {
        return data[(y * width + x) * channels + c];
    }

    const uint8_t& at(int y, int x, int c) const {
        return data[(y * width + x) * channels + c];
    }

};

// One saliency map  (H×W floats in [0,1])
struct SaliencyMap {
    int width, height;
    std::vector<float> data;

    float& at(int y, int x){ 
        return data[y * width + x]; 
    }

    const float& at(int y, int x) const { 
        return data[y * width + x]; 
    }

};

// Semantic segmentation mask (H×W uint8 class labels 0..255)
struct SegmentationMask {
    int width, height;
    std::vector<uint8_t> labels;     // class per pixel
    std::vector<float>   classWeight;// importance weight per class

    uint8_t& at(int y, int x){ 
        return labels[y * width + x]; 
    }

    const uint8_t& at(int y, int x) const{ 
        return labels[y * width + x]; 
    }

};

// Combined per-pixel importance [0,1]
struct ImportanceMap {
    int width, height;
    std::vector<float> score;       // fused importance
    std::vector<uint8_t> tier;      // 1, 2, or 3

    float& scoreAt(int y, int x){ 
        return score[y * width + x]; 
    }

    uint8_t& tierAt(int y, int x){ 
        return tier[y * width + x]; 
    }

    const uint8_t& tierAt(int y, int x) const{ 
        return tier[y * width + x]; 
    }
    
};

// Prediction residuals, one byte per pixel per coded channel.
// Residuals are zig-zag mapped so small errors have small magnitude and
// therefore leave the high bit planes empty 

struct ResidualPlanes {
    int width = 0, height = 0;
    std::array<std::vector<uint8_t>, NUM_CHANNELS> r;
};

// One slice = (bit plane, channel, tier).  Slices are coded in this order:
// plane-major (MSB first), then channel, then tier (1 = most important).
struct SliceId {
    int plane;    // 1..8  (1 = most significant)
    int channel;  // 0 = G, 1 = R-G, 2 = B-G
    int tier;     // 1..3
};

// Result of entropy coding
struct EncodedStream {
    std::vector<uint8_t> tierMapBytes;   // compressed tier map
    std::vector<uint8_t> payload;        // compressed residual bit planes
    // ideal (model) cost in bits spent on each tier, for reporting
    std::array<double, NUM_TIERS + 1> tierBits{};
    std::array<uint64_t, NUM_TIERS + 1> tierPixels{};
};
