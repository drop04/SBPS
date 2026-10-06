#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  STAGE 3 — Decorrelation + 2D Bit-Plane Slicing
//
//  v1 sliced the raw pixel bytes.  Raw bit planes of a natural image are
//  nearly incompressible below the top 2-3 planes, so v1 expanded the data.
//  v2 first removes the redundancy between neighbouring pixels and between
//  channels, and only then slices:
//
//    1. Reversible colour transform:   G,  R-G (mod 256),  B-G (mod 256)
//    2. MED (LOCO-I / JPEG-LS) predictor per channel, causal neighbours only
//    3. Residual  e = (x - pred) mod 256, reinterpreted as signed, then
//       zig-zag mapped so |e| small  ->  value small  ->  high planes all 0
//    4. The residual bytes are what get cut into bit planes (MSB first).
//
//  The slice grid is still  tier x bit-plane x channel  (72 slices).  The
//  slices are scheduled plane-major so every pixel's more significant planes
//  are known (to encoder AND decoder) before its less significant planes are
//  coded — that is what the entropy coder's contexts rely on.
// ═══════════════════════════════════════════════════════════════════════════
#include "pipeline.h"
#include "image_io.h"

// ─── Integrity checksum (CRC-32) of the original RGB data ──────────────────
inline uint32_t crc32(const std::vector<uint8_t>& d) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (uint8_t b : d) c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ─── MED predictor (LOCO-I / JPEG-LS) ──────────────────────────────────────
// Uses only already-coded neighbours: `p` is a W-wide plane in which every
// pixel before (x,y) in raster order is valid.  Used identically by encoder
// and decoder.
//
// (Experiment log: an adaptive blend of 5 predictors weighted by local error
//  was tried and rejected — at best 0.7% smaller on a noisy photo-like image
//  and up to 2x larger on gradients and hard edges.)
static inline int medPredict(const uint8_t* p, int W, int x, int y) {
    int a, b, c;
    if (y == 0) {
        a = x > 0 ? p[x - 1] : 0;  b = a;  c = a;
    } else if (x == 0) {
        b = p[(y - 1) * W];  a = b;  c = b;
    } else {
        a = p[y * W + x - 1];
        b = p[(y - 1) * W + x];
        c = p[(y - 1) * W + x - 1];
    }
    int mx = std::max(a, b), mn = std::min(a, b);
    if (c >= mx) return mn;
    if (c <= mn) return mx;
    return a + b - c;
}

static inline uint8_t zigzag(int e /* signed, -128..127 */) {
    return (uint8_t)(((e << 1) ^ (e >> 7)) & 0xFF);
}
static inline int unzigzag(uint8_t z) {
    return (int)(z >> 1) ^ -(int)(z & 1);
}

// ─── Image -> residual planes ──────────────────────────────────────────────
inline ResidualPlanes decorrelate(const Image& img) {
    const int W = img.width, H = img.height, N = W * H;
    // Reversible colour transform into three planes: [G, R-G, B-G]
    std::array<std::vector<uint8_t>, NUM_CHANNELS> t;
    for (auto& v : t) v.resize(N);
    for (int i = 0; i < N; ++i) {
        uint8_t R = img.data[i * 3], G = img.data[i * 3 + 1], B = img.data[i * 3 + 2];
        t[0][i] = G;
        t[1][i] = (uint8_t)(R - G);
        t[2][i] = (uint8_t)(B - G);
    }
    ResidualPlanes rp;
    rp.width = W; rp.height = H;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        rp.r[c].resize(N);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int pred = medPredict(t[c].data(), W, x, y);
                int e = (int)(int8_t)(uint8_t)(t[c][y * W + x] - pred);
                rp.r[c][y * W + x] = zigzag(e);
            }
    }
    return rp;
}

// ─── Residual planes -> image (exact inverse) ──────────────────────────────
inline Image reconstruct(const ResidualPlanes& rp) {
    const int W = rp.width, H = rp.height, N = W * H;
    std::array<std::vector<uint8_t>, NUM_CHANNELS> t;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        t[c].assign(N, 0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int pred = medPredict(t[c].data(), W, x, y);
                t[c][y * W + x] = (uint8_t)(pred + unzigzag(rp.r[c][y * W + x]));
            }
    }
    Image img;
    img.width = W; img.height = H; img.channels = 3;
    img.data.resize((size_t)N * 3);
    for (int i = 0; i < N; ++i) {
        uint8_t G = t[0][i];
        img.data[i * 3]     = (uint8_t)(t[1][i] + G);
        img.data[i * 3 + 1] = G;
        img.data[i * 3 + 2] = (uint8_t)(t[2][i] + G);
    }
    return img;
}

// ─── Slice grid ────────────────────────────────────────────────────────────
// Plane-major order (MSB first), then channel, then tier.
inline std::vector<SliceId> sliceSchedule() {
    std::vector<SliceId> s;
    for (int p = 1; p <= NUM_PLANES; ++p)
        for (int c = 0; c < NUM_CHANNELS; ++c)
            for (int t = 1; t <= NUM_TIERS; ++t)
                s.push_back({p, c, t});
    return s;
}

// Raster-order pixel indices for each tier (index 1..3; 0 unused).
inline std::array<std::vector<int>, NUM_TIERS + 1>
tierPixelLists(const std::vector<uint8_t>& tier) {
    std::array<std::vector<int>, NUM_TIERS + 1> L;
    for (int i = 0; i < (int)tier.size(); ++i) {
        int t = tier[i];
        if (t < 1 || t > NUM_TIERS) throw std::runtime_error("invalid tier value");
        L[t].push_back(i);
    }
    return L;
}

inline bool verifyLossless(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height && a.data == b.data;
}

// ─── Debug visualisation: amplified |residual| of the G channel ────────────
inline void saveResidualPNG(const std::string& path, const ResidualPlanes& rp) {
    Image v;
    v.width = rp.width; v.height = rp.height; v.channels = 3;
    v.data.resize((size_t)rp.width * rp.height * 3);
    for (int i = 0; i < rp.width * rp.height; ++i) {
        int m = std::min(255, (int)rp.r[0][i] * 16);
        v.data[i * 3] = v.data[i * 3 + 1] = v.data[i * 3 + 2] = (uint8_t)m;
    }
    savePNG(path, v);
}
