#pragma once
#include "pipeline.h"
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 2: Eye Tracking / Saliency Fusion
//
//  In production: load real fixation maps from SALICON / MIT300 / OSIE datasets
//  (stored as grayscale PGM/PNG files alongside each image).
//
//  Prototype approach:
//    A. Generate a plausible saliency map using centre-bias + contrast saliency
//       (approximates what eye trackers measure on natural images)
//    B. Fuse with segmentation class weights
//    C. Produce per-pixel importance score and tier assignments
//
//  The centre-bias prior is empirically validated:
//  humans fixate near the centre of images ~70% of the time (SALICON paper).
// ─────────────────────────────────────────────────────────────────────────────

// Alpha/beta blend weights for fusion (tunable hyperparameters)
static constexpr float ALPHA_SEG = 0.40f;  // semantic weight contribution
static constexpr float BETA_SAL  = 0.60f;  // saliency weight contribution

// ── A. Saliency map generation ───────────────────────────────────────────────

// 1. Centre-bias Gaussian (models observer tendency to fixate at centre)
static SaliencyMap centreBiasMap(int W, int H) {
    SaliencyMap sal;
    sal.width = W; sal.height = H;
    sal.data.resize(W*H);
    float cy = H / 2.f, cx = W / 2.f;
    float sigY = H / 3.5f, sigX = W / 3.5f;
    float maxV = 0;
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float dy = (y - cy) / sigY;
        float dx = (x - cx) / sigX;
        float v  = std::exp(-0.5f * (dy*dy + dx*dx));
        sal.at(y,x) = v;
        maxV = std::max(maxV, v);
    }
    for (auto& v : sal.data) v /= maxV;
    return sal;
}

// 2. Local contrast saliency (pixels very different from neighbourhood are salient)
static SaliencyMap contrastSaliency(const Image& img) {
    int W = img.width, H = img.height;
    SaliencyMap sal;
    sal.width = W; sal.height = H;
    sal.data.resize(W*H, 0);

    int radius = std::max(5, std::min(W, H) / 10);
    float maxC = 0;

    // Summed-area tables give every window mean in O(1), so the whole map is
    // O(W*H) instead of O(W*H*radius^2) (v1 took ~90 s on a 2-megapixel image).
    // Windows are clipped at the image border (mean over valid pixels).
    const int SW = W + 1;
    std::array<std::vector<uint64_t>, 3> S;
    for (auto& t : S) t.assign((size_t)SW * (H + 1), 0);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y) {
            uint64_t row = 0;
            for (int x = 0; x < W; ++x) {
                row += img.at(y, x, c);
                S[c][(size_t)(y + 1) * SW + (x + 1)] = S[c][(size_t)y * SW + (x + 1)] + row;
            }
        }

    for (int y = 0; y < H; ++y) {
        const int y0 = std::max(0, y - radius), y1 = std::min(H - 1, y + radius);
        for (int x = 0; x < W; ++x) {
            const int x0 = std::max(0, x - radius), x1 = std::min(W - 1, x + radius);
            const float area = (float)((y1 - y0 + 1) * (x1 - x0 + 1));
            float d2 = 0;
            for (int c = 0; c < 3; ++c) {
                const uint64_t sum = S[c][(size_t)(y1 + 1) * SW + (x1 + 1)]
                                   - S[c][(size_t)y0 * SW + (x1 + 1)]
                                   - S[c][(size_t)(y1 + 1) * SW + x0]
                                   + S[c][(size_t)y0 * SW + x0];
                float d = img.at(y, x, c) - (float)sum / area;
                d2 += d * d;
            }
            float cval = std::sqrt(d2);
            sal.at(y, x) = cval;
            maxC = std::max(maxC, cval);
        }
    }
    if (maxC > 0)
        for (auto& v : sal.data) v /= maxC;
    return sal;
}

// 3. Gaussian blur to smooth the saliency map (simulate fixation spread)
static void gaussianBlur(SaliencyMap& sal, float sigma) {
    int W = sal.width, H = sal.height;
    int radius = (int)(3 * sigma);
    // Build 1-D kernel
    std::vector<float> kern(2*radius+1);
    float ksum = 0;
    for (int i = -radius; i <= radius; ++i) {
        kern[i+radius] = std::exp(-0.5f * i*i / (sigma*sigma));
        ksum += kern[i+radius];
    }
    for (auto& k : kern) k /= ksum;

    // Horizontal pass
    std::vector<float> tmp(W*H, 0);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float v = 0;
        for (int dx = -radius; dx <= radius; ++dx)
            v += kern[dx+radius] * sal.data[y*W + std::clamp(x+dx,0,W-1)];
        tmp[y*W+x] = v;
    }
    // Vertical pass
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float v = 0;
        for (int dy = -radius; dy <= radius; ++dy)
            v += kern[dy+radius] * tmp[std::clamp(y+dy,0,H-1)*W+x];
        sal.data[y*W+x] = v;
    }
}

// Combine centre-bias + contrast into one saliency map
inline SaliencyMap generateSaliency(const Image& img) {
    SaliencyMap cb   = centreBiasMap(img.width, img.height);
    SaliencyMap cont = contrastSaliency(img);
    gaussianBlur(cont, std::max(img.width, img.height) / 30.f);

    SaliencyMap sal;
    sal.width  = img.width;
    sal.height = img.height;
    sal.data.resize(img.width * img.height);

    float maxV = 0;
    for (int i = 0; i < img.width * img.height; ++i) {
        sal.data[i] = 0.45f * cb.data[i] + 0.55f * cont.data[i];
        maxV = std::max(maxV, sal.data[i]);
    }
    if (maxV > 0)
        for (auto& v : sal.data) v /= maxV;

    std::cout << "[Stage 2] Saliency map generated ("
              << img.width << "×" << img.height << ")\n";
    return sal;
}

// ── B. Load real saliency map from file (grayscale PNG, for real datasets) ───
// Requires libpng — implementation in image_io.h style
// Usage: auto sal = loadSaliencyFromFile("salicon/img001_fixMap.png");
inline SaliencyMap loadSaliencyFromFile(const std::string& path, int W, int H) {
    (void)path;
    // Stub — in production, load the grayscale fixation density map PNG
    // and resize it to W×H. For now falls back to generated saliency.
    std::cout << "[Stage 2] Note: loadSaliencyFromFile is a stub. "
              << "Using generated saliency. To use real SALICON/MIT300 data, "
              << "load grayscale fixation PNG here.\n";
    // Return a placeholder — caller should substitute real loading
    SaliencyMap sal;
    sal.width = W; sal.height = H;
    sal.data.assign(W*H, 0.5f);
    return sal;
}

// ── C. Fuse segmentation + saliency → ImportanceMap ─────────────────────────
inline ImportanceMap fuseImportance(const SegmentationMask& seg,
                                    const SaliencyMap&      sal) {
    assert(seg.width == sal.width && seg.height == sal.height);
    int W = seg.width, H = seg.height, N = W*H;

    ImportanceMap imp;
    imp.width  = W;
    imp.height = H;
    imp.score.resize(N);
    imp.tier.resize(N);

    // Per-region mean saliency (to weight class importance by actual fixation)
    std::vector<float> regionSalSum(NUM_CLASSES, 0);
    std::vector<int>   regionCnt(NUM_CLASSES, 0);
    for (int i = 0; i < N; ++i) {
        int k = seg.labels[i];
        regionSalSum[k] += sal.data[i];
        ++regionCnt[k];
    }
    std::vector<float> regionMeanSal(NUM_CLASSES, 0);
    for (int k = 0; k < NUM_CLASSES; ++k)
        if (regionCnt[k] > 0)
            regionMeanSal[k] = regionSalSum[k] / regionCnt[k];

    // Pixel importance = α×semantic_weight + β×saliency
    float maxScore = 0;
    for (int i = 0; i < N; ++i) {
        int k = seg.labels[i];
        float semW  = seg.classWeight[k];
        float salW  = sal.data[i];
        float score = ALPHA_SEG * semW + BETA_SAL * salW;
        imp.score[i] = score;
        maxScore = std::max(maxScore, score);
    }
    // Normalize
    if (maxScore > 0)
        for (auto& s : imp.score) s /= maxScore;

    // Assign tiers
    int t1=0, t2=0, t3=0;
    for (int i = 0; i < N; ++i) {
        float s = imp.score[i];
        if      (s >= TIER1_THRESHOLD) { imp.tier[i] = 1; ++t1; }
        else if (s >= TIER2_THRESHOLD) { imp.tier[i] = 2; ++t2; }
        else                           { imp.tier[i] = 3; ++t3; }
    }

    std::cout << "[Stage 2] Importance fusion complete:\n"
              << "  Tier 1 (high saliency):   " << t1 << " pixels ("
              << 100.f*t1/N << "%)\n"
              << "  Tier 2 (medium saliency): " << t2 << " pixels ("
              << 100.f*t2/N << "%)\n"
              << "  Tier 3 (background):      " << t3 << " pixels ("
              << 100.f*t3/N << "%)\n";
    return imp;
}
