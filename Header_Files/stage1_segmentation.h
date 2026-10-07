#pragma once
#include "pipeline.h"
#include <cmath>
 
//  Stage 1: Semantic Segmentation
//
//  In a production system this would call SAM2 or DeepLabV3+ via ONNX Runtime.
//  Here we implement a self-contained prototype segmentation that:
//    1. Computes local variance (texture energy) per pixel using a 7×7 window
//    2. Computes edge magnitude via Sobel
//    3. K-means clusters pixels into NUM_CLASSES regions (K=5)
//    4. Assigns a semantic importance weight per class based on variance rank
//
//  The class weights feed directly into Stage 2's importance fusion.
//  Replace segment() with your ONNX/OpenCV DNN call for a real deployment. 

static constexpr int NUM_CLASSES     = 5;
static constexpr int VARIANCE_WINDOW = 7;
static constexpr int KMEANS_ITERS    = 15;

// Luminance of a pixel
static inline float luma(const Image& img, int y, int x) {
    float r = img.at(y, x, 0);
    float g = img.at(y, x, 1);
    float b = img.at(y, x, 2);
    return 0.299f*r + 0.587f*g + 0.114f*b;
}

// Local variance in a window (measure of texture complexity)
static float localVariance(const Image& img, int cy, int cx, int half) {
    float sum = 0, sum2 = 0;
    int count = 0;

    for (int dy = -half; dy <= half; ++dy)

    for (int dx = -half; dx <= half; ++dx) {
        int y = std::clamp(cy+dy, 0, img.height-1);
        int x = std::clamp(cx+dx, 0, img.width-1);
        float v = luma(img, y, x);
        sum  += v;
        sum2 += v*v;
        ++count;
    }
    float mean = sum / count;
    return sum2/count - mean*mean;
}

// Sobel edge magnitude at (y,x)
static float sobelMag(const Image& img, int y, int x) {

    auto L = [&](int dy, int dx) {
        return luma(img, std::clamp(y+dy,0,img.height-1), std::clamp(x+dx,0,img.width-1));
    };

    float gx = -L(-1,-1) + L(-1,1) - 2*L(0,-1) + 2*L(0,1) - L(1,-1) + L(1,1);
    float gy = -L(-1,-1) - 2*L(-1,0) - L(-1,1) + L(1,-1) + 2*L(1,0) + L(1,1);
    return std::sqrt(gx*gx + gy*gy);
}

//K-means on 5-feature vectors [R,G,B,variance,edge]
struct FeatureVec { float f[5]; };

static float featureDist(const FeatureVec& a, const FeatureVec& b) {
    float d = 0;

    for (int i = 0; i < 5; ++i) 
        d += (a.f[i]-b.f[i])*(a.f[i]-b.f[i]);

    return d;
}

//Main segmentation function 
inline SegmentationMask segment(const Image& img) {
    int W = img.width, H = img.height, N = W*H;

    // feature vectors per pixel
    std::vector<FeatureVec> feats(N);
    float maxVar = 1.f, maxEdge = 1.f;

    // First pass: compute raw values
    std::vector<float> vars(N), edges(N);

    for (int y = 0; y < H; ++y)

    for (int x = 0; x < W; ++x) {
        int i = y*W + x;
        vars[i]  = localVariance(img, y, x, VARIANCE_WINDOW/2);
        edges[i] = sobelMag(img, y, x);
        maxVar   = std::max(maxVar,  vars[i]);
        maxEdge  = std::max(maxEdge, edges[i]);
    }

    // Second pass: normalize and pack
    for (int y = 0; y < H; ++y)

    for (int x = 0; x < W; ++x) {
        int i = y*W + x;
        feats[i].f[0] = img.at(y,x,0) / 255.f;
        feats[i].f[1] = img.at(y,x,1) / 255.f;
        feats[i].f[2] = img.at(y,x,2) / 255.f;
        feats[i].f[3] = vars[i]  / maxVar;
        feats[i].f[4] = edges[i] / maxEdge;
    }

    // K-means (K = NUM_CLASSES)
    // Seed centroids uniformly across pixels
    std::vector<FeatureVec> centroids(NUM_CLASSES);

    for (int k = 0; k < NUM_CLASSES; ++k) {
        int idx = (k * N) / NUM_CLASSES;
        centroids[k] = feats[idx];
    }

    std::vector<uint8_t> assignment(N, 0);
    for (int iter = 0; iter < KMEANS_ITERS; ++iter) {
        // Assign step
        for (int i = 0; i < N; ++i){
            float best = 1e30f;
            for (int k = 0; k < NUM_CLASSES; ++k){
                float d = featureDist(feats[i], centroids[k]);
                if (d < best){ 
                    best = d; 
                    assignment[i] = (uint8_t)k; 
                }
            }
        }
        // Update step
        std::vector<FeatureVec> newC(NUM_CLASSES, {0,0,0,0,0});
        std::vector<int> cnt(NUM_CLASSES, 0);
        for (int i = 0; i < N; ++i) {
            int k = assignment[i];

            for (int f = 0; f < 5; ++f) 
                newC[k].f[f] += feats[i].f[f];

            ++cnt[k];
        }
        for (int k = 0; k < NUM_CLASSES; ++k){
            if (cnt[k] > 0)
                for (int f = 0; f < 5; ++f)
                    centroids[k].f[f] = newC[k].f[f] / cnt[k];
        }
    }

    // Rank classes by texture complexity (variance + edge) -> importance weight
    //    High texture/edge -> likely foreground/object -> higher weight
    std::vector<float> classComplexity(NUM_CLASSES, 0);

    for (int k = 0; k < NUM_CLASSES; ++k)
        classComplexity[k] = 0.6f * centroids[k].f[3] + 0.4f * centroids[k].f[4]; 

    // Normalize weights to [0.1, 1.0]
    float cMin = *std::min_element(classComplexity.begin(), classComplexity.end());
    float cMax = *std::max_element(classComplexity.begin(), classComplexity.end());
    float cRange = std::max(cMax - cMin, 1e-6f);

    SegmentationMask mask;
    mask.width  = W;
    mask.height = H;
    mask.labels.resize(N);
    mask.classWeight.resize(NUM_CLASSES);

    for (int k = 0; k < NUM_CLASSES; ++k)
        mask.classWeight[k] = 0.1f + 0.9f * (classComplexity[k] - cMin) / cRange;

    for (int i = 0; i < N; ++i)
        mask.labels[i] = assignment[i];

    std::cout << "[Stage 1] Segmentation complete — " << NUM_CLASSES << " classes, weights: ";

    for (int k = 0; k < NUM_CLASSES; ++k)
        std::cout << std::fixed << std::setprecision(2) << mask.classWeight[k] << " ";

    std::cout << "\n";
    return mask;
}

// Save segmentation as colour-coded PNG
#include "image_io.h"
static const std::array<std::array<uint8_t,3>, NUM_CLASSES> CLASS_COLORS = {{
    {230, 25,  75},   // class 0 – red
    {60,  180, 75},   // class 1 – green
    {67,  99,  216},  // class 2 – blue
    {255, 225, 25},   // class 3 – yellow
    {145, 30,  180},  // class 4 – purple
}};

inline void saveSegPNG(const std::string& path, const SegmentationMask& mask) {
    Image img;
    img.width = mask.width; 
    img.height = mask.height; 
    img.channels = 3;
    img.data.resize(mask.width * mask.height * 3);
    for (int i = 0; i < mask.width * mask.height; ++i) {
        int k = mask.labels[i] % NUM_CLASSES;
        img.data[i*3+0] = CLASS_COLORS[k][0];
        img.data[i*3+1] = CLASS_COLORS[k][1];
        img.data[i*3+2] = CLASS_COLORS[k][2];
    }
    savePNG(path, img);
}
