//  STAGE 4 — Context-modelled binary range coding


#pragma once
#include "pipeline.h"
#include "stage3_slicing.h"

// Tunables
#ifndef ADAPT_LIMIT_VAL
#define ADAPT_LIMIT_VAL 90
#endif

static constexpr int  ADAPT_LIMIT = ADAPT_LIMIT_VAL;   // higher = slower, more precise adaptation
static constexpr int  NUM_CTX     = 512;  // contexts per slice model (see ctx below)

// Adaptive probability for one binary context 
struct BitModel {
    uint16_t p = 32768;   // P(bit == 1), 16-bit fixed point
    uint8_t  n = 0;       // observations so far (capped) -> adaptation rate

    static const int* rateTable() {
        static int t[ADAPT_LIMIT + 1];
        static bool init = false;

        if (!init) {
            for (int i = 0; i <= ADAPT_LIMIT; ++i) 
                t[i] = (int)(65536.0 / (i + 1.6));

            init = true;
        }

        return t;
    }

    void update(int bit) {
        int64_t target = bit ? 65535 : 0;
        int64_t pi = p;
        pi += ((target - pi) * rateTable()[n]) >> 16;

        if (pi < 24) 
            pi = 24;

        if (pi > 65535 - 24) 
            pi = 65535 - 24;

        p = (uint16_t)pi;

        if (n < ADAPT_LIMIT) 
            ++n;
    }
};

// Binary range coder 

class RangeEncoder {

    uint32_t x1 = 0, x2 = 0xFFFFFFFFu;

    public:
        std::vector<uint8_t> out;
        void encode(int bit, uint32_t p1) {          // p1 = P(bit==1) in (0,65536)
            uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * p1) >> 16);

            if (bit) 
                x2 = xmid; 
            else 
                x1 = xmid + 1;

            while (((x1 ^ x2) & 0xFF000000u) == 0) {
                out.push_back((uint8_t)(x2 >> 24));
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
            }
        }

        void finish() {
            for (int i = 0; i < 4; ++i) { 
                out.push_back((uint8_t)(x1 >> 24)); 
                x1 <<= 8; 
            }
        }
};

class RangeDecoder {
    uint32_t x1 = 0, x2 = 0xFFFFFFFFu, x = 0;
    const uint8_t* buf; 
    size_t n, pos = 0;
    
    uint8_t next() { 
        return pos < n ? buf[pos++] : 0; 
    }   // zero-pad past the end

    public:
        RangeDecoder(const uint8_t* b, size_t len) : buf(b), n(len) {
            for (int i = 0; i < 4; ++i) {
                x = (x << 8) | next();
            }
        }

        int decode(uint32_t p1) {
            uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * p1) >> 16);
            int bit = x <= xmid;
            if (bit) 
                x2 = xmid; 
            else 
                x1 = xmid + 1;

            while (((x1 ^ x2) & 0xFF000000u) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
                x = (x << 8) | next();
            }
            return bit;
        }
};

// Encoder / decoder "operations" so one template serves both
struct EncOps {
    RangeEncoder& rc;
    double* costSink = nullptr;                  // optional: ideal bits spent
    int code(BitModel& m, int bit) {
        if (costSink) {
            double p = (bit ? m.p : 65536 - m.p) / 65536.0;
            *costSink -= std::log2(p);
        }
        rc.encode(bit, m.p);
        m.update(bit);
        return bit;
    }
};

struct DecOps {
    RangeDecoder& rc;
    double* costSink = nullptr;                  // unused
    int code(BitModel& m, int /*ignored*/) {
        int bit = rc.decode(m.p);
        m.update(bit);
        return bit;
    }
};

// Tier map coding 
// Labels are spatially smooth, so context = labels of W, N, NW, NE.
template <class Ops>

void codeTierMap(Ops& ops, std::vector<uint8_t>& tier, int W, int H) {
    std::vector<BitModel> m0(81), m1(81);
    auto lab = [&](int x, int y) -> int {
        if (x < 0 || x >= W || y < 0) 
            return 0;

        return tier[(size_t)y * W + x] - 1;
    };

    for (int y = 0; y < H; ++y){
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y * W + x;
            int ctx = ((lab(x - 1, y) * 3 + lab(x, y - 1)) * 3 + lab(x - 1, y - 1)) * 3 + lab(x + 1, y - 1);
            int l = tier[i] - 1;   // only for encoder
            int b0 = ops.code(m0[ctx], l > 0);
            int b1 = b0 ? ops.code(m1[ctx], l > 1) : 0;
            tier[i] = (uint8_t)(b0 + b1 + 1);
        }
    }
}

// Residual bit-plane coding 
template <class Ops>

void codeResiduals(Ops& ops, ResidualPlanes& rp, const std::vector<uint8_t>& tier, std::array<double, NUM_TIERS + 1>* tierBits = nullptr) {
    const int W = rp.width, H = rp.height;
    (void)H;
    const size_t N = (size_t)W * rp.height;
    auto lists = tierPixelLists(tier);

    std::array<std::vector<uint8_t>, NUM_CHANNELS> known;

    for (auto& k : known) 
        k.assign(N, 0);

    std::vector<BitModel> models((size_t)NUM_TIERS * NUM_CHANNELS * NUM_PLANES * NUM_CTX);

    auto lvl = [](unsigned v) { 
        return v == 0 ? 0 : v == 1 ? 1 : v < 4 ? 2 : 3; 
    };

    for (const SliceId& s : sliceSchedule()) {
        const int sh = 8 - s.plane;           // bit position of this plane
        const int c  = s.channel;
        // Measured: one shared model set is ~1% smaller than per-tier models
        // (neighbourhood context already captures what the tier would say, so
        // splitting just dilutes statistics).  Build with -DTIERED_MODELS to
        // give every tier its own adaptive statistics instead.
#ifdef TIERED_MODELS
        const int tm = s.tier - 1;
#else
        const int tm = 0;
#endif
        BitModel* base = &models[(((size_t)tm * NUM_CHANNELS + c) * NUM_PLANES + (s.plane - 1)) * NUM_CTX];
        uint8_t* kc = known[c].data();

        if (tierBits) 
            ops.costSink = &(*tierBits)[s.tier];

        for (int i : lists[s.tier]) {
            const int x = i % W, y = i / W;

            unsigned own = kc[i] >> (sh + 1);                  // planes 1..p-1
            unsigned qW = 0, qN = 0, qNW = 0, qNE = 0;
            int bW = 0, bN = 0;

            if (x > 0) { 
                uint8_t v = kc[i - 1]; 
                qW = v >> (sh + 1); 
                bW = (v >> sh) & 1; 
            }

            if (y > 0) {
                uint8_t v = kc[i - W]; 
                qN = v >> (sh + 1); 
                bN = (v >> sh) & 1;

                if (x > 0)     
                    qNW = kc[i - W - 1] >> (sh + 1);
                if (x < W - 1) 
                    qNE = kc[i - W + 1] >> (sh + 1);

            }

            unsigned act = 2 * qW + 2 * qN + qNW + qNE;
            int bl = act ? 32 - __builtin_clz(act) : 0;

            if (bl > 7) 
                bl = 7;

            int X = c > 0 ? lvl(known[c - 1][i] >> sh) : 0;   // incl. plane p of prev channel

            int ctx = (((lvl(own) * 8 + bl) * 2 + bW) * 2 + bN) * 4 + X;

            int bit = (rp.r[c].empty() ? 0 : (rp.r[c][i] >> sh) & 1);   // encoder only
            bit = ops.code(base[ctx], bit);
            kc[i] |= (uint8_t)(bit << sh);
        }
    }

    for (int c = 0; c < NUM_CHANNELS; ++c) 
        rp.r[c] = std::move(known[c]);

}

//  API 
inline EncodedStream encodeStream(ResidualPlanes rp, std::vector<uint8_t> tier) {
    EncodedStream es;
    {   // tier map
        RangeEncoder rc; EncOps ops{rc};
        std::vector<uint8_t> t = tier;
        codeTierMap(ops, t, rp.width, rp.height);
        rc.finish();
        es.tierMapBytes = std::move(rc.out);
    }
    {   // residual planes
        RangeEncoder rc; EncOps ops{rc};
        codeResiduals(ops, rp, tier, &es.tierBits);
        rc.finish();
        es.payload = std::move(rc.out);
    }

    for (uint8_t t : tier) 
        es.tierPixels[t]++;

    return es;
}

inline void decodeStream(const std::vector<uint8_t>& tierMapBytes, const std::vector<uint8_t>& payload, int W, int H, ResidualPlanes& rp, std::vector<uint8_t>& tier) {

    tier.assign((size_t)W * H, 1);

    {
        RangeDecoder rc(tierMapBytes.data(), tierMapBytes.size());
        DecOps ops{rc};
        codeTierMap(ops, tier, W, H);
    }

    rp.width = W; rp.height = H;

    for (auto& r : rp.r) {
        r.clear();    
    }           // empty => decoder mode
    
    {
        RangeDecoder rc(payload.data(), payload.size());
        DecOps ops{rc};
        codeResiduals(ops, rp, tier);
    }
}
