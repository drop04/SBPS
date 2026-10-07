//  STAGE 5 — Bitstream container (.sbps, version 2)
//
//  Layout (all integers little-endian, written byte-by-byte so the format is
//  independent of host endianness):
//
//    "SBP2"                       4   magic
//    width, height                4+4
//    crc32 of original RGB data   4
//    tierMapBytes, payloadBytes   4+4
//    tier map                     tierMapBytes
//    residual payload             payloadBytes
//
//  v1 stored three raw int32 pixel-index tables (4 bytes per pixel — more
//  than the raw image).  v2 stores the tier map as a context-coded label
//  image, typically a fraction of a percent of the file.
//
//  Reading is defensive: every size is validated against the real file
//  length, and the CRC detects corruption after decoding. 

#pragma once
#include "pipeline.h"
#include <iomanip>
#include <filesystem>

static const char MAGIC[4]        = {'S', 'B', 'P', '2'};   // modelled
static const char MAGIC_STORED[4] = {'S', 'B', 'P', 'R'};   // raw fallback
static constexpr size_t STORED_HEADER_BYTES = 4 + 4 + 4 + 4;
static constexpr size_t HEADER_BYTES = 4 + 4 + 4 + 4 + 4 + 4;

static void putU32(std::vector<uint8_t>& o, uint32_t v) {

    for (int i = 0; i < 4; ++i) 
        o.push_back((uint8_t)(v >> (8 * i)));

}

static uint32_t getU32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline void writeCompressedFile(const std::string& path, int W, int H, uint32_t crc, const EncodedStream& es) {
    std::vector<uint8_t> o;
    o.reserve(HEADER_BYTES + es.tierMapBytes.size() + es.payload.size());
    o.insert(o.end(), MAGIC, MAGIC + 4);
    putU32(o, (uint32_t)W);
    putU32(o, (uint32_t)H);
    putU32(o, crc);
    putU32(o, (uint32_t)es.tierMapBytes.size());
    putU32(o, (uint32_t)es.payload.size());
    o.insert(o.end(), es.tierMapBytes.begin(), es.tierMapBytes.end());
    o.insert(o.end(), es.payload.begin(), es.payload.end());

    std::ofstream f(path, std::ios::binary);

    if (!f) 
        throw std::runtime_error("Cannot write: " + path);

    f.write((const char*)o.data(), (std::streamsize)o.size());

    if (!f) 
        throw std::runtime_error("Write failed: " + path);

}

// Fallback for incompressible images: header + raw RGB.  Bounds worst-case
// expansion to STORED_HEADER_BYTES.
inline void writeStoredFile(const std::string& path, int W, int H, uint32_t crc, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> o;
    o.insert(o.end(), MAGIC_STORED, MAGIC_STORED + 4);
    putU32(o, (uint32_t)W);
    putU32(o, (uint32_t)H);
    putU32(o, crc);
    o.insert(o.end(), rgb.begin(), rgb.end());
    std::ofstream f(path, std::ios::binary);

    if (!f) 
        throw std::runtime_error("Cannot write: " + path);

    f.write((const char*)o.data(), (std::streamsize)o.size());

    if (!f) 
        throw std::runtime_error("Write failed: " + path);

}

struct DeserializedStream {
    bool stored = false;                 // raw fallback mode
    std::vector<uint8_t> raw;            // RGB data when stored
    int W = 0, H = 0;
    uint32_t crc = 0;
    std::vector<uint8_t> tierMapBytes, payload;
};

inline DeserializedStream readCompressedFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) 
        throw std::runtime_error("Cannot open: " + path);

    std::streamsize size = f.tellg();
    f.seekg(0);

    if (size < (std::streamsize)STORED_HEADER_BYTES) 
        throw std::runtime_error("File too short to be a .sbps file");

    std::vector<uint8_t> buf((size_t)size);
    f.read((char*)buf.data(), size);

    if (!f) 
        throw std::runtime_error("Read failed: " + path);

    DeserializedStream ds;

    if (std::equal(MAGIC_STORED, MAGIC_STORED + 4, buf.begin())) {
        uint32_t W = getU32(&buf[4]), H = getU32(&buf[8]);
        ds.crc = getU32(&buf[12]);

        if (W == 0 || H == 0 || W > MAX_DIM || H > MAX_DIM || (uint64_t)W * H > MAX_PIXELS)
            throw std::runtime_error("Invalid image dimensions in header");

        if (STORED_HEADER_BYTES + (uint64_t)W * H * 3 != (uint64_t)size)
            throw std::runtime_error("File is truncated or corrupt (size mismatch)");

        ds.stored = true; ds.W = (int)W; ds.H = (int)H;
        ds.raw.assign(buf.begin() + STORED_HEADER_BYTES, buf.end());
        return ds;
    }

    if (size < (std::streamsize)HEADER_BYTES) 
        throw std::runtime_error("File too short to be a .sbps file");

    if (!std::equal(MAGIC, MAGIC + 4, buf.begin()))
        throw std::runtime_error("Not a .sbps (v2) file");

    uint32_t W = getU32(&buf[4]), H = getU32(&buf[8]);
    ds.crc = getU32(&buf[12]);
    uint64_t tmLen = getU32(&buf[16]), plLen = getU32(&buf[20]);

    if (W == 0 || H == 0 || W > MAX_DIM || H > MAX_DIM || (uint64_t)W * H > MAX_PIXELS)
        throw std::runtime_error("Invalid image dimensions in header");

    if (HEADER_BYTES + tmLen + plLen != (uint64_t)size)
        throw std::runtime_error("File is truncated or corrupt (size mismatch)");

    ds.W = (int)W; ds.H = (int)H;
    ds.tierMapBytes.assign(buf.begin() + HEADER_BYTES, buf.begin() + HEADER_BYTES + tmLen);
    ds.payload.assign(buf.begin() + HEADER_BYTES + tmLen, buf.end());
    return ds;
}

// Reporting 
inline void printStats(int W, int H, const EncodedStream& es, const std::string& outPath, bool stored = false) {
    size_t raw  = (size_t)W * H * 3;
    size_t file = (size_t)std::filesystem::file_size(outPath);
    double ratio = (double)raw / (double)file;
    double totalBits = 0;

    for (int t = 1; t <= NUM_TIERS; ++t) 
        totalBits += es.tierBits[t];

    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  COMPRESSION SUMMARY\n";
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  Image size       : " << W << " × " << H << "  (" << raw << " bytes raw)\n";
    std::cout << "  File size        : " << file << " bytes\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Compression ratio: " << ratio << "x   ("
              << (8.0 * file / ((double)W * H)) << " bits/pixel)\n";

    if (stored) 
        std::cout << "  Mode             : STORED (image is incompressible; modelled coding would be larger)\n";

    std::cout << "  Tier map         : " << es.tierMapBytes.size() << " bytes\n";
    std::cout << "  Residual payload : " << es.payload.size() << " bytes\n";

    for (int t = 1; t <= NUM_TIERS; ++t) {
        double bytes = es.tierBits[t] / 8.0;
        double bpp = es.tierPixels[t] ? es.tierBits[t] / (double)es.tierPixels[t] / 3.0 : 0.0;
        std::cout << "  Tier " << t << " payload  : " << std::setw(9) << std::setprecision(0)
                  << bytes << " bytes (" << std::setprecision(1)
                  << (totalBits > 0 ? 100.0 * es.tierBits[t] / totalBits : 0.0) << "% of payload, "
                  << std::setprecision(2) << bpp << " bits/sample, "
                  << es.tierPixels[t] << " px)\n";
    }
    
    std::cout << "══════════════════════════════════════════\n\n";
}
