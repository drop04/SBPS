#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#include "Header_Files/pipeline.h"
#include "Header_Files/image_io.h"
#include "Header_Files/stage1_segmentation.h"
#include "Header_Files/stage2_saliency.h"
#include "Header_Files/stage3_slicing.h"
#include "Header_Files/stage4_entropy.h"
#include "Header_Files/stage5_bitstream.h"

using Clock = std::chrono::high_resolution_clock;

static auto tic() { 
    return Clock::now(); 
}

static double toc(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
 
//  Encode pipeline 
static void encode(const std::string& inputPNG, const std::string& outputSBPS, const std::string& debugDir = "") {

    auto T0 = tic();

    std::cout << std::endl <<  "ENCODE: " << inputPNG << std::endl;
    
    Image img = loadPNG(inputPNG);
    std::cout << "[Input]   " << img.width << "×" << img.height << " RGB  (" << (size_t)img.width * img.height * 3 << " bytes)\n\n";
    
    if (!debugDir.empty()) 
        std::filesystem::create_directories(debugDir);

    // Stage 1: segmentation
    auto t1 = tic();
    SegmentationMask seg = segment(img);
    std::cout << "          [" << toc(t1) << " ms]\n\n";
    if (!debugDir.empty()) 
        saveSegPNG(debugDir + "/seg.png", seg);

    // Stage 2: saliency + importance fusion -> 3 tiers
    auto t2 = tic();
    SaliencyMap sal = generateSaliency(img);
    ImportanceMap imp = fuseImportance(seg, sal);
    std::cout << "          [" << toc(t2) << " ms]\n\n";
    if (!debugDir.empty()){
        saveSaliencyPNG(debugDir + "/saliency.png", sal);
        saveTierMapPNG(debugDir + "/tiermap.png", imp);
    }

    // Stage 3: decorrelate (colour transform + MED prediction) -> residual planes
    auto t3 = tic();
    ResidualPlanes rp = decorrelate(img);
    std::cout << "[Stage 3] Decorrelation: G / R-G / B-G, MED prediction, zig-zag residuals\n";
    std::cout << "          [" << toc(t3) << " ms]\n\n";
    if (!debugDir.empty()) 
        saveResidualPNG(debugDir + "/residual.png", rp);

    {   // Cheap self-check of this stage before spending time on entropy coding
        bool ok = verifyLossless(img, reconstruct(rp));
        std::cout << "[Verify] Decorrelation round-trip lossless: " << (ok ? "✓ PASS" : "✗ FAIL") << "\n\n";
        if(!ok) 
            throw std::runtime_error("decorrelation is not invertible (bug)");
    }

    // Stage 4: context-modelled range coding
    auto t4 = tic();
    EncodedStream es = encodeStream(std::move(rp), imp.tier);
    std::cout << "[Stage 4] Entropy coding: " << es.payload.size() << " payload bytes + " << es.tierMapBytes.size() << " tier-map bytes\n";
    std::cout << "          [" << toc(t4) << " ms]\n\n";

    // Stage 5: container
    auto t5 = tic();
    const size_t modelledBytes = HEADER_BYTES + es.tierMapBytes.size() + es.payload.size();
    const bool useStored = STORED_HEADER_BYTES + img.data.size() <= modelledBytes;

    if (useStored) 
        writeStoredFile(outputSBPS, img.width, img.height, crc32(img.data), img.data);
    else           
        writeCompressedFile(outputSBPS, img.width, img.height, crc32(img.data), es);

    std::cout << "[Stage 5] Written to " << outputSBPS << (useStored ? "  (stored mode)" : "") << "\n";
    std::cout << "          [" << toc(t5) << " ms]\n\n";

    printStats(img.width, img.height, es, outputSBPS, useStored);
    std::cout << "[Total encode time] " << toc(T0) << " ms\n\n";
}
 
//  Decode pipeline 
static void decode(const std::string& inputSBPS, const std::string& outputPNG) {

    auto T0 = tic();

    std::cout << std::endl <<   "  DECODE: " << inputSBPS << std::endl;
    std::cout << std::endl;

    DeserializedStream ds = readCompressedFile(inputSBPS);
    std::cout << "[Stage 5] Read " << inputSBPS << " — " << ds.W << "×" << ds.H << "\n";

    Image img;
    if (ds.stored) {
        img.width = ds.W; img.height = ds.H; img.channels = 3;
        img.data = std::move(ds.raw);
    } 
    else {
        ResidualPlanes rp;
        std::vector<uint8_t> tier;
        decodeStream(ds.tierMapBytes, ds.payload, ds.W, ds.H, rp, tier);
        img = reconstruct(rp);
    }
    if (crc32(img.data) != ds.crc)
        throw std::runtime_error("checksum mismatch — file is corrupt");

    savePNG(outputPNG, img);
    std::cout << "[Output]  Saved to " << outputPNG << "  (checksum OK)\n";
    std::cout << "[Total decode time] " << toc(T0) << " ms\n\n";
}

// converts the input image to PNG 
static std::string toPNG(const std::string& inputPath){
    std::filesystem::path input = inputPath;

    if(!std::filesystem::exists(input)){
        std::cerr << "Invalid Input !!" << std::endl << "Please check File Path" << std::endl;
        return "";
    }

    std::string cmd1 = "file --mime-type -b " + inputPath;
    FILE* pipe = popen(cmd1.c_str(), "r");
    if(!pipe){
        std::cerr << "Failed to execute --mime-type command" << std::endl;
        return "";
    }
    char buffer[150];
    std::string result = "";

    while(fgets(buffer, sizeof(buffer), pipe) != nullptr){
        result += buffer;
    }

    pclose(pipe);

    if(result.find("Image/") != 0){
        std::cerr << "Input is NOT AN IMAGE" << std::endl;
        return "";
    }

    std::filesystem::path output = input;
    output.replace_extension(".png");

    std::string cmd = "python3 -c \"from PIL import Image; Image.open('" + std::string(input) 
                       + "').convert('RGB').save('" + std::string(output) + "')\"";
    
    system(cmd.c_str());
    return std::string(output);
}

//  Main
static int usage() {
    std::cerr << "Usage:\n"
              << "  compress encode <input.png> <output.sbps> [debug_dir]\n"
              << "  compress decode <input.sbps> <output.png>\n"
              << "  compress roundtrip <input.png>   (encode+decode+verify)\n";
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage();
    }

    std::string mode = argv[1];
    try {
        if (mode == "encode" && argc >= 4) {
            std::string output = toPNG(argv[2]);

            encode(output, argv[3], argc >= 5 ? argv[4] : "");
        } 
        else if (mode == "decode" && argc >= 4) {
            decode(argv[2], argv[3]);
        } 
        else if (mode == "roundtrip" && argc >= 3) {
            namespace fs = std::filesystem;
            fs::path tmp = fs::temp_directory_path() / "sbps_roundtrip";
            fs::create_directories(tmp);
            std::string sbps = (tmp / "rt.sbps").string();
            std::string outPNG = (tmp / "rt_decoded.png").string();
            
            std::string output = toPNG(argv[2]);

            encode(output, sbps, (tmp / "debug").string());
            decode(sbps, outPNG);

            Image original = loadPNG(std::string(output));
            Image decoded  = loadPNG(outPNG);
            bool lossless = verifyLossless(original, decoded);

            std::cout << std::endl << "  LOSSLESS VERIFICATION  " << std::endl;
            std::cout << "  Result: " << (lossless ? "✓ PERFECT LOSSLESS RECONSTRUCTION"
                                                   : "✗ MISMATCH — BUG IN PIPELINE");
            std::cout << std::endl;

            if (!lossless) {
                return 1;
            }
        } 
        else {
            return usage();
        }
    } 
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
