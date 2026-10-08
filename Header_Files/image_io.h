#pragma once
#include "pipeline.h"
#include <png.h>
#include <stdexcept>
#include <cstdio>

//  PNG loader (RGB only, 8-bit)
inline Image loadPNG(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");

    if (!fp) {
        throw std::runtime_error("Cannot open: " + path);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png) { 
        fclose(fp); throw std::runtime_error("png_create_read_struct failed"); 
    }

    png_infop info = png_create_info_struct(png);

    if (!info) { 
        fclose(fp); 
        throw std::runtime_error("png_create_info_struct failed"); 
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        throw std::runtime_error("Error during PNG init_io");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    Image img;
    img.width    = (int)png_get_image_width(png, info);
    img.height   = (int)png_get_image_height(png, info);
    img.channels = 3;

    // Force 8-bit RGB
    png_byte colorType = png_get_color_type(png, info);
    png_byte bitDepth  = png_get_bit_depth(png, info);

    if (bitDepth == 16) {
        png_set_strip_16(png);
    }

    if (colorType == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    if (colorType == PNG_COLOR_TYPE_RGBA || colorType == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_strip_alpha(png);
    }

    if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    img.data.resize(img.width * img.height * 3);
    std::vector<png_bytep> rowPtrs(img.height);

    for (int y = 0; y < img.height; ++y){
        rowPtrs[y] = &img.data[y * img.width * 3];
    }

    png_read_image(png, rowPtrs.data());
    png_destroy_read_struct(&png, &info, nullptr);

    fclose(fp);

    return img;
}

//  PNG writer
inline void savePNG(const std::string& path, const Image& img) {
    FILE* fp = fopen(path.c_str(), "wb");
    
    if(!fp) {
        throw std::runtime_error("Cannot write: " + path);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info  = png_create_info_struct(png);

    if (setjmp(png_jmpbuf(png))) { 
        fclose(fp); 
        throw std::runtime_error("PNG write error"); 
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, img.width, img.height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, info);

    for (int y = 0; y < img.height; ++y){
        png_write_row(png, (png_bytep)&img.data[y * img.width * 3]);
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);

    fclose(fp);
}

//  Save a saliency map as grayscale PNG
inline void saveSaliencyPNG(const std::string& path, const SaliencyMap& sal) {
    Image img;
    img.width = sal.width; 
    img.height = sal.height; 
    img.channels = 3;
    img.data.resize(sal.width * sal.height * 3);

    for (int i = 0; i < sal.width * sal.height; ++i){
        uint8_t v = (uint8_t)(std::clamp(sal.data[i], 0.f, 1.f) * 255.f);
        img.data[i*3] = img.data[i*3+1] = img.data[i*3+2] = v;
    }

    savePNG(path, img);
}

//  Save a tier map as colour-coded PNG
inline void saveTierMapPNG(const std::string& path, const ImportanceMap& imp) {
    Image img;
    img.width = imp.width; img.height = imp.height; img.channels = 3;
    img.data.resize(imp.width * imp.height * 3);

    for (int i = 0; i < imp.width * imp.height; ++i) {
        uint8_t t = imp.tier[i];
        // Tier 1 = red, Tier 2 = green, Tier 3 = blue
        img.data[i*3+0] = (t == 1) ? 220 : 0;
        img.data[i*3+1] = (t == 2) ? 220 : 0;
        img.data[i*3+2] = (t == 3) ? 220 : 50;
    }
    
    savePNG(path, img);
}
