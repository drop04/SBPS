CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I/usr/include/libpng16
LDFLAGS  := -lpng16

compress: main.cpp pipeline.h image_io.h stage1_segmentation.h stage2_saliency.h \
          stage3_slicing.h stage4_entropy.h stage5_bitstream.h
	$(CXX) $(CXXFLAGS) main.cpp $(LDFLAGS) -o compress

# Variant: one shared model set for all tiers (for experiments)
compress_shared: main.cpp
	$(CXX) $(CXXFLAGS) -DSHARE_TIER_MODELS main.cpp $(LDFLAGS) -o compress_shared

test: compress
	bash tests/run_tests.sh

clean:
	rm -f compress compress_shared

.PHONY: test clean
