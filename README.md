# SBPS v2 — saliency-tiered lossless image compressor

Same 5-stage architecture as v1; stages 3–5 rewritten so it actually compresses.

    make            # build ./compress  (needs g++ and libpng)
    make test       # fuzz + round-trip + robustness suite
    ./compress encode in.png out.sbps [debug_dir]
    ./compress decode out.sbps out.png
    ./compress roundtrip in.png

## What changed

| Problem in v1 | Fix in v2 |
|---|---|
| Tier index tables cost 4 bytes/pixel (more than the raw image) | Tier map is context-coded as a label image (≈0.1–0.5 % of the file) |
| Raw bit planes of natural images are ~incompressible | Reversible colour transform (G, R-G, B-G) + MED prediction + zig-zag residuals, *then* bit-plane slicing |
| Contexts = previous 5 bits in a 1-D list | 2-D contexts: own higher planes, W/N/NW/NE neighbour magnitudes, W/N same-plane bits, previous channel |
| Hand-rolled 64-bit carry-less coder, separate enc/dec logic | Standard 32-bit range coder; encoder and decoder share one templated code path |
| Truncated/corrupt files crashed (`bad_alloc`, uncaught exceptions) | All header sizes validated, CRC-32 of the pixels, clean `error:` messages |
| Incompressible images expanded ~2.8x | "Stored" fallback caps expansion at 16 bytes |
| Stage 2 was O(N·r²): 92 s on a 2 MP image | Summed-area tables: 3.9 s total encode |
| `make test` needed a missing `test.png` | Self-contained test suite in `tests/` |

## Format (.sbps v2)
`"SBP2" | W | H | crc32 | tierMapBytes | payloadBytes | tier map | payload` (little-endian),
or `"SBPR" | W | H | crc32 | raw RGB` in stored mode.

## Notes / honest limits
* Lossless coding means tiers change *order and statistics*, not quality. Slices are
  scheduled plane-major (MSB first) and tier 1 first within each plane, but there is no
  partial/progressive decoder yet.
* Per-tier model sets (`-DTIERED_MODELS`) measured ~1 % *worse* than shared models, so
  shared is the default.
* An adaptive blend of 5 predictors was tried and rejected (see comment in `stage3_slicing.h`).
* Stages 1–2 (k-means "segmentation", centre-bias + contrast "saliency") are still stand-ins
  for real models.
