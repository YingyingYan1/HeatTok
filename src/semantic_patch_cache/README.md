# semantic_patch_cache

This directory stores **HeatTok thermodiffusion semantic patch preprocessing outputs** (`.pt` cache files).

## Contents

- After FastSAM initial segmentation + thermodiffusion merging, each image's semantic patches, grid metadata, and Gaussian position parameters are written to `.pt` files.
- Training and pregeneration scripts automatically read/write cache files in this directory.
- Filename format:
  `{image_hash}_g{0|1}_gd{8}_s{semantic_patch_size}.pt`
  - `g`: whether `global_downsample` is enabled
  - `gd`: global downsample divisor (currently 8)
  - `s`: semantic patch size (e.g., 28)

## How To Generate

Use either of the following:

```bash
# Single-GPU / sharded pregeneration
python scripts/pregenerate_semantic_cache.py --config examples/train_lora/qwen2_5vl_lora_sft.yaml

# Multi-GPU parallel pregeneration
bash scripts/parallel_pregenerate.sh 8
```

If you skip pregeneration, training will compute and write cache files online (slower).

## Notes

- `.pt` files in this directory can be large and are usually not committed to version control.
- If cache files are deleted, they will be recomputed during the next training/pregeneration run.
