# semantic_patch_cache

This directory stores **HeatTok thermodiffusion semantic patch preprocessing outputs** (`.pt` cache files).

## Contents

- After SAM initial segmentation + thermodiffusion merging, each image's semantic patches, grid metadata, and Gaussian position parameters are written to `.pt` files.
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

## Dataset Parameter Sets

HeatTok heat-diffusion / post-processing defaults live in:

- `src/llamafactory/utils/semantic_patch_utils.py` (training / cache generation)
- `src/4heat_sam_chicun.py` (standalone demo defaults)
- `visualization/visualize_heatok.py` (visualization demo)

Different datasets usually need different parameter settings; please tune them for your own data to obtain better results.

### EarthVQA (current training / cache / visualization defaults)


| Parameter                 | Value   | Notes                                        |
| ------------------------- | ------- | -------------------------------------------- |
| `sam_backend`             | `fastsam` | Default segmentation backend                 |
| `pred_iou_thresh`         | `0.85`  | SAM AMG                                      |
| `stability_score_thresh`  | `0.90`  | SAM AMG                                      |
| `sam_morph_kernel`        | `5`     | Morphological cleanup                        |
| `min_size_threshold`      | `50`    | Drop tiny regions before merge               |
| `post_min_size_threshold` | `500`   | Post-merge min region size                   |
| `post_max_size_threshold` | `40000` | Post-merge max region size                   |
| `target_split_size`       | `15000` | Target size when splitting oversized regions |
| `sigma_T`                 | `5.0`   | Temperature scale                            |
| `sigma_C`                 | `5.0`   | Color sensitivity                            |
| `alpha`                   | `0.5`   | Complexity factor                            |
| `K0`                      | `0.5`   | Base diffusivity                             |
| `delta_t`                 | `0.001` | Diffusion step size                          |
| `diffusion_iterations`    | `30`    | Max diffusion steps                          |
| `merge_threshold`         | `0.01`  | Temperature merge threshold                  |
| `semantic_patch_size`     | `28`    | From training yaml                           |
| `global_downsample`       | `true`  | From training yaml                           |


### VRSBench 


| Parameter                 | Value   | Notes                                           |
| ------------------------- | ------- | ----------------------------------------------- |
| `sam_morph_kernel`        | `5`     | Morphological cleanup                           |
| `min_size_threshold`      | `50`    | Pre-merge tiny region cleanup                   |
| `post_min_size_threshold` | `200`   | Keep more small regions on ~512x512 images      |
| `post_max_size_threshold` | `20000` | Split only larger regions                       |
| `target_split_size`       | `2500`  | Larger split pieces                             |
| `sigma_T`                 | `9.0`   | Temperature scale                               |
| `sigma_C`                 | `12.0`  | Color sensitivity                               |
| `alpha`                   | `0.4`   | Complexity factor                               |
| `K0`                      | `1.0`   | Base diffusivity                                |
| `delta_t`                 | `0.001` | Diffusion step size                             |
| `diffusion_iterations`    | `20`    | Max diffusion steps                             |
| `merge_threshold`         | `0.01`  | Temperature merge threshold                     |


## Notes

- `.pt` files in this directory can be large and are usually not committed to version control.
- If cache files are deleted, they will be recomputed during the next training/pregeneration run.
- Changing heat-diffusion parameters usually requires regenerating the corresponding cache files.

