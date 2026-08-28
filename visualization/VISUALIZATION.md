# HeatTok Visualization

This folder provides a lightweight demo for visualizing **HeatTok** tokens.

HeatTok first obtains fine-grained regions with a segmentation backend, then merges adjacent homogeneous areas via **heat diffusion**, producing irregular, object-aligned visual tokens. The script draws white boundaries of the merged tokens on the original image and prints the final token count.

> Training defaults to **FastSAM**. This visualization demo currently uses **Meta SAM** (optional).

## Quick Start

1. Place test images under `visualization/examples/` (or set `HEATOK_VIS_INPUT_DIR` / `HEATOK_VIS_IMAGE`).
2. Make sure Meta SAM is installed and the checkpoint exists (default: `segment-anything-main/models/sam_vit_h_4b8939.pth`).
3. Run from the repo root:

```bash
CUDA_VISIBLE_DEVICES=0 python visualization/visualize_heatok.py
```


## Note on Parameters

The current default parameter set in `visualize_heatok.py` is tuned for **EarthVQA** experiments.

Default heat-diffusion and post-processing parameters work reasonably on many remote-sensing examples, but **results on different images may vary**. For **VRSBench** or other datasets, you may need to tune parameters in `visualize_heatok.py` (e.g. `merge_threshold`, `sigma_C` / `sigma_T`, `post_min_size` / `post_max_size`, `target_split_size`, and SAM thresholds) for your own data.
