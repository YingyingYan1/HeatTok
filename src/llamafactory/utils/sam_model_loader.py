# -*- coding: utf-8 -*-
"""English docstring."""

import os
import torch
from typing import Optional, Tuple

def load_sam_model(
    sam_checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    sam_backend: str = "fastsam",
    device_id: int = 0,
    gpus_to_use: Optional[list] = None,
) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """English docstring."""
    if gpus_to_use is None or len(gpus_to_use) == 0:
        gpus_to_use = [device_id]

    sam_backend = (sam_backend or "fastsam").strip().lower()
    if sam_backend == "auto" and sam_checkpoint:
        if sam_checkpoint.endswith(".pt"):
            sam_backend = "fastsam"
        else:
            sam_backend = "sam"
    elif sam_checkpoint:
        if sam_backend == "sam" and sam_checkpoint.endswith(".pt"):
            print("Warning: .pt checkpoint detected, switching backend to fastsam.")
            sam_backend = "fastsam"
        elif sam_backend == "fastsam" and sam_checkpoint.endswith(".pth"):
            print("Warning: .pth checkpoint detected, switching backend to sam.")
            sam_backend = "sam"
    if sam_backend not in {"sam", "fastsam"}:
        print(f"Warning: unknown sam_backend={sam_backend}, fallback to fastsam")
        sam_backend = "fastsam"

    if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
        return None, None, None

    try:
        import sys

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        repo_root = os.path.dirname(base_dir)

        if not torch.cuda.is_available():
            print("Warning: CUDA is unavailable; cannot load segmentation model")
            return None, None, None

        device_str = f"cuda:{device_id}"
        if sam_backend == "fastsam":
            fastsam_dir = os.path.join(repo_root, "FastSAM-main")
            if os.path.isdir(fastsam_dir) and fastsam_dir not in sys.path:
                sys.path.insert(0, fastsam_dir)
            from fastsam import FastSAM

            print(f"Loading FastSAM model from: {sam_checkpoint}")
            fastsam_model = FastSAM(sam_checkpoint)
            setattr(fastsam_model, "backend_name", "FASTSAM")
            print(f"FastSAM model loaded (inference device: {device_str})")
            return fastsam_model, fastsam_model, fastsam_model

        sam_dir = os.path.join(repo_root, "segment-anything-main")
        if os.path.isdir(sam_dir) and sam_dir not in sys.path:
            sys.path.insert(0, sam_dir)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        print(f"Loading SAM ({model_type}) from: {sam_checkpoint}")
        sam_raw = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam_raw.to(device=device_str)
        sam_dp = torch.nn.DataParallel(sam_raw, device_ids=gpus_to_use)
        mask_generator = SamAutomaticMaskGenerator(
            sam_raw,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.90,
        )
        setattr(mask_generator, "backend_name", "SAM")
        print(f"SAM model loaded (inference device: {device_str})")
        return sam_raw, sam_dp, mask_generator

    except ImportError as e:
        print(f"Warning: failed to import backend dependency: {e}")
        return None, None, None
    except Exception as e:
        print(f"Warning: failed to load segmentation model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

