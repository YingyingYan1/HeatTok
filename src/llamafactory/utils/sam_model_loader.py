# -*- coding: utf-8 -*-
"""English docstring."""

import os
import torch
from typing import Optional, Tuple

def load_sam_model(
    sam_checkpoint: Optional[str] = None,
    device_id: int = 0,
    gpus_to_use: Optional[list] = None,
) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """English docstring."""
    del gpus_to_use  # unused, kept for call-site compatibility
    if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
        return None, None, None

    try:
        import sys

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        repo_root = os.path.dirname(base_dir)
        fastsam_dir = os.path.join(repo_root, "FastSAM-main")
        if os.path.isdir(fastsam_dir) and fastsam_dir not in sys.path:
            sys.path.insert(0, fastsam_dir)

        from fastsam import FastSAM

        if not torch.cuda.is_available():
            print("Warning: CUDA is unavailable; cannot load FastSAM model")
            return None, None, None

        device_str = f"cuda:{device_id}"
        print(f"Loading FastSAM model from: {sam_checkpoint}")

        fastsam_model = FastSAM(sam_checkpoint)
        print(f"FastSAM model loaded (inference device: {device_str})")
        return fastsam_model, fastsam_model, fastsam_model

    except ImportError as e:
        print(f"Warning: failed to import fastsam; please ensure FastSAM is installed: {e}")
        return None, None, None
    except Exception as e:
        print(f"Warning: failed to load FastSAM model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

