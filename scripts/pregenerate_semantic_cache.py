#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pregenerate semantic patch cache files.

Usage:
    python scripts/pregenerate_semantic_cache.py \
        --config examples/train_lora/qwen2_5vl_lora_sft.yaml \
        --dataset_dir data \
        --dataset_info data/dataset_info.json

Notes:
    - Loads only FastSAM + ImageProcessor (not the full Qwen model), so memory usage is lower.
    - Iterates all training images and generates one `.pt` cache per image.
    - After pregeneration, training loads cache files directly instead of generating online.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml
import torch
from PIL import Image
from functools import partial
from tqdm import tqdm as _tqdm
from transformers import Qwen2VLImageProcessor

tqdm = partial(_tqdm, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_info(dataset_info_path: str) -> dict:
    """Load dataset_info.json."""
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_image_paths(
    dataset_name: str,
    dataset_dir: str,
    dataset_info: dict,
    max_samples: Optional[int] = None,
) -> List[str]:
    """Collect all image paths from the dataset."""
    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.json")

    ds_config = dataset_info[dataset_name]
    file_name = ds_config.get("file_name")
    if not file_name:
        raise ValueError(f"No 'file_name' specified for dataset '{dataset_name}'")

    data_file = os.path.join(dataset_dir, file_name)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"[Dataset] Loading from: {data_file}")

    image_column = ds_config.get("columns", {}).get("images", "images")
    image_paths = []

    # Read the JSONL file
    with open(data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            try:
                record = json.loads(line.strip())
                images = record.get(image_column, [])
                if isinstance(images, str):
                    images = [images]
                for img_path in images:
                    if img_path:
                        # Convert relative paths to absolute paths
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(dataset_dir, img_path)
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                        else:
                            print(f"[Warning] Image not found: {img_path}")
            except Exception as e:
                print(f"[Warning] Failed to parse line {idx+1}: {e}")
                continue

    # Deduplicate paths
    image_paths = list(set(image_paths))
    print(f"[Dataset] Found {len(image_paths)} unique images")
    return image_paths


def get_cache_path(img_path: str, global_downsample: bool, semantic_patch_size: int) -> str:
    """Compute the cache file path for one image."""
    import hashlib
    
    # Read image and compute hash (same logic as semantic_patch_utils.py)
    image = Image.open(img_path).convert("RGB")
    image_bytes = image.tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
    
    # Cache directory
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "semantic_patch_cache"
    )
    
    # Cache filename format (consistent with semantic_patch_utils.py)
    # GLOBAL_DOWNSAMPLE_DIVISOR = 8 (paper: global branch uses 8x downsampling)
    cache_filename = f"{image_hash}_g{int(global_downsample)}_gd8_s{semantic_patch_size}.pt"
    
    return os.path.join(cache_dir, cache_filename)


def check_cache_exists(img_path: str, global_downsample: bool, semantic_patch_size: int) -> bool:
    """Check whether cache for one image already exists."""
    try:
        cache_path = get_cache_path(img_path, global_downsample, semantic_patch_size)
        return os.path.exists(cache_path)
    except Exception:
        return False


def pregenerate_cache(
    image_paths: List[str],
    image_processor: Qwen2VLImageProcessor,
    sam_model_raw,
    sam_model_dataparallel,
    mask_generator,
    semantic_patch_size: int = 32,
    global_downsample: bool = False,
    skip_existing: bool = True,
):
    """Pregenerate semantic patch cache for all images."""
    from llamafactory.utils.semantic_patch_utils import process_image_with_semantic_patches

    print(f"\n[Cache Generation] Starting...")
    print(f"  - semantic_patch_size: {semantic_patch_size}")
    print(f"  - global_downsample: {global_downsample}")
    print(f"  - total images: {len(image_paths)}")
    print(f"  - sam_model available: {sam_model_raw is not None}")
    print(f"  - skip_existing: {skip_existing}")

    # Pre-check: skip images that are already cached
    if skip_existing:
        print(f"\n[Cache Check] Checking existing cache...")
        uncached_paths = []
        cached_count = 0
        for img_path in tqdm(image_paths, desc="Checking cache"):
            if check_cache_exists(img_path, global_downsample, semantic_patch_size):
                cached_count += 1
            else:
                uncached_paths.append(img_path)
        
        print(f"[Cache Check] Already cached: {cached_count}")
        print(f"[Cache Check] Need to generate: {len(uncached_paths)}")
        
        if len(uncached_paths) == 0:
            print(f"\n[Cache Generation] All images already cached, nothing to do!")
            return
        
        image_paths = uncached_paths

    success_count = 0
    fail_count = 0

    print(f"\n[Cache Generation] Generating {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Generating cache"):
        try:
            image = Image.open(img_path).convert("RGB")
            _ = process_image_with_semantic_patches(
                image,
                image_processor,
                sam_model_raw=sam_model_raw,
                sam_model_dataparallel=sam_model_dataparallel,
                mask_generator=mask_generator,
                use_semantic_patches=True,
                global_downsample=global_downsample,
                semantic_patch_size=semantic_patch_size,
            )
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"\n[Error] Failed to process {img_path}: {e}")
            continue

    print(f"\n[Cache Generation] Completed!")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description="Pregenerate semantic patch cache")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config YAML path (e.g. examples/train_lora/qwen2_5vl_lora_sft.yaml)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Dataset directory (default: data)",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default="data/dataset_info.json",
        help="Path to dataset_info.json",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing; default: all)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID used by this process (for multi-GPU parallel runs)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of GPUs used (for multi-GPU parallel runs)",
    )
    args = parser.parse_args()

    # 1. Load config
    print(f"[Config] Loading from: {args.config}")
    config = load_yaml_config(args.config)

    # 2. Extract key parameters
    dataset_name = config.get("dataset")
    semantic_patch_size = config.get("semantic_patch_size", 32)
    global_downsample = config.get("global_downsample", False)
    sam_checkpoint = config.get("sam_checkpoint")
    sam_model_type = config.get("sam_model_type", "vit_h")
    sam_backend = config.get("sam_backend", "fastsam")
    model_name_or_path = config.get("model_name_or_path")
    max_samples_config = config.get("max_samples")

    if args.max_samples is None and max_samples_config is not None:
        args.max_samples = max_samples_config

    print(f"[Config] Dataset: {dataset_name}")
    print(f"[Config] semantic_patch_size: {semantic_patch_size}")
    print(f"[Config] global_downsample: {global_downsample}")
    print(f"[Config] sam_checkpoint: {sam_checkpoint}")
    print(f"[Config] sam_backend: {sam_backend}")
    print(f"[Config] sam_model_type: {sam_model_type}")

    # 3. Configure GPU (if gpu_id is provided)
    if args.gpu_id is not None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            # If CUDA_VISIBLE_DEVICES is already set externally, keep it and treat gpu_id as local rank
            device_id = args.gpu_id
            print(
                f"\n[GPU] Using local GPU {args.gpu_id} "
                f"(rank {args.gpu_id}/{args.num_gpus}, CUDA_VISIBLE_DEVICES={visible_devices})"
            )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            device_id = 0  # CUDA_VISIBLE_DEVICES is set, so actual local device becomes 0
            print(f"\n[GPU] Using GPU {args.gpu_id} (rank {args.gpu_id}/{args.num_gpus})")
    else:
        device_id = 0
        print(f"\n[GPU] Using default GPU 0")

    # 3.1 Set CUDA device for this process to avoid defaulting to cuda:0
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device_id)
        except Exception as e:
            print(f"[Warning] Failed to set CUDA device {device_id}: {e}")

    # 4. Initialize segmentation model (SAM / FastSAM)
    sam_model_raw = None
    sam_model_dataparallel = None
    mask_generator = None
    
    if sam_checkpoint and os.path.exists(sam_checkpoint):
        print(f"\n[{sam_backend.upper()}] Loading from: {sam_checkpoint}")
        from llamafactory.utils.sam_model_loader import load_sam_model
        
        sam_model_raw, sam_model_dataparallel, mask_generator = load_sam_model(
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            sam_backend=sam_backend,
            device_id=device_id,
        )
        
        if sam_model_raw is not None:
            print(f"[{sam_backend.upper()}] Loaded successfully")
        else:
            print(f"[Warning] {sam_backend.upper()} failed to load")
            print(f"[Warning] Will use regular tiles instead of semantic patches")
    else:
        print(f"[Warning] Segmentation checkpoint not found: {sam_checkpoint}")
        print(f"[Warning] Will use regular tiles instead of semantic patches")

    # 5. Initialize ImageProcessor (without loading the full model)
    if not model_name_or_path or not os.path.exists(model_name_or_path):
        raise ValueError(f"model_name_or_path not found: {model_name_or_path}")

    print(f"\n[ImageProcessor] Loading from: {model_name_or_path}")
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    print(f"[ImageProcessor] Loaded successfully")

    # 6. Collect image paths
    dataset_info = load_dataset_info(args.dataset_info)
    all_image_paths = collect_image_paths(
        dataset_name,
        args.dataset_dir,
        dataset_info,
        max_samples=args.max_samples,
    )

    if not all_image_paths:
        print("[Error] No images found in dataset!")
        return

    # 7. Shard images by GPU ID (each GPU handles a subset)
    if args.gpu_id is not None and args.num_gpus > 1:
        total_images = len(all_image_paths)
        images_per_gpu = (total_images + args.num_gpus - 1) // args.num_gpus
        start_idx = args.gpu_id * images_per_gpu
        end_idx = min(start_idx + images_per_gpu, total_images)
        image_paths = all_image_paths[start_idx:end_idx]
        print(f"\n[Distribution] Total images: {total_images}")
        print(f"[Distribution] This GPU ({args.gpu_id}) processes: {start_idx} ~ {end_idx-1} ({len(image_paths)} images)")
    else:
        image_paths = all_image_paths
        print(f"\n[Distribution] Processing all {len(image_paths)} images on single GPU")

    # 8. Pregenerate cache
    pregenerate_cache(
        image_paths,
        image_processor,
        sam_model_raw,
        sam_model_dataparallel,
        mask_generator,
        semantic_patch_size=semantic_patch_size,
        global_downsample=global_downsample,
    )


if __name__ == "__main__":
    main()
