# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Union
import importlib.util

# Import 4heat_sam_chicun.py
import os
# semantic_patch_utils.py is in src/llamafactory/utils/
# Need to import src/4heat_sam_chicun.py
# __file__ -> src/llamafactory/utils/semantic_patch_utils.py
# dirname 3 times -> src/
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAM_HEAT_PATH = os.path.join(_base_dir, "4heat_sam_chicun.py")
POSITION_QUANTIZATION = 4096
GLOBAL_DOWNSAMPLE_DIVISOR = 8  # paper: global branch downsamples by a factor of 8
_SEMANTIC_PATCH_CACHE: dict[tuple, dict] = {}
_CACHE_DIR = os.path.join(_base_dir, "semantic_patch_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# Global flag to track if orientation debug message has been printed (for multi-process safety)
_ORIENTATION_DEBUG_PRINTED = False

# Global flag to enable/disable debug prints (can be toggled by env vars)
# - HEATTOK_DEBUG=1: enable verbose debug prints (includes cache + gaussian/grid debug)
# - HEATTOK_CACHE_DEBUG=1: only show cache hit/miss logs (works even when HEATTOK_QUIET=1)
_ENABLE_DEBUG_PRINTS = os.environ.get("HEATTOK_DEBUG", "0").strip().lower() in ("1", "true")
_HEATTOK_CACHE_DEBUG = os.environ.get("HEATTOK_CACHE_DEBUG", "0").strip().lower() in ("1", "true")

# English comment.
_HEATTOK_QUIET = os.environ.get("HEATTOK_QUIET", "1").strip().lower() not in ("0", "false")


def _quantize_position(y_norm: float, x_norm: float) -> list[int]:
    y_idx = int(np.clip(round(y_norm * (POSITION_QUANTIZATION - 1)), 0, POSITION_QUANTIZATION - 1))
    x_idx = int(np.clip(round(x_norm * (POSITION_QUANTIZATION - 1)), 0, POSITION_QUANTIZATION - 1))
    return [y_idx, x_idx]


def _get_processor_patch_size(image_processor) -> Optional[int]:
    patch_size = getattr(image_processor, "patch_size", None)
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size
    config = getattr(image_processor, "image_processor_config", None)
    patch_size = getattr(config, "patch_size", None)
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size
    return None


def _normalize_tile_size(tile_size: int, image_processor) -> int:
    patch_size = _get_processor_patch_size(image_processor)
    if patch_size and tile_size % patch_size != 0:
        tile_size = int(math.ceil(tile_size / patch_size) * patch_size)
    return tile_size


def _check_cache_compatibility(cached_result: dict, global_downsample: bool) -> dict:
    """
    Check if old cache file correctly distinguishes global tokens and semantic tokens
    Returns a dictionary with:
    - is_compatible: Whether the cache is compatible
    - num_global_tokens: Inferred number of global tokens
    - num_semantic_tokens: Inferred number of semantic tokens
    - issues: List of issues found
    """
    result = {
        "is_compatible": True,
        "num_global_tokens": 0,
        "num_semantic_tokens": 0,
        "issues": []
    }
    
    if "patch_positions" not in cached_result or cached_result["patch_positions"] is None:
        result["is_compatible"] = False
        result["issues"].append("Missing patch_positions field")
        return result
    
    if "image_grid_thw" not in cached_result:
        result["is_compatible"] = False
        result["issues"].append("Missing image_grid_thw field")
        return result
    
    if "gaussian_params" not in cached_result or cached_result["gaussian_params"] is None:
        result["is_compatible"] = False
        result["issues"].append("Missing gaussian_params field")
        return result
    
    patch_positions = cached_result["patch_positions"]
    patch_extents = cached_result.get("patch_extents")
    image_grid_thw = cached_result["image_grid_thw"]
    gaussian_params = cached_result["gaussian_params"]
    
    # Calculate expected token counts
    total_tokens = patch_positions.shape[0]
    total_semantic_patches = gaussian_params.shape[0]
    total_grid_rows = image_grid_thw.shape[0]
    
    # Infer global token count
    num_global_rows = 0
    if global_downsample and total_grid_rows > total_semantic_patches:
        num_global_rows = total_grid_rows - total_semantic_patches
    
    # English comment.
    expected_global_tokens = 0
    if num_global_rows > 0:
        for idx in range(num_global_rows):
            grid_t, grid_h, grid_w = image_grid_thw[idx]
            expected_global_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
    
    # English comment.
    expected_semantic_tokens = 0
    for idx in range(num_global_rows, total_grid_rows):
        grid_t, grid_h, grid_w = image_grid_thw[idx]
        expected_semantic_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
    
    result["num_global_tokens"] = expected_global_tokens
    result["num_semantic_tokens"] = expected_semantic_tokens
    
    # English comment.
    expected_total = expected_global_tokens + expected_semantic_tokens
    if expected_total != total_tokens:
        result["is_compatible"] = False
        result["issues"].append(
            f"Token count mismatch: expected {expected_total}, actual {total_tokens} "
            f"(global:{expected_global_tokens}, semantic:{expected_semantic_tokens})"
        )
        return result
    
    # English comment.
    if expected_global_tokens > 0:
        global_positions = patch_positions[:expected_global_tokens].float() / 4095.0  # normalized positions
        global_extents = patch_extents[:expected_global_tokens].float() / 4095.0 if patch_extents is not None else None
        
        # English comment.
        # English comment.
        y_coords = global_positions[:, 0]
        x_coords = global_positions[:, 1]
        
        # English comment.
        y_std = y_coords.std().item()
        x_std = x_coords.std().item()
        
        # English comment.
        if y_std > 0.4 or x_std > 0.4:
            result["is_compatible"] = False
            result["issues"].append(
                f"Global token position encoding may be incorrect: y_std={y_std:.3f}, x_std={x_std:.3f} "
                f"(regular grid should be more concentrated)"
            )
        
        # Check extent: For regular grid, extent should be 1/grid_h * 1/grid_w
        if global_extents is not None:
            # English comment.
            extent_y = global_extents[:, 0]
            extent_x = global_extents[:, 1]
            extent_y_std = extent_y.std().item()
            extent_x_std = extent_x.std().item()
            
            # English comment.
            if extent_y_std > 0.01 or extent_x_std > 0.01:
                result["is_compatible"] = False
                result["issues"].append(
                    f"Global token extent may be incorrect: y_std={extent_y_std:.4f}, x_std={extent_x_std:.4f} "
                    f"(regular grid extent should be very consistent)"
                )
    
    # English comment.
    if expected_semantic_tokens > 0:
        semantic_positions = patch_positions[expected_global_tokens:].float() / 4095.0
        semantic_extents = patch_extents[expected_global_tokens:].float() / 4095.0 if patch_extents is not None else None
        
        # English comment.
        semantic_y_coords = semantic_positions[:, 0]
        semantic_x_coords = semantic_positions[:, 1]
        
        # English comment.
        # English comment.
        unique_positions = len(torch.unique(semantic_positions, dim=0))
        expected_unique = total_semantic_patches
        
        # English comment.
        if unique_positions > expected_unique * 2:
            result["is_compatible"] = False
            result["issues"].append(
                f"Semantic token position encoding may be incorrect: unique_positions={unique_positions}, "
                f"expected about {expected_unique} semantic patches"
            )
    
    return result


def _add_orientations_to_cached_result(cached_result: dict, original_image: Image.Image, global_downsample: bool) -> dict:
    """
    Add patch_orientations field to old cache file (compatibility handling)
    Calculate orientation angles from gaussian_params and distinguish global tokens and semantic tokens
    """
    if "gaussian_params" not in cached_result or cached_result["gaussian_params"] is None:
        return cached_result
    
    if "patch_positions" not in cached_result or cached_result["patch_positions"] is None:
        return cached_result
    
    if "image_grid_thw" not in cached_result:
        return cached_result
    
    gaussian_params = cached_result["gaussian_params"]
    image_grid_thw = cached_result["image_grid_thw"]
    patch_positions = cached_result["patch_positions"]
    
    # Calculate orientation angles
    gaussian_np = gaussian_params.detach().cpu().numpy()
    if gaussian_np.shape[1] < 5:
        return cached_result
    
    mu_x = gaussian_np[:, 0]
    mu_y = gaussian_np[:, 1]
    sigma_x = gaussian_np[:, 2]
    sigma_y = gaussian_np[:, 3]
    rho = gaussian_np[:, 4]
    
    sigma_x_sq = sigma_x ** 2
    sigma_y_sq = sigma_y ** 2
    denominator = sigma_x_sq - sigma_y_sq
    numerator = 2.0 * rho * sigma_x * sigma_y
    
    phi_k = np.where(
        np.abs(denominator) > 1e-6,
        0.5 * np.arctan2(numerator, denominator),
        np.zeros_like(mu_x)
    )
    
    # English comment.
    # English comment.
    num_global_rows = 0
    if global_downsample and image_grid_thw.shape[0] > 0:
        total_semantic_patches = gaussian_np.shape[0]
        total_grid_rows = image_grid_thw.shape[0]
        
        # English comment.
        total_tokens_from_grid = 0
        for idx in range(image_grid_thw.shape[0]):
            grid_t, grid_h, grid_w = image_grid_thw[idx]
            total_tokens_from_grid += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
        
        # English comment.
        # English comment.
        if total_grid_rows > total_semantic_patches:
            # English comment.
            num_global_rows = total_grid_rows - total_semantic_patches
            # English comment.
            num_global_rows = min(num_global_rows, total_grid_rows)
    
    # Build orientation angles list
    orientation_angles_list = []
    
    # 1. Add orientation angle 0 for global tokens
    if num_global_rows > 0:
        for idx in range(num_global_rows):
            grid_t, grid_h, grid_w = image_grid_thw[idx]
            num_global_tokens = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
            for _ in range(num_global_tokens):
                orientation_angles_list.append(0.0)
    
    # English comment.
    semantic_idx = 0
    for idx in range(num_global_rows, image_grid_thw.shape[0]):
        grid_t, grid_h, grid_w = image_grid_thw[idx]
        num_tokens = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
        if semantic_idx < len(phi_k):
            phi_k_value = phi_k[semantic_idx]
            for _ in range(num_tokens):
                orientation_angles_list.append(phi_k_value)
            semantic_idx += 1
    
    # Verify length (be tolerant: pad/truncate to match patch_positions)
    target_len = int(patch_positions.shape[0])
    if len(orientation_angles_list) != target_len:
        # English comment.
        # English comment.
        if _ENABLE_DEBUG_PRINTS:
            print(
                f"[Cache Compatibility Warning] patch_orientations length mismatch, auto-aligning "
                f"(orientations: {len(orientation_angles_list)}, positions: {target_len})"
            )
        if len(orientation_angles_list) < target_len:
            orientation_angles_list.extend([0.0] * (target_len - len(orientation_angles_list)))
        else:
            orientation_angles_list = orientation_angles_list[:target_len]

    cached_result["patch_orientations"] = torch.tensor(orientation_angles_list, dtype=torch.float32)
    # Count how many tokens have non-zero orientation (semantic tokens) - use numpy for efficiency
    orientation_array = np.array(orientation_angles_list)
    num_with_orientation = int(np.sum(np.abs(orientation_array) > 1e-6))
    num_zero_orientation = int(len(orientation_angles_list) - num_with_orientation)
    # Only print detailed info once per run to avoid flooding logs (use global flag for multi-process safety)
    global _ORIENTATION_DEBUG_PRINTED
    if _ENABLE_DEBUG_PRINTS and not _ORIENTATION_DEBUG_PRINTED:
        _ORIENTATION_DEBUG_PRINTED = True
        print(f"[Cache Compatibility] Adding patch_orientations to cached results:")
        print(
            f"  Example: {len(orientation_angles_list)} tokens "
            f"(global: {num_zero_orientation} with phi=0 -> regular grid encoding, "
            f"semantic: {num_with_orientation} with phi!=0 -> Gaussian+orientation encoding)"
        )
        print(f"[Cache Compatibility] (Subsequent additions will be silent to reduce log spam)")
    
    return cached_result


def _build_online_global_prefix(image: Image.Image, image_processor, global_downsample: bool) -> Optional[dict]:
    if not global_downsample:
        return None

    if GLOBAL_DOWNSAMPLE_DIVISOR == 1:
        image_inputs_global = image
    else:
        i_wh = image.size
        i_wh = [max(x // GLOBAL_DOWNSAMPLE_DIVISOR, 1) for x in i_wh]
        image_inputs_global = image.resize(i_wh)

    outputs_global = image_processor(image_inputs_global, return_tensors="pt")
    global_pixel_values = outputs_global["pixel_values"]
    global_grid_thw = outputs_global["image_grid_thw"]

    patch_position_entries: list[list[int]] = []
    patch_extent_entries: list[list[int]] = []
    orientation_entries: list[float] = []

    for grid_t, grid_h, grid_w in global_grid_thw.numpy():
        grid_t = int(grid_t)
        grid_h = max(int(grid_h), 1)
        grid_w = max(int(grid_w), 1)
        for _ in range(grid_t):
            for h_idx in range(grid_h):
                for w_idx in range(grid_w):
                    y_norm = (h_idx + 0.5) / grid_h
                    x_norm = (w_idx + 0.5) / grid_w
                    quantized = _quantize_position(y_norm, x_norm)
                    patch_position_entries.append(quantized)
                    # Keep global tokens equivalent to vanilla center-only MRoPE.
                    patch_extent_entries.append(quantized)
                    orientation_entries.append(0.0)

    return {
        "pixel_values": global_pixel_values,
        "image_grid_thw": global_grid_thw,
        "patch_positions": torch.tensor(patch_position_entries, dtype=torch.long),
        "patch_extents": torch.tensor(patch_extent_entries, dtype=torch.long),
        "patch_orientations": torch.tensor(orientation_entries, dtype=torch.float32),
    }


def _replace_cached_global_with_online(
    cached_result: dict, image: Image.Image, image_processor, global_downsample: bool
) -> dict:
    """
    Keep semantic tokens from cache, but always rebuild global tokens online.
    This allows reusing old semantic cache files while changing global resolution strategy.
    """
    if not global_downsample:
        return cached_result

    if "pixel_values" not in cached_result or "image_grid_thw" not in cached_result:
        return cached_result

    global_prefix = _build_online_global_prefix(image, image_processor, global_downsample)
    if global_prefix is None:
        return cached_result

    image_grid_thw = cached_result["image_grid_thw"]
    gaussian_params = cached_result.get("gaussian_params")
    total_grid_rows = int(image_grid_thw.shape[0]) if hasattr(image_grid_thw, "shape") else 0

    num_global_rows = 0
    if total_grid_rows > 0 and gaussian_params is not None and hasattr(gaussian_params, "shape"):
        semantic_rows = int(gaussian_params.shape[0])
        if total_grid_rows > semantic_rows:
            num_global_rows = total_grid_rows - semantic_rows
    if num_global_rows == 0 and total_grid_rows > 0 and cached_result.get("num_global_tokens", 0):
        # Backward-compatible fallback: old caches typically have one global row.
        num_global_rows = 1
    num_global_rows = max(min(num_global_rows, total_grid_rows), 0)

    if num_global_rows > 0:
        cached_global_grid = image_grid_thw[:num_global_rows]
        semantic_grid = image_grid_thw[num_global_rows:]
        cached_global_patch_tokens = int(cached_global_grid.prod(-1).sum().item())
    else:
        semantic_grid = image_grid_thw
        cached_global_patch_tokens = 0

    cached_pixels = cached_result["pixel_values"]
    if cached_global_patch_tokens > cached_pixels.shape[0]:
        return cached_result
    semantic_pixels = cached_pixels[cached_global_patch_tokens:]

    def _slice_semantic_tensor(name: str) -> Optional[torch.Tensor]:
        value = cached_result.get(name)
        if not isinstance(value, torch.Tensor):
            return None
        if cached_global_patch_tokens <= value.shape[0]:
            return value[cached_global_patch_tokens:]
        return value

    semantic_positions = _slice_semantic_tensor("patch_positions")
    semantic_extents = _slice_semantic_tensor("patch_extents")
    semantic_orientations = _slice_semantic_tensor("patch_orientations")

    merged = {
        "pixel_values": torch.cat([global_prefix["pixel_values"], semantic_pixels], dim=0),
        "image_grid_thw": torch.cat([global_prefix["image_grid_thw"], semantic_grid], dim=0),
    }
    if gaussian_params is not None:
        merged["gaussian_params"] = gaussian_params

    if semantic_positions is not None:
        merged["patch_positions"] = torch.cat([global_prefix["patch_positions"], semantic_positions], dim=0)
    if semantic_extents is not None:
        merged["patch_extents"] = torch.cat([global_prefix["patch_extents"], semantic_extents], dim=0)
    if semantic_orientations is not None:
        merged["patch_orientations"] = torch.cat([global_prefix["patch_orientations"], semantic_orientations], dim=0)

    merge_size = getattr(image_processor, "merge_size", 2)
    merge_square = max(int(merge_size) ** 2, 1)
    num_global_tokens = int(global_prefix["image_grid_thw"].prod(-1).sum().item()) // merge_square
    num_semantic_tokens = (
        int(semantic_grid.prod(-1).sum().item()) // merge_square if semantic_grid.numel() > 0 else 0
    )
    merged["num_global_tokens"] = num_global_tokens
    merged["num_semantic_tokens"] = num_semantic_tokens

    return merged


def load_sam_heat_module():
    """Load 4heat_sam_chicun.py module"""
    # Check if file exists
    if not os.path.exists(SAM_HEAT_PATH):
        # English comment.
        alt_base_dir = os.path.dirname(_base_dir)
        alt_path = os.path.join(alt_base_dir, "src", "4heat_sam_chicun.py")
        if os.path.exists(alt_path):
            actual_path = alt_path
        else:
            raise ImportError(f"Cannot find 4heat_sam_chicun.py. Tried: {SAM_HEAT_PATH}, {alt_path}")
    else:
        actual_path = SAM_HEAT_PATH
    
    spec = importlib.util.spec_from_file_location("sam_heat_module", actual_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {actual_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_image_with_semantic_patches(
    image: Image.Image,
    image_processor,
    sam_model_raw=None,
    sam_model_dataparallel=None,
    mask_generator=None,
    use_semantic_patches: bool = True,
    global_downsample: bool = True,
    semantic_patch_size: int = 32,
    merge_params: Optional[dict] = None,
) -> dict:

    if not use_semantic_patches:
        # English comment.
        return image_processor(image, return_tensors="pt")
    
    # English comment.
    try:
        sam_heat_module = load_sam_heat_module()
        SAMHeatDiffusionProcessor = sam_heat_module.SAMHeatDiffusionProcessor
    except Exception as e:
        print(f"Warning: Failed to load SAM+HeatDiffusion module, using tiles: {e}")
        return process_image_with_regular_tiles(image, image_processor, global_downsample)
    
    # English comment.
    if sam_model_raw is None or mask_generator is None:
        print("Warning: SAM model not available, using tiles")
        return process_image_with_regular_tiles(image, image_processor, global_downsample)
    
    import hashlib
    import os
    import torch
    
    # English comment.
    # English comment.
    # English comment.
    # English comment.
    
    # English comment.
    try:
        from torch.distributed import is_initialized, get_rank
        if is_initialized():
            rank = get_rank()
    except ImportError:
        rank = None
    
    # English comment.
    # English comment.
    image_bytes = image.tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
    
    # English comment.
    process_id = os.getpid()
    env_tmp_root = os.getenv("SAM_HEAT_TMP_DIR")
    if env_tmp_root:
        tmp_dir = os.path.join(env_tmp_root, f"semantic_patches_{process_id}")
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        # Use system temp; do not write visualization under semantic_patch_cache/.
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix=f"semantic_patches_{process_id}_")
    tmp_path = os.path.join(tmp_dir, f"temp_image_{image_hash}.png")

    cache_key = (
        image_hash,
        bool(global_downsample),
        GLOBAL_DOWNSAMPLE_DIVISOR,
        semantic_patch_size,
        True,
    )
    # English comment.
    # English comment.
    # if cache_key in _SEMANTIC_PATCH_CACHE:
    #     cached = _SEMANTIC_PATCH_CACHE[cache_key]
    #     if _ENABLE_DEBUG_PRINTS or _HEATTOK_CACHE_DEBUG:
    #         print(f"[SemanticPatch Cache] Reusing cached result for image {image_hash}")
    #     result = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cached.items()}
    if False:  # English comment.
        result = {}
        # Compatible with old cache files: if patch_orientations is missing, calculate from gaussian_params
        if "patch_orientations" not in result and "gaussian_params" in result and result["gaussian_params"] is not None:
            # Check compatibility (only print warnings to reduce log spam)
            compatibility_check = _check_cache_compatibility(result, global_downsample)
            if not compatibility_check["is_compatible"]:
                if _ENABLE_DEBUG_PRINTS:
                    num_global = compatibility_check["num_global_tokens"]
                    num_semantic = compatibility_check["num_semantic_tokens"]
                    total_tokens = result["patch_positions"].shape[0] if "patch_positions" in result else 0
                    print(f"[Cache Compatibility Warning] Old cache file may be incompatible:")
                    print(f"  Check result: global_tokens={num_global}, semantic_tokens={num_semantic}, total={total_tokens}")
                    for issue in compatibility_check["issues"]:
                        print(f"  - {issue}")
                    print(f"[Cache Compatibility] Suggestion: Delete old cache files and regenerate for best results")
            # Always add orientations (silently if compatible)
            result = _add_orientations_to_cached_result(result, image, global_downsample)
        result = _replace_cached_global_with_online(result, image, image_processor, global_downsample)
        return result

    cache_filename = (
        f"{image_hash}_g{int(global_downsample)}_gd{GLOBAL_DOWNSAMPLE_DIVISOR}_s{semantic_patch_size}.pt"
    )
    cache_path = os.path.join(_CACHE_DIR, cache_filename)
    legacy_cache_filename = f"{image_hash}_g{int(global_downsample)}_s{semantic_patch_size}.pt"
    legacy_cache_path = os.path.join(_CACHE_DIR, legacy_cache_filename)

    def _load_disk_cache(retries: int = 20, wait_seconds: float = 0.5):
        for attempt in range(retries + 1):
            for candidate_cache_path in (cache_path, legacy_cache_path):
                if os.path.exists(candidate_cache_path):
                    try:
                        if _ENABLE_DEBUG_PRINTS or _HEATTOK_CACHE_DEBUG:
                            print(f"[SemanticPatch Cache] Loading cached result from disk: {candidate_cache_path}")
                        disk_data = torch.load(candidate_cache_path, map_location="cpu", weights_only=False)
                        if isinstance(disk_data, dict):
                            # English comment.
                            result = {
                                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                for k, v in disk_data.items()
                            }
                            if "patch_orientations" not in result and "gaussian_params" in result and result["gaussian_params"] is not None:
                                result_before_add = result
                                result = _add_orientations_to_cached_result(result, image, global_downsample)
                                if "patch_orientations" not in result or result.get("patch_orientations") is None:
                                    print(f"[Cache Compatibility] Failed to add orientations to old cache, deleting: {candidate_cache_path}")
                                    try:
                                        os.remove(candidate_cache_path)
                                    except Exception:
                                        pass
                                    continue
                            result = _replace_cached_global_with_online(result, image, image_processor, global_downsample)
                            return result
                    except Exception as cache_err:
                        print(f"Error: Failed to load cached patch ({candidate_cache_path}): {cache_err}")
                        continue
            if attempt < retries:
                time.sleep(wait_seconds)
        return None

    disk_cached = _load_disk_cache()
    if disk_cached is not None:
        # Check cache compatibility
        if "patch_orientations" not in disk_cached:
            compatibility_check = _check_cache_compatibility(disk_cached, global_downsample)
            # Only print warnings to reduce log spam
            if not compatibility_check["is_compatible"]:
                num_global = compatibility_check["num_global_tokens"]
                num_semantic = compatibility_check["num_semantic_tokens"]
                total_tokens = disk_cached["patch_positions"].shape[0] if "patch_positions" in disk_cached else 0
                if _ENABLE_DEBUG_PRINTS:
                    print(f"[Cache Compatibility Warning] Old cache file may be incompatible:")
                    print(f"  Check result: global_tokens={num_global}, semantic_tokens={num_semantic}, total={total_tokens}")
                    for issue in compatibility_check["issues"]:
                        print(f"  - {issue}")
                    print(f"[Cache Compatibility] Deleting incompatible cache and regenerating")
                try:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    if os.path.exists(legacy_cache_path):
                        os.remove(legacy_cache_path)
                except Exception as del_err:
                    print(f"[Cache Compatibility] Failed to delete old cache: {del_err}")
                disk_cached = None
            else:
                # Always add orientations (silently if compatible)
                disk_cached = _add_orientations_to_cached_result(disk_cached, image, global_downsample)
                if "patch_orientations" not in disk_cached or disk_cached.get("patch_orientations") is None:
                    print(f"[Cache Compatibility] Failed to add orientations, deleting cache")
                    try:
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                        if os.path.exists(legacy_cache_path):
                            os.remove(legacy_cache_path)
                    except Exception:
                        pass
                    disk_cached = None
        if disk_cached is not None:
            disk_cached = _replace_cached_global_with_online(disk_cached, image, image_processor, global_downsample)
            return disk_cached

    # English comment.
    if not os.path.exists(tmp_path):
        image.save(tmp_path)
    
    try:
        worker_info = torch.utils.data.get_worker_info()
        force_online_in_worker = os.environ.get("HEATTOK_FORCE_ONLINE_IN_WORKER", "0").strip().lower() in ("1", "true")
        if worker_info is not None and not force_online_in_worker:
            cached = _load_disk_cache()
            if cached is not None:
                return cached
            if _ENABLE_DEBUG_PRINTS or _HEATTOK_CACHE_DEBUG:
                print(
                    "[SemanticPatch Cache] Worker mode with no cache; fallback to regular tiles. "
                    "Set HEATTOK_FORCE_ONLINE_IN_WORKER=1 to force online SAM+Heat generation in workers."
                )
            return process_image_with_regular_tiles(image, image_processor, global_downsample)
        # English comment.
        # English comment.
        # English comment.
        # English comment.
        # English comment.
        try:
            device_index = (
                sam_model_raw.device.index
                if hasattr(sam_model_raw, "device") and sam_model_raw.device and sam_model_raw.device.type == "cuda"
                else torch.cuda.current_device() if torch.cuda.is_available() else 0
            )
            processor = SAMHeatDiffusionProcessor(
                sam_model_raw=sam_model_raw,
                sam_model_dataparallel=sam_model_dataparallel,
                mask_generator=mask_generator,
                filename=tmp_path,
                output_dir_base=tmp_dir,
                device_id=device_index,
            )
        except RuntimeError as e:
            if "Cannot re-initialize CUDA" in str(e) or "fork" in str(e).lower():
                # English comment.
                print("Warning: Cannot re-initialize CUDA in fork/worker, trying cached patch")
                cached = _load_disk_cache()
                if cached is not None:
                    return cached
                print("Warning: No cached patch available, using tiles")
                return process_image_with_regular_tiles(image, image_processor, global_downsample)
            else:
                # English comment.
                raise
        
        # English comment.
        if merge_params is None:
            merge_params = {
                'sam_morph_kernel': 5,
                'min_size_threshold': 50,
                'post_min_size_threshold': 500,
                'post_max_size_threshold': 40000,
                'target_split_size': 15000,
                'sigma_T': 5.0,
                'sigma_C': 5.0,
                'alpha': 0.5,
                'K0': 0.5,
                'delta_t': 0.001,
                'diffusion_iterations': 30,
                'merge_threshold': 0.01,
                'save_visualization': False,
            }
        
        # English comment.
        merge_params['target_size'] = semantic_patch_size
        # Cache path: only keep .pt files; disable tmp visualization outputs.
        merge_params['save_visualization'] = False
        merge_params.pop('patch_output_dir', None)
        
        # Run SAM + merge pipeline
        patches_tensor, positions_tensor, gaussian_tensor = processor.run_sam_and_merge(**merge_params)
        
        if patches_tensor is None or patches_tensor.numel() == 0:
            print("Warning: No semantic patches generated, using tiles")
            return process_image_with_regular_tiles(image, image_processor, global_downsample)
        
        # English comment.
        result = convert_semantic_patches_to_qwen2vl_format(
            patches_tensor,
            image_processor,
            image,
            global_downsample=global_downsample,
            semantic_patch_size=semantic_patch_size,
            positions_tensor=positions_tensor,
            gaussian_params_tensor=gaussian_tensor,
        )
        cached_result = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in result.items()
        }
        # English comment.
        if (
            "patch_orientations" not in cached_result
            and cached_result.get("gaussian_params") is not None
            and "patch_positions" in cached_result
            and cached_result.get("patch_positions") is not None
        ):
            cached_result = _add_orientations_to_cached_result(cached_result, image, global_downsample)
        # English comment.
        # _SEMANTIC_PATCH_CACHE[cache_key] = cached_result

        try:
            disk_payload = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in cached_result.items()
            }
            os.makedirs(_CACHE_DIR, exist_ok=True)
            cache_name = os.path.basename(cache_path)
            tmp_cache_path = os.path.join(_CACHE_DIR, f".{cache_name}.{os.getpid()}.tmp")
            torch.save(disk_payload, tmp_cache_path)
            os.replace(tmp_cache_path, cache_path)
            print(f"[HeatTok] Saved cache: {cache_name}")
        except Exception as save_err:
            print(f"Error: Failed to save cached patch ({cache_path}): {save_err}")
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cached_result.items()}
    
    except Exception as e:
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return process_image_with_regular_tiles(image, image_processor, global_downsample)
    
    finally:
        # Remove temporary workspace; keep only .pt under semantic_patch_cache/.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            import shutil

            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def convert_semantic_patches_to_qwen2vl_format(
    semantic_patches: torch.Tensor,
    image_processor,
    original_image: Image.Image,
    global_downsample: bool = True,
    semantic_patch_size: int = 32,
    positions_tensor: Optional[torch.Tensor] = None,
    gaussian_params_tensor: Optional[torch.Tensor] = None,
) -> dict:

    # English comment.
    original_do_rescale = image_processor.do_rescale
    original_do_resize = image_processor.do_resize

    # English comment.
    image_processor.do_rescale = False
    image_processor.do_resize = False

    # Convert tensor patches to PIL images
    semantic_patches_list = []
    N, C, H, W = semantic_patches.shape

    # Normalize to [0, 1] if needed
    if semantic_patches.max() > 1.0:
        semantic_patches = semantic_patches / 255.0

    for i in range(N):
        # Convert one patch to PIL image
        patch_np = semantic_patches[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        patch_np = (patch_np * 255).astype(np.uint8)
        patch_img = Image.fromarray(patch_np)
        semantic_patches_list.append(patch_img)

    # English comment.
    merge_size = getattr(image_processor, "merge_size", 2)
    merge_square = max(int(merge_size) ** 2, 1)
    num_patches = len(semantic_patches_list)
    if num_patches > 0 and num_patches % merge_square != 0:
        padding_needed = merge_square - (num_patches % merge_square)
        last_patch = semantic_patches_list[-1]
        for _ in range(padding_needed):
            semantic_patches_list.append(last_patch.copy())

    # English comment.
    new_pixel_values = []
    semantic_grid_list: list[np.ndarray] = []

    for patch_img in semantic_patches_list:
        temp_out = image_processor(patch_img, return_tensors="pt")
        new_pixel_values.append(temp_out["pixel_values"])
        if "image_grid_thw" in temp_out:
            grid_np = temp_out["image_grid_thw"].numpy()
            if grid_np.ndim == 1:
                grid_np = grid_np.reshape(1, -1)
            semantic_grid_list.append(grid_np.astype(np.int64))

    # English comment.
    image_processor.do_rescale = original_do_rescale
    image_processor.do_resize = original_do_resize

    # English comment.
    if new_pixel_values:
        new_pixel_np_list = [pv.numpy() for pv in new_pixel_values]
        new_pixel_np = np.concatenate(new_pixel_np_list, axis=0)  # (total_patches, 588)
    else:
        new_pixel_np = np.empty((0, 588), dtype=np.float32)

    # English comment.
    outputs_global = None
    if global_downsample:
        if GLOBAL_DOWNSAMPLE_DIVISOR == 1:
            image_inputs_global = original_image
        else:
            i_wh = original_image.size
            i_wh = [max(x // GLOBAL_DOWNSAMPLE_DIVISOR, 1) for x in i_wh]
            image_inputs_global = original_image.resize(i_wh)

        outputs_global = image_processor(image_inputs_global, return_tensors="pt")
        global_pixel_values = outputs_global["pixel_values"].numpy()

        new_pixel_np = np.concatenate([global_pixel_values, new_pixel_np], axis=0)

    # Build image_grid_thw
    if semantic_grid_list:
        semantic_grid_thw = np.concatenate(semantic_grid_list, axis=0)
    else:
        semantic_grid_thw = np.empty((0, 3), dtype=np.int64)

    if global_downsample and outputs_global is not None:
        global_grid_thw = outputs_global["image_grid_thw"].numpy()
        final_grid_thw = np.concatenate([global_grid_thw, semantic_grid_thw], axis=0)
    else:
        final_grid_thw = semantic_grid_thw

    # English comment.
    # English comment.
    # Should equal sum(t * h * w for each grid_thw row)
    expected_patches = int(final_grid_thw.prod(-1).sum())
    actual_patches = new_pixel_np.shape[0]
    if expected_patches != actual_patches:
        print(f"Warning: pixel_values count ({actual_patches}) doesn't match image_grid_thw expected ({expected_patches})")
        print(f"  image_grid_thw shape: {final_grid_thw.shape}")
        print(f"  image_grid_thw: {final_grid_thw}")
        print(f"  pixel_values shape: {new_pixel_np.shape}")
        print(f"  semantic patches: {len(semantic_patches_list)}")
        print(f"  global_downsample: {global_downsample}")

        # English comment.
        # English comment.
        # English comment.
        if actual_patches > 0:
            # English comment.
            # For 32x32: 2x2=4 patches (patch_size=16)
            # For 64x64: 4x4=16 patches (patch_size=16)
            patch_size = _get_processor_patch_size(image_processor) or 16
            if semantic_patch_size == 32:
                patches_per_semantic = 2 * 2  # 2x2 grid
            elif semantic_patch_size == 64:
                patches_per_semantic = 4 * 4  # 4x4 grid
            else:
                patches_per_semantic = (semantic_patch_size // patch_size) ** 2

            # English comment.
            if global_downsample and outputs_global is not None:
                global_patches = outputs_global["pixel_values"].shape[0]
            else:
                global_patches = 0

            expected_total = len(semantic_patches_list) * patches_per_semantic + global_patches
            print(f"  Expected total patches: {expected_total} (semantic: {len(semantic_patches_list)}*{patches_per_semantic}, global: {global_patches})")

            if actual_patches == expected_total:
                # English comment.
                # English comment.
                print("  Fixing: Recalculating image_grid_thw to match patches")
                # English comment.
                semantic_grid_thw_fixed = []
                for _ in range(len(semantic_patches_list)):
                    if semantic_patch_size == 32:
                        semantic_grid_thw_fixed.append([1, 2, 2])
                    elif semantic_patch_size == 64:
                        semantic_grid_thw_fixed.append([1, 4, 4])
                    else:
                        grid_size = semantic_patch_size // patch_size
                        semantic_grid_thw_fixed.append([1, grid_size, grid_size])

                if global_downsample and outputs_global is not None:
                    global_grid_thw = outputs_global["image_grid_thw"].numpy()
                    if global_grid_thw.ndim == 1:
                        global_grid_thw = global_grid_thw.reshape(1, -1)
                    final_grid_thw = np.concatenate([global_grid_thw, np.array(semantic_grid_thw_fixed)], axis=0)
                else:
                    final_grid_thw = np.array(semantic_grid_thw_fixed)

                # Verify
                expected_patches_fixed = int(final_grid_thw.prod(-1).sum())
                if expected_patches_fixed == actual_patches:
                    print(f"  Fixed: Updated image_grid_thw, patches={expected_patches_fixed}")
                else:
                    print(f"  Error: Updated image_grid_thw patches={expected_patches_fixed} != actual={actual_patches}")
                    raise ValueError(f"Failed to fix pixel_values and image_grid_thw mismatch")
            else:
                raise ValueError(f"pixel_values count ({actual_patches}) doesn't match expected ({expected_total})")
        else:
            raise ValueError("pixel_values is empty")

    pixel_values_tensor = torch.from_numpy(new_pixel_np)
    image_grid_thw_tensor = torch.from_numpy(final_grid_thw)

    patch_positions_tensor = None
    original_width, original_height = original_image.size

    # Build per-token patch_positions and patch_extents
    # English comment.
    # English comment.
    patch_position_entries: list[list[int]] = []
    patch_extent_entries: list[list[int]] = []
    num_global_rows = 0
    
    # English comment.
    if global_downsample and outputs_global is not None:
        global_grid_thw = outputs_global["image_grid_thw"].numpy()
        num_global_rows = global_grid_thw.shape[0]
        # English comment.
        # English comment.
        # Global tokens use regular grid position encoding (e.g., 14x14 or 28x28 patches)
        for grid_t, grid_h, grid_w in global_grid_thw:
            grid_t = int(grid_t)
            grid_h = max(int(grid_h), 1)
            grid_w = max(int(grid_w), 1)
            for _ in range(grid_t):
                for h_idx in range(grid_h):
                    for w_idx in range(grid_w):
                        # English comment.
                        y_norm = (h_idx + 0.5) / grid_h
                        x_norm = (w_idx + 0.5) / grid_w
                        # English comment.
                        patch_position_entries.append(_quantize_position(y_norm, x_norm))
                        # Keep global tokens equivalent to vanilla center-only MRoPE:
                        # set extent == center so alpha/beta mixing collapses to center encoding.
                        patch_extent_entries.append(_quantize_position(y_norm, x_norm))

    # English comment.
    gaussian_np = None
    if gaussian_params_tensor is not None and gaussian_params_tensor.numel() != 0:
        gaussian_np = gaussian_params_tensor.detach().cpu().numpy()

    centers_np = None
    extents_np = None
    if gaussian_np is not None:
        mu_x = gaussian_np[:, 0]
        mu_y = gaussian_np[:, 1]
        sigma_x = gaussian_np[:, 2]
        sigma_y = gaussian_np[:, 3]
        # English comment.
        rho = gaussian_np[:, 4] if gaussian_np.shape[1] >= 5 else np.zeros_like(mu_x)
        
        # English comment.
        safe_width = max(float(original_width), 1.0)
        safe_height = max(float(original_height), 1.0)
        centers_np = np.stack([mu_y / safe_height, mu_x / safe_width], axis=1)
        # English comment.
        extent_x_norm = np.clip((2.0 * sigma_x) / safe_width, 0.0, 1.0)
        extent_y_norm = np.clip((2.0 * sigma_y) / safe_height, 0.0, 1.0)
        extents_np = np.stack([extent_y_norm, extent_x_norm], axis=1)
    elif positions_tensor is not None and positions_tensor.numel() != 0:
        centers_np = positions_tensor.detach().cpu().numpy()

    # English comment.
    # English comment.
    orientation_angles_np = None
    original_centers_np = None  # English comment.
    original_extents_np = None  # English comment.
    
    if gaussian_np is not None and gaussian_np.shape[1] >= 5:
        # English comment.
        mu_x_orig = gaussian_np[:, 0]
        mu_y_orig = gaussian_np[:, 1]
        sigma_x_orig = gaussian_np[:, 2]
        sigma_y_orig = gaussian_np[:, 3]
        rho_orig = gaussian_np[:, 4]
        
        sigma_x_sq_orig = sigma_x_orig ** 2
        sigma_y_sq_orig = sigma_y_orig ** 2
        denominator_orig = sigma_x_sq_orig - sigma_y_sq_orig
        numerator_orig = 2.0 * rho_orig * sigma_x_orig * sigma_y_orig
        
        phi_k_orig = np.where(
            np.abs(denominator_orig) > 1e-6,
            0.5 * np.arctan2(numerator_orig, denominator_orig),
            np.zeros_like(mu_x_orig)
        )
        orientation_angles_np = phi_k_orig
        
        # English comment.
        safe_width = max(float(original_width), 1.0)
        safe_height = max(float(original_height), 1.0)
        original_centers_np = np.stack([mu_y_orig / safe_height, mu_x_orig / safe_width], axis=1)
        extent_x_norm_orig = np.clip((2.0 * sigma_x_orig) / safe_width, 0.0, 1.0)
        extent_y_norm_orig = np.clip((2.0 * sigma_y_orig) / safe_height, 0.0, 1.0)
        original_extents_np = np.stack([extent_y_norm_orig, extent_x_norm_orig], axis=1)
    
    # English comment.
    semantic_index = 0
    for idx in range(num_global_rows, len(final_grid_thw)):
        grid_t, grid_h, grid_w = final_grid_thw[idx]
        grid_t = int(grid_t)
        grid_h = max(int(grid_h), 1)
        grid_w = max(int(grid_w), 1)
        num_tokens = grid_t * grid_h * grid_w
        
        # English comment.
        if original_centers_np is not None and semantic_index < len(original_centers_np):
            center_y, center_x = original_centers_np[semantic_index]
            y_norm = float(np.clip(center_y, 0.0, 1.0))
            x_norm = float(np.clip(center_x, 0.0, 1.0))
        elif centers_np is not None and semantic_index < len(centers_np):
            center_y, center_x = centers_np[semantic_index]
            y_norm = float(np.clip(center_y, 0.0, 1.0))
            x_norm = float(np.clip(center_x, 0.0, 1.0))
        else:
            y_norm = 0.5
            x_norm = 0.5
        
        if original_extents_np is not None and semantic_index < len(original_extents_np):
            extent_y_norm, extent_x_norm = map(float, original_extents_np[semantic_index])
        elif extents_np is not None and semantic_index < len(extents_np):
            extent_y_norm, extent_x_norm = map(float, extents_np[semantic_index])
        else:
            extent_y_norm = 0.0
            extent_x_norm = 0.0
        
        for _ in range(num_tokens):
            # English comment.
            patch_position_entries.append(_quantize_position(y_norm, x_norm))
            patch_extent_entries.append(_quantize_position(extent_y_norm, extent_x_norm))
        semantic_index += 1

    # English comment.
    if patch_position_entries:
        if len(patch_position_entries) == pixel_values_tensor.shape[0]:
            patch_positions_tensor = torch.tensor(patch_position_entries, dtype=torch.long)
            if not hasattr(convert_semantic_patches_to_qwen2vl_format, "_position_encoding_debug_printed"):
                convert_semantic_patches_to_qwen2vl_format._position_encoding_debug_printed = True
                # Calculate actual token counts
                num_global_tokens = 0
                if global_downsample and outputs_global is not None:
                    global_grid_thw = outputs_global["image_grid_thw"].numpy()
                    for grid_t, grid_h, grid_w in global_grid_thw:
                        num_global_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                num_semantic_tokens = pixel_values_tensor.shape[0] - num_global_tokens
                if _ENABLE_DEBUG_PRINTS and not _HEATTOK_QUIET:
                    print(f"[Position Encoding] Token distribution:")
                    print(f"  - Global tokens: {num_global_tokens} (use regular grid position encoding, no orientation)")
                    print(f"  - Semantic tokens: {num_semantic_tokens} (use Gaussian position encoding with orientation)")
                    print(f"  - Total tokens: {pixel_values_tensor.shape[0]}")
                    print(f"[Position Encoding] Position encoding types are correctly separated (not mixed)")
        else:
            print(
                "Warning: patch_positions count doesn't match pixel_values, "
                f"positions={len(patch_position_entries)}, pixels={pixel_values_tensor.shape[0]}"
            )

    result = {
        "pixel_values": pixel_values_tensor,
        "image_grid_thw": image_grid_thw_tensor,
    }
    # English comment.
    # English comment.
    # English comment.
    # English comment.
    if patch_positions_tensor is not None:
        result["patch_positions"] = patch_positions_tensor
    if patch_extent_entries and len(patch_extent_entries) == pixel_values_tensor.shape[0]:
        result["patch_extents"] = torch.tensor(patch_extent_entries, dtype=torch.long)
    if gaussian_params_tensor is not None:
        result["gaussian_params"] = gaussian_params_tensor
    # English comment.
    # English comment.
    if orientation_angles_np is not None:
        orientation_angles_list = []
        
        # English comment.
        if global_downsample and outputs_global is not None:
            global_grid_thw = outputs_global["image_grid_thw"].numpy()
            for grid_t, grid_h, grid_w in global_grid_thw:
                grid_t = int(grid_t)
                grid_h = max(int(grid_h), 1)
                grid_w = max(int(grid_w), 1)
                num_global_tokens = grid_t * grid_h * grid_w
                # English comment.
                for _ in range(num_global_tokens):
                    orientation_angles_list.append(0.0)
        
        # 2. Add calculated orientation angles for semantic patches tokens
        semantic_idx = 0
        for idx in range(num_global_rows, len(final_grid_thw)):
            grid_t, grid_h, grid_w = final_grid_thw[idx]
            num_tokens = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
            if semantic_idx < len(orientation_angles_np):
                phi_k_value = orientation_angles_np[semantic_idx]
                for _ in range(num_tokens):
                    orientation_angles_list.append(phi_k_value)
                semantic_idx += 1
        
        if orientation_angles_list:
            # English comment.
            target_len = int(pixel_values_tensor.shape[0])
            if len(orientation_angles_list) != target_len:
                if _ENABLE_DEBUG_PRINTS:
                    print(
                        f"[Orientation Align] patch_orientations length mismatch, auto-aligning "
                        f"(orientations: {len(orientation_angles_list)}, pixels: {target_len})"
                    )
                if len(orientation_angles_list) < target_len:
                    orientation_angles_list.extend([0.0] * (target_len - len(orientation_angles_list)))
                else:
                    orientation_angles_list = orientation_angles_list[:target_len]
            result["patch_orientations"] = torch.tensor(orientation_angles_list, dtype=torch.float32)

    # English comment.
    pixel_count = result["pixel_values"].shape[0]
    if "patch_positions" in result and result["patch_positions"].shape[0] != pixel_count:
        raise ValueError(
            f"[SelfCheck] patch_positions length ({result['patch_positions'].shape[0]}) "
            f"!= pixel_values length ({pixel_count})"
        )
    if "patch_extents" in result and result["patch_extents"].shape[0] != pixel_count:
        raise ValueError(
            f"[SelfCheck] patch_extents length ({result['patch_extents'].shape[0]}) "
            f"!= pixel_values length ({pixel_count})"
        )
    if "patch_orientations" in result and result["patch_orientations"].shape[0] != pixel_count:
        raise ValueError(
            f"[SelfCheck] patch_orientations length ({result['patch_orientations'].shape[0]}) "
            f"!= pixel_values length ({pixel_count})"
        )

    # LLM-side token counts (after merge_size aggregation).
    merge_size = getattr(image_processor, "merge_size", 2)
    merge_square = max(int(merge_size) ** 2, 1)
    num_global_tokens = 0
    if global_downsample and outputs_global is not None:
        global_grid_thw = outputs_global["image_grid_thw"].numpy()
        for grid_t, grid_h, grid_w in global_grid_thw:
            num_global_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
    num_global_tokens = num_global_tokens // merge_square
    num_semantic_tokens = max(pixel_count // merge_square - num_global_tokens, 0)
    result["num_global_tokens"] = int(num_global_tokens)
    result["num_semantic_tokens"] = int(num_semantic_tokens)

    return result


def process_image_with_regular_tiles(
    image: Image.Image,
    image_processor,
    global_downsample: bool = True,
    tile_size: int = 56,
) -> dict:
    tile_size = _normalize_tile_size(tile_size, image_processor)
    image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
    cache_key = (
        image_hash,
        bool(global_downsample),
        GLOBAL_DOWNSAMPLE_DIVISOR,
        tile_size,
        False,
    )
    cache_filename = (
        f"{image_hash}_g{int(global_downsample)}_gd{GLOBAL_DOWNSAMPLE_DIVISOR}_t{tile_size}.pt"
    )
    cache_path = os.path.join(_CACHE_DIR, cache_filename)
    # English comment.
    # if cache_key in _SEMANTIC_PATCH_CACHE:
    #     cached = _SEMANTIC_PATCH_CACHE[cache_key]
    if False:  # English comment.
        cached = {}
        if _ENABLE_DEBUG_PRINTS or _HEATTOK_CACHE_DEBUG:
            print(f"[SemanticPatch Cache] Reusing cached tile result for image {cache_key[0]}")
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cached.items()}

    # English comment.
    image_inputs_patches, cols, rows = split_image_to_tiles(image, tile_size)
    
    # English comment.
    original_do_rescale = image_processor.do_rescale
    original_do_resize = image_processor.do_resize
    
    # English comment.
    image_processor.do_rescale = False
    image_processor.do_resize = False
    
    # English comment.
    new_pixel = []
    tile_grid_list: list[np.ndarray] = []
    for tile in image_inputs_patches:
        temp_out = image_processor(tile, return_tensors="pt")
        new_pixel.append(temp_out["pixel_values"])
        if "image_grid_thw" in temp_out:
            grid_np = temp_out["image_grid_thw"].numpy()
            if grid_np.ndim == 1:
                grid_np = grid_np.reshape(1, -1)
            tile_grid_list.append(grid_np.astype(np.int64))

    # English comment.
    image_processor.do_rescale = original_do_rescale
    image_processor.do_resize = original_do_resize

    # Flatten pixel_values
    new_pixel = np.array([pv.numpy() for pv in new_pixel])  # (N, num_patches_per_tile, 588)
    new_pixel = new_pixel.reshape(-1, new_pixel.shape[-1])  # (total_patches, 588)

    # English comment.
    if global_downsample:
        if GLOBAL_DOWNSAMPLE_DIVISOR == 1:
            image_inputs_global = image
        else:
            i_wh = image.size
            i_wh = [max(x // GLOBAL_DOWNSAMPLE_DIVISOR, 1) for x in i_wh]
            image_inputs_global = image.resize(i_wh)
        outputs_global = image_processor(image_inputs_global, return_tensors="pt")
        global_pixel_values = outputs_global["pixel_values"].numpy()

        new_pixel = np.concatenate([global_pixel_values, new_pixel], axis=0)
    else:
        outputs_global = None

    # English comment.
    if tile_grid_list:
        tile_grid_thw = np.concatenate(tile_grid_list, axis=0)
    else:
        tile_grid_thw = np.empty((0, 3), dtype=np.int64)

    if global_downsample and outputs_global is not None:
        global_grid_thw = outputs_global["image_grid_thw"].numpy()
        final_grid_thw = np.concatenate([global_grid_thw, tile_grid_thw], axis=0)
    else:
        global_grid_thw = np.empty((0, 3))
        final_grid_thw = tile_grid_thw

    patch_position_entries: list[list[int]] = []
    patch_extent_entries: list[list[int]] = []
    # English comment.
    for grid_t, grid_h, grid_w in global_grid_thw:
        grid_t = int(grid_t)
        grid_h = max(int(grid_h), 1)
        grid_w = max(int(grid_w), 1)
        for _ in range(grid_t):
            for h_idx in range(grid_h):
                for w_idx in range(grid_w):
                    y_norm = (h_idx + 0.5) / grid_h
                    x_norm = (w_idx + 0.5) / grid_w
                    patch_position_entries.append(_quantize_position(y_norm, x_norm))
                    # Keep global tokens equivalent to vanilla center-only MRoPE.
                    patch_extent_entries.append(_quantize_position(y_norm, x_norm))

    image_width, image_height = image.size
    for tile_idx in range(len(tile_grid_thw)):
        row = tile_idx // cols
        col = tile_idx % cols
        left = col * tile_size
        upper = row * tile_size
        tile_width = min(tile_size, image_width - left) if image_width > 0 else tile_size
        tile_height = min(tile_size, image_height - upper) if image_height > 0 else tile_size
        center_x = left + tile_width / 2.0
        center_y = upper + tile_height / 2.0
        if image_width > 0:
            x_norm = float(np.clip(center_x / float(image_width), 0.0, 1.0))
        else:
            x_norm = 0.5
        if image_height > 0:
            y_norm = float(np.clip(center_y / float(image_height), 0.0, 1.0))
        else:
            y_norm = 0.5
        extent_x_norm = float(np.clip(tile_width / float(image_width) if image_width > 0 else 0.0, 0.0, 1.0))
        extent_y_norm = float(np.clip(tile_height / float(image_height) if image_height > 0 else 0.0, 0.0, 1.0))
        grid_t, grid_h, grid_w = tile_grid_thw[tile_idx]
        grid_t = int(grid_t)
        grid_h = max(int(grid_h), 1)
        grid_w = max(int(grid_w), 1)
        token_extent_y = extent_y_norm / float(grid_h)
        token_extent_x = extent_x_norm / float(grid_w)
        for _ in range(grid_t * grid_h * grid_w):
            patch_position_entries.append(_quantize_position(y_norm, x_norm))
            patch_extent_entries.append(_quantize_position(token_extent_y, token_extent_x))

    patch_positions_tensor = None
    if patch_position_entries and len(patch_position_entries) == new_pixel.shape[0]:
        patch_positions_tensor = torch.tensor(patch_position_entries, dtype=torch.long)
    patch_extents_tensor = None
    if patch_extent_entries and len(patch_extent_entries) == new_pixel.shape[0]:
        patch_extents_tensor = torch.tensor(patch_extent_entries, dtype=torch.long)

    # Calculate global and semantic token counts
    num_global_tokens = 0
    num_semantic_tokens = 0
    if global_downsample and outputs_global is not None:
        global_grid_thw = outputs_global["image_grid_thw"].numpy()
        for grid_t, grid_h, grid_w in global_grid_thw:
            num_global_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
    
    for grid_t, grid_h, grid_w in tile_grid_thw:
        num_semantic_tokens += int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
    
    # Apply merge_size to get actual token counts
    merge_size = getattr(image_processor, "merge_size", 2)
    merge_square = merge_size ** 2
    num_global_tokens = num_global_tokens // merge_square
    num_semantic_tokens = num_semantic_tokens // merge_square
    
    result = {
        "pixel_values": torch.from_numpy(new_pixel),
        "image_grid_thw": torch.from_numpy(final_grid_thw),
        "num_global_tokens": num_global_tokens,
        "num_semantic_tokens": num_semantic_tokens,
    }
    if patch_positions_tensor is not None:
        result["patch_positions"] = patch_positions_tensor
    if patch_extents_tensor is not None:
        result["patch_extents"] = patch_extents_tensor

    cached_result = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
    # English comment.
    # _SEMANTIC_PATCH_CACHE[cache_key] = cached_result

    # English comment.
    try:
        disk_payload = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in cached_result.items()}
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_name = os.path.basename(cache_path)
        tmp_cache_path = os.path.join(_CACHE_DIR, f".{cache_name}.{os.getpid()}.tmp")
        torch.save(disk_payload, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
    except Exception as save_err:
        print(f"Error: Failed to save cached patch ({cache_path}): {save_err}")

    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cached_result.items()}


def split_image_to_tiles(image: Image.Image, tile_size: int = 56) -> Tuple[List[Image.Image], int, int]:
    """English docstring."""
    width, height = image.size
    
    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    
    tiles = []
    
    for row in range(rows):
        for col in range(cols):
            left = col * tile_size
            upper = row * tile_size
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)
            
            tile = image.crop((left, upper, right, lower))
            
            # English comment.
            if tile.size != (tile_size, tile_size):
                new_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                new_tile.paste(tile, (0, 0))
                tile = new_tile
            
            tiles.append(tile)
    
    return tiles, cols, rows

