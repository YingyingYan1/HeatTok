# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, csv, json, time, pathlib, threading, sys
# Avoid OpenBLAS segfault: set before any numpy/scipy load
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "4")
# Reduce 4heat_sam_chicun print spam (WARN/ERROR only; set to INFO/DEBUG for verbose)
os.environ["SAM_HEAT_LOG_LEVEL"] = os.environ.get("SAM_HEAT_LOG_LEVEL"
, "ERROR")
# UTF-8 stdout for correct console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import importlib.util
import inspect
import math
import warnings
import logging
from PIL import Image

# Allow local absolute paths in HuggingFace Hub (avoid HFValidationError for local checkpoint)
try:
    import huggingface_hub.utils._validators as _hf_validators
    _validate_repo_id_orig = getattr(_hf_validators, "validate_repo_id", None)
    if _validate_repo_id_orig is not None:
        def _validate_repo_id_patch(repo_id: str):
            if not repo_id:
                _validate_repo_id_orig(repo_id)
                return
            if repo_id.startswith("/") or "\\" in repo_id or (len(repo_id) > 1 and repo_id[1] == ":"):
                return  # treat as local path, skip validation
            _validate_repo_id_orig(repo_id)
        _hf_validators.validate_repo_id = _validate_repo_id_patch
except Exception:
    pass

# Add src for llamafactory imports when profiling
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LOCAL_QWEN2_5_VL_PATH = _PROJECT_ROOT / "qwen2_5_vl"
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _patch_local_qwen2_5_vl() -> None:
    """Load customized qwen2_5_vl (G-MRoPE) from project root."""
    local_dir = _LOCAL_QWEN2_5_VL_PATH.resolve()
    required_files = (
        "__init__.py",
        "configuration_qwen2_5_vl.py",
        "modeling_qwen2_5_vl.py",
        "processing_qwen2_5_vl.py",
    )
    missing = [name for name in required_files if not (local_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing qwen2_5_vl files under {local_dir}: {missing}")

    package_name = "transformers.models.qwen2_5_vl"
    for module_name in list(sys.modules.keys()):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        package_name,
        local_dir / "__init__.py",
        submodule_search_locations=[str(local_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local qwen2_5_vl from {local_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    import transformers.models as transformers_models

    setattr(transformers_models, "qwen2_5_vl", module)

# Suppress Transformers warnings about unused model_kwargs (image_patch_orientations is used via **kwargs)
# This warning is harmless as the parameter is correctly passed through **kwargs
warnings.filterwarnings("ignore", message=".*model_kwargs.*")
warnings.filterwarnings("ignore", message=".*not used by the model.*")
warnings.filterwarnings("ignore", message=".*image_patch_orientations.*")
# Also suppress at logger level - set before importing transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- GPU & OpenBLAS ---
# Default to 8-GPU evaluation; can override from shell via CUDA_VISIBLE_DEVICES.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
REQUIRED_NUM_GPUS = 8

import torch
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Please check your PyTorch CUDA installation.")

_patch_local_qwen2_5_vl()
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.utils import logging as transformers_logging
from peft import PeftModel
from tqdm import tqdm

# --- Distributed eval (torchrun) ---
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Suppress Transformers warnings about unused model_kwargs (image_patch_orientations is used via **kwargs)
# This warning is harmless as the parameter is correctly passed through **kwargs
# Set verbosity to ERROR to suppress warnings
transformers_logging.set_verbosity_error()
# Also suppress specific loggers
transformers_logging.get_logger("transformers.modeling_utils").setLevel(transformers_logging.ERROR)
transformers_logging.get_logger("transformers.generation.utils").setLevel(transformers_logging.ERROR)
transformers_logging.disable_default_handler()
transformers_logging.enable_explicit_format()

# --- Path config (relative to project root) ---
BASE_MODEL_PATH = _PROJECT_ROOT / "Qwen2.5-VL-7B-Instruct"
LORA_MODEL_ID = _PROJECT_ROOT / "outputs/VRSBENCH-qwen2.5vl-7B-heatok/checkpoint-7865"
HF_MODEL_ID = LORA_MODEL_ID
DATA_JSON = _PROJECT_ROOT / "data/VRSBench_EVAL_vqa.json"
IMAGES_DIR = _PROJECT_ROOT / "VRSBench/Images_val/Images_val"
OUT_DIR = _PROJECT_ROOT / "val/vrsbench-qwen2.5vl-7B-heatok-checkpoint-7865-5epoch"
BATCH_SIZE = 1
MAX_WORKERS = 1
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
RETRIES = 1
BACKOFF = 1.5
TASK_MODE = "generic"  # full VRSBench_EVAL_vqa.json (mixed question types)
ENFORCE_INTEGER_ONLY_PROMPT = (TASK_MODE == "quantity")
ENFORCE_CATEGORY_ONLY_PROMPT = (TASK_MODE == "category")

# HeatTok semantic patch / VRSBench settings (aligned with training yaml + SEMANTIC_CACHE.md)
SEMANTIC_CACHE_DIR = _PROJECT_ROOT / "src/semantic_patch_cache_vrsbench"
os.environ.setdefault("HEATTOK_SEMANTIC_CACHE_DIR", str(SEMANTIC_CACHE_DIR))
SEMANTIC_GLOBAL_DOWNSAMPLE = True  # g1: global branch enabled
GLOBAL_DOWNSAMPLE_DIVISOR = 8  # gd8: global branch downsampled by 8x
SEMANTIC_PATCH_SIZE = 28
SEMANTIC_IMAGE_MAX_PIXELS = 512 * 512  # 262144, VRSBench training resolution
SEMANTIC_IMAGE_MIN_PIXELS = 32 * 32

# VRSBench heat-diffusion defaults (SEMANTIC_CACHE.md)
VRSBENCH_MERGE_PARAMS = {
    "sam_morph_kernel": 5,
    "min_size_threshold": 50,
    "post_min_size_threshold": 200,
    "post_max_size_threshold": 20000,
    "target_split_size": 2500,
    "sigma_T": 9.0,
    "sigma_C": 12.0,
    "alpha": 0.4,
    "K0": 1.0,
    "delta_t": 0.001,
    "diffusion_iterations": 20,
    "merge_threshold": 0.01,
}

# FastSAM (VRSBench default backend in SEMANTIC_CACHE.md)
SAM_BACKEND = "fastsam"
SAM_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT",
    str(_PROJECT_ROOT / "FastSAM-main/weights/FastSAM-x.pt"),
)
SAM_MODEL_TYPE = "vit_h"
# --- Profiling: per-step time and GPU memory ---
PROFILE_INFERENCE_STEPS = False
NUM_PROFILE_IMAGES = 5

MODEL = None
PROCESSOR = None
MODEL_LOCK = threading.Lock()  # Serialize concurrent generate() calls
SAM_RAW = None
SAM_DATAPARALLEL = None
MASK_GENERATOR = None


def _get_model_primary_device() -> torch.device:
    if MODEL is not None:
        device = getattr(MODEL, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(MODEL.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_dtype() -> torch.dtype:
    if MODEL is not None:
        model_dtype = getattr(MODEL, "dtype", None)
        if model_dtype is not None:
            return model_dtype
        try:
            return next(MODEL.parameters()).dtype
        except StopIteration:
            pass
    return torch.float32


def _preprocess_image_for_cache(image_path: pathlib.Path) -> Image.Image:
    if PROCESSOR is None:
        raise RuntimeError("Processor must be loaded before computing semantic cache hashes.")

    with Image.open(image_path) as img:
        image = img.copy()

    image_processor = getattr(PROCESSOR, "image_processor", None)
    image_max_pixels = getattr(image_processor, "image_max_pixels", SEMANTIC_IMAGE_MAX_PIXELS)
    image_min_pixels = getattr(image_processor, "image_min_pixels", SEMANTIC_IMAGE_MIN_PIXELS)

    width, height = image.size
    if (width * height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (width * height))
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        image = image.resize((width, height))
        width, height = image.size

    if (width * height) < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / (width * height))
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _semantic_cache_candidates(image_hash: str) -> List[pathlib.Path]:
    g = int(SEMANTIC_GLOBAL_DOWNSAMPLE)
    s = SEMANTIC_PATCH_SIZE
    names = [
        f"{image_hash}_g{g}_gd{GLOBAL_DOWNSAMPLE_DIVISOR}_s{s}.pt",
        f"{image_hash}_g{g}_s{s}.pt",  # legacy cache name (VRSBench caches use this)
    ]
    return [SEMANTIC_CACHE_DIR / name for name in names]


def _compute_semantic_cache_path(image_path: pathlib.Path) -> pathlib.Path:
    if not SEMANTIC_CACHE_DIR.exists():
        SEMANTIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    processed_image = _preprocess_image_for_cache(image_path)
    image_hash = hashlib.md5(processed_image.tobytes()).hexdigest()[:8]
    for cache_path in _semantic_cache_candidates(image_hash):
        if cache_path.exists():
            return cache_path
    return _semantic_cache_candidates(image_hash)[0]


def _load_semantic_cache(image_path: pathlib.Path) -> dict:
    if not SEMANTIC_CACHE_DIR.exists():
        SEMANTIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    processed_image = _preprocess_image_for_cache(image_path)
    image_hash = hashlib.md5(processed_image.tobytes()).hexdigest()[:8]
    cache_path = None
    for candidate in _semantic_cache_candidates(image_hash):
        if candidate.exists():
            cache_path = candidate
            break
    if cache_path is None:
        tried = ", ".join(p.name for p in _semantic_cache_candidates(image_hash))
        raise FileNotFoundError(f"Semantic cache not found for {image_path.name}. Tried: {tried}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid semantic cache format: {cache_path}")
    # [HeatTok] Verify cache contains G-MRoPE patch metadata
    has_positions = "patch_positions" in cache
    has_extents = "patch_extents" in cache
    if has_positions and has_extents:
        if not hasattr(_load_semantic_cache, "_cache_verified"):
            # print(f"[SemanticPatch Cache] Loaded semantic patch (HeatTok): {cache_path.name}")
            # print(f"[SemanticPatch Cache]   - Has patch_positions and patch_extents (G-MRoPE ready)")
            _load_semantic_cache._cache_verified = True
    elif has_positions or has_extents:
        if not hasattr(_load_semantic_cache, "_cache_partial"):
            # print(f"[SemanticPatch Cache] Warning: partial patch metadata in cache: {cache_path.name}")
            _load_semantic_cache._cache_partial = True
    return cache


def _build_semantic_tensors(cache: dict) -> Dict[str, torch.Tensor]:
    if "pixel_values" not in cache or "image_grid_thw" not in cache:
        raise KeyError("Cache must contain 'pixel_values' and 'image_grid_thw'")

    pixel_values = cache["pixel_values"]
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.clone().detach()
    else:
        pixel_values = torch.as_tensor(pixel_values)
    pixel_values = pixel_values.to(torch.float32).contiguous()

    image_grid_thw = cache["image_grid_thw"]
    if isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.clone().detach()
    else:
        image_grid_thw = torch.as_tensor(image_grid_thw)
    image_grid_thw = image_grid_thw.to(torch.long)

    merge_size = getattr(getattr(PROCESSOR, "image_processor", None), "merge_size", 2) if PROCESSOR else 2
    merge_square = merge_size ** 2 if merge_size else 4
    if image_grid_thw.numel() > 0:
        tokens_per_entry = torch.prod(image_grid_thw, dim=1)
        tokens_total = int(tokens_per_entry.sum().item())
        tokens_after_merge = torch.div(tokens_per_entry + merge_square - 1, merge_square, rounding_mode="floor")
        token_total_after_merge = int(tokens_after_merge.sum().item())
    else:
        tokens_total = 0
        token_total_after_merge = 0
    if tokens_total != pixel_values.shape[0]:
        print(
            f"Warning: token count mismatch (pixel_values={pixel_values.shape[0]}, grid_total={tokens_total})"
        )

    result: Dict[str, torch.Tensor] = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    result["image_token_count"] = token_total_after_merge

    patch_positions = cache.get("patch_positions")
    if patch_positions is not None:
        if isinstance(patch_positions, torch.Tensor):
            patch_positions = patch_positions.clone().detach()
        else:
            patch_positions = torch.as_tensor(patch_positions)
        if patch_positions.numel() > 0 and patch_positions.shape[0] == pixel_values.shape[0]:
            result["image_patch_positions"] = patch_positions.to(torch.long)
        else:
            print(
                "Warning: patch_positions length does not match pixel_values, skipping G-MRoPE positions",
                patch_positions.shape if patch_positions.numel() > 0 else None,
                pixel_values.shape,
            )

    patch_extents = cache.get("patch_extents")
    if patch_extents is not None:
        if isinstance(patch_extents, torch.Tensor):
            patch_extents = patch_extents.clone().detach()
        else:
            patch_extents = torch.as_tensor(patch_extents)
        if patch_extents.numel() > 0 and patch_extents.shape[0] == pixel_values.shape[0]:
            result["image_patch_extents"] = patch_extents.to(torch.long)
        else:
            print(
                "Warning: patch_extents length does not match pixel_values, skipping G-MRoPE extents",
                patch_extents.shape if patch_extents.numel() > 0 else None,
                pixel_values.shape,
            )

    # [!! G-MRoPE Orientation !!] Handle patch_orientations (support old cache compatibility)
    patch_orientations = cache.get("patch_orientations")
    if patch_orientations is None:
        # Try to calculate from gaussian_params (compatibility with old cache)
        gaussian_params = cache.get("gaussian_params")
        if gaussian_params is not None:
            try:
                import numpy as np
                gaussian_np = gaussian_params.detach().cpu().numpy() if isinstance(gaussian_params, torch.Tensor) else gaussian_params
                if gaussian_np.shape[1] >= 5:
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
                    
                    # Build orientation angles list (global tokens: 0.0, semantic tokens: phi_k)
                    orientation_angles_list = []
                    total_semantic_patches = gaussian_np.shape[0]
                    total_grid_rows = image_grid_thw.shape[0]
                    num_global_rows = max(0, total_grid_rows - total_semantic_patches) if SEMANTIC_GLOBAL_DOWNSAMPLE else 0
                    
                    # Add 0.0 for global tokens
                    if num_global_rows > 0:
                        for idx in range(num_global_rows):
                            grid_t, grid_h, grid_w = image_grid_thw[idx]
                            num_global_tokens = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                            for _ in range(num_global_tokens):
                                orientation_angles_list.append(0.0)
                    
                    # Add phi_k for semantic tokens
                    semantic_idx = 0
                    for idx in range(num_global_rows, total_grid_rows):
                        grid_t, grid_h, grid_w = image_grid_thw[idx]
                        num_tokens = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                        if semantic_idx < len(phi_k):
                            phi_k_value = phi_k[semantic_idx]
                            for _ in range(num_tokens):
                                orientation_angles_list.append(phi_k_value)
                            semantic_idx += 1
                    
                    if len(orientation_angles_list) == pixel_values.shape[0]:
                        patch_orientations = torch.tensor(orientation_angles_list, dtype=torch.float32)
                        
                        # Also calculate global and semantic token counts for text sequence separation
                        merge_size = getattr(getattr(PROCESSOR, "image_processor", None), "merge_size", 2) if PROCESSOR else 2
                        merge_square = merge_size ** 2 if merge_size else 4
                        
                        num_global_tokens_after_merge = 0
                        num_semantic_tokens_after_merge = 0
                        
                        # Calculate global tokens after merge
                        if num_global_rows > 0:
                            for idx in range(num_global_rows):
                                grid_t, grid_h, grid_w = image_grid_thw[idx]
                                tokens_before_merge = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                                tokens_after_merge = tokens_before_merge // merge_square
                                num_global_tokens_after_merge += tokens_after_merge
                        
                        # Calculate semantic tokens after merge
                        for idx in range(num_global_rows, total_grid_rows):
                            grid_t, grid_h, grid_w = image_grid_thw[idx]
                            tokens_before_merge = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                            tokens_after_merge = tokens_before_merge // merge_square
                            num_semantic_tokens_after_merge += tokens_after_merge
                        
                        # Store in result for text sequence processing
                        result["num_global_tokens"] = num_global_tokens_after_merge
                        result["num_semantic_tokens"] = num_semantic_tokens_after_merge
            except Exception as e:
                # If calculation fails, patch_orientations remains None
                pass
    
    # Handle case when cache already has patch_orientations
    if patch_orientations is not None:
        # Cache already has patch_orientations, but we still need to calculate token counts if not present
        if "num_global_tokens" not in result or "num_semantic_tokens" not in result:
            gaussian_params = cache.get("gaussian_params")
            if gaussian_params is not None:
                try:
                    import numpy as np
                    gaussian_np = gaussian_params.detach().cpu().numpy() if isinstance(gaussian_params, torch.Tensor) else gaussian_params
                    total_semantic_patches = gaussian_np.shape[0] if len(gaussian_np.shape) > 0 else 0
                    total_grid_rows = image_grid_thw.shape[0]
                    num_global_rows = max(0, total_grid_rows - total_semantic_patches) if SEMANTIC_GLOBAL_DOWNSAMPLE else 0
                    
                    merge_size = getattr(getattr(PROCESSOR, "image_processor", None), "merge_size", 2) if PROCESSOR else 2
                    merge_square = merge_size ** 2 if merge_size else 4
                    
                    num_global_tokens_after_merge = 0
                    num_semantic_tokens_after_merge = 0
                    
                    # Calculate global tokens after merge
                    if num_global_rows > 0:
                        for idx in range(num_global_rows):
                            grid_t, grid_h, grid_w = image_grid_thw[idx]
                            tokens_before_merge = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                            tokens_after_merge = tokens_before_merge // merge_square
                            num_global_tokens_after_merge += tokens_after_merge
                    
                    # Calculate semantic tokens after merge
                    for idx in range(num_global_rows, total_grid_rows):
                        grid_t, grid_h, grid_w = image_grid_thw[idx]
                        tokens_before_merge = int(grid_t) * max(int(grid_h), 1) * max(int(grid_w), 1)
                        tokens_after_merge = tokens_before_merge // merge_square
                        num_semantic_tokens_after_merge += tokens_after_merge
                    
                    result["num_global_tokens"] = num_global_tokens_after_merge
                    result["num_semantic_tokens"] = num_semantic_tokens_after_merge
                except Exception:
                    pass
    
    if patch_orientations is not None:
        if isinstance(patch_orientations, torch.Tensor):
            patch_orientations = patch_orientations.clone().detach()
        else:
            patch_orientations = torch.as_tensor(patch_orientations)
        if patch_orientations.numel() > 0 and patch_orientations.shape[0] == pixel_values.shape[0]:
            result["image_patch_orientations"] = patch_orientations.to(torch.float32)
        else:
            print(
                "Warning: patch_orientations length does not match pixel_values, skipping orientations",
                patch_orientations.shape if patch_orientations.numel() > 0 else None,
                pixel_values.shape,
            )

    return result

# ----------------------------- Load model -----------------------------

def load_model_once(device: str | None = None):
    global MODEL, PROCESSOR
    if MODEL is not None:
        return

    model_path = str(pathlib.Path(HF_MODEL_ID).resolve())
    use_local_dir = os.path.isdir(model_path)

    # When loading from local dir: patch cached_file/cached_files so transformers read from disk
    _cached_file_orig = _cached_files_orig = None
    if use_local_dir:
        from transformers.utils import hub as tf_hub
        _cached_file_orig = tf_hub.cached_file
        _cached_files_orig = getattr(tf_hub, "cached_files", None)
        def _is_local_dir(p):
            if p is None:
                return False
            p = os.path.abspath(os.path.normpath(str(p).rstrip("/")))
            return os.path.isdir(p) and (p == model_path or p.startswith("/") or "\\" in str(p))
        def _cached_file_local(path_or_repo_id, filename, *args, **kwargs):
            if _is_local_dir(path_or_repo_id):
                full = os.path.join(path_or_repo_id, filename)
                if os.path.isfile(full):
                    return full
            return _cached_file_orig(path_or_repo_id, filename, *args, **kwargs)
        tf_hub.cached_file = _cached_file_local
        if _cached_files_orig is not None:
            def _cached_files_local(path_or_repo_id, filenames, *args, **kwargs):
                if _is_local_dir(path_or_repo_id):
                    out = [os.path.join(path_or_repo_id, f) for f in filenames if os.path.isfile(os.path.join(path_or_repo_id, f))]
                    if len(out) == len(filenames):
                        return out
                return _cached_files_orig(path_or_repo_id, filenames, *args, **kwargs)
            tf_hub.cached_files = _cached_files_local

    if device is None:
        device = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            torch.cuda.set_device(int(str(device).split(":")[1]))
        except Exception:
            pass
    # Log visible GPUs (commented out to reduce multi-process log spam)
    # if torch.cuda.is_available():
    #     print(f"CUDA visible GPU count: {torch.cuda.device_count()}")
    #     if torch.cuda.device_count() < 2:
    #          print("Note: CUDA_VISIBLE_DEVICES may be set to fewer devices; PyTorch sees "
    #                f"{torch.cuda.device_count()} GPU(s)")
    #     for i in range(torch.cuda.device_count()):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # else:
    #     print("Warning: CUDA unavailable, falling back to CPU (not recommended)")
    
    # Detect LoRA checkpoint and load base model + adapter
    # LoRA checkpoints include adapter_config.json
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    processor_source = str(BASE_MODEL_PATH.resolve())
    if os.path.exists(adapter_config_path):
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", str(BASE_MODEL_PATH))
        if not os.path.isabs(base_model_path):
            base_model_path = str((_PROJECT_ROOT / base_model_path).resolve())
        processor_source = base_model_path
        # print(f"[LoRA] Detected LoRA checkpoint, loading base model: {base_model_path}")
        base_local = os.path.isdir(base_model_path) or base_model_path.startswith("/") or "\\" in base_model_path
        
        # Load base model
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=base_local,
        )
        # Load LoRA adapter
        MODEL = PeftModel.from_pretrained(base_model, model_path, local_files_only=use_local_dir)
        # print(f"[LoRA] Loaded LoRA adapter from {model_path}")
    else:
        # Full model checkpoint (no LoRA)
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=use_local_dir,
        )
        # print(f"[Full Model] Loaded full model from {model_path}")
    
    PROCESSOR = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=os.path.isdir(processor_source),
    )

    # Restore original cached_file/cached_files so other code is unaffected
    if use_local_dir and _cached_file_orig is not None:
        from transformers.utils import hub as tf_hub
        tf_hub.cached_file = _cached_file_orig
        if _cached_files_orig is not None:
            tf_hub.cached_files = _cached_files_orig

    image_processor = getattr(PROCESSOR, "image_processor", None)
    if image_processor is not None:
        image_processor.image_max_pixels = SEMANTIC_IMAGE_MAX_PIXELS
        image_processor.image_min_pixels = SEMANTIC_IMAGE_MIN_PIXELS
    if hasattr(PROCESSOR, "image_max_pixels"):
        PROCESSOR.image_max_pixels = SEMANTIC_IMAGE_MAX_PIXELS
    if hasattr(PROCESSOR, "image_min_pixels"):
        PROCESSOR.image_min_pixels = SEMANTIC_IMAGE_MIN_PIXELS

    # flash_attention_2 is set via attn_implementation in from_pretrained

    # Log model device placement (commented out to reduce multi-process log spam)
    # print("Model loaded.")
    # print("Parameter devices (sample):")
    # device_set = set()
    # for name, param in MODEL.named_parameters():
    #     device_set.add(str(param.device))
    # print(f"Unique parameter devices: {device_set}")
    
    # [HeatTok] Verify customized G-MRoPE model is loaded
    try:
        model_file = inspect.getfile(Qwen2_5_VLForConditionalGeneration)
    except Exception:
        model_file = None
    if model_file:
        # print(f"[HeatTok] Model source file: {model_file}")
        with open(model_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'image_patch_positions' in content and 'image_patch_extents' in content:
                pass  # print(f"[HeatTok] Confirmed HeatTok G-MRoPE model is active")
            else:
                print(f"[HeatTok] Warning: model file may not include HeatTok G-MRoPE support")
    else:
        print("[HeatTok] Warning: could not locate model file; HeatTok support not verified")
    
    # [HeatTok] Report semantic cache directory status
    # print(f"[HeatTok] Semantic cache dir: {SEMANTIC_CACHE_DIR}")
    if SEMANTIC_CACHE_DIR.exists():
        pass  # cache_files = list(SEMANTIC_CACHE_DIR.glob("*.pt"))
        # print(f"[HeatTok] Found {len(cache_files)} cache file(s)")
    else:
        print(f"[HeatTok] Cache directory does not exist: {SEMANTIC_CACHE_DIR}")


def _abs_file_url(p: pathlib.Path) -> str:
    return f"file://{p.resolve().as_posix()}"

def _build_user_text(question: str, q_type: str | None = None) -> str:
    text = (
        f"Task: Answer the following question about the image.\n"
        f"Question: {question}\n"
        f"Return only the final answer."
    )
    q_type_text = (q_type or "").lower()
    if ENFORCE_INTEGER_ONLY_PROMPT or ("quantity" in q_type_text):
        text += "\nAnswer with one integer only."
    if ENFORCE_CATEGORY_ONLY_PROMPT or ("category" in q_type_text):
        text += "\nAnswer with one category label only."
    return text


def vqa_infer(image_path: pathlib.Path, question: str, q_type: str) -> str:
    user_text = _build_user_text(question, q_type)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": _abs_file_url(image_path)},
        ],
    }]

    semantic_data = None
    try:
        semantic_cache = _load_semantic_cache(image_path)
        # [HeatTok] Build vision tensors from semantic cache
        if semantic_cache is not None:
            try:
                semantic_data = _build_semantic_tensors(semantic_cache)
                # Verify HeatTok G-MRoPE tensors are present
                has_positions = "image_patch_positions" in semantic_data
                has_extents = "image_patch_extents" in semantic_data
                if has_positions and has_extents:
                    if not hasattr(vqa_infer, "_heatok_loaded"):
                        # print(f"[HeatTok] Using semantic cache with G-MRoPE: {image_path.name}")
                        vqa_infer._heatok_loaded = True
                else:
                    if not hasattr(vqa_infer, "_heatok_warned"):
                        # print(f"[HeatTok] Warning: incomplete semantic cache: {image_path.name}")
                        vqa_infer._heatok_warned = True
            except Exception as err:
                if not hasattr(vqa_infer, "_heatok_error"):
                    print(f"[HeatTok] Failed to build semantic tensors ({image_path.name}): {err}")
                    vqa_infer._heatok_error = True
                semantic_data = None
    except FileNotFoundError:
        if not hasattr(vqa_infer, "_heatok_missing"):
            print(f"[HeatTok] Semantic cache not found, falling back to regular patches: {image_path.name}")
            vqa_infer._heatok_missing = True
    except Exception as cache_err:
        if not hasattr(vqa_infer, "_heatok_cache_err"):
            print(f"[HeatTok] Cache load error ({image_path.name}): {cache_err}")
            vqa_infer._heatok_cache_err = True

    # Build prompt and tokenize
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # [!! G-MRoPE Orientation + Token Separation !!] Handle text sequence with separated global and semantic tokens
    if semantic_data is not None:
        num_global_tokens = semantic_data.get("num_global_tokens")
        num_semantic_tokens = semantic_data.get("num_semantic_tokens")
        
        # Check if we should separate global and semantic tokens in text sequence
        if num_global_tokens is not None and num_semantic_tokens is not None and num_global_tokens > 0 and num_semantic_tokens > 0:
            # Separate global and semantic tokens with two pairs of vision tokens
            vision_start = "<|vision_start|>"
            vision_end = "<|vision_end|>"
            image_token = "<|image_pad|>"
            placeholder = f"{vision_start}{image_token}{vision_end}"
            replacement = (
                f"{vision_start}{image_token * num_global_tokens}{vision_end}"
                f"{vision_start}{image_token * num_semantic_tokens}{vision_end}"
            )
            text = text.replace(placeholder, replacement, 1)
        else:
            # Fallback to original behavior: single pair of vision tokens
            image_token_count = semantic_data.get("image_token_count")
            if image_token_count is not None and image_token_count > 0:
                vision_start = "<|vision_start|>"
                vision_end = "<|vision_end|>"
                image_token = "<|image_pad|>"
                placeholder = f"{vision_start}{image_token}{vision_end}"
                replacement = f"{vision_start}{image_token * image_token_count}{vision_end}"
                text = text.replace(placeholder, replacement, 1)
    inputs = PROCESSOR.tokenizer(text=[text], padding=True, return_tensors="pt")

    target_device = _get_model_primary_device()
    model_dtype = _get_model_dtype()

    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(target_device)

    if "cache_position" not in inputs:
        seq_len = inputs.get("input_ids", torch.empty(1, 0, device=target_device)).shape[-1]
        if seq_len > 0:
            cache_position = torch.arange(seq_len, device=target_device, dtype=torch.long)
            inputs["cache_position"] = cache_position

    if "position_ids" not in inputs and "input_ids" in inputs:
        seq_len = inputs["input_ids"].shape[-1]
        position_ids = torch.arange(seq_len, device=target_device, dtype=torch.long).unsqueeze(0)
        inputs["position_ids"] = position_ids

    if semantic_data is not None:
        # [HeatTok] Inject semantic patch pixel values and G-MRoPE tensors
        pixel_values = semantic_data.pop("pixel_values").to(target_device, dtype=model_dtype).contiguous()
        inputs["pixel_values"] = pixel_values
        semantic_data.pop("image_token_count", None)
        # Remove non-tensor values that shouldn't be passed to the model
        semantic_data.pop("num_global_tokens", None)
        semantic_data.pop("num_semantic_tokens", None)
        
        # [HeatTok G-MRoPE] Log tensor shapes once
        has_positions = "image_patch_positions" in semantic_data
        has_extents = "image_patch_extents" in semantic_data
        has_orientations = "image_patch_orientations" in semantic_data
        
        if has_positions and has_extents:
            if not hasattr(vqa_infer, "_heatok_gmrope_used"):
                # print(f"[HeatTok] Using semantic patches (HeatTok) + G-MRoPE encoding")
                # print(f"[HeatTok]   - patch_positions: {semantic_data['image_patch_positions'].shape}")
                # print(f"[HeatTok]   - patch_extents: {semantic_data['image_patch_extents'].shape}")
                # if has_orientations:
                #     print(f"[HeatTok]   - patch_orientations: {semantic_data['image_patch_orientations'].shape} (with orientation-aware encoding)")
                # else:
                #     print(f"[HeatTok]   - patch_orientations: None (using basic G-MRoPE without orientation)")
                vqa_infer._heatok_gmrope_used = True
        
        for key, tensor in semantic_data.items():
            if isinstance(tensor, torch.Tensor):
                inputs[key] = tensor.to(target_device)
            # Skip non-tensor values (should already be removed above, but check for safety)
    else:
        # [Fallback] Regular grid patches without HeatTok
        if not hasattr(vqa_infer, "_regular_patch_used"):
            # print(f"[HeatTok] Using regular patches (no HeatTok semantic cache)")
            vqa_infer._regular_patch_used = True

    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(target_device, dtype=model_dtype).contiguous()

    with MODEL_LOCK, torch.inference_mode():
        # [HeatTok G-MRoPE] Pass patch metadata through generate()
        # image_patch_positions and image_patch_extents are consumed during prefill only
        # Transformers may warn they are unused in later decode steps; that is expected

        # Build generate() kwargs
        generate_params = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        }
        # Only pass temperature when sampling is enabled
        if TEMPERATURE > 0:
            generate_params["temperature"] = TEMPERATURE
        
        # Standard text inputs for generate()
        standard_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "position_ids": inputs.get("position_ids"),
            "cache_position": inputs.get("cache_position"),
        }
        # Drop None entries
        standard_inputs = {k: v for k, v in standard_inputs.items() if v is not None}
        
        # Vision inputs (passed via model_kwargs to prepare_inputs_for_generation)
        model_kwargs = {
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }
        # [G-MRoPE] Positions, extents, and orientations
        if "image_patch_positions" in inputs:
            model_kwargs["image_patch_positions"] = inputs["image_patch_positions"]
        if "image_patch_extents" in inputs:
            model_kwargs["image_patch_extents"] = inputs["image_patch_extents"]
        if "image_patch_orientations" in inputs:
            model_kwargs["image_patch_orientations"] = inputs["image_patch_orientations"]
        
        # Drop None entries
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        
        # [HeatTok G-MRoPE] Retry without patch params if Transformers rejects them
        # Transformers may not accept image_patch_positions/extents in all generation steps
        # Strategy: Try with all parameters first, if warning/error occurs, retry without problematic parameters
        # These parameters are needed in prefill stage but may cause warnings in subsequent generation steps
        gen_kwargs = {**standard_inputs, **generate_params}
        gen_kwargs.update(model_kwargs)
        
        # Parameters that might cause "not used by the model" warnings in generation steps
        # They are needed in prefill but may not be needed in subsequent steps
        patch_params = ["image_patch_positions", "image_patch_extents", "image_patch_orientations"]
        
        # First attempt: Use all parameters (including patch parameters for prefill)
        # These parameters are needed in the prefill stage but may cause warnings/errors in subsequent steps
        try:
            out_ids = MODEL.generate(**gen_kwargs)
        except (ValueError, TypeError) as e:
            # If error occurs related to unused model_kwargs, retry without patch parameters
            error_msg = str(e)
            if "model_kwargs" in error_msg and ("not used by the model" in error_msg or "unexpected keyword" in error_msg):
                # Remove patch parameters and retry
                # This works because these parameters are only needed in prefill, not in subsequent generation steps
                gen_kwargs_retry = {k: v for k, v in gen_kwargs.items() if k not in patch_params}
                out_ids = MODEL.generate(**gen_kwargs_retry)
            else:
                # Re-raise if it's a different error
                raise
    
    # Decode generated tokens
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    output = PROCESSOR.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output

# ----------------------------- Profiling: per-step time and model params -----------------------------
# Hand-calculated params for non-neural steps (from 4heat_sam_chicun.py):
# - diffusion_merge: 18 (Sobel 3x3 x2) + 6 (sigma_T, sigma_C, alpha, K0, delta_t, merge_threshold) = 24, non-learnable
# - clipping_kmeans: 0 (sklearn KMeans has no fixed weights; centers fitted per run)
# - tensor_conversion: 0 (format conversion only)
DIFFUSION_MERGE_PARAMS = 24
CLIPPING_KMEANS_PARAMS = 0
TENSOR_CONVERSION_PARAMS = 0


def _format_params(num_params: int) -> str:
    """Format param count as human-readable string (e.g. 136M, 7B)."""
    if num_params >= 1e9:
        return "{:.1f}B".format(num_params / 1e9)
    if num_params >= 1e6:
        return "{:.0f}M".format(num_params / 1e6)
    if num_params >= 1e3:
        return "{:.1f}K".format(num_params / 1e3)
    return str(num_params)


def profile_inference_steps(data: List[Dict]) -> None:
    """
    Run full pipeline (FastSAM + heat diffusion + tensor conversion + model forward)
    on NUM_PROFILE_IMAGES images at VRSBench resolution (512x512).
    """
    global SAM_RAW, SAM_DATAPARALLEL, MASK_GENERATOR
    try:
        from llamafactory.utils.semantic_patch_utils import process_image_with_semantic_patches
        from llamafactory.utils.sam_model_loader import load_sam_model
    except ImportError as e:
        print(f"[Profile] Cannot import: {e}. Ensure src/ is in path and llamafactory is available.")
        return

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"[Profile] FastSAM checkpoint not found: {SAM_CHECKPOINT}. Set SAM_CHECKPOINT env or edit script.")
        return

    print(f"[Profile] Loading {SAM_BACKEND} from {SAM_CHECKPOINT} ...")
    SAM_RAW, SAM_DATAPARALLEL, MASK_GENERATOR = load_sam_model(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=SAM_MODEL_TYPE,
        sam_backend=SAM_BACKEND,
        device_id=0,
    )
    if SAM_RAW is None:
        print("[Profile] Segmentation backend load failed, abort.")
        return
    # Model size: official SAM-B is 136M params
    sam_params = sum(p.numel() for p in SAM_RAW.parameters())
    sam_params_str = _format_params(sam_params)

    image_processor = getattr(PROCESSOR, "image_processor", None)
    if image_processor is None:
        print("[Profile] PROCESSOR.image_processor not found.")
        return

    # Pick first N images that exist
    profile_items = []
    for qa in data:
        img_path = IMAGES_DIR / str(qa.get("image_id", ""))
        if img_path.exists() and len(profile_items) < NUM_PROFILE_IMAGES:
            profile_items.append((img_path, qa.get("question", "Describe the image.")))
        if len(profile_items) >= NUM_PROFILE_IMAGES:
            break
    if not profile_items:
        print("[Profile] No images found.")
        return

    image_side = int(math.sqrt(SEMANTIC_IMAGE_MAX_PIXELS))
    print(f"[Profile] Running pipeline on {len(profile_items)} images ({image_side}x{image_side})...")
    merge_params = dict(VRSBENCH_MERGE_PARAMS)
    merge_params["target_size"] = SEMANTIC_PATCH_SIZE

    step_keys = ["sam_s", "sam_mem_mb", "sam_mem_avg_mb", "diffusion_merge_s", "diffusion_merge_mem_mb", "diffusion_merge_mem_avg_mb",
                 "clipping_kmeans_s", "clipping_kmeans_mem_mb", "clipping_kmeans_mem_avg_mb", "tensor_conversion_s"]
    all_timings: List[Dict[str, float]] = []
    semantic_results: List[Tuple[Dict, Optional[Dict]]] = []

    for img_path, question in profile_items:
        with Image.open(img_path) as img:
            image = img.copy().convert("RGB")
        w, h = image.size
        if w * h > SEMANTIC_IMAGE_MAX_PIXELS or w * h < SEMANTIC_IMAGE_MIN_PIXELS:
            image = _preprocess_image_for_cache(img_path)
        try:
            out = process_image_with_semantic_patches(
                image,
                image_processor,
                sam_model_raw=SAM_RAW,
                sam_model_dataparallel=SAM_DATAPARALLEL,
                mask_generator=MASK_GENERATOR,
                use_semantic_patches=True,
                global_downsample=SEMANTIC_GLOBAL_DOWNSAMPLE,
                semantic_patch_size=SEMANTIC_PATCH_SIZE,
                merge_params=merge_params,
                return_timings=True,
            )
            if isinstance(out, tuple) and len(out) == 2:
                result, timings = out
                all_timings.append(timings)
                semantic_results.append((result, timings))
            else:
                semantic_results.append((out, None))
        except Exception as e:
            print(f"[Profile] Pipeline failed for {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_timings:
        print("[Profile] No timings collected.")
        return

    # Model forward: time generate (prefill + decode). Model params = main model (Qwen) size.
    qwen_params = sum(p.numel() for p in MODEL.parameters())
    qwen_params_str = _format_params(qwen_params)

    target_device = _get_model_primary_device()
    model_dtype = _get_model_dtype()
    forward_times: List[float] = []

    for (semantic_data, _), (img_path, question) in zip(semantic_results, profile_items):
        if semantic_data is None or "pixel_values" not in semantic_data:
            continue
        user_text = _build_user_text(question, TASK_MODE)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": f"file://{img_path.resolve().as_posix()}"},
            ],
        }]
        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Required image token count = sum(grid_t*grid_h*grid_w) / merge_size^2 (must match pixel_values length)
        merge_size = getattr(getattr(PROCESSOR, "image_processor", None), "merge_size", 2) or 2
        merge_square = merge_size ** 2
        grid_thw = semantic_data["image_grid_thw"]
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        num_image_tokens = int(grid_thw.prod(-1).sum().item() // merge_square)
        num_global = semantic_data.get("num_global_tokens")
        num_semantic = semantic_data.get("num_semantic_tokens")
        vision_start, vision_end, image_token = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
        if num_global is not None and num_semantic is not None and num_global > 0 and num_semantic > 0:
            placeholder = f"{vision_start}{image_token}{vision_end}"
            replacement = (
                f"{vision_start}{image_token * num_global}{vision_end}"
                f"{vision_start}{image_token * num_semantic}{vision_end}"
            )
            if placeholder in text:
                text = text.replace(placeholder, replacement, 1)
            else:
                text = text.replace(image_token, image_token * num_image_tokens, 1)
        else:
            replacement = f"{vision_start}{image_token * num_image_tokens}{vision_end}"
            placeholder = f"{vision_start}{image_token}{vision_end}"
            if placeholder in text:
                text = text.replace(placeholder, replacement, 1)
            elif image_token in text:
                text = text.replace(image_token, image_token * num_image_tokens, 1)
        inputs = PROCESSOR.tokenizer(text=[text], padding=True, return_tensors="pt")
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(target_device)
        seq_len = inputs.get("input_ids", torch.empty(1, 0)).shape[-1]
        if seq_len > 0:
            inputs["cache_position"] = torch.arange(seq_len, device=target_device, dtype=torch.long)
        if "position_ids" not in inputs:
            inputs["position_ids"] = torch.arange(seq_len, device=target_device, dtype=torch.long).unsqueeze(0)
        pixel_values = semantic_data["pixel_values"].to(target_device, dtype=model_dtype).contiguous()
        inputs["pixel_values"] = pixel_values
        inputs["image_grid_thw"] = semantic_data["image_grid_thw"].to(target_device)
        for k in ["image_patch_positions", "image_patch_extents", "image_patch_orientations"]:
            if k in semantic_data and semantic_data[k] is not None:
                inputs[k] = semantic_data[k].to(target_device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with MODEL_LOCK, torch.inference_mode():
            MODEL.generate(
                **{k: v for k, v in inputs.items() if v is not None},
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - t0)

    # Aggregate
    n = len(all_timings)
    avg = {}
    for k in step_keys:
        vals = [t.get(k, 0) for t in all_timings if k in t and t.get(k) is not None]
        avg[k] = sum(vals) / len(vals) if vals else 0.0
    avg["model_forward_s"] = sum(forward_times) / len(forward_times) if forward_times else 0.0

    def _s_to_ms(t_s: float) -> float:
        return t_s * 1000.0

    # Report: time (ms) + model params. diffusion/clipping/tensor use hand-calculated counts (see constants above).
    image_side = int(math.sqrt(SEMANTIC_IMAGE_MAX_PIXELS))
    image_size_label = f"{image_side}x{image_side}"
    report = {
        "image_size": image_size_label,
        "num_images": n,
        "steps": {
            "sam_proposal": {"time_ms": _s_to_ms(avg["sam_s"]), "params": sam_params_str},
            "diffusion_merge": {"time_ms": _s_to_ms(avg["diffusion_merge_s"]), "params": str(DIFFUSION_MERGE_PARAMS)},
            "clipping_kmeans": {"time_ms": _s_to_ms(avg["clipping_kmeans_s"]), "params": str(CLIPPING_KMEANS_PARAMS)},
            "tensor_conversion": {"time_ms": _s_to_ms(avg.get("tensor_conversion_s", 0)), "params": str(TENSOR_CONVERSION_PARAMS)},
            "model_forward": {"time_ms": _s_to_ms(avg["model_forward_s"]), "params": qwen_params_str},
        },
        "raw_avg": avg,
    }

    out_path = OUT_DIR / "inference_steps_profile.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print("Inference step timing & model params (avg over {} images, {})".format(n, image_size_label))
    print("=" * 60)
    for name, d in report["steps"].items():
        print("  {}: time = {:.1f} ms, params = {}".format(name, d["time_ms"], d["params"]))
    print("  (diffusion_merge: params = 18 Sobel + 6 hyperparams, non-learnable; clipping/tensor: hand-calc)")
    print("Memory (MB) - peak / avg:")
    print("  sam_proposal:       peak = {:.1f}, avg = {:.1f}".format(avg.get("sam_mem_mb", 0), avg.get("sam_mem_avg_mb", 0)))
    print("  diffusion_merge:   peak = {:.1f}, avg = {:.1f}".format(avg.get("diffusion_merge_mem_mb", 0), avg.get("diffusion_merge_mem_avg_mb", 0)))
    print("  clipping_kmeans:   peak = {:.1f}, avg = {:.1f}".format(avg.get("clipping_kmeans_mem_mb", 0), avg.get("clipping_kmeans_mem_avg_mb", 0)))
    print("Saved: {}".format(out_path))
    print("=" * 60 + "\n")


# ----------------------------- Records -----------------------------
@dataclass
class Record:
    image_id: str
    question: str
    ground_truth: str
    dataset: str
    question_id: int | str
    type: str
    answer: str             # Model prediction

# ----------------------------- Worker -----------------------------

def _process_one(qa_item: Dict) -> Record:
    
    # Parse fields from QA item
    image_id = str(qa_item.get("image_id", ""))
    question = str(qa_item.get("question", ""))
    ground_truth = str(qa_item.get("ground_truth", ""))
    dataset = str(qa_item.get("dataset", ""))
    question_id = qa_item.get("question_id", -1)
    q_type = str(qa_item.get("type", ""))

    # [VRSBench] Resolve image path from IMAGES_DIR / image_id filename
    img_path = IMAGES_DIR / image_id

    if not img_path.exists():
        answer = f"[ERROR] not found: {img_path}"
    else:
        last_exc = None
        for attempt in range(RETRIES + 1):
            try:
                answer = vqa_infer(img_path, question, q_type)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < RETRIES:
                    print(f"Warning: {image_id} inference failed, retrying... error: {e}")
                    time.sleep(BACKOFF ** attempt)
        if last_exc is not None:
            answer = f"[EXCEPTION] {last_exc}"

    rec = Record(
        image_id=image_id,
        question=question,
        ground_truth=ground_truth,
        dataset=dataset,
        question_id=question_id,
        type=q_type,
        answer=answer,
    )

    return rec

# ----------------------------- Main -----------------------------

def main():
    
    if torch.cuda.is_available():
        visible_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        # print(f"PyTorch visible GPUs: {visible_gpus}")
        if not visible_gpus:
             print("Error: CUDA available but PyTorch sees no GPU.")
             return
        if torch.cuda.device_count() < REQUIRED_NUM_GPUS:
            print(
                f"Error: found {torch.cuda.device_count()} GPU(s), "
                f"but {REQUIRED_NUM_GPUS} GPUs required (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
            )
            return
    else:
        print("Error: CUDA not available.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(f"Error: {DATA_JSON} is not a JSON list.")
            return
        print(f"Loaded {len(data)} VQA items.")
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse {DATA_JSON}: {e}")
        return
    except FileNotFoundError:
        print(f"Error: file not found {DATA_JSON}")
        return

    target_device = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
    print(f"[Rank {RANK}] Loading model on {target_device} ...")
    load_model_once(device=target_device)
    print(f"[Rank {RANK}] Model loaded.")

    if PROFILE_INFERENCE_STEPS:
        profile_inference_steps(data)
        print("Profiling done. Set PROFILE_INFERENCE_STEPS=False to run full evaluation.")
        return

    items: List[Dict] = [x for i, x in enumerate(data) if i % WORLD_SIZE == RANK]
    print(f"[Rank {RANK}] Processing {len(items)} / {len(data)} items (world_size={WORLD_SIZE})")
    
    all_results: List[Record] = []

    pbar = tqdm(items, total=len(items), desc=f"Processing VQA (rank {RANK})")
    for qa_item in pbar:
        try:
            result_record = _process_one(qa_item)
            all_results.append(result_record)
        except Exception as e:
            img_id_for_error = qa_item.get("image_id", "unknown")
            print(f"[Rank {RANK}] Error on {img_id_for_error}: {e}")
    
    print(f"[Rank {RANK}] Finished {len(all_results)} items.")

    if not all_results:
        print(f"[Rank {RANK}] No results, skip writing.")
        return

    rank_json_path = OUT_DIR / f"predictions_rank{RANK}.json"
    print(f"[Rank {RANK}] Writing partial results to {rank_json_path} ...")
    try:
        results_as_dicts = [rec.__dict__ for rec in all_results]
        with open(rank_json_path, "w", encoding="utf-8") as f:
            json.dump(results_as_dicts, f, indent=4, ensure_ascii=False) 
    except Exception as e:
        print(f"[Rank {RANK}] Failed to write JSON: {e}")

    if WORLD_SIZE > 1 and torch.distributed.is_available():
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            torch.distributed.barrier()
        except Exception as e:
            print(f"[Rank {RANK}] Warning: barrier failed: {e}")

    if RANK != 0:
        return

    merged_results: List[Dict] = []
    for r in range(WORLD_SIZE):
        part_path = OUT_DIR / f"predictions_rank{r}.json"
        if not part_path.exists():
            print(f"[Rank 0] Warning: missing {part_path}")
            continue
        try:
            with open(part_path, "r", encoding="utf-8") as f:
                part = json.load(f)
                if isinstance(part, list):
                    merged_results.extend(part)
        except Exception as e:
            print(f"[Rank 0] Failed to read {part_path}: {e}")

    if not merged_results:
        print("[Rank 0] No merged results to write.")
        return

    json_path = OUT_DIR / "predictions.json"
    print(f"[Rank 0] Writing merged results to {json_path} ...")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Rank 0] Failed to write JSON: {e}")

    csv_path = OUT_DIR / "results.csv"
    print(f"[Rank 0] Writing merged results to {csv_path} ...")
    csv_header = ["image_id", "question", "ground_truth", "dataset", "question_id", "type", "answer"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
            csvw = csv.writer(csvf)
            csvw.writerow(csv_header)
            for rec in merged_results:
                csvw.writerow([
                    rec.get("image_id", ""),
                    rec.get("question", ""),
                    rec.get("ground_truth", ""),
                    rec.get("dataset", ""),
                    rec.get("question_id", ""),
                    rec.get("type", ""),
                    rec.get("answer", ""),
                ])
    except Exception as e:
        print(f"[Rank 0] Failed to write CSV: {e}")

    print(f"[Rank 0] Done! Results saved to {OUT_DIR}")
    print(f"[Rank 0] JSON: {json_path}")
    print(f"[Rank 0] CSV: {csv_path}")

if __name__ == "__main__":
    main()