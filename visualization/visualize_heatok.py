# -*- coding: utf-8 -*-
"""
HeatTok token visualization (open-source demo).

Pipeline:
1) Segment with Meta SAM
2) Heat-diffusion merge of regions
3) Save only: <name>_SAM_merged_boundary.png
   Prints only the final merged region/token count.
"""

import os
# Limit BLAS/OpenMP threads early to avoid OpenBLAS thread-metadata overflow / segfault
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import builtins
from collections import defaultdict
import time
import threading
from pathlib import Path
from tqdm import trange as _trange, tqdm as _tqdm

# Quiet demo: hide tqdm bars and common library noise.
warnings.filterwarnings("ignore", message=".*low contrast image.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"skimage\..*")


def tqdm(*args, **kwargs):
    kwargs.setdefault("disable", True)
    return _tqdm(*args, **kwargs)


def trange(*args, **kwargs):
    kwargs.setdefault("disable", True)
    return _trange(*args, **kwargs)
from sklearn.cluster import KMeans

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except Exception:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    SamPredictor = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    print("="*50)
    print("Warning: missing 'torch_scatter'")
    print("GPU scatter ops will fall back to slower CPU logic.")
    print("Install with: pip install torch-scatter (match your PyTorch/CUDA)")
    print("="*50)
    scatter_mean = None
    scatter_add = None

from torchvision import io as tv_io
from torchvision.transforms.functional import rgb_to_grayscale, gaussian_blur
from skimage import io as ski_io
from skimage import color as ski_color
from skimage.segmentation import mark_boundaries, find_boundaries
from scipy.ndimage import distance_transform_edt

# Visualization alpha for avg-color token map.
# 1.0 = fully token color, 0.0 = fully original image.
AVG_COLOR_ALPHA = 0.65
VIS_DASH_LEN = 8
VIS_GAP_LEN = 6
VIS_BOUNDARY_WIDTH = 3

# Colored boundary options:
# - "white": solid white boundaries as before
# - "multi": use fixed palette below (cyclic)
# - "multi_rand": reproducible random palette per label using uniform[0,1) with a fixed seed
BOUNDARY_COLOR_MODE = os.getenv("HEATOK_BOUNDARY_MODE", "white").lower()  # "white" | "multi" | "multi_rand"
# Default seed for reproducible random palette (matches your sample when seed=7)
HEATOK_COLOR_SEED = int(os.getenv("HEATOK_COLOR_SEED", "7"))

# Fixed palette (used when mode == "multi"); values are in [0, 1] for float images
BOUNDARY_COLOR_PALETTE = np.array([
    [0.90, 0.40, 0.40],  # red
    [0.40, 0.75, 0.95],  # light blue
    [0.30, 0.80, 0.55],  # green
    [0.95, 0.75, 0.30],  # orange
    [0.65, 0.45, 0.85],  # purple
    [0.30, 0.65, 0.95],  # blue
    [0.95, 0.55, 0.30],  # coral
    [0.20, 0.75, 0.75],  # teal
    [0.85, 0.85, 0.35],  # olive
    [0.55, 0.55, 0.95],  # periwinkle
], dtype=np.float32)

# torch_scatter availability check
if scatter_mean is None:
    print("Warning: torch_scatter not available, using slower fallbacks")
_LOG_LEVEL_MAP = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
_DEFAULT_LOG_LEVEL = os.getenv("SAM_HEAT_LOG_LEVEL", "WARN").upper()
_BUILTIN_PRINT = builtins.print


def _should_log(level: str) -> bool:
    return _LOG_LEVEL_MAP.get(level, 2) <= _LOG_LEVEL_MAP.get(_DEFAULT_LOG_LEVEL, 2)


def print(*args, **kwargs):  # type: ignore[override]
    level = kwargs.pop("level", None)
    if level is None:
        merged = " ".join(str(arg) for arg in args)
        if any(token in merged for token in ["ERROR", "Error"]):
            level = "ERROR"
        elif any(token in merged for token in ["WARNING", "Warning"]):
            level = "WARN"
        else:
            level = "INFO"
    if _should_log(level.upper()):
        _BUILTIN_PRINT(*args, **kwargs)


def _gen_reproducible_palette(num_labels: int, seed: int = HEATOK_COLOR_SEED) -> np.ndarray:
    """
    Generate a reproducible random color palette in [0,1] using numpy's Generator with given seed.
    Colors are drawn sequentially as uniform[0,1) triples per label id: [r,g,b].
    """
    if num_labels <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    pal = rng.random((num_labels, 3), dtype=np.float32)
    return pal


class _GPUMemoryAvgSampler:
    """Sampling GPU memory at intervals to compute average (MB). Use as context manager."""
    def __init__(self, interval_s=0.02):
        self.interval_s = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self.avg_mb = 0.0

    def _sample_loop(self):
        while not self._stop.wait(timeout=self.interval_s):
            if torch.cuda.is_available():
                self._samples.append(torch.cuda.memory_allocated())

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._samples:
            self.avg_mb = (sum(self._samples) / len(self._samples)) / (1024 ** 2)
        return False


class _NullSampler:
    """No-op context manager; avg_mb = 0 when GPU avg sampling is disabled."""
    avg_mb = 0.0
    def __enter__(self): return self
    def __exit__(self, *args): return False


# --- 2. UnionFind class (from heat_diffusion_code.py) ---
class UnionFind:
    """Disjoint-set structure used to merge connected regions."""
    def __init__(self, cluster_ids):
        self.parent = {i: i for i in cluster_ids}
        self.rank = {i: 0 for i in cluster_ids}
    def find(self, i):
        if i not in self.parent: return i
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank.get(root_i, 0) < self.rank.get(root_j, 0):
                self.parent[root_i] = root_j
            elif self.rank.get(root_i, 0) > self.rank.get(root_j, 0):
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] = self.rank.get(root_i, 0) + 1
            return True
        return False


# --- 3. Cluster class (from heat_diffusion_code.py) ---
class Cluster(object):
    """Store the center and heat-diffusion features of one region."""
    cluster_index = 1
    def __init__(self, h, w, l=0, a=0, b=0):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
        self.no = Cluster.cluster_index
        Cluster.cluster_index += 1
        self.avg_color_lab = np.array([l, a, b])
        self.complexity = 0.0
        self.temperature = 0.0

    def update(self, h, w, l, a, b):
        self.h = h; self.w = w
        self.l = l; self.a = a; self.b = b
        
    def __repr__(self):
        return f"C[{self.no}]@({self.h:.1f},{self.w:.1f})"


# --- 4. SLICProcessor class (from heat_diffusion_code.py) ---
class SLICProcessor(object):

    @staticmethod
    def open_image(path, device):
        """Load RGB input and return blurred LAB data plus the original RGB array."""
        try:
            # 1. Read image with torchvision.io as (C, H, W), [0, 255], uint8
            rgb_tensor_c_first = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
            rgb_tensor_c_first_blurred = gaussian_blur(rgb_tensor_c_first, kernel_size=[7, 7])
            
            # 2. Permute to (H, W, C) [0, 255] uint8 for skimage.color
            rgb_uint8_hwc_blurred = rgb_tensor_c_first_blurred.permute(1, 2, 0).contiguous()
            
            # 3. Convert to float [0, 1] for skimage.color
            rgb_float_hwc_blurred = (rgb_uint8_hwc_blurred.cpu() / 255.0).numpy()
            
            # 4. RGB -> LAB using scikit-image
            lab_arr_hwc_blurred = ski_color.rgb2lab(rgb_float_hwc_blurred)
            lab_tensor_c_first = torch.from_numpy(lab_arr_hwc_blurred).permute(2, 0, 1).float().to(device)
            rgb_uint8_hwc_original = rgb_tensor_c_first.permute(1, 2, 0).cpu().numpy()
            
            return lab_tensor_c_first, rgb_uint8_hwc_original
            
        except FileNotFoundError:
             raise FileNotFoundError(f"Image file not found at: {path}")
        except Exception as e:
             raise RuntimeError(f"Error reading/processing image {path}: {e}") from e

    def __init__(self, filename, K, M, output_dir_base=None, device_id=1):
        if output_dir_base is None:
            output_dir_base = str(_REPO_ROOT / "visualization" / "outputs")
        
        self.K = K
        self.M = M
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")
        
        print(f"SLICProcessor: using device {self.device}")

        self.filename = filename
        self.original_filename_base = os.path.splitext(os.path.basename(filename))[0]
        # Keep each image run in its own SAM heat-merge output directory.
        self.output_dir = os.path.join(output_dir_base, f"{self.original_filename_base}_SAM_HeatMerge")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        try:
            # self.data: LAB tensor (C=3, H, W) on self.device
            # self.rgb_data_uint8: RGB numpy array (H, W, C) on CPU
            self.data, self.rgb_data_uint8 = self.open_image(filename, self.device)
        except Exception as e:
            print(f"Error initializing SLICProcessor: {e}")
            raise

        self.C, self.image_height, self.image_width = self.data.shape
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / max(self.K, 1))) # SLIC grid interval
        if self.S == 0: self.S = 1

        print(f"Image size: {self.image_height} H x {self.image_width} W")

        self.clusters = []      # (CPU) Cluster objects in Python list
        self.cluster_map = {}   # (CPU) ID -> Cluster object mapping
        self.K_actual = 0      
        self.slic_labels = torch.full((self.image_height, self.image_width), -1, 
                                      dtype=torch.long, device=self.device)
        self.dis = torch.full((self.image_height, self.image_width), float('inf'), 
                              dtype=torch.float32, device=self.device)
        # Initialized later by _populate_clusters_from_labels
        self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device) 
        h_coords = torch.arange(self.image_height, device=self.device, dtype=torch.float32)
        w_coords = torch.arange(self.image_width, device=self.device, dtype=torch.float32)
        self.pixel_coords = torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), dim=-1) # Shape (H, W, 2)
        self.gradient_map_tensor = None 
        self.neighbors = defaultdict(set)
        self.temperatures = {}
        self.diffusion_coeffs = {}
        self.final_merged_labels = None 

        Cluster.cluster_index = 1

    # ... (make_cluster_and_init_centers, init_clusters, get_gradient, move_clusters, assignment, update_cluster) ...
    def make_cluster_and_init_centers(self, h, w, cluster_idx):
        # Legacy SLIC path
        h = int(h); w = int(w)
        h = np.clip(h, 0, self.image_height - 1)
        w = np.clip(w, 0, self.image_width - 1)
        l = self.data[0, h, w].item()
        a = self.data[1, h, w].item()
        b = self.data[2, h, w].item()
        cluster_obj = Cluster(h, w, l, a, b)
        self.cluster_centers[cluster_idx, 0] = l
        self.cluster_centers[cluster_idx, 1] = a
        self.cluster_centers[cluster_idx, 2] = b
        self.cluster_centers[cluster_idx, 3] = h
        self.cluster_centers[cluster_idx, 4] = w
        return cluster_obj

    def init_clusters(self):
        # Legacy SLIC path
        print("SLIC init_clusters (unused for SAM path)")
        pass

    def get_gradient(self, h, w):
        if not hasattr(self, 'data_cpu'):
             self.data_cpu = self.data.permute(1, 2, 0).cpu().numpy()
        _h = int(h); _w = int(w)
        if _w + 1 >= self.image_width: _w = self.image_width - 2
        if _h + 1 >= self.image_height: _h = self.image_height - 2
        _h = max(0, _h); _w = max(0, _w)
        gradient = self.data_cpu[_h + 1, _w + 1, 0] - self.data_cpu[_h, _w, 0] + \
                     self.data_cpu[_h + 1, _w + 1, 1] - self.data_cpu[_h, _w, 1] + \
                     self.data_cpu[_h + 1, _w + 1, 2] - self.data_cpu[_h, _w, 2]
        return gradient

    def move_clusters(self):
        # Legacy SLIC path
        print("SLIC move_clusters (unused for SAM path)")
        pass

    def assignment(self):
        # Legacy SLIC path
        print("SLIC assignment (unused for SAM path)")
        pass

    def update_cluster(self):
        # Legacy SLIC path
        print("SLIC update_cluster (unused for SAM path)")
        pass
    def get_gradient_tensor(self):
        """Compute and cache the Sobel gradient magnitude of the LAB lightness channel."""
        if self.gradient_map_tensor is not None:
            return self.gradient_map_tensor
            
        print("Computing Sobel gradient on GPU...")
        l_channel_batch = self.data[0:1, :, :].unsqueeze(0)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).reshape((1, 1, 3, 3))
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).reshape((1, 1, 3, 3))
        grad_x = F.conv2d(l_channel_batch, sobel_x, padding=1)
        grad_y = F.conv2d(l_channel_batch, sobel_y, padding=1)
        self.gradient_map_tensor = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        print("Gradient map ready")
        return self.gradient_map_tensor

    def _assign_unlabeled_pixels(self):
        """Fill unlabeled pixels from their nearest labeled pixels."""
        if self.slic_labels is None: return

        unassigned_mask_gpu = (self.slic_labels == -1)
        num_unassigned = unassigned_mask_gpu.sum().item()

        if num_unassigned == 0: print("No unassigned pixels found"); return
        elif num_unassigned == self.N: print("Warning: all pixels are unassigned"); return
        print(f"Assigning {num_unassigned} unlabeled pixels...")
        
        unassigned_mask_cpu = unassigned_mask_gpu.cpu().numpy()
        distances_cpu, indices_cpu = distance_transform_edt(unassigned_mask_cpu, return_indices=True)
        
        indices_gpu = torch.from_numpy(indices_cpu).long().to(self.device) # (2, H, W)
        
        nearest_labels = self.slic_labels[indices_gpu[0], indices_gpu[1]]
        
        self.slic_labels[unassigned_mask_gpu] = nearest_labels[unassigned_mask_gpu]

        remaining = (self.slic_labels == -1).sum().item()
        if remaining > 0: print(f"Warning: {remaining} pixels remain unassigned")
        else: print("All unassigned pixels filled")


    def _calculate_complexities(self):
        """Compute cluster complexity from mean gradient magnitude."""
        if self.gradient_map_tensor is None: self.get_gradient_tensor()
        if scatter_mean is None:
            print("Warning: torch_scatter unavailable; using CPU fallback")
            return self._calculate_complexities_cpu()  # Use the slower CPU path.

        print("Computing cluster features on GPU (avg color + complexity)...")
        
        flat_gradients = self.gradient_map_tensor.view(self.N)
        flat_labels = self.slic_labels.view(self.N) # (N)
        
        valid_mask = flat_labels >= 0
        if not valid_mask.any():
            print("Warning: no valid labeled pixels; setting complexities to zero")
            for cluster in self.clusters: cluster.complexity = 0.0
            return

        flat_gradients_valid = flat_gradients[valid_mask]
        flat_labels_valid = flat_labels[valid_mask]
        
        # (K_actual)
        avg_gradients = scatter_mean(flat_gradients_valid, flat_labels_valid, dim=0, dim_size=self.K_actual)
        
        avg_gradients_cpu = avg_gradients.cpu().numpy()
        
        for i in range(self.K_actual):
            if np.isnan(avg_gradients_cpu[i]):
                self.clusters[i].complexity = 0.0
            else:
                self.clusters[i].complexity = avg_gradients_cpu[i]
                
        print("Cluster features ready (GPU)")


    def _calculate_complexities_cpu(self):
        """Compute cluster complexity with the CPU fallback."""
        print("Computing complexity on CPU...")
        if self.gradient_map_tensor is None: self.get_gradient_tensor()
        
        gradient_map_cpu = self.gradient_map_tensor.cpu().numpy()
        labels_cpu = self.slic_labels.cpu().numpy()
        
        self.cluster_map = {c.no: c for c in self.clusters}
        
        temp_pixels = defaultdict(list)
        for h in range(self.image_height):
            for w in range(self.image_width):
                label_idx = labels_cpu[h, w] # 0..K_actual-1
                if 0 <= label_idx < self.K_actual:
                    temp_pixels[label_idx].append((h,w))

        for cluster_id_no, cluster in tqdm(self.cluster_map.items(), desc="Calculating Complexity (CPU)"):
           cluster_idx = -1
           for i, c in enumerate(self.clusters):
               if c.no == cluster_id_no:
                   cluster_idx = i
                   break
           if cluster_idx == -1: continue

           pixels = temp_pixels.get(cluster_idx, [])
           if not pixels: cluster.complexity = 0; continue
           
           pixel_count = len(pixels)
           pixel_coords = np.array(pixels)
           rows, cols = pixel_coords[:, 0], pixel_coords[:, 1]
           gradient_values = gradient_map_cpu[rows, cols]
           sum_gradient_mag = np.sum(gradient_values)
           cluster.complexity = sum_gradient_mag / pixel_count if pixel_count > 0 else 0
        print("Complexity computed (CPU)")

    def _build_adjacency(self):
        """Build the undirected four-neighbor region adjacency graph."""
        print("Building adjacency on GPU...")
        self.neighbors = defaultdict(set)

        label_array = self.slic_labels.cpu().numpy().astype(np.int64, copy=False)
        if label_array.size == 0:
            print("Warning: label map is empty; skipping adjacency construction")
            return

        max_label_idx = int(label_array.max())
        if max_label_idx < 0:
            print("Warning: all labels are -1; skipping adjacency construction")
            return

        # label idx (0..K_actual-1) -> cluster no
        idx_to_no = np.full(max_label_idx + 1, -1, dtype=np.int64)
        for idx, cluster in enumerate(self.clusters):
            if idx <= max_label_idx:
                idx_to_no[idx] = int(cluster.no)

        def _collect_pairs(src_idx: np.ndarray, dst_idx: np.ndarray):
            valid = (src_idx >= 0) & (dst_idx >= 0) & (src_idx != dst_idx)
            if not np.any(valid):
                return np.empty((0, 2), dtype=np.int64)

            src_no = idx_to_no[src_idx[valid]]
            dst_no = idx_to_no[dst_idx[valid]]
            mapped_valid = (src_no >= 0) & (dst_no >= 0) & (src_no != dst_no)
            if not np.any(mapped_valid):
                return np.empty((0, 2), dtype=np.int64)

            pairs = np.stack([src_no[mapped_valid], dst_no[mapped_valid]], axis=1)
            pairs.sort(axis=1)  # Canonicalize each undirected edge.
            return pairs
        right_pairs = _collect_pairs(label_array[:, :-1], label_array[:, 1:])
        down_pairs = _collect_pairs(label_array[:-1, :], label_array[1:, :])

        if right_pairs.shape[0] == 0 and down_pairs.shape[0] == 0:
            print("Warning: no adjacent region boundaries found")
            return

        all_pairs = np.concatenate([right_pairs, down_pairs], axis=0)
        unique_pairs = np.unique(all_pairs, axis=0)

        for no_a, no_b in unique_pairs.tolist():
            self.neighbors[int(no_a)].add(int(no_b))
            self.neighbors[int(no_b)].add(int(no_a))

        num_clusters_with_neighbors = len(self.neighbors)
        total_clusters = len(self.cluster_map) if self.cluster_map else len(self.clusters)
        print(f"Adjacency built: {num_clusters_with_neighbors}/{total_clusters} clusters have neighbors")


    def _calculate_average_colors(self):
        print("Average colors already computed during population")
        pass 

    def _calculate_initial_temperatures(self, sigma_T=10.0):
        print("Computing temperatures on GPU...")
        self.temperatures = {}
        if not self.neighbors: self._build_adjacency()
        if not self.cluster_map: self.cluster_map = {c.no: c for c in self.clusters}
        
        for cluster_id in tqdm(self.cluster_map.keys(), desc="Calculating Temps"):
            neighbor_nos = self.neighbors.get(cluster_id, set())
            if cluster_id not in self.cluster_map: continue 
            cluster_i = self.cluster_map[cluster_id]
            neighbor_count = len(neighbor_nos)
            if neighbor_count == 0: self.temperatures[cluster_id] = 0.0; continue
            
            sum_similarity = 0.0; valid_neighbor_count = 0
            for cluster_j_no in neighbor_nos:
                cluster_j = self.cluster_map.get(cluster_j_no)
                if cluster_j:
                    color_dist = np.linalg.norm(cluster_i.avg_color_lab - cluster_j.avg_color_lab)
                    similarity = math.exp(-color_dist / max(sigma_T, 1e-6))
                    sum_similarity += similarity; valid_neighbor_count += 1
            self.temperatures[cluster_id] = sum_similarity / valid_neighbor_count if valid_neighbor_count > 0 else 0.0
        print("Temperatures ready (GPU)")


    def _calculate_diffusion_coeffs(self, K0=1.0, sigma_C=10.0, alpha=0.5):
        print("Computing diffusion coefficients on GPU...")
        self.diffusion_coeffs = {}
        if not self.neighbors: self._build_adjacency()
        if not self.cluster_map: self.cluster_map = {c.no: c for c in self.clusters}
        
        processed_pairs = set(); max_k_ij = 0.0
        for cluster_i_no, neighbor_nos in tqdm(self.neighbors.items(), desc="Calculating K_ij"):
            cluster_i = self.cluster_map.get(cluster_i_no)
            if not cluster_i: continue
            
            for cluster_j_no in neighbor_nos:
                pair = tuple(sorted((cluster_i_no, cluster_j_no)))
                if pair in processed_pairs: continue
                
                cluster_j = self.cluster_map.get(cluster_j_no)
                if not cluster_j: continue
                
                color_dist_sq = np.sum((cluster_i.avg_color_lab - cluster_j.avg_color_lab)**2)
                color_term = math.exp(-color_dist_sq / (2 * max(sigma_C**2, 1e-6)))
                complexity_term = 1.0 + alpha * min(cluster_i.complexity, cluster_j.complexity)
                k_ij = K0 * color_term * complexity_term
                self.diffusion_coeffs[pair] = k_ij; processed_pairs.add(pair)
                max_k_ij = max(max_k_ij, k_ij)
        print("Diffusion matrix ready (GPU)")

    def _simulate_heat_diffusion(self, delta_t=0.001, max_iterations=50):
        print(f"Simulating heat diffusion for up to {max_iterations} iterations (CPU)...")
        if not self.temperatures: self._calculate_initial_temperatures()
        if not self.diffusion_coeffs: self._calculate_diffusion_coeffs()
        if not self.cluster_map: self.cluster_map = {c.no: c for c in self.clusters}
        
        current_temps = self.temperatures.copy()
        for t in trange(max_iterations, desc="Heat Diffusion (CPU)"):
            next_temps = current_temps.copy(); max_temp_change_iter = 0.0
            
            for cluster_i_no in self.cluster_map.keys():
                temp_i = current_temps.get(cluster_i_no, 0)
                sum_heat_exchange = 0.0
                neighbor_nos = self.neighbors.get(cluster_i_no, set())
                
                for cluster_j_no in neighbor_nos:
                    temp_j = current_temps.get(cluster_j_no, 0)
                    pair = tuple(sorted((cluster_i_no, cluster_j_no)))
                    k_ij = self.diffusion_coeffs.get(pair, 0)
                    sum_heat_exchange += k_ij * (temp_j - temp_i)
                    
                temp_change = delta_t * sum_heat_exchange
                next_temps[cluster_i_no] = temp_i + temp_change
                max_temp_change_iter = max(max_temp_change_iter, abs(temp_change))
                
            current_temps = next_temps
            if max_temp_change_iter < 1e-5:
                print(f"Heat diffusion converged after {t + 1} iterations")
                break
        self.temperatures = current_temps
        print("Heat diffusion simulation complete")

    def _merge_clusters(self, merge_threshold=0.03):
        print("Merging clusters by temperature difference (CPU)...")
        if not self.temperatures: print("Warning: temperatures are unavailable"); return None
        cluster_ids = list(self.cluster_map.keys())
        if not cluster_ids: print("Warning: no cluster IDs found"); return None
        
        uf = UnionFind(cluster_ids); merged_count = 0; checked_pairs = 0
        processed_pairs = set()
        
        for cluster_i_no, neighbor_nos in self.neighbors.items():
            temp_i = self.temperatures.get(cluster_i_no, -1)
            if temp_i == -1: continue
            
            for cluster_j_no in neighbor_nos:
                pair = tuple(sorted((cluster_i_no, cluster_j_no)))
                if pair not in processed_pairs:
                    checked_pairs += 1
                    processed_pairs.add(pair)
                    temp_j = self.temperatures.get(cluster_j_no, -1)
                    
                    if temp_j != -1:
                        temp_diff = abs(temp_i - temp_j)
                        if temp_diff < merge_threshold:
                            if uf.union(cluster_i_no, cluster_j_no):
                                merged_count += 1
                                
        print("Heat diffusion finished (GPU)")
        return uf

    def _generate_final_labels(self, uf_structure):
        """Map union-find roots to contiguous final labels."""
        print("Generating final labels (GPU)...")
        idx_to_no = {i: c.no for i, c in enumerate(self.clusters)}
        
        root_map_no = {}
        if uf_structure:
            all_nos = list(self.cluster_map.keys())
            for no in all_nos:
                if no not in uf_structure.parent: 
                    uf_structure.parent[no] = no
                root_map_no[no] = uf_structure.find(no)
        else:
            root_map_no = {c.no: c.no for c in self.clusters if c}

        unique_roots = sorted(list(set(root_map_no.values())))
        root_renumber_map = {root_id: i for i, root_id in enumerate(unique_roots)}
        num_final_segments = len(unique_roots)
        final_map = {}
        for idx, no in idx_to_no.items():
            root = root_map_no.get(no, no)
            final_label = root_renumber_map.get(root, -1)
            final_map[idx] = final_label
        lut = torch.full((self.K_actual,), -1, dtype=torch.long, device=self.device)
        for idx, final_label in final_map.items():
            if 0 <= idx < self.K_actual:
                lut[idx] = final_label
        temp_labels = self.slic_labels.clone()
        valid_mask = (temp_labels >= 0) & (temp_labels < self.K_actual)
        valid_indices_in_lut = temp_labels[valid_mask]
        if valid_indices_in_lut.numel() > 0:
            if valid_indices_in_lut.max() >= self.K_actual:
                print(f"Warning: label index {valid_indices_in_lut.max()} exceeds LUT size {self.K_actual}")
                valid_indices_in_lut = torch.clamp(valid_indices_in_lut, 0, self.K_actual - 1)
                temp_labels[valid_mask] = lut[valid_indices_in_lut]
            else:
                temp_labels[valid_mask] = lut[temp_labels[valid_mask]]
        self.final_merged_labels = temp_labels.cpu().numpy()
        
        print(f"Generated final labels with {num_final_segments} regions")
        unassigned_mask = (self.final_merged_labels == -1)
        num_unassigned = np.sum(unassigned_mask)

        if num_unassigned > 0:
            print(f"Filling {num_unassigned} unassigned pixels...")
            if np.all(unassigned_mask):
                print("Warning: all pixels are unassigned")
            else:
                distances, indices = distance_transform_edt(unassigned_mask, return_indices=True)
                nearest_labels = self.final_merged_labels[indices[0], indices[1]]
                self.final_merged_labels[unassigned_mask] = nearest_labels[unassigned_mask]
                remaining_unassigned = np.sum(self.final_merged_labels == -1)
                if remaining_unassigned > 0: print(f"Warning: {remaining_unassigned} pixels remain unassigned")
                else: print("All unassigned final-label pixels filled")
        else:
            print("No unassigned pixels found")
    def _enforce_min_size(self, MIN_SIZE):
        """Merge regions smaller than the requested minimum size."""
        if MIN_SIZE <= 0:
            return

        print(f"\n--- Step: enforce min_size={MIN_SIZE} [GPU] ---")

        labels_cpu = self.slic_labels.cpu().numpy()
        idx_to_cluster = {i: c for i, c in enumerate(self.clusters)}
        
        for iter_num in range(5): 
            print(f"  Minimum-size merge iteration {iter_num + 1}/5...")
            
            unique_labels_idx, counts = np.unique(labels_cpu[labels_cpu != -1], return_counts=True)
            size_map = {label_idx: count for label_idx, count in zip(unique_labels_idx, counts)}
            
            small_regions_idx = {idx for idx, count in size_map.items() if count < MIN_SIZE}
            small_regions_static_copy = small_regions_idx.copy()  # Keep the iteration target set fixed.
            
            if not small_regions_idx:
                print("  No regions smaller than MIN_SIZE remain")
                break
                
            print(f"    Found {len(small_regions_idx)} small regions to merge...")

            neighbors_idx = defaultdict(set)
            for h in range(self.image_height - 1):
                for w in range(self.image_width - 1):
                    current_idx = labels_cpu[h, w]
                    if current_idx == -1: continue
                    
                    right_idx = labels_cpu[h, w + 1]
                    if right_idx != -1 and right_idx != current_idx:
                        neighbors_idx[current_idx].add(right_idx)
                        neighbors_idx[right_idx].add(current_idx)
                        
                    down_idx = labels_cpu[h + 1, w]
                    if down_idx != -1 and down_idx != current_idx:
                        neighbors_idx[current_idx].add(down_idx)
                        neighbors_idx[down_idx].add(current_idx)

            num_merged_this_iter = 0
            small_regions_list = sorted(list(small_regions_idx), key=lambda idx: size_map[idx])
            
            for label_i_idx in small_regions_list:
                if labels_cpu[labels_cpu == label_i_idx].size == 0:
                    continue 

                cluster_i = idx_to_cluster.get(label_i_idx)
                if not cluster_i: continue 

                neighbors = neighbors_idx.get(label_i_idx, set())
                if not neighbors: continue 

                color_i = cluster_i.avg_color_lab
                best_neighbor_idx = -1
                min_dist = float('inf')

                for label_j_idx in neighbors:
                    if label_j_idx in small_regions_static_copy:
                        continue
                        
                    cluster_j = idx_to_cluster.get(label_j_idx)
                    if not cluster_j: continue

                    color_j = cluster_j.avg_color_lab
                    dist = np.linalg.norm(color_i - color_j) 
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor_idx = label_j_idx
                
                if best_neighbor_idx != -1:
                    labels_cpu[labels_cpu == label_i_idx] = best_neighbor_idx
                    num_merged_this_iter += 1
                    
                    size_map[best_neighbor_idx] += size_map[label_i_idx]
                    del size_map[label_i_idx]
                    small_regions_idx.discard(label_i_idx) 
            
            print(f"    Merged {num_merged_this_iter} small regions this iteration")
            if num_merged_this_iter == 0: 
                print("  No further small-region merges are possible")
                break
        
        print("  Rebuilding clusters after minimum-size enforcement...")
        
        final_unique_idx = np.unique(labels_cpu[labels_cpu != -1])
        self.K_actual = len(final_unique_idx)  # Number of remaining regions.
        
        old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(final_unique_idx)}
        
        new_clusters = []       # Rebuilt Cluster list.
        new_cluster_map = {}    # Rebuilt cluster number-to-object map.
        new_cluster_centers_list = []  # Rebuilt cluster centers.

        for old_idx in final_unique_idx:
            cluster_obj = idx_to_cluster[old_idx]
            new_clusters.append(cluster_obj)
            new_cluster_map[cluster_obj.no] = cluster_obj
            
            # Reuse the center corresponding to the old label index.
            if old_idx < self.cluster_centers.shape[0]:
                new_cluster_centers_list.append(self.cluster_centers[old_idx].cpu().numpy())
            else:
                print(f"Warning: old label {old_idx} exceeds cluster center count {self.cluster_centers.shape[0]}")
                # Preserve shape with a zero center for an invalid index.
                new_cluster_centers_list.append(np.zeros(5))


        self.clusters = new_clusters
        self.cluster_map = new_cluster_map
        
        new_labels_cpu = np.full_like(labels_cpu, -1)
        for old_idx, new_idx in old_to_new_idx_map.items():
            new_labels_cpu[labels_cpu == old_idx] = new_idx
            
        self.slic_labels = torch.from_numpy(new_labels_cpu).long().to(self.device)
        self.cluster_centers = torch.from_numpy(np.array(new_cluster_centers_list, dtype=np.float32)).float().to(self.device)
        
        print(f"  Remaining regions: {self.K_actual}")
    def _enforce_size_constraints(self, L_merged, POST_MIN_SIZE, POST_MAX_SIZE, TARGET_SPLIT_SIZE):
        """Merge undersized regions and split oversized regions."""
        if POST_MIN_SIZE <= 0 and POST_MAX_SIZE <= 0:
            return L_merged

        # Convert to GPU tensor
        if not hasattr(self, 'data_cpu'):
            print("  Creating CPU copy of LAB data for post-processing...")
            self.data_cpu = self.data.permute(1, 2, 0).cpu().numpy()
        lab_data_cpu = self.data_cpu

        # Pixel coordinates have shape (H, W, 2).
        pixel_coords_cpu = self.pixel_coords.cpu().numpy()

        # Work on a copy of the merged NumPy label map.
        L_clean = L_merged.copy() 

        # =======================================================
        # =======================================================
        if POST_MIN_SIZE > 0:
            print(f"\n--- Running Post-processing (Step 3): Enforcing POST_MIN_SIZE constraint of {POST_MIN_SIZE} pixels ---")
            # Repeat merging until stable or the iteration limit is reached.
            for iter_num in range(5):  # Limit postprocessing to five passes.
                print(f"  Post-processing iteration {iter_num + 1}/5...")
                
                # 1. Identify undersized regions.
                unique_labels, counts = np.unique(L_clean[L_clean != -1], return_counts=True)
                size_map = {label: count for label, count in zip(unique_labels, counts)}
                
                small_regions = {label for label, count in size_map.items() if count < POST_MIN_SIZE}
                
                if not small_regions:
                    print("  No regions smaller than POST_MIN_SIZE found. Stopping iteration.")
                    break
                    
                print(f"    Found {len(small_regions)} small final regions to merge...")

                # 2.1 Compute each region's average LAB color.
                avg_colors = {}
                for label in unique_labels:
                    mask = (L_clean == label)
                    avg_colors[label] = np.mean(lab_data_cpu[mask], axis=0)

                # 2.2 Build the region adjacency graph.
                neighbors_clean = defaultdict(set)
                for h in range(self.image_height - 1):
                    for w in range(self.image_width - 1):
                        current_label = L_clean[h, w]
                        if current_label == -1: continue
                        right_label = L_clean[h, w + 1]
                        if right_label != -1 and right_label != current_label:
                            neighbors_clean[current_label].add(right_label)
                            neighbors_clean[right_label].add(current_label)
                        down_label = L_clean[h + 1, w]
                        if down_label != -1 and down_label != current_label:
                            neighbors_clean[current_label].add(down_label)
                            neighbors_clean[down_label].add(current_label)

                # 3. Merge each small region into its closest-color neighbor.
                num_merged_this_iter = 0
                small_regions_list = sorted(list(small_regions), key=lambda label: size_map[label])
                
                for label_i in small_regions_list:
                    if label_i not in size_map: continue  # It may have merged earlier in this pass.

                    neighbors = neighbors_clean.get(label_i, set())
                    if not neighbors: continue 

                    color_i = avg_colors[label_i]
                    best_neighbor = -1
                    min_dist = float('inf')

                    for label_j in neighbors:
                        # Small neighbors remain valid candidates in postprocessing.
                        
                        color_j = avg_colors.get(label_j)
                        if color_j is None: continue

                        dist = np.linalg.norm(color_i - color_j)
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = label_j
                    
                    # 4. Apply the selected merge.
                    if best_neighbor != -1:
                        L_clean[L_clean == label_i] = best_neighbor
                        num_merged_this_iter += 1
                        # Update the selected neighbor's accumulated size.
                        if best_neighbor in size_map:
                            size_map[best_neighbor] += size_map[label_i]
                        else:
                            # The neighbor may not yet have a size entry.
                            size_map[best_neighbor] = size_map[label_i]
                        
                        del size_map[label_i]
                        
                print(f"    Merged {num_merged_this_iter} small regions in this iteration.")
                if num_merged_this_iter == 0:
                    print("  No more merges possible. Stopping.")
                    break
            
            print("  Post-processing min-size enforcement complete.")
        
        # =======================================================
        # =======================================================
        if POST_MAX_SIZE > 0 and KMeans is not None:
            print(f"\n--- Running Post-processing (Step 4): Enforcing POST_MAX_SIZE constraint of {POST_MAX_SIZE} pixels ---")

            # 1. Identify oversized regions.
            # Recompute sizes after minimum-size merging.
            unique_labels, counts = np.unique(L_clean[L_clean != -1], return_counts=True)
            size_map = {label: count for label, count in zip(unique_labels, counts)}
            
            large_regions = [label for label, count in size_map.items() if count > POST_MAX_SIZE]
            
            if large_regions:
                print(f"    Found {len(large_regions)} large regions to split using K-Means...")
                # Use the configured target split size.
                
                # Allocate fresh labels above the current maximum.
                next_new_label = L_clean.max() + 1
                
                # Retain SLIC scaling factors for the optional feature construction.
                M_inv = (1.0 / max(self.M, 1e-6))
                S_inv = (1.0 / max(self.S, 1e-6))

                for label_i in large_regions:
                    current_size = size_map[label_i]
                    # Choose the split count from the configured target size.
                    K_new = int(round(current_size / float(TARGET_SPLIT_SIZE)))
                    K_new = max(2, K_new)  # An oversized region splits into at least two.
                    
                    print(f"      Splitting region {label_i} (size {current_size}) into {K_new} new regions (target size {TARGET_SPLIT_SIZE})...")

                    # 3. Split the region with K-Means.
                    mask = (L_clean == label_i)
                    lab_data = lab_data_cpu[mask] # (N_pixels, 3)
                    xy_data = pixel_coords_cpu[mask]  # (N_pixels, 2)
                    
                    # Optional 5D K-Means features follow SLIC-style scaling.
                    # The intended metric combines color and spatial distances.
                    # Feature order: (L/M, a/M, b/M, h/S, w/S).
#                    features = np.hstack([
#                        lab_data * M_inv, 
#                        xy_data * S_inv
#                    ])
                    features = xy_data
                    
                    # Fit K-Means on the active feature array.
                    kmeans = KMeans(n_clusters=K_new, n_init=5, random_state=42)
                    new_sub_labels = kmeans.fit_predict(features)  # Shape (N_pixels,), values 0..K_new-1.

                    # 4. Remap local K-Means labels to global labels.
                    new_sub_labels_remapped = new_sub_labels + next_new_label
                    
                    # Write the new labels into the working map.
                    L_clean[mask] = new_sub_labels_remapped
                    
                    # Reserve the next range of global label IDs.
                    next_new_label += K_new
            else:
                print("    No regions larger than POST_MAX_SIZE found.")
        
        elif POST_MAX_SIZE > 0 and KMeans is None:
            print("Warning: Skipping max-size enforcement because scikit-learn (KMeans) is not imported.")

        return L_clean  # Return the postprocessed NumPy label map.
    def save_final_image(self, name, labels_to_draw):
        if labels_to_draw is None:
            return
        try:
            # self.rgb_data_uint8 has shape (H, W, C) and dtype uint8.
            rgb_float = self.rgb_data_uint8.astype(np.float32) / 255.0
            labels_int = labels_to_draw.astype(int)
            # Draw boundaries with palette or white based on mode
            boundary_mask = find_boundaries(labels_int, mode="inner").astype(np.uint8)
            if boundary_mask.any():
                kernel = np.ones((3, 3), np.uint8)
                boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=max(1, VIS_BOUNDARY_WIDTH - 1)).astype(bool)
                if BOUNDARY_COLOR_MODE == "white":
                    rgb_float[boundary_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                elif BOUNDARY_COLOR_MODE == "multi":
                    color_map = BOUNDARY_COLOR_PALETTE
                    color_per_pixel = color_map[labels_int % len(color_map)]
                    rgb_float[boundary_mask] = color_per_pixel[boundary_mask]
                else:  # "multi_rand"
                    n_labels = int(labels_int.max()) + 1
                    color_map = _gen_reproducible_palette(n_labels, seed=HEATOK_COLOR_SEED)
                    color_per_pixel = color_map[labels_int % len(color_map)]
                    rgb_float[boundary_mask] = color_per_pixel[boundary_mask]
            boundary_image_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
            ski_io.imsave(name, boundary_image_uint8, check_contrast=False)
        except Exception as e:
            print(f"Failed to save {name}: {e}", level="ERROR")

    def save_average_color_image(self, name, labels_to_draw):
        if labels_to_draw is None: print(f"Warning: labels are None; skipping {name}"); return
        print(f"Saving average-color image: {name}")
        try:
            labels_int = labels_to_draw.astype(int)
            labels_plus_one = labels_int + 1 # Map background label -1 to 0
            # skimage.label2rgb expects float image in [0, 1] for stable averaging.
            # Passing uint8 [0, 255] can lead to values being clipped to white.
            rgb_float = self.rgb_data_uint8.astype(np.float32) / 255.0
            
            # Use per-region mean color from the original image (paper-style token visualization).
            # This produces blocks close to original appearance instead of random pseudo-colors.
            avg_color_image = ski_color.label2rgb(
                labels_plus_one,
                image=rgb_float,
                kind='avg',
                                                     bg_label=0,
                                                     bg_color=(0, 0, 0),
                                                    )
            # Blend token mean-color map with original image so background details remain visible.
            alpha = float(np.clip(AVG_COLOR_ALPHA, 0.0, 1.0))
            avg_color_image = alpha * avg_color_image + (1.0 - alpha) * rgb_float

            # Add colored or white solid token boundaries.
            boundary_mask = find_boundaries(labels_plus_one, mode="inner").astype(np.uint8)
            if boundary_mask.any():
                kernel = np.ones((3, 3), np.uint8)
                boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=max(1, VIS_BOUNDARY_WIDTH - 1)).astype(bool)
                if BOUNDARY_COLOR_MODE == "white":
                    avg_color_image[boundary_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                elif BOUNDARY_COLOR_MODE == "multi":
                    color_map = BOUNDARY_COLOR_PALETTE
                    color_per_pixel = color_map[labels_int % len(color_map)]
                    avg_color_image[boundary_mask] = color_per_pixel[boundary_mask]
                else:  # "multi_rand"
                    n_labels = int(labels_int.max()) + 1
                    color_map = _gen_reproducible_palette(n_labels, seed=HEATOK_COLOR_SEED)
                    color_per_pixel = color_map[labels_int % len(color_map)]
                    avg_color_image[boundary_mask] = color_per_pixel[boundary_mask]
            
            output_image_uint8 = (np.clip(avg_color_image, 0, 1) * 255).astype(np.uint8)
            
            ski_io.imsave(name, output_image_uint8)
            print(f"Saved avg-color image: {name}")
        except Exception as e:
            print(f"Failed to save {name}: {e}")
            import traceback
            traceback.print_exc()
    def _extract_and_save_patches(self, final_labels_map, output_patch_dir, base_filename):
        """Extract, square-pad, resize, and save one RGB patch per region."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, skip patch extraction")
            return
            
        print(f"\n--- Extracting patches to {output_patch_dir} ---")
        os.makedirs(output_patch_dir, exist_ok=True)
        original_rgb = self.rgb_data_uint8
        unique_labels = np.unique(final_labels_map)
        num_patches_saved = 0
        
        for label_id in tqdm(unique_labels, desc="Extracting Patches"):
            if label_id == -1:
                continue
            # The mask has shape (H, W).
            mask = (final_labels_map == label_id)
            masked_image = np.zeros_like(original_rgb)
            masked_image[mask] = original_rgb[mask] # (H, W, 3)
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
                
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            # The cropped patch has shape (h, w, 3).
            cropped_patch = masked_image[y1:y2+1, x1:x2+1]
            h, w, _ = cropped_patch.shape
            max_dim = max(h, w)
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            padded_square_patch = np.pad(
                cropped_patch, 
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                'constant', 
                constant_values=0
            )
            # cv2.resize uses (width, height)
            target_size = (28, 28) 
            resized_patch = cv2.resize(
                padded_square_patch, 
                target_size, 
                interpolation=cv2.INTER_AREA 
            )
            try:
                patch_filename = f"{base_filename}_patch_{label_id}.png"
                patch_filepath = os.path.join(output_patch_dir, patch_filename)
#                ski_io.imsave(patch_filepath, padded_square_patch)
                ski_io.imsave(patch_filepath, resized_patch)
                num_patches_saved += 1
            except Exception as e:
                print(f"Warning: failed to save patch {label_id}: {e}")
                
        print(f"--- Saved {num_patches_saved} patches ---")
            
            
    
    def get_semantic_patches_tensor(self, final_labels_map, target_size=28):
        """Return square region patches as an (N, 3, H, W) tensor."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, cannot build patch tensor")
            return torch.empty(0, 3, target_size, target_size)
            
        print(f"\n--- Generating patch tensor ({target_size}x{target_size}) ---")
        original_rgb_np = self.rgb_data_uint8
        original_image_tensor = torch.from_numpy(original_rgb_np).permute(2, 0, 1).float() / 255.0
        unique_labels = np.unique(final_labels_map[final_labels_map != -1])
        num_labels = len(unique_labels)
        
        patches_list = []
        
        for label_id in tqdm(unique_labels, desc="Generating Patch Tensors"):
            mask = torch.from_numpy(final_labels_map == label_id) # (H, W)
            masked_image_tensor = original_image_tensor.clone() # (3, H, W)
            masked_image_tensor[0][~mask] = 0
            masked_image_tensor[1][~mask] = 0
            masked_image_tensor[2][~mask] = 0
            rows, cols = np.where(mask)
            if len(rows) == 0: continue
                
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            patch = masked_image_tensor[:, y1 : y2 + 1, x1 : x2 + 1] # (3, h, w)
            h, w = patch.shape[1], patch.shape[2]
            max_dim = max(h, w)
            
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            padded_patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
            resized_patch = F.interpolate(
                padded_patch.unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ) # (1, 3, 28, 28)
            
            patches_list.append(resized_patch.squeeze(0)) # (3, 28, 28)

        if not patches_list:
            print("    [Tensor Gen] no valid patches generated")
            return torch.empty(0, 3, target_size, target_size)
        final_patches_tensor = torch.stack(patches_list)
        print(f"--- Patch tensor shape: {final_patches_tensor.shape} ---")
        return final_patches_tensor
    
    def get_semantic_positions_tensor(self, final_labels_map):
        """Return region centers and Gaussian position parameters."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, cannot build position tensor")
            empty_tensor = torch.empty(0, 2)
            empty_gaussian = torch.empty(0, 5)
            return empty_tensor, empty_gaussian
        
        print(f"\n--- Generating position tensor ---")
        
        labels = np.asarray(final_labels_map)
        h, w = labels.shape[:2]
        labels_flat = labels.reshape(-1)
        valid = labels_flat >= 0
        if not np.any(valid):
            print("    [Position Gen] no valid centers")
            empty_tensor = torch.empty(0, 2)
            empty_gaussian = torch.empty(0, 5)
            return empty_tensor, empty_gaussian

        # Vectorize statistics to avoid O(K * H * W) per-label searches.
        # Recover (row, col) from flat indices without allocating np.indices arrays.
        label_vals = labels_flat[valid].astype(np.int64, copy=False)
        flat_pos = np.nonzero(valid)[0].astype(np.int64, copy=False)
        row_vals = (flat_pos // w).astype(np.float64, copy=False)
        col_vals = (flat_pos % w).astype(np.float64, copy=False)

        counts_full = np.bincount(label_vals).astype(np.float64)
        present_mask = counts_full > 0
        if not np.any(present_mask):
            empty_tensor = torch.empty(0, 2)
            empty_gaussian = torch.empty(0, 5)
            return empty_tensor, empty_gaussian

        sum_rows_full = np.bincount(label_vals, weights=row_vals, minlength=counts_full.shape[0])
        sum_cols_full = np.bincount(label_vals, weights=col_vals, minlength=counts_full.shape[0])
        sum_rows_sq_full = np.bincount(label_vals, weights=row_vals * row_vals, minlength=counts_full.shape[0])
        sum_cols_sq_full = np.bincount(label_vals, weights=col_vals * col_vals, minlength=counts_full.shape[0])
        sum_row_col_full = np.bincount(label_vals, weights=row_vals * col_vals, minlength=counts_full.shape[0])

        counts = counts_full[present_mask]
        mu_y = sum_rows_full[present_mask] / counts
        mu_x = sum_cols_full[present_mask] / counts

        e_yy = sum_rows_sq_full[present_mask] / counts
        e_xx = sum_cols_sq_full[present_mask] / counts
        e_xy = sum_row_col_full[present_mask] / counts

        cov_yy = e_yy - mu_y * mu_y
        cov_xx = e_xx - mu_x * mu_x
        cov_xy = e_xy - mu_x * mu_y

        # Preserve the original covariance fallback for single-pixel regions.
        singleton = counts <= 1.0
        cov_xx[singleton] = 1e-6
        cov_yy[singleton] = 1e-6
        cov_xy[singleton] = 0.0

        sigma_x = np.sqrt(np.maximum(cov_xx, 1e-12))
        sigma_y = np.sqrt(np.maximum(cov_yy, 1e-12))
        sigma_x = np.maximum(sigma_x, 1e-6)
        sigma_y = np.maximum(sigma_y, 1e-6)

        denom = sigma_x * sigma_y
        rho = np.zeros_like(denom, dtype=np.float64)
        nz = denom > 0
        rho[nz] = cov_xy[nz] / denom[nz]
        rho = np.clip(rho, -0.999, 0.999)

        centers_np = np.stack([mu_y, mu_x], axis=1).astype(np.float32, copy=False)
        gaussian_np = np.stack([mu_x, mu_y, sigma_x, sigma_y, rho], axis=1).astype(np.float32, copy=False)

        final_centers_tensor = torch.from_numpy(centers_np)
        final_gaussian_tensor = torch.from_numpy(gaussian_np)
        print(f"--- Position tensor shape: {final_centers_tensor.shape} ---")

        return final_centers_tensor, final_gaussian_tensor
    def run_slic_and_merge(self, slic_iterations=10, **merge_params):
        print("--- (SLIC path not used in SAM mode) ---")
        pass



# --- 5. SAMHeatDiffusionProcessor ---

class SAMHeatDiffusionProcessor(SLICProcessor):
    
    def __init__(self, sam_model_raw, sam_model_dataparallel, mask_generator, 
                 filename, output_dir_base, device_id=0):
        
        print(f"SAMHeatDiffusionProcessor: loading {filename} on cuda:{device_id}")
        
        super().__init__(filename, K=1, M=10, 
                         output_dir_base=output_dir_base, 
                         device_id=device_id)

        self.sam_raw = sam_model_raw
        self.sam_dp = sam_model_dataparallel
        self.mask_generator = mask_generator
        # Remap mask generator model when using DataParallel.
        if self.sam_dp is not None and hasattr(self.mask_generator, "model"):
            try:
                self.mask_generator.model = self.sam_dp
            except Exception:
                pass

        print("Loading unblurred RGB image for segmentation...")
        original_bgr = cv2.imread(filename)
        if original_bgr is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        self.image_rgb_numpy_unblurred = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        print("SAMHeatDiffusionProcessor initialized")

    def generate_labels_from_sam(self, morph_kernel_size=5):
        """Run SAM and build label map."""
        backend_name = getattr(self.mask_generator, "backend_name", "SAM")
        print(f"--- Running {backend_name} segmentation... ---")
        masks = self.mask_generator.generate(self.image_rgb_numpy_unblurred)
        print(f"{backend_name} produced {len(masks)} masks")

        if len(masks) == 0:
            print("Warning: SAM masks list is empty")
            self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                           dtype=torch.long, device=self.device)
            self.K_actual = 0
            self.clusters = []
            self.cluster_map = {}
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=False)
        sam_labels_cpu = np.full((self.image_height, self.image_width), -1, dtype=np.int32)
        
        # Assign SAM labels (0 to K_sam-1)
        K_sam = 0
        for i, ann in enumerate(tqdm(sorted_anns, desc="Assigning SAM labels")):
            m = ann['segmentation']
            sam_labels_cpu[m] = i
            K_sam = i + 1
        
        print(f"Assigned {K_sam} SAM regions")
        background_mask_cpu = (sam_labels_cpu == -1).astype(np.uint8)
        
        K_total = K_sam
        
        if np.sum(background_mask_cpu) > 0:
            print("Processing background regions...")
            if morph_kernel_size > 0:
                print(f"Applying {morph_kernel_size}x{morph_kernel_size} morphology open...")
                kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                opened_background_mask_cpu = cv2.morphologyEx(background_mask_cpu, cv2.MORPH_OPEN, kernel)
            else:
                opened_background_mask_cpu = background_mask_cpu

            num_bg_labels, bg_labels_matrix = cv2.connectedComponents(opened_background_mask_cpu, connectivity=8)
            
            print(f"Found {num_bg_labels - 1} connected background regions")
            if num_bg_labels > 1:
                for i in range(1, num_bg_labels): # 0 is background
                    component_mask = (bg_labels_matrix == i)
                    sam_labels_cpu[component_mask] = K_sam + i - 1
                K_total = K_sam + (num_bg_labels - 1)
        
        print(f"Total regions (SAM + background): {K_total}")
        self.K_actual = K_total
        self.slic_labels = torch.from_numpy(sam_labels_cpu).long().to(self.device)

    def _populate_clusters_from_labels(self):
        """Build cluster features and objects from segmentation labels."""
        if self.K_actual == 0:
            print("Warning: K_actual is 0, skip cluster population")
            self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device)
            self.clusters = []
            self.cluster_map = {}
            return
            
        print("Populating clusters from labels")
        if scatter_mean is None:
            raise ImportError("torch_scatter is required for cluster population")
        lab_pixels_hwc = self.data.permute(1, 2, 0) # HWC LAB
        all_pixels_data = torch.cat((lab_pixels_hwc, self.pixel_coords), dim=-1)
        
        # 2. Flatten tensors
        flat_pixels = all_pixels_data.view(self.N, 5)
        flat_labels = self.slic_labels.view(self.N)
        valid_mask = flat_labels >= 0
        if not valid_mask.any():
            print("Warning: no valid labels found")
            self.K_actual = 0
            self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device)
            self.clusters = []
            self.cluster_map = {}
            return
        
        flat_pixels_valid = flat_pixels[valid_mask]
        flat_labels_valid = flat_labels[valid_mask]
        try:
            new_centers_gpu = scatter_mean(flat_pixels_valid, flat_labels_valid, dim=0, dim_size=self.K_actual)
        except RuntimeError as e:
            print(f"torch_scatter error: {e}")
            print(f"label max: {flat_labels_valid.max()}, dim_size: {self.K_actual}")
            print("Falling back to CPU for cluster centers...")
            new_centers_list = []
            for i in range(self.K_actual):
                mask_i = (flat_labels_valid == i)
                if mask_i.any():
                    center_i = torch.mean(flat_pixels_valid[mask_i], dim=0)
                    new_centers_list.append(center_i)
                else:
                    new_centers_list.append(torch.zeros(5, device=self.device, dtype=torch.float32))
            new_centers_gpu = torch.stack(new_centers_list)
        nan_mask = torch.isnan(new_centers_gpu[:, 0])
        if nan_mask.any():
            print(f"Warning: {nan_mask.sum().item()} NaN centers found")
            # Replace NaNs with zeros
            new_centers_gpu[nan_mask] = 0.0 

        # 6. Keep centers on GPU
        self.cluster_centers = new_centers_gpu
        
        # 7. Build Python-side cluster objects from CPU copy
        new_centers_cpu = new_centers_gpu.cpu().numpy()
        
        self.clusters = []
        self.cluster_map = {}
        Cluster.cluster_index = 1 # Cluster id is 1-based

        for i in range(self.K_actual):
            l, a, b, h, w = new_centers_cpu[i]
            
            # Create Python Cluster object
            c = Cluster(h, w, l, a, b) # c.no is 1, 2, 3...
            c.avg_color_lab = np.array([l, a, b])
            
            self.clusters.append(c)
            self.cluster_map[c.no] = c
        
        print(f"Built {len(self.clusters)} clusters")
    def run_sam_and_merge(self, return_timings=False, **merge_params):
        """Run segmentation, heat-diffusion merging, visualization, and tensor extraction."""
        timings = {} if return_timings else None
        if return_timings and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        total_start_time = time.time()

        print(f"\n--- Step 1: SAM segmentation [GPU] ---")
        t_sam_start = time.perf_counter()
        sam_mem_start_mb = 0.0
        if return_timings and torch.cuda.is_available():
            torch.cuda.synchronize()
            sam_mem_start_mb = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
        sam_avg = _GPUMemoryAvgSampler(0.02) if (return_timings and torch.cuda.is_available()) else _NullSampler()
        sam_avg.__enter__()
        try:
            # Run SAM to get self.slic_labels and self.K_actual
            self.generate_labels_from_sam(
                morph_kernel_size=merge_params.get('sam_morph_kernel', 5)
            )

            # 2. Populate self.clusters / self.cluster_map / self.cluster_centers
            self._populate_clusters_from_labels()
            if self.K_actual == 0:
                print("Warning: SAM produced zero regions, skipping")
                if return_timings:
                    timings["sam_s"] = time.perf_counter() - t_sam_start
                    timings["sam_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
                    timings["sam_mem_inc_mb"] = max(timings["sam_mem_mb"] - sam_mem_start_mb, 0.0) if torch.cuda.is_available() else 0.0
                    timings["sam_mem_avg_mb"] = sam_avg.avg_mb
                    timings["diffusion_merge_s"] = 0.0
                    timings["diffusion_merge_mem_mb"] = 0.0
                    timings["diffusion_merge_mem_inc_mb"] = 0.0
                    timings["diffusion_merge_mem_avg_mb"] = 0.0
                    timings["clipping_kmeans_s"] = 0.0
                    timings["clipping_kmeans_mem_mb"] = 0.0
                    timings["clipping_kmeans_mem_inc_mb"] = 0.0
                    timings["clipping_kmeans_mem_avg_mb"] = 0.0
                    return None, None, None, timings
                return None, None, None
            print("\n--- SAM postprocess: assign unlabeled pixels [GPU/CPU] ---")
            self._assign_unlabeled_pixels()
        finally:
            sam_avg.__exit__(None, None, None)
        
        if return_timings and torch.cuda.is_available():
            torch.cuda.synchronize()
        sam_end_time = time.perf_counter()
        if return_timings:
            timings["sam_s"] = sam_end_time - t_sam_start
            timings["sam_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            timings["sam_mem_inc_mb"] = max(timings["sam_mem_mb"] - sam_mem_start_mb, 0.0) if torch.cuda.is_available() else 0.0
            timings["sam_mem_avg_mb"] = sam_avg.avg_mb
        print(f"--- SAM complete ({sam_end_time - t_sam_start:.2f}s): {self.K_actual} initial regions ---")
        min_size_threshold = merge_params.get('min_size_threshold', 0)
        if min_size_threshold > 0:
            self._enforce_min_size(min_size_threshold)
        else:
            print("\n--- Skip min-size filter (threshold=0) ---")

        # --- Step 2: heat-diffusion merge using SLICProcessor helpers ---
        print(f"\n--- Step 2: heat-diffusion merge [GPU/CPU] ---")
        merge_start_time = time.time()
        if return_timings:
            t_diffusion_start = time.perf_counter()
            diff_mem_start_mb = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                diff_mem_start_mb = torch.cuda.memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
        diff_avg = _GPUMemoryAvgSampler(0.02) if (return_timings and torch.cuda.is_available()) else _NullSampler()
        diff_avg.__enter__()
        try:
            # 1. Compute complexities
            self._calculate_complexities() # GPU + CPU pipeline
            
            # 2. Build adjacency
            # self._calculate_average_colors() # already done in _populate...
            self._build_adjacency()
            
            if not self.neighbors:
                print("Warning: no adjacency edges found")
                self.final_merged_labels = self.slic_labels.cpu().numpy()
            else:
                print("Running heat-diffusion merge...")
                self._calculate_initial_temperatures(sigma_T=merge_params.get('sigma_T', 5.0))
                self._calculate_diffusion_coeffs(K0=merge_params.get('K0', 0.5), 
                                               sigma_C=merge_params.get('sigma_C', 5.0), 
                                               alpha=merge_params.get('alpha', 0.5))
                self._simulate_heat_diffusion(delta_t=merge_params.get('delta_t', 0.001), 
                                              max_iterations=merge_params.get('diffusion_iterations', 30))
                merge_structure = self._merge_clusters(merge_threshold=merge_params.get('merge_threshold', 0.01))
                self._generate_final_labels(merge_structure)
        finally:
            diff_avg.__exit__(None, None, None)
        
        if return_timings:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings["diffusion_merge_s"] = time.perf_counter() - t_diffusion_start
            timings["diffusion_merge_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            timings["diffusion_merge_mem_inc_mb"] = max(timings["diffusion_merge_mem_mb"] - diff_mem_start_mb, 0.0) if torch.cuda.is_available() else 0.0
            timings["diffusion_merge_mem_avg_mb"] = diff_avg.avg_mb
            t_clipping_start = time.perf_counter()
            clip_mem_start_mb = 0.0
            if torch.cuda.is_available():
                clip_mem_start_mb = torch.cuda.memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
        clip_avg = _GPUMemoryAvgSampler(0.02) if (return_timings and torch.cuda.is_available()) else _NullSampler()
        clip_avg.__enter__()
        try:
            post_min_size_threshold = merge_params.get('post_min_size_threshold', 0)
            post_max_size_threshold = merge_params.get('post_max_size_threshold', 0)
            target_split_size = merge_params.get('target_split_size', 2500)  # Target size for split regions.

            if (post_min_size_threshold > 0 or post_max_size_threshold > 0) and self.final_merged_labels is not None:
                self.final_merged_labels = self._enforce_size_constraints(
                    self.final_merged_labels, 
                    post_min_size_threshold,
                    post_max_size_threshold,  # Maximum allowed region size.
                    target_split_size          # Target size after splitting.
                )
            elif (post_min_size_threshold > 0 or post_max_size_threshold > 0):
                print("Warning: final_merged_labels is None, skip post size constraints")
        finally:
            clip_avg.__exit__(None, None, None)
        
        if return_timings:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings["clipping_kmeans_s"] = time.perf_counter() - t_clipping_start
            timings["clipping_kmeans_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            timings["clipping_kmeans_mem_inc_mb"] = max(timings["clipping_kmeans_mem_mb"] - clip_mem_start_mb, 0.0) if torch.cuda.is_available() else 0.0
            timings["clipping_kmeans_mem_avg_mb"] = clip_avg.avg_mb

        patches_tensor = None
        positions_tensor = None
        gaussian_tensor = None
        if self.final_merged_labels is not None:
            num_final_segments = len(np.unique(self.final_merged_labels[self.final_merged_labels != -1]))
            # Always show final merged token count for the visualization demo.
            _BUILTIN_PRINT(
                f"[HeatTok] Final tokens after merge: {num_final_segments}"
            )

            merged_output_name_boundary = f"{self.original_filename_base}_SAM_merged_boundary.png"
            merged_full_output_path_boundary = os.path.join(self.output_dir, merged_output_name_boundary)
            self.save_final_image(merged_full_output_path_boundary, self.final_merged_labels)
            
        merge_end_time = time.time()
        print(f"--- Merge pipeline complete ({merge_end_time - merge_start_time:.2f}s) ---")
        total_end_time = time.time()
        print(f"\n{'='*10} Processing complete ({total_end_time - total_start_time:.2f}s) {'='*10}")
        if return_timings:
            return patches_tensor, positions_tensor, gaussian_tensor, timings
        return patches_tensor, positions_tensor, gaussian_tensor

# --- 6. Standalone HeatTok visualization demo ---

if __name__ == '__main__':
    # Paths relative to HeatTok repo root
    image_dir = os.environ.get(
        "HEATOK_VIS_INPUT_DIR",
        str(_REPO_ROOT / "visualization" / "examples"),
    )
    image_path = os.environ.get("HEATOK_VIS_IMAGE", "")  # optional single image
    sam_checkpoint = os.environ.get(
        "SAM_CHECKPOINT",
        str(_REPO_ROOT / "segment-anything-main" / "models" / "sam_vit_h_4b8939.pth"),
    )
    output_dir = os.environ.get(
        "HEATOK_VIS_OUTPUT_DIR",
        str(_REPO_ROOT / "visualization" / "outputs"),
    )

    # Heat-diffusion / postprocess params (same as original visualization defaults)
    param_sam_pred_iou = 0.85
    param_sam_stability = 0.90
    param_sam_morph_kernel = 5
    param_min_size = 50
    param_post_min_size = 500
    param_post_max_size = int(os.environ.get("POST_MAX_SIZE", "40000"))
    param_target_size = int(os.environ.get("TARGET_SPLIT_SIZE", "15000"))
    param_merge_threshold = 0.01

    heat_params = {
        'sigma_C': 5.0,
        'alpha': 0.5,
        'K0': 0.5,
        'sigma_T': 5.0,
        'delta_t': 0.001,
        'diffusion_iterations': 30,
        'merge_threshold': param_merge_threshold,
        'sam_morph_kernel': param_sam_morph_kernel,
        'min_size_threshold': param_min_size,
        'post_min_size_threshold': param_post_min_size,
        'post_max_size_threshold': param_post_max_size,
        'target_split_size': param_target_size,
    }

    print("--- HeatTok visualization (SAM) ---")
    print(f"Image dir: {image_dir}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    print(f"Output dir: {output_dir}")

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        raise SystemExit(1)

    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise ImportError(
            "segment_anything is not installed. Install Meta SAM with:\n"
            "  cd segment-anything-main && pip install -e ."
        )
    if not os.path.exists(sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {sam_checkpoint}")
        raise SystemExit(1)

    n_gpus = torch.cuda.device_count()
    gpus_to_use = list(range(min(2, max(n_gpus, 1))))
    primary_device_id = gpus_to_use[0]
    primary_device_str = f"cuda:{primary_device_id}"
    model_type = "vit_h"

    print(f"Loading SAM ({model_type}) on {primary_device_str} ...")
    sam_raw = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_raw.to(device=primary_device_str)
    sam_dataparallel = torch.nn.DataParallel(sam_raw, device_ids=gpus_to_use)
    mask_generator = SamAutomaticMaskGenerator(
        sam_raw,
        pred_iou_thresh=param_sam_pred_iou,
        stability_score_thresh=param_sam_stability,
    )
    print(f"SAM ready on GPUs {gpus_to_use}")

    if image_path:
        images_to_process = [image_path]
    else:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}\n"
                "Put images under visualization/examples/ or set HEATOK_VIS_INPUT_DIR / HEATOK_VIS_IMAGE."
            )
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        images_to_process = [
            os.path.join(image_dir, fn)
            for fn in sorted(os.listdir(image_dir))
            if fn.lower().endswith(valid_exts)
        ]

    if not images_to_process:
        raise RuntimeError(f"No images found in: {image_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(images_to_process)} image(s)")
    for idx, current_image_path in enumerate(images_to_process, start=1):
        print(f"[{idx}/{len(images_to_process)}] {os.path.basename(current_image_path)}")
        try:
            processor = SAMHeatDiffusionProcessor(
                sam_model_raw=sam_raw,
                sam_model_dataparallel=sam_dataparallel,
                mask_generator=mask_generator,
                filename=current_image_path,
                output_dir_base=output_dir,
                device_id=primary_device_id,
            )
            processor.run_sam_and_merge(**heat_params)
        except Exception as e:
            print(f"Failed on {current_image_path}: {e}", level="ERROR")
            import traceback
            traceback.print_exc()

    print(f"Done. Outputs under: {output_dir}")
