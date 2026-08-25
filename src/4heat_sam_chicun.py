# -*- coding: utf-8 -*-
"""English docstring."""

# English comment.
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import builtins
from collections import defaultdict
import time
from tqdm import trange, tqdm
from sklearn.cluster import KMeans

# English comment.
import sys
_src_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_src_dir)
fastsam_dir = os.path.join(_repo_root, "FastSAM-main")
if fastsam_dir not in sys.path:
    sys.path.insert(0, fastsam_dir)
try:
    from fastsam import FastSAM, FastSAMPrompt
    FASTSAM_AVAILABLE = True
except ImportError:
    print("Warning: failed to import fastsam, FastSAM will be unavailable")
    FASTSAM_AVAILABLE = False
    FastSAM = None
    FastSAMPrompt = None

# English comment.
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
from skimage.segmentation import mark_boundaries
from scipy.ndimage import distance_transform_edt

# torch_scatter availability check
if scatter_mean is None:
    print("Warning: torch_scatter not available, using slower fallbacks")
    # English comment.


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


# English comment.
class UnionFind:
    """English docstring."""
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


# English comment.
class GPUHeatDiffusion:
    """English docstring."""
    
    def __init__(self, device):
        self.device = device
    
    @staticmethod
    def build_adjacency_gpu(labels: torch.Tensor) -> tuple:
        """English docstring."""
        device = labels.device
        H, W = labels.shape
        
        # English comment.
        left = labels[:, :-1]   # (H, W-1)
        right = labels[:, 1:]   # (H, W-1)
        
        # English comment.
        top = labels[:-1, :]    # (H-1, W)
        bottom = labels[1:, :]  # (H-1, W)
        
        # English comment.
        h_valid = (left >= 0) & (right >= 0) & (left != right)
        v_valid = (top >= 0) & (bottom >= 0) & (top != bottom)
        
        # Collect candidate edges
        h_edges = torch.stack([left[h_valid], right[h_valid]], dim=0)  # (2, E_h)
        v_edges = torch.stack([top[v_valid], bottom[v_valid]], dim=0)  # (2, E_v)
        
        # English comment.
        if h_edges.numel() > 0 and v_edges.numel() > 0:
            all_edges = torch.cat([h_edges, v_edges], dim=1)  # (2, E)
        elif h_edges.numel() > 0:
            all_edges = h_edges
        elif v_edges.numel() > 0:
            all_edges = v_edges
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device), 0
        
        # English comment.
        sorted_edges = torch.sort(all_edges, dim=0)[0]  # English comment.
        
        # Deduplicate edges
        unique_edges = torch.unique(sorted_edges, dim=1)  # (2, E_unique)
        
        # English comment.
        edge_index = torch.cat([unique_edges, unique_edges.flip(0)], dim=1)  # (2, 2*E)
        
        num_clusters = labels.max().item() + 1 if labels.numel() > 0 else 0
        
        return edge_index, num_clusters
    
    @staticmethod
    def compute_cluster_features_gpu(labels: torch.Tensor, 
                                     lab_data: torch.Tensor,
                                     gradient_map: torch.Tensor,
                                     num_clusters: int) -> tuple:
        """English docstring."""
        device = labels.device
        H, W = labels.shape
        
        # Flatten arrays
        flat_labels = labels.view(-1)  # (N,)
        flat_lab = lab_data.view(3, -1).t()  # (N, 3)
        flat_grad = gradient_map.view(-1)  # (N,)
        
        # English comment.
        valid_mask = flat_labels >= 0
        valid_labels = flat_labels[valid_mask]  # (M,)
        valid_lab = flat_lab[valid_mask]  # (M, 3)
        valid_grad = flat_grad[valid_mask]  # (M,)
        
        if valid_labels.numel() == 0:
            return (torch.zeros((num_clusters, 3), device=device),
                    torch.zeros(num_clusters, device=device))
        
        # English comment.
        if scatter_mean is not None:
            avg_colors = scatter_mean(valid_lab, valid_labels, dim=0, dim_size=num_clusters)
            complexities = scatter_mean(valid_grad, valid_labels, dim=0, dim_size=num_clusters)
        else:
            # English comment.
            avg_colors = torch.zeros((num_clusters, 3), device=device)
            complexities = torch.zeros(num_clusters, device=device)
            counts = torch.zeros(num_clusters, device=device)
            
            avg_colors.scatter_add_(0, valid_labels.unsqueeze(1).expand(-1, 3), valid_lab)
            complexities.scatter_add_(0, valid_labels, valid_grad)
            counts.scatter_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=torch.float))
            
            counts = counts.clamp(min=1)
            avg_colors = avg_colors / counts.unsqueeze(1)
            complexities = complexities / counts
        
        return avg_colors, complexities
    
    @staticmethod
    def compute_temperatures_gpu(avg_colors: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  num_clusters: int,
                                  sigma_T: float = 5.0) -> torch.Tensor:
        """English docstring."""
        device = avg_colors.device
        
        if edge_index.numel() == 0:
            return torch.zeros(num_clusters, device=device)
        
        # English comment.
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # English comment.
        color_src = avg_colors[src]  # (E, 3)
        color_dst = avg_colors[dst]  # (E, 3)
        color_dist = torch.norm(color_src - color_dst, dim=1)  # (E,)
        
        # English comment.
        similarity = torch.exp(-color_dist / max(sigma_T, 1e-6))  # (E,)
        
        # English comment.
        if scatter_mean is not None:
            temperatures = scatter_mean(similarity, src, dim=0, dim_size=num_clusters)
        else:
            sum_sim = torch.zeros(num_clusters, device=device)
            counts = torch.zeros(num_clusters, device=device)
            sum_sim.scatter_add_(0, src, similarity)
            counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
            counts = counts.clamp(min=1)
            temperatures = sum_sim / counts
        
        return temperatures
    
    @staticmethod
    def compute_diffusion_matrix_gpu(avg_colors: torch.Tensor,
                                      complexities: torch.Tensor,
                                      edge_index: torch.Tensor,
                                      num_clusters: int,
                                      K0: float = 1.0,
                                      sigma_C: float = 5.0,
                                      alpha: float = 0.5) -> torch.Tensor:
        """English docstring."""
        device = avg_colors.device
        
        if edge_index.numel() == 0:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device),
                (num_clusters, num_clusters)
            )
        
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # English comment.
        color_src = avg_colors[src]  # (E, 3)
        color_dst = avg_colors[dst]  # (E, 3)
        color_dist_sq = torch.sum((color_src - color_dst) ** 2, dim=1)  # (E,)
        color_term = torch.exp(-color_dist_sq / (2 * max(sigma_C ** 2, 1e-6)))  # (E,)
        
        # English comment.
        comp_src = complexities[src]  # (E,)
        comp_dst = complexities[dst]  # (E,)
        min_comp = torch.min(comp_src, comp_dst)  # (E,)
        complexity_term = 1.0 + alpha * min_comp  # (E,)
        
        # English comment.
        k_values = K0 * color_term * complexity_term  # (E,)
        
        # English comment.
        diffusion_sparse = torch.sparse_coo_tensor(
            edge_index,
            k_values,
            (num_clusters, num_clusters)
        ).coalesce()
        
        return diffusion_sparse
    
    @staticmethod
    def simulate_diffusion_gpu(temperatures: torch.Tensor,
                                diffusion_sparse: torch.Tensor,
                                delta_t: float = 0.001,
                                max_iterations: int = 50,
                                convergence_threshold: float = 1e-5) -> torch.Tensor:
        """English docstring."""
        device = temperatures.device
        K = temperatures.shape[0]
        
        if K == 0 or diffusion_sparse._nnz() == 0:
            return temperatures.clone()
        
        # English comment.
        # English comment.
        ones = torch.ones(K, device=device)
        degrees = torch.sparse.mm(diffusion_sparse, ones.unsqueeze(1)).squeeze()  # (K,)
        
        current_temps = temperatures.clone()
        
        for iteration in range(max_iterations):
            # Compute K @ T
            diffused = torch.sparse.mm(diffusion_sparse, current_temps.unsqueeze(1)).squeeze()  # (K,)
            
            # English comment.
            temp_change = delta_t * (diffused - current_temps * degrees)
            next_temps = current_temps + temp_change
            
            # English comment.
            max_change = temp_change.abs().max().item()
            if max_change < convergence_threshold:
                print(f"  Converged at iter {iteration + 1}")
                break
            
            current_temps = next_temps
        
        return current_temps
    
    @staticmethod
    def merge_by_temperature_gpu(temperatures: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  merge_threshold: float = 0.03) -> torch.Tensor:
        """English docstring."""
        device = temperatures.device
        K = temperatures.shape[0]
        
        if K == 0 or edge_index.numel() == 0:
            return torch.arange(K, device=device)
        
        # English comment.
        labels = torch.arange(K, device=device)
        
        # English comment.
        src = edge_index[0]
        dst = edge_index[1]
        temp_diff = torch.abs(temperatures[src] - temperatures[dst])
        merge_mask = temp_diff < merge_threshold
        
        merge_src = src[merge_mask]
        merge_dst = dst[merge_mask]
        
        if merge_src.numel() == 0:
            return labels
        
        # English comment.
        # English comment.
        for _ in range(int(math.log2(K)) + 2):
            # English comment.
            # labels[max(i,j)] = labels[min(i,j)]
            
            src_labels = labels[merge_src]
            dst_labels = labels[merge_dst]
            
            # English comment.
            update_mask = src_labels != dst_labels
            if not update_mask.any():
                break
            
            # English comment.
            max_labels = torch.max(src_labels[update_mask], dst_labels[update_mask])
            min_labels = torch.min(src_labels[update_mask], dst_labels[update_mask])
            
            # Path compression scatter step
            labels.scatter_(0, max_labels, min_labels)
            
            # English comment.
            labels = labels[labels]
        
        # English comment.
        unique_labels = torch.unique(labels)
        new_labels = torch.zeros_like(labels)
        for new_id, old_id in enumerate(unique_labels):
            new_labels[labels == old_id] = new_id
        
        return new_labels

    @staticmethod
    def compute_region_sizes_gpu(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
        """English docstring."""
        device = labels.device
        flat_labels = labels.view(-1)
        valid_mask = flat_labels >= 0
        valid_labels = flat_labels[valid_mask]
        
        if valid_labels.numel() == 0:
            return torch.zeros(num_labels, device=device)
        
        sizes = torch.zeros(num_labels, device=device)
        ones = torch.ones_like(valid_labels, dtype=torch.float)
        sizes.scatter_add_(0, valid_labels, ones)
        
        return sizes

    @staticmethod
    def compute_region_avg_colors_gpu(labels: torch.Tensor, 
                                       lab_data: torch.Tensor,
                                       num_labels: int) -> torch.Tensor:
        """English docstring."""
        device = labels.device
        flat_labels = labels.view(-1)
        flat_lab = lab_data.view(3, -1).t()  # (N, 3)
        
        valid_mask = flat_labels >= 0
        valid_labels = flat_labels[valid_mask]
        valid_lab = flat_lab[valid_mask]
        
        if valid_labels.numel() == 0:
            return torch.zeros((num_labels, 3), device=device)
        
        if scatter_mean is not None:
            avg_colors = scatter_mean(valid_lab, valid_labels, dim=0, dim_size=num_labels)
        else:
            avg_colors = torch.zeros((num_labels, 3), device=device)
            counts = torch.zeros(num_labels, device=device)
            avg_colors.scatter_add_(0, valid_labels.unsqueeze(1).expand(-1, 3), valid_lab)
            counts.scatter_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=torch.float))
            counts = counts.clamp(min=1)
            avg_colors = avg_colors / counts.unsqueeze(1)
        
        return avg_colors

    @staticmethod
    def enforce_min_size_gpu(labels: torch.Tensor,
                              lab_data: torch.Tensor,
                              min_size: int,
                              max_iterations: int = 5) -> torch.Tensor:
        """English docstring."""
        device = labels.device
        H, W = labels.shape
        current_labels = labels.clone()
        
        for iteration in range(max_iterations):
            # English comment.
            valid_mask = current_labels >= 0
            if not valid_mask.any():
                break
                
            unique_labels = torch.unique(current_labels[valid_mask])
            num_labels = unique_labels.max().item() + 1 if unique_labels.numel() > 0 else 0
            
            if num_labels == 0:
                break
            
            # English comment.
            sizes = GPUHeatDiffusion.compute_region_sizes_gpu(current_labels, num_labels)
            
            # English comment.
            small_mask = (sizes > 0) & (sizes < min_size)
            if not small_mask.any():
                print(f"  Iter {iteration + 1}: no regions below min_size={min_size}")
                break
            
            small_count = small_mask.sum().item()
            print(f"  Iter {iteration + 1}: small regions count = {small_count}")
            
            # English comment.
            edge_index, _ = GPUHeatDiffusion.build_adjacency_gpu(current_labels)
            
            if edge_index.numel() == 0:
                break
            
            # English comment.
            avg_colors = GPUHeatDiffusion.compute_region_avg_colors_gpu(
                current_labels, lab_data, num_labels
            )
            avg_colors = torch.nan_to_num(avg_colors, nan=0.0)
            
            # English comment.
            src = edge_index[0]
            dst = edge_index[1]
            
            # English comment.
            src_is_small = small_mask[src]
            dst_is_large = ~small_mask[dst] & (sizes[dst] > 0)
            valid_edges = src_is_small & dst_is_large
            
            if not valid_edges.any():
                print(f"  Iter {iteration + 1}: no valid merge edges")
                break
            
            valid_src = src[valid_edges]
            valid_dst = dst[valid_edges]
            
            # English comment.
            color_src = avg_colors[valid_src]
            color_dst = avg_colors[valid_dst]
            color_dist = torch.norm(color_src - color_dst, dim=1)
            
            # English comment.
            # English comment.
            merge_map = torch.full((num_labels,), -1, dtype=torch.long, device=device)
            
            # English comment.
            sorted_indices = torch.argsort(color_dist)
            sorted_src = valid_src[sorted_indices]
            sorted_dst = valid_dst[sorted_indices]
            
            # English comment.
            processed = torch.zeros(num_labels, dtype=torch.bool, device=device)
            
            for i in range(sorted_src.numel()):
                s = sorted_src[i].item()
                d = sorted_dst[i].item()
                if not processed[s] and sizes[s] > 0 and sizes[d] > 0:
                    merge_map[s] = d
                    processed[s] = True
            
            # English comment.
            merge_count = (merge_map >= 0).sum().item()
            if merge_count == 0:
                break
            
            # English comment.
            lut = torch.arange(num_labels, device=device)
            needs_merge = merge_map >= 0
            lut[needs_merge] = merge_map[needs_merge]
            
            # Build label lookup table
            valid_pixels = current_labels >= 0
            current_labels[valid_pixels] = lut[current_labels[valid_pixels]]
            
            print(f"  Iter {iteration + 1}: merged {merge_count} regions")
        
        # English comment.
        valid_mask = current_labels >= 0
        if valid_mask.any():
            unique_new = torch.unique(current_labels[valid_mask])
            relabel_lut = torch.full((current_labels.max().item() + 1,), -1, 
                                      dtype=torch.long, device=device)
            for new_id, old_id in enumerate(unique_new):
                relabel_lut[old_id] = new_id
            current_labels[valid_mask] = relabel_lut[current_labels[valid_mask]]
        
        return current_labels

    @staticmethod
    def enforce_max_size_gpu(labels: torch.Tensor,
                              pixel_coords: torch.Tensor,
                              max_size: int,
                              target_size: int) -> torch.Tensor:
        """English docstring."""
        device = labels.device
        current_labels = labels.clone()
        
        valid_mask = current_labels >= 0
        if not valid_mask.any():
            return current_labels
        
        unique_labels = torch.unique(current_labels[valid_mask])
        num_labels = unique_labels.max().item() + 1 if unique_labels.numel() > 0 else 0
        
        # English comment.
        sizes = GPUHeatDiffusion.compute_region_sizes_gpu(current_labels, num_labels)
        
        # English comment.
        large_mask = sizes > max_size
        large_labels = torch.where(large_mask)[0]
        
        if large_labels.numel() == 0:
            print(f"  No regions above max_size={max_size}")
            return current_labels
        
        print(f"  Large regions to split: {large_labels.numel()}")
        
        # English comment.
        next_label = current_labels.max().item() + 1
        
        # English comment.
        coords_cpu = pixel_coords.cpu().numpy()  # (H, W, 2)
        labels_cpu = current_labels.cpu().numpy()
        
        for label_id in large_labels:
            label_id = label_id.item()
            region_size = int(sizes[label_id].item())
            
            # English comment.
            k_new = max(2, int(round(region_size / target_size)))
            
            print(f"    Split label {label_id} (size {region_size}) into {k_new} parts")
            
            # English comment.
            mask = labels_cpu == label_id
            xy_data = coords_cpu[mask]  # (N, 2)
            
            if xy_data.shape[0] < k_new:
                continue
            
            # Run K-Means split
            try:
                kmeans = KMeans(n_clusters=k_new, n_init=5, random_state=42)
                sub_labels = kmeans.fit_predict(xy_data)
                
                # English comment.
                new_sub_labels = sub_labels + next_label
                labels_cpu[mask] = new_sub_labels
                next_label += k_new
            except Exception as e:
                print(f"    K-Means failed: {e}")
                continue
        
        # Move back to GPU
        current_labels = torch.from_numpy(labels_cpu).long().to(device)
        
        return current_labels


# --- 3. Cluster class (from heat_diffusion_code.py) ---
# English comment.
class Cluster(object):
    """English docstring."""
    cluster_index = 1
    def __init__(self, h, w, l=0, a=0, b=0):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
        self.no = Cluster.cluster_index
        Cluster.cluster_index += 1
        
        # English comment.
        self.avg_color_lab = np.array([l, a, b])
        self.complexity = 0.0
        self.temperature = 0.0

    def update(self, h, w, l, a, b):
        self.h = h; self.w = w
        self.l = l; self.a = a; self.b = b
        
    def __repr__(self):
        return f"C[{self.no}]@({self.h:.1f},{self.w:.1f})"


# --- 4. SLICProcessor class (from heat_diffusion_code.py) ---
# English comment.
class SLICProcessor(object):

    @staticmethod
    def open_image(path, device):
        """English docstring."""
        try:
            # 1. Read image with torchvision.io as (C, H, W), [0, 255], uint8
            rgb_tensor_c_first = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
            
            # English comment.
            # English comment.
            rgb_tensor_c_first_blurred = gaussian_blur(rgb_tensor_c_first, kernel_size=[7, 7])
            
            # 2. Permute to (H, W, C) [0, 255] uint8 for skimage.color
            rgb_uint8_hwc_blurred = rgb_tensor_c_first_blurred.permute(1, 2, 0).contiguous()
            
            # 3. Convert to float [0, 1] for skimage.color
            rgb_float_hwc_blurred = (rgb_uint8_hwc_blurred.cpu() / 255.0).numpy()
            
            # 4. RGB -> LAB using scikit-image
            lab_arr_hwc_blurred = ski_color.rgb2lab(rgb_float_hwc_blurred)
            
            # English comment.
            lab_tensor_c_first = torch.from_numpy(lab_arr_hwc_blurred).permute(2, 0, 1).float().to(device)

            # English comment.
            # English comment.
            rgb_uint8_hwc_original = rgb_tensor_c_first.permute(1, 2, 0).cpu().numpy()
            
            return lab_tensor_c_first, rgb_uint8_hwc_original
            
        except FileNotFoundError:
             raise FileNotFoundError(f"Image file not found at: {path}")
        except Exception as e:
             raise RuntimeError(f"Error reading/processing image {path}: {e}") from e

    def __init__(self, filename, K, M, output_dir_base=None, device_id=1):
        
        self.K = K # English comment.
        self.M = M # English comment.
        
        # English comment.
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")
        
        print(f"SLICProcessor: using device {self.device}")

        self.filename = filename
        self.original_filename_base = os.path.splitext(os.path.basename(filename))[0]
        if output_dir_base is None:
            output_dir_base = os.path.join(_repo_root, "outputs", "heatsam_output")
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
        self.K_actual = 0       # English comment.
        
        # English comment.
        self.slic_labels = torch.full((self.image_height, self.image_width), -1, 
                                      dtype=torch.long, device=self.device)
        self.dis = torch.full((self.image_height, self.image_width), float('inf'), 
                              dtype=torch.float32, device=self.device)
        # Initialized later by _populate_clusters_from_labels
        self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device) 
        
        # English comment.
        h_coords = torch.arange(self.image_height, device=self.device, dtype=torch.float32)
        w_coords = torch.arange(self.image_width, device=self.device, dtype=torch.float32)
        self.pixel_coords = torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), dim=-1) # Shape (H, W, 2)

        # English comment.
        self.gradient_map_tensor = None 
        self.neighbors = defaultdict(set)
        self.temperatures = {}
        self.diffusion_coeffs = {}
        self.final_merged_labels = None 

        Cluster.cluster_index = 1 # English comment.

    # ... (make_cluster_and_init_centers, init_clusters, get_gradient, move_clusters, assignment, update_cluster) ...
    # English comment.
    # English comment.
    
    # English comment.
    # English comment.
    
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
        # English comment.
        # English comment.
        print("SLIC init_clusters (unused for SAM path)")
        pass

    def get_gradient(self, h, w):
        # English comment.
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
        # English comment.
        print("SLIC move_clusters (unused for SAM path)")
        pass

    def assignment(self):
        # Legacy SLIC path
        # English comment.
        print("SLIC assignment (unused for SAM path)")
        pass

    def update_cluster(self):
        # Legacy SLIC path
        # English comment.
        print("SLIC update_cluster (unused for SAM path)")
        pass
        

    # English comment.

    def get_gradient_tensor(self):
        """English docstring."""
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
        """English docstring."""
        if self.slic_labels is None: return

        unassigned_mask_gpu = (self.slic_labels == -1)
        num_unassigned = unassigned_mask_gpu.sum().item()

        if num_unassigned == 0:
            print("No unlabeled pixels")
            return
        elif num_unassigned == self.N:
            print("Warning: all pixels are unlabeled")
            return
        print(f"Assigning {num_unassigned} unlabeled pixels...")
        
        unassigned_mask_cpu = unassigned_mask_gpu.cpu().numpy()
        distances_cpu, indices_cpu = distance_transform_edt(unassigned_mask_cpu, return_indices=True)
        
        indices_gpu = torch.from_numpy(indices_cpu).long().to(self.device) # (2, H, W)
        
        nearest_labels = self.slic_labels[indices_gpu[0], indices_gpu[1]]
        
        self.slic_labels[unassigned_mask_gpu] = nearest_labels[unassigned_mask_gpu]

        remaining = (self.slic_labels == -1).sum().item()
        if remaining > 0:
            print(f"Warning: {remaining} pixels still unlabeled")
        else:
            print("All unlabeled pixels assigned")


    def _calculate_complexities(self):
        """English docstring."""
        if self.gradient_map_tensor is None: 
            self.get_gradient_tensor()
        
        print("Computing cluster features on GPU (avg color + complexity)...")
        
        # English comment.
        self.avg_colors_tensor, self.complexities_tensor = GPUHeatDiffusion.compute_cluster_features_gpu(
            self.slic_labels,
            self.data,  # LAB data (3, H, W)
            self.gradient_map_tensor,
            self.K_actual
        )
        
        # Remove NaN values
        self.complexities_tensor = torch.nan_to_num(self.complexities_tensor, nan=0.0)
        self.avg_colors_tensor = torch.nan_to_num(self.avg_colors_tensor, nan=0.0)
        
        # English comment.
        complexities_cpu = self.complexities_tensor.cpu().numpy()
        avg_colors_cpu = self.avg_colors_tensor.cpu().numpy()
        
        for i in range(min(self.K_actual, len(self.clusters))):
            self.clusters[i].complexity = complexities_cpu[i]
            self.clusters[i].avg_color_lab = avg_colors_cpu[i]
                
        print("Cluster features ready (GPU)")


    def _calculate_complexities_cpu(self):
        """English docstring."""
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
        """English docstring."""
        print("Building adjacency on GPU...")
        
        # English comment.
        self.edge_index, _ = GPUHeatDiffusion.build_adjacency_gpu(self.slic_labels)
        
        # English comment.
        self.neighbors = defaultdict(set)
        idx_to_no = {i: c.no for i, c in enumerate(self.clusters)}

        if self.edge_index.numel() > 0:
            edge_cpu = self.edge_index.cpu().numpy()
            for i in range(edge_cpu.shape[1]):
                src_idx, dst_idx = edge_cpu[0, i], edge_cpu[1, i]
                src_no = idx_to_no.get(src_idx, -1)
                dst_no = idx_to_no.get(dst_idx, -1)
                if src_no != -1 and dst_no != -1:
                    self.neighbors[src_no].add(dst_no)

        num_clusters_with_neighbors = len(self.neighbors)
        total_clusters = len(self.cluster_map)
        print(f"Adjacency built: {num_clusters_with_neighbors}/{total_clusters} clusters have neighbors")


    def _calculate_average_colors(self):
        # English comment.
        print("Average colors already computed during population")
        pass 

    def _calculate_initial_temperatures(self, sigma_T=5.0):
        """English docstring."""
        print("Computing temperatures on GPU...")
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            self._build_adjacency()
        if not self.cluster_map:
            self.cluster_map = {c.no: c for c in self.clusters}
        
        # English comment.
        self.temperatures_tensor = GPUHeatDiffusion.compute_temperatures_gpu(
            self.avg_colors_tensor,
            self.edge_index,
            self.K_actual,
            sigma_T=sigma_T
        )
        
        # English comment.
        self.temperatures = {}
        temps_cpu = self.temperatures_tensor.cpu().numpy()
        for i, c in enumerate(self.clusters):
            self.temperatures[c.no] = temps_cpu[i] if i < len(temps_cpu) else 0.0
        
        print("Temperatures ready (GPU)")


    def _calculate_diffusion_coeffs(self, K0=1.0, sigma_C=5.0, alpha=0.5):
        """English docstring."""
        print("Computing diffusion coefficients on GPU...")
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            self._build_adjacency()
        if not self.cluster_map:
            self.cluster_map = {c.no: c for c in self.clusters}
        
        # English comment.
        self.diffusion_sparse = GPUHeatDiffusion.compute_diffusion_matrix_gpu(
            self.avg_colors_tensor,
            self.complexities_tensor,
            self.edge_index,
            self.K_actual,
            K0=K0,
            sigma_C=sigma_C,
            alpha=alpha
        )
        
        # English comment.
        self.diffusion_coeffs = {}
        
        print("Diffusion matrix ready (GPU)")

    def _simulate_heat_diffusion(self, delta_t=0.001, max_iterations=50):
        """English docstring."""
        print(f"Simulating heat diffusion on GPU ({max_iterations} iterations)...")
        
        # English comment.
        if not hasattr(self, 'temperatures_tensor') or self.temperatures_tensor is None:
            self._calculate_initial_temperatures()
        if not hasattr(self, 'diffusion_sparse') or self.diffusion_sparse is None:
            self._calculate_diffusion_coeffs()
        if not self.cluster_map:
            self.cluster_map = {c.no: c for c in self.clusters}
        
        # English comment.
        self.temperatures_tensor = GPUHeatDiffusion.simulate_diffusion_gpu(
            self.temperatures_tensor,
            self.diffusion_sparse,
            delta_t=delta_t,
            max_iterations=max_iterations,
            convergence_threshold=1e-5
        )
        
        # English comment.
        temps_cpu = self.temperatures_tensor.cpu().numpy()
        self.temperatures = {}
        for i, c in enumerate(self.clusters):
            self.temperatures[c.no] = temps_cpu[i] if i < len(temps_cpu) else 0.0
        
        print("Heat diffusion finished (GPU)")

    def _merge_clusters(self, merge_threshold=0.03):
        """English docstring."""
        print(f"Merging by temperature on GPU (threshold={merge_threshold})...")
        
        if not hasattr(self, 'temperatures_tensor') or self.temperatures_tensor is None:
            print("Warning: temperatures not computed")
            return None
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            self._build_adjacency()
        
        # English comment.
        self.gpu_merge_labels = GPUHeatDiffusion.merge_by_temperature_gpu(
            self.temperatures_tensor,
            self.edge_index,
            merge_threshold=merge_threshold
        )
        
        # English comment.
        cluster_ids = list(self.cluster_map.keys())
        uf = UnionFind(cluster_ids)
        
        # English comment.
        merge_labels_cpu = self.gpu_merge_labels.cpu().numpy()
        for i, c in enumerate(self.clusters):
            if i < len(merge_labels_cpu):
                new_label = merge_labels_cpu[i]
                if new_label != i and new_label < len(self.clusters):
                    target_no = self.clusters[new_label].no
                    uf.union(c.no, target_no)
        
        # English comment.
        unique_labels = len(np.unique(merge_labels_cpu))
        merged_count = self.K_actual - unique_labels
        print(f"Merge labels: {self.K_actual} -> {unique_labels} (GPU)")
        
        return uf

    def _generate_final_labels(self, uf_structure):
        """English docstring."""
        print("Generating final labels (GPU)...")
        
        # English comment.
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

        # English comment.
        final_map = {}
        for idx, no in idx_to_no.items():
            root = root_map_no.get(no, no)
            final_label = root_renumber_map.get(root, -1)
            final_map[idx] = final_label
        
        # English comment.
        lut = torch.full((self.K_actual,), -1, dtype=torch.long, device=self.device)
        for idx, final_label in final_map.items():
            if 0 <= idx < self.K_actual:
                lut[idx] = final_label
        
        # English comment.
        temp_labels = self.slic_labels.clone()
        valid_mask = (temp_labels >= 0) & (temp_labels < self.K_actual)
        
        # English comment.
        # English comment.
        valid_indices_in_lut = temp_labels[valid_mask]
        if valid_indices_in_lut.numel() > 0:
            if valid_indices_in_lut.max() >= self.K_actual:
                print(f"Warning: label index {valid_indices_in_lut.max()} exceeds LUT size {self.K_actual}")
                # English comment.
                valid_indices_in_lut = torch.clamp(valid_indices_in_lut, 0, self.K_actual - 1)
                temp_labels[valid_mask] = lut[valid_indices_in_lut]
            else:
                temp_labels[valid_mask] = lut[temp_labels[valid_mask]]
        
        # English comment.
        self.final_merged_labels = temp_labels.cpu().numpy()
        
        print(f"Final segments after merge: {num_final_segments}")

        # English comment.
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
                if remaining_unassigned > 0:
                    print(f"Warning: {remaining_unassigned} pixels still unassigned")
                else:
                    print("All unassigned pixels filled")
        else:
            print("No unassigned pixels found")

    # English comment.
    def _enforce_min_size(self, MIN_SIZE):
        """English docstring."""
        if MIN_SIZE <= 0:
            return

        print(f"\n--- Step: enforce min_size={MIN_SIZE} [GPU] ---")
        
        # English comment.
        self.slic_labels = GPUHeatDiffusion.enforce_min_size_gpu(
            self.slic_labels,
            self.data,  # LAB data (3, H, W)
            min_size=MIN_SIZE,
            max_iterations=5
        )
        
        # English comment.
        valid_mask = self.slic_labels >= 0
        if valid_mask.any():
            unique_labels = torch.unique(self.slic_labels[valid_mask])
            self.K_actual = unique_labels.numel()
        else:
            self.K_actual = 0
        
        print(f"  Remaining regions: {self.K_actual}")
        
        # English comment.
        self._rebuild_clusters_from_labels()

    def _rebuild_clusters_from_labels(self):
        """English docstring."""
        valid_mask = self.slic_labels >= 0
        if not valid_mask.any():
            self.clusters = []
            self.cluster_map = {}
            self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device)
            self.K_actual = 0
            return
        
        unique_labels = torch.unique(self.slic_labels[valid_mask])
        self.K_actual = unique_labels.numel()
        
        # English comment.
        avg_colors = GPUHeatDiffusion.compute_region_avg_colors_gpu(
            self.slic_labels, self.data, self.K_actual
        )
        avg_colors = torch.nan_to_num(avg_colors, nan=0.0)
        
        # English comment.
        flat_labels = self.slic_labels.view(-1)
        flat_coords = self.pixel_coords.view(-1, 2)
        valid_flat = flat_labels >= 0
        valid_labels = flat_labels[valid_flat]
        valid_coords = flat_coords[valid_flat]
        
        if scatter_mean is not None:
            avg_coords = scatter_mean(valid_coords, valid_labels, dim=0, dim_size=self.K_actual)
        else:
            avg_coords = torch.zeros((self.K_actual, 2), device=self.device)
            counts = torch.zeros(self.K_actual, device=self.device)
            avg_coords.scatter_add_(0, valid_labels.unsqueeze(1).expand(-1, 2), valid_coords)
            counts.scatter_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=torch.float))
            counts = counts.clamp(min=1)
            avg_coords = avg_coords / counts.unsqueeze(1)
        
        avg_coords = torch.nan_to_num(avg_coords, nan=0.0)
        
        # Rebuild clusters
        Cluster.cluster_index = 1
        self.clusters = []
        self.cluster_map = {}
        
        avg_colors_cpu = avg_colors.cpu().numpy()
        avg_coords_cpu = avg_coords.cpu().numpy()
        
        for i in range(self.K_actual):
            h, w = avg_coords_cpu[i]
            l, a, b = avg_colors_cpu[i]
            cluster_obj = Cluster(h, w, l, a, b)
            cluster_obj.avg_color_lab = avg_colors_cpu[i]
            self.clusters.append(cluster_obj)
            self.cluster_map[cluster_obj.no] = cluster_obj
        
        # Update cluster_centers (L, a, b, h, w)
        self.cluster_centers = torch.cat([avg_colors, avg_coords], dim=1)

    # English comment.
    def _enforce_size_constraints(self, L_merged, POST_MIN_SIZE, POST_MAX_SIZE, TARGET_SPLIT_SIZE):
        """English docstring."""
        if POST_MIN_SIZE <= 0 and POST_MAX_SIZE <= 0:
            return L_merged

        # Convert to GPU tensor
        L_tensor = torch.from_numpy(L_merged).long().to(self.device)

        # =======================================================
        # English comment.
        # =======================================================
        if POST_MIN_SIZE > 0:
            print(f"\n--- Postprocess 3: enforce min_size={POST_MIN_SIZE} [GPU] ---")
            L_tensor = GPUHeatDiffusion.enforce_min_size_gpu(
                L_tensor,
                self.data,
                min_size=POST_MIN_SIZE,
                max_iterations=5
            )
            print("  Min-size enforcement done")
        
        # =======================================================
        # English comment.
        # =======================================================
        if POST_MAX_SIZE > 0 and KMeans is not None:
            print(f"\n--- Postprocess 4: enforce max_size={POST_MAX_SIZE} [GPU] ---")
            L_tensor = GPUHeatDiffusion.enforce_max_size_gpu(
                L_tensor,
                self.pixel_coords,
                max_size=POST_MAX_SIZE,
                target_size=TARGET_SPLIT_SIZE
            )
            print("  Max-size enforcement done")
        elif POST_MAX_SIZE > 0 and KMeans is None:
            print("Warning: sklearn.KMeans not available, skip max-size split")

        # English comment.
        return L_tensor.cpu().numpy()


    # English comment.
    def save_final_image(self, name, labels_to_draw):
        if labels_to_draw is None:
            print(f"Warning: labels is None, skip saving {name}")
            return
        print(f"Saving boundary image: {name}")
        try:
            # self.rgb_data_uint8 ? (H, W, C) numpy uint8
            rgb_float = self.rgb_data_uint8.astype(float) / 255.0
            labels_int = labels_to_draw.astype(int)
            boundary_image = mark_boundaries(rgb_float, labels_int, color=(1, 1, 0.7), mode='inner')
            boundary_image_uint8 = (np.clip(boundary_image, 0, 1) * 255).astype(np.uint8)
            ski_io.imsave(name, boundary_image_uint8)
            print(f"Saved: {name}")
        except Exception as e:
            print(f"Failed to save {name}: {e}")

    def save_average_color_image(self, name, labels_to_draw):
        if labels_to_draw is None:
            print(f"Warning: labels is None, skip saving {name}")
            return
        print(f"Saving average-color image: {name}")
        try:
            labels_int = labels_to_draw.astype(int)
            labels_plus_one = labels_int + 1 # Map background label -1 to 0
            
            random_color_image = ski_color.label2rgb(labels_plus_one,
#                                                     image=self.rgb_data_uint8,
#                                                     kind='avg',
                                                     bg_label=0,
                                                     bg_color=(0, 0, 0),
                                                    )
            
            output_image_uint8 = (np.clip(random_color_image, 0, 1) * 255).astype(np.uint8)
            
            ski_io.imsave(name, output_image_uint8)
            print(f"Saved avg-color image: {name}")
        except Exception as e:
            print(f"Failed to save {name}: {e}")
            import traceback
            traceback.print_exc()

    # English comment.
    def _extract_and_save_patches(self, final_labels_map, output_patch_dir, base_filename):
        """English docstring."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, skip patch extraction")
            return
            
        print(f"\n--- Extracting patches to {output_patch_dir} ---")
        
        # English comment.
        os.makedirs(output_patch_dir, exist_ok=True)
        
        # English comment.
        original_rgb = self.rgb_data_uint8
        
        # English comment.
        unique_labels = np.unique(final_labels_map)
        num_patches_saved = 0
        
        for label_id in tqdm(unique_labels, desc="Extracting Patches"):
            # English comment.
            if label_id == -1:
                continue
                
            # English comment.
            # mask ? (H, W)
            mask = (final_labels_map == label_id)
            
            # English comment.
            # English comment.
            masked_image = np.zeros_like(original_rgb)
            masked_image[mask] = original_rgb[mask] # (H, W, 3)
            
            # English comment.
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue # English comment.
                
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            
            # English comment.
            # cropped_patch ? (h, w, 3)
            cropped_patch = masked_image[y1:y2+1, x1:x2+1]
            
            # English comment.
            h, w, _ = cropped_patch.shape
            max_dim = max(h, w)
            
            # English comment.
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            
            # English comment.
            padded_square_patch = np.pad(
                cropped_patch, 
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                'constant', 
                constant_values=0
            )
            
            
            # English comment.
            # cv2.resize uses (width, height)
            # English comment.
            target_size = (32, 32) 
            # English comment.
            resized_patch = cv2.resize(
                padded_square_patch, 
                target_size, 
                interpolation=cv2.INTER_AREA 
            )
            
            
            
            # English comment.
            try:
                patch_filename = f"{base_filename}_patch_{label_id}.png"
                patch_filepath = os.path.join(output_patch_dir, patch_filename)
#                ski_io.imsave(patch_filepath, padded_square_patch)
                ski_io.imsave(patch_filepath, resized_patch) # English comment.
                num_patches_saved += 1
            except Exception as e:
                print(f"Warning: failed to save patch {label_id}: {e}")
                
        print(f"--- Saved {num_patches_saved} patches ---")
            
            
    
    def get_semantic_patches_tensor(self, final_labels_map, target_size=32):
        """English docstring."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, cannot build patch tensor")
            return torch.empty(0, 3, target_size, target_size)
            
        print(f"\n--- Generating patch tensor ({target_size}x{target_size}) ---")
        
        # English comment.
        original_rgb_np = self.rgb_data_uint8
        
        # English comment.
        original_image_tensor = torch.from_numpy(original_rgb_np).permute(2, 0, 1).float() / 255.0
        
        # English comment.
        unique_labels = np.unique(final_labels_map[final_labels_map != -1])
        num_labels = len(unique_labels)
        
        patches_list = []
        
        for label_id in tqdm(unique_labels, desc="Generating Patch Tensors"):
            # English comment.
            mask = torch.from_numpy(final_labels_map == label_id) # (H, W)
            
            # English comment.
            masked_image_tensor = original_image_tensor.clone() # (3, H, W)
            masked_image_tensor[0][~mask] = 0
            masked_image_tensor[1][~mask] = 0
            masked_image_tensor[2][~mask] = 0
            
            # English comment.
            rows, cols = np.where(mask)
            if len(rows) == 0: continue
                
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            
            # English comment.
            patch = masked_image_tensor[:, y1 : y2 + 1, x1 : x2 + 1] # (3, h, w)
            
            # English comment.
            h, w = patch.shape[1], patch.shape[2]
            max_dim = max(h, w)
            
            pad_top = (max_dim - h) // 2
            pad_bottom = max_dim - h - pad_top
            pad_left = (max_dim - w) // 2
            pad_right = max_dim - w - pad_left
            
            # English comment.
            padded_patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
            
            # English comment.
            resized_patch = F.interpolate(
                padded_patch.unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ) # (1, 3, 32, 32)
            
            patches_list.append(resized_patch.squeeze(0)) # (3, 32, 32)

        if not patches_list:
            print("    [Tensor Gen] no valid patches generated")
            return torch.empty(0, 3, target_size, target_size)
            
        # English comment.
        final_patches_tensor = torch.stack(patches_list)
        print(f"--- Patch tensor shape: {final_patches_tensor.shape} ---")
        return final_patches_tensor
    
    def get_semantic_positions_tensor(self, final_labels_map):
        """English docstring."""
        if final_labels_map is None:
            print("Warning: final_merged_labels is None, cannot build position tensor")
            empty_tensor = torch.empty(0, 2)
            empty_gaussian = torch.empty(0, 5)
            return empty_tensor, empty_gaussian
        
        print(f"\n--- Generating position tensor ---")
        
        # English comment.
        unique_labels = np.unique(final_labels_map[final_labels_map != -1])
        num_labels = len(unique_labels)
        
        centers_list = []
        gaussian_list = []
        
        for label_id in tqdm(unique_labels, desc="Generating Position Tensors"):
            # English comment.
            rows, cols = np.where(final_labels_map == label_id)
            if len(rows) == 0:
                continue
            
            rows = rows.astype(np.float64)
            cols = cols.astype(np.float64)
            
            # English comment.
            mu_y = rows.mean()
            mu_x = cols.mean()
            
            # English comment.
            if rows.size > 1:
                centered_cols = cols - mu_x
                centered_rows = rows - mu_y
                cov_xx = np.mean(centered_cols * centered_cols)
                cov_yy = np.mean(centered_rows * centered_rows)
                cov_xy = np.mean(centered_cols * centered_rows)
            else:
                cov_xx = cov_yy = 1e-6
                cov_xy = 0.0
            
            sigma_x = max(np.sqrt(max(cov_xx, 1e-12)), 1e-6)
            sigma_y = max(np.sqrt(max(cov_yy, 1e-12)), 1e-6)
            if sigma_x > 0 and sigma_y > 0:
                rho = cov_xy / (sigma_x * sigma_y)
                rho = float(np.clip(rho, -0.999, 0.999))
            else:
                rho = 0.0
            
            centers_list.append(torch.tensor([mu_y, mu_x], dtype=torch.float32))
            gaussian_list.append(torch.tensor([mu_x, mu_y, sigma_x, sigma_y, rho], dtype=torch.float32))

        if not centers_list:
            print("    [Position Gen] no valid centers")
            empty_tensor = torch.empty(0, 2)
            empty_gaussian = torch.empty(0, 5)
            return empty_tensor, empty_gaussian

        final_centers_tensor = torch.stack(centers_list)
        final_gaussian_tensor = torch.stack(gaussian_list)
        print(f"--- Position tensor shape: {final_centers_tensor.shape} ---")

        return final_centers_tensor, final_gaussian_tensor
        
        
            

    # English comment.
    def run_slic_and_merge(self, slic_iterations=10, **merge_params):
        print("--- (SLIC path not used in SAM mode) ---")
        # English comment.
        pass


# --- 5. SAMHeatDiffusionProcessor class ---
# English comment.

class SAMHeatDiffusionProcessor(SLICProcessor):
    """English docstring."""
    
    def __init__(self, sam_model_raw, sam_model_dataparallel, mask_generator, 
                 filename, output_dir_base, device_id=0):
        
        print(f"SAMHeatDiffusionProcessor (FastSAM): loading {filename} on cuda:{device_id}")
        
        # English comment.
        # super().__init__ ?:
        # 1. Call open_image()
        # English comment.
        #    -> self.data (LAB tensor, GPU)
        # 3. open_image() also returns original RGB
        #    -> self.rgb_data_uint8 (RGB uint8, CPU) for visualization
        super().__init__(filename, K=1, M=10, 
                         output_dir_base=output_dir_base, 
                         device_id=device_id)

        # English comment.
        # English comment.
        self.fastsam_model = sam_model_raw  # FastSAM model
        self.sam_raw = sam_model_raw  # English comment.
        self.sam_dp = sam_model_dataparallel  # English comment.
        self.mask_generator = mask_generator  # English comment.

        # English comment.
        # English comment.
        print("Loading image for FastSAM segmentation...")
        original_bgr = cv2.imread(filename)
        if original_bgr is None:
            raise FileNotFoundError(f"FastSAM image not found: {filename}")
        self.image_rgb_numpy_unblurred = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        # Keep a PIL image for FastSAM inference
        from PIL import Image
        self.image_pil = Image.fromarray(self.image_rgb_numpy_unblurred)
        
        print("SAMHeatDiffusionProcessor (FastSAM) initialized.")

    def generate_labels_from_sam(self, morph_kernel_size=5):
        """English docstring."""
        print("--- Running FastSAM inference... ---")
        
        if not FASTSAM_AVAILABLE or self.fastsam_model is None:
            print("Warning: FastSAM is not available")
            self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                           dtype=torch.long, device=self.device)
            self.K_actual = 0
            self.clusters = []
            self.cluster_map = {}
            return
        
        # English comment.
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # English comment.
            # English comment.
            # English comment.
            # English comment.
            everything_results = self.fastsam_model(
                self.image_pil,
                device=device_str,
                retina_masks=True,
                imgsz=1024,
                conf=0.1,  # English comment.
                iou=0.5    # English comment.
            )
            
            # English comment.
            if everything_results is None or len(everything_results) == 0:
                print("Warning: FastSAM returned empty results")
                self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                               dtype=torch.long, device=self.device)
                self.K_actual = 0
                self.clusters = []
                self.cluster_map = {}
                return
            
            # English comment.
            from fastsam import FastSAMPrompt
            prompt_process = FastSAMPrompt(self.image_pil, everything_results, device=device_str)
            
            # English comment.
            if everything_results[0] is None:
                print("Warning: FastSAM results[0] is None")
                self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                               dtype=torch.long, device=self.device)
                self.K_actual = 0
                self.clusters = []
                self.cluster_map = {}
                return
            
            masks = prompt_process._format_results(everything_results[0], filter=0)
            
            print(f"FastSAM produced {len(masks)} masks")

            if len(masks) == 0:
                print("Warning: FastSAM masks list is empty")
                self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                               dtype=torch.long, device=self.device)
                self.K_actual = 0
                self.clusters = []
                self.cluster_map = {}
                return

            # English comment.
            # English comment.
            # English comment.
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=False)
            
            # English comment.
            sam_labels_cpu = np.full((self.image_height, self.image_width), -1, dtype=np.int32)
            
            # 3. Assign FastSAM labels (0 to K_sam-1)
            K_sam = 0
            # English comment.
            # English comment.
            for i, ann in enumerate(tqdm(sorted_anns, desc="Assigning FastSAM masks")):
                m = ann['segmentation']
                # English comment.
                if isinstance(m, np.ndarray):
                    sam_labels_cpu[m] = i # English comment.
                else:
                    # English comment.
                    sam_labels_cpu[m.astype(bool)] = i
                K_sam = i + 1
            
            print(f"Collected {K_sam} FastSAM masks")

            # English comment.
            background_mask_cpu = (sam_labels_cpu == -1).astype(np.uint8)
            
            K_total = K_sam
            
            if np.sum(background_mask_cpu) > 0:
                print("Processing background regions...")
                
                # English comment.
                if morph_kernel_size > 0:
                    print(f"Applying {morph_kernel_size}x{morph_kernel_size} morphology open...")
                    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                    opened_background_mask_cpu = cv2.morphologyEx(background_mask_cpu, cv2.MORPH_OPEN, kernel)
                else:
                    opened_background_mask_cpu = background_mask_cpu

                num_bg_labels, bg_labels_matrix = cv2.connectedComponents(opened_background_mask_cpu, connectivity=8)
                
                print(f"Found {num_bg_labels - 1} background components")

                # English comment.
                if num_bg_labels > 1:
                    for i in range(1, num_bg_labels): # 0 is background
                        component_mask = (bg_labels_matrix == i)
                        sam_labels_cpu[component_mask] = K_sam + i - 1
                    K_total = K_sam + (num_bg_labels - 1)
            
            print(f"Total regions (FastSAM + background): {K_total}")

            # English comment.
            self.K_actual = K_total
            self.slic_labels = torch.from_numpy(sam_labels_cpu).long().to(self.device)
            
        except Exception as e:
            print(f"Warning: FastSAM processing failed: {e}")
            import traceback
            traceback.print_exc()
            self.slic_labels = torch.zeros((self.image_height, self.image_width), 
                                           dtype=torch.long, device=self.device)
            self.K_actual = 0
            self.clusters = []
            self.cluster_map = {}
            return

    def _populate_clusters_from_labels(self):
        """English docstring."""
        if self.K_actual == 0:
            print("Warning: K_actual is 0, skip cluster population")
            self.cluster_centers = torch.empty((0, 5), dtype=torch.float32, device=self.device)
            self.clusters = []
            self.cluster_map = {}
            return
            
        print(f"Populating clusters from labels: {self.K_actual} regions")
        if scatter_mean is None:
            raise ImportError("torch_scatter is required for cluster population")
        
        # English comment.
        lab_pixels_hwc = self.data.permute(1, 2, 0) # HWC LAB
        all_pixels_data = torch.cat((lab_pixels_hwc, self.pixel_coords), dim=-1)
        
        # 2. Flatten tensors
        flat_pixels = all_pixels_data.view(self.N, 5)
        flat_labels = self.slic_labels.view(self.N)
        
        # English comment.
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

        # English comment.
        try:
            new_centers_gpu = scatter_mean(flat_pixels_valid, flat_labels_valid, dim=0, dim_size=self.K_actual)
        except RuntimeError as e:
            print(f"torch_scatter error: {e}")
            print(f"label max: {flat_labels_valid.max()}, dim_size: {self.K_actual}")
            print("Falling back to CPU for cluster centers...")
            
            # English comment.
            new_centers_list = []
            for i in range(self.K_actual):
                mask_i = (flat_labels_valid == i)
                if mask_i.any():
                    center_i = torch.mean(flat_pixels_valid[mask_i], dim=0)
                    new_centers_list.append(center_i)
                else:
                    new_centers_list.append(torch.zeros(5, device=self.device, dtype=torch.float32))
            new_centers_gpu = torch.stack(new_centers_list)


        # English comment.
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
        # English comment.
        # English comment.

    def run_sam_and_merge(self, **merge_params):
        """English docstring."""
        
        print(f"\n{'='*10} FastSAM + Heat Diffusion: {self.original_filename_base} {'='*10}")
        total_start_time = time.time()

        print(f"\n--- Step 1: FastSAM segmentation [GPU] ---")
        start_time = time.time()
        
        # 1. Run FastSAM to get self.slic_labels and self.K_actual
        self.generate_labels_from_sam(
            morph_kernel_size=merge_params.get('sam_morph_kernel', 5)
        )

        # 2. Populate self.clusters / self.cluster_map / self.cluster_centers
        self._populate_clusters_from_labels()
        
        # English comment.
        if self.K_actual == 0:
            print("Warning: FastSAM produced zero regions, skipping")
            # English comment.
            return None, None, None
            
        # English comment.
        print("\n--- FastSAM postprocess: assign unlabeled pixels [GPU/CPU] ---")
        self._assign_unlabeled_pixels()
        
        sam_end_time = time.time()
        print(f"--- FastSAM done ({sam_end_time - start_time:.2f}s), regions: {self.K_actual} ---")

        # English comment.
        sam_labels_cpu = self.slic_labels.cpu().numpy()
        sam_output_name = f"{self.original_filename_base}_sam_only_boundary.png"
        sam_full_output_path = os.path.join(self.output_dir, sam_output_name)
        self.save_final_image(sam_full_output_path, sam_labels_cpu)
        
        sam_avg_color_output_name = f"{self.original_filename_base}_sam_only_avg_color.png"
        sam_avg_color_full_output_path = os.path.join(self.output_dir, sam_avg_color_output_name)
        # English comment.
        self.save_average_color_image(sam_avg_color_full_output_path, sam_labels_cpu)

        # English comment.
        # English comment.
        min_size_threshold = merge_params.get('min_size_threshold', 0)
        if min_size_threshold > 0:
            self._enforce_min_size(min_size_threshold)
            
            # English comment.
            sam_labels_clean_cpu = self.slic_labels.cpu().numpy()
            sam_clean_output_name = f"{self.original_filename_base}_sam_clean_boundary.png"
            sam_clean_full_output_path = os.path.join(self.output_dir, sam_clean_output_name)
            self.save_final_image(sam_clean_full_output_path, sam_labels_clean_cpu)
            
            sam_clean_color_output_name = f"{self.original_filename_base}_sam_clean_avg_color.png"
            sam_clean_color_full_output_path = os.path.join(self.output_dir, sam_clean_color_output_name)
            self.save_average_color_image(sam_clean_color_full_output_path, sam_labels_clean_cpu)
            
        else:
            print("\n--- Skip min-size filter (threshold=0) ---")

        # English comment.
        num_segments_pre_merge = self.K_actual
        print(f"[FastSAM+Merge] regions before merge: {num_segments_pre_merge}")

        # --- Step 2: heat diffusion merge ---
        print(f"\n--- Step 2: Heat diffusion merge [GPU/CPU] ---")
        merge_start_time = time.time()
        
        # 1. Compute complexities
        self._calculate_complexities() # GPU + CPU pipeline
        
        # 2. Build adjacency
        # self._calculate_average_colors() # already done in _populate...
        self._build_adjacency() # English comment.
        print(f"[Adjacency] edges: {len(self.neighbors)}")
        
        if not self.neighbors:
            print("Warning: no adjacency edges found")
            self.final_merged_labels = self.slic_labels.cpu().numpy()
        else:
            print("Running heat diffusion...")
            # HeatTok paper Optimal defaults: κ0=1.0, σC=5.0, σT=5.0, α=0.5, τm=0.03
            print(f"[Heat Params] sigma_T={merge_params.get('sigma_T', 5.0)}, "
                  f"sigma_C={merge_params.get('sigma_C', 5.0)}, "
                  f"alpha={merge_params.get('alpha', 0.5)}, "
                  f"K0={merge_params.get('K0', 1.0)}, "
                  f"delta_t={merge_params.get('delta_t', 0.001)}, "
                  f"iterations={merge_params.get('diffusion_iterations', 30)}, "
                  f"merge_threshold={merge_params.get('merge_threshold', 0.03)}")
            self._calculate_initial_temperatures(sigma_T=merge_params.get('sigma_T', 5.0))
            self._calculate_diffusion_coeffs(K0=merge_params.get('K0', 1.0), 
                                           sigma_C=merge_params.get('sigma_C', 5.0), 
                                           alpha=merge_params.get('alpha', 0.5))
            self._simulate_heat_diffusion(delta_t=merge_params.get('delta_t', 0.001), 
                                          max_iterations=merge_params.get('diffusion_iterations', 30))
            merge_structure = self._merge_clusters(merge_threshold=merge_params.get('merge_threshold', 0.03))
            
            # English comment.
            if merge_structure is not None:
                unique_roots = set()
                for cluster_id in self.cluster_map.keys():
                    root = merge_structure.find(cluster_id)
                    unique_roots.add(root)
                num_merged_groups = len(unique_roots)
                print(f"[Merge] merged groups: {num_merged_groups} (before: {len(self.cluster_map)})")
            else:
                print("[Merge] no merge structure returned")
            
            # English comment.
            self._generate_final_labels(merge_structure)
        
# English comment.
#        post_min_size_threshold = merge_params.get('post_min_size_threshold', 0)
#        if post_min_size_threshold > 0 and self.final_merged_labels is not None:
#            self.final_merged_labels = self._post_enforce_min_size(
#                self.final_merged_labels, 
#                post_min_size_threshold
#            )
#        elif post_min_size_threshold > 0:
# English comment.


        # English comment.
        # English comment.
        post_min_size_threshold = merge_params.get('post_min_size_threshold', 0)
        post_max_size_threshold = merge_params.get('post_max_size_threshold', 0)
        target_split_size = merge_params.get('target_split_size', 2500) # English comment.

        if (post_min_size_threshold > 0 or post_max_size_threshold > 0) and self.final_merged_labels is not None:
            # English comment.
            self.final_merged_labels = self._enforce_size_constraints(
                self.final_merged_labels, 
                post_min_size_threshold,
                post_max_size_threshold, # (?)
                target_split_size      # (?)
            )
        elif (post_min_size_threshold > 0 or post_max_size_threshold > 0):
            print("Warning: final_merged_labels is None, skip post size constraints")
            
            
        # English comment.
        if self.final_merged_labels is not None:
            num_final_segments = len(np.unique(self.final_merged_labels[self.final_merged_labels != -1]))
            reduction_ratio = (1 - num_final_segments / num_segments_pre_merge) * 100 if num_segments_pre_merge > 0 else 0
            print(f"--- Final segments: {num_final_segments} (from {num_segments_pre_merge}, -{reduction_ratio:.1f}%) ---")
            merge_end_time = time.time()
            print(f"--- Merge done ({merge_end_time - merge_start_time:.2f}s) ---")

            merged_output_name_boundary = f"{self.original_filename_base}_SAM_merged_boundary.png"
            merged_output_name_color = f"{self.original_filename_base}_SAM_merged_avg_color.png"
            merged_full_output_path_boundary = os.path.join(self.output_dir, merged_output_name_boundary)
            merged_full_output_path_color = os.path.join(self.output_dir, merged_output_name_color)
            
            self.save_final_image(merged_full_output_path_boundary, self.final_merged_labels)
            self.save_average_color_image(merged_full_output_path_color, self.final_merged_labels)
    
            # English comment.
            
            # English comment.
            patch_output_dir = merge_params.get('patch_output_dir') # English comment.
            if patch_output_dir and self.final_merged_labels is not None:
                self._extract_and_save_patches(
                    self.final_merged_labels, 
                    patch_output_dir, 
                    self.original_filename_base # e.g. "a177"
                )

            # English comment.
            # English comment.
            target_size = merge_params.get('target_size', 32)
            patches_tensor = self.get_semantic_patches_tensor(
                self.final_merged_labels,
                target_size=target_size
            )
            positions_tensor, gaussian_tensor = self.get_semantic_positions_tensor(
                self.final_merged_labels
            )

            # English comment.
            # English comment.
            merge_square = int(merge_params.get('merge_square', 4))
            merge_square = max(merge_square, 1)
            semantic_patch_tokens = int(patches_tensor.shape[0]) if patches_tensor is not None else 0
            if semantic_patch_tokens % merge_square != 0:
                print(f"[Token Warning] patch_tokens ({semantic_patch_tokens}) not divisible by merge_square ({merge_square})")
            semantic_visual_tokens = int(math.ceil(semantic_patch_tokens / float(merge_square)))
            global_visual_tokens = int(merge_params.get('global_visual_tokens', 0))
            total_visual_tokens = global_visual_tokens + semantic_visual_tokens
            print(
                f"[Token] visual tokens -> global={global_visual_tokens}, "
                f"semantic={semantic_visual_tokens}, total={total_visual_tokens} "
                f"(patch_tokens={semantic_patch_tokens}, merge_square={merge_square}, patch_size={target_size})"
            )
            # English comment.
            
            total_end_time = time.time()
            print(f"\n{'='*10} FastSAM + Heat Diffusion done (elapsed: {total_end_time - total_start_time:.2f}s) {'='*10}\n")
            return patches_tensor, positions_tensor, gaussian_tensor
        else:
            # final_merged_labels is None: return empty outputs
            print("Warning: final_merged_labels is None, skipping patch tensor")
            total_end_time = time.time()
            print(f"\n{'='*10} FastSAM + Heat Diffusion done (elapsed: {total_end_time - total_start_time:.2f}s) {'='*10}\n")
            return None, None, None

# --- 6. Standalone demo: FastSAM + Heat Diffusion ---
# LLaMA-Factory training path loads FastSAM via sam_model_loader instead.

if __name__ == '__main__':
    
    # Paths relative to repo root (VRSBench is sibling of this project by default)
    image_path = os.path.join(_repo_root, "..", "VRSBench", "Images_train", "Images_train", "00027_0000.png")
    sam_checkpoint = os.path.join(_repo_root, "FastSAM-main", "weights", "FastSAM-x.pt")
    output_dir = os.path.join(_repo_root, "outputs", "heatsam_output")
    patch_output_dir = os.path.join(_repo_root, "outputs", "patch_output")

    param_sam_morph_kernel = 5
    param_min_size = 50
    param_post_min_size = 100
    param_post_max_size = 10000
    param_target_size = 1000

    # HeatTok paper Optimal (Ours): κ0=1.0, σC=5.0, σT=5.0, α=0.5, τm=0.03
    param_merge_threshold = 0.03
    
    heat_params = {
        'sigma_C': 5.0,
        'alpha': 0.5,
        'K0': 1.0, 
        'sigma_T': 5.0,
        'delta_t': 0.001, 
        'diffusion_iterations': 30,
        'merge_threshold': param_merge_threshold,
        'sam_morph_kernel': param_sam_morph_kernel,
        'min_size_threshold': param_min_size,
        'post_min_size_threshold': param_post_min_size,
        'post_max_size_threshold': param_post_max_size,
        'target_split_size': param_target_size,
        'patch_output_dir': patch_output_dir,
    }

    print(f"--- Run FastSAM + Heat Diffusion ---")
    print(f"Image: {image_path}")
    print(f"FastSAM checkpoint: {sam_checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"min_size = {param_min_size}")
    print(f"Post min_size = {param_post_min_size}")
    print(f"Post max_size = {param_post_max_size}")
    print(f"Target split size = {param_target_size}")
    print(f"Patch output dir = {patch_output_dir}")
    print(f"Merge threshold = {param_merge_threshold}")
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        exit()

    if FastSAM is None:
        print("Error: FastSAM is unavailable. Please install/import FastSAM.")
        exit()

    if not os.path.exists(sam_checkpoint):
        print(f"Error: FastSAM checkpoint not found: {sam_checkpoint}")
        exit()

    primary_device_id = 0
    print(f"Loading FastSAM model on cuda:{primary_device_id} ...")
    fastsam_model = FastSAM(sam_checkpoint)
    print("FastSAM ready")

    try:
        processor = SAMHeatDiffusionProcessor(
            sam_model_raw=fastsam_model,
            sam_model_dataparallel=fastsam_model,
            mask_generator=fastsam_model,
            filename=image_path,
            output_dir_base=output_dir,
            device_id=primary_device_id,
        )
        
        patches, positions, gaussian = processor.run_sam_and_merge(**heat_params)
        
        print("\n--- Result ---")
        if patches is not None and positions is not None:
            print(f"Patch tensor shape: {patches.shape}")
            print(f"Position tensor shape: {positions.shape}")
            if gaussian is not None:
                print(f"Gaussian tensor shape: {gaussian.shape}")
        else:
            print("Warning: patch or position tensor is None")

    except FileNotFoundError as e:
        print(f"\nError: missing file: {e}")
    except ImportError as e:
        print(f"\nError: missing Python module: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback; traceback.print_exc()

    print(f"\n--- Done ---")


