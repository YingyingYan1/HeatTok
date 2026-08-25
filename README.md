<div align="center">

# [ACM MM 2026] HeatTok: Enhancing Remote Sensing Image Understanding via Thermodiffusion-based Tokenization

<p>
  Yingying Yan<sup>∗</sup>,
  Jiaqi Tang<sup>∗</sup>,
  Wei Wei<sup>†</sup>,
  Qianzhou Wang,
  Jinjian Wu,
  Botong Geng,
  Jianmin Chen,
  Yuyang Xia,
  Lei Zhang
</p>

<p>
Northwestern Polytechnical University, Hong Kong University of Science and Technology
</p>

<p><sup>∗</sup> Equal contribution. <sup>†</sup> Corresponding author.</p>

<a href="https://github.com/YingyingYan1/HeatTok"><img src="https://img.shields.io/badge/Code-GitHub-111111"></a>
<a href="https://arxiv.org/abs/2608.22485"><img src="https://img.shields.io/badge/cs.CV-Paper-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
<a href="#models"><img src="https://img.shields.io/badge/Model-Hugging%20Face-facc15"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>

Official repository for **HeatTok**.

</div>

---

## 📰 News

- [2026.08] [Code](https://github.com/YingyingYan1/HeatTok) released.
- [2026.08] [Paper](https://arxiv.org/abs/2608.22485) released on arXiv.
- [2026.07] 🎉 HeatTok was accepted by ACM MM 2026!

---

## 🔎 Overview

Current visual tokenizers in Multimodal Large Language Models (MLLMs) predominantly rely on patch-based partitioning, which causes severe semantic mixture and object fragmentation in remote sensing imagery due to the irregular contours of geo-objects. Moreover, existing adaptive methods struggle to extract precise object-level tokens and lack dedicated geometric positional encodings for irregular regions.

We propose **HeatTok**, a semantic-aware tokenizer driven by thermodiffusion aggregation. Inspired by the physical principles of heat conduction, HeatTok adaptively merges adjacent homogeneous regions to generate semantically independent, object-aligned irregular tokens. To enable MLLMs to perceive these irregular shapes, we design the **Gaussian Multimodal Rotary Positional Embedding (G-MRoPE)**, which models token spatial distributions via 2D Gaussians and explicitly injects center, scale, and orientation cues.


### Key Contributions

- **HeatTok**: A semantic-aligned tokenizer that utilizes a thermodiffusion mechanism to adaptively aggregate fine-grained regions. By merging adjacent homogeneous areas, it generates boundary-adherent tokens, substantially reducing semantic mixture and fragmentation.
- **G-MRoPE**: Injects the Gaussian center, scale, and orientation parameters of irregular tokens into M-RoPE. This endows MLLMs with robust positional representations and precise perception of scale and orientation.

<p align="center">
  <img src="./frame.png" alt="Overview of the HeatTok framework" width="100%">
</p>

---

## 🛠️ Installation

HeatTok is built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Follow these steps to set up the environment:

```bash
git clone https://github.com/YingyingYan1/HeatTok.git
cd HeatTok
```

```bash
conda create -n heatok python=3.10 -y
conda activate heatok
pip install -r requirements.txt
pip install -e ".[torch,metrics]"
```

### Additional Dependencies

HeatTok requires FastSAM for initial segmentation:

```bash
# Clone FastSAM
git clone https://github.com/CASIA-LMC-Lab/FastSAM.git FastSAM-main
cd FastSAM-main
pip install -r requirements.txt
```



The default launcher uses DeepSpeed ZeRO-3 and FlashAttention 2. Install versions compatible with your CUDA and PyTorch environment.

---

## 🤖 Models

| Model | Dataset | Checkpoint |
|:------|:--------|:-----------|
| HeatTok-Qwen2.5-VL-7B | VRSBench | [Download]() |
| HeatTok-Qwen2.5-VL-7B | EarthVQA | [Download]() |

> Model weights will be released soon.

---

## 📦 Data Preparation

Place the datasets under the repository root with the following layout (relative paths):

```text
.
├── VRSBench/          # VRSBench images and annotations
├── EarthVQA/          # EarthVQA images and annotations
└── data/
    ├── dataset_info.json
    ├── vrsbench.jsonl
    └── EarthVQA.jsonl
```

### VRSBench

VRSBench is a visual question answering benchmark for remote sensing images, covering question types including category, existence, quantity, color, shape, size, position, direction, scene, and reasoning.



### EarthVQA

EarthVQA is a relation-reasoning VQA dataset for queryable Earth, containing basic judgment, relation judgment, basic counting, relation counting, object analysis, and comprehensive analysis question types.


---

## 🔥 Preprocessing: Semantic Patch Cache

HeatTok uses thermodiffusion-based preprocessing to generate semantic patches. You can **pre-generate** the cache files before training for faster training speed, or let the training process compute them online (slower).

### Pre-generate Cache (Recommended)

```bash
# Single-GPU pregeneration
python scripts/pregenerate_semantic_cache.py \
    --config examples/train_lora/qwen2_5vl_lora_sft.yaml

# Multi-GPU parallel pregeneration (e.g., 8 GPUs)
bash scripts/parallel_pregenerate.sh 8

# Specify GPU IDs (e.g., use GPUs 4,5,6,7)
bash scripts/parallel_pregenerate.sh 4 "4,5,6,7"
```

Cache files are saved to `src/semantic_patch_cache/` with the naming format:
```
{image_hash}_g{0|1}_gd{8}_s{semantic_patch_size}.pt
```

Where:
- `g`: whether `global_downsample` is enabled (0 or 1)
- `gd`: global downsample divisor (default: 8)
- `s`: semantic patch size (e.g., 28)

### Cache Contents

Each `.pt` file contains:
- Semantic patches after FastSAM segmentation + thermodiffusion merging
- Grid metadata
- Gaussian position parameters for G-MRoPE


---

## 🚀 Training

### Configuration

The main training configuration is at `examples/train_lora/qwen2_5vl_lora_sft.yaml`:


### Launch Training

```bash
# Single GPU
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml

# Multi-GPU with DeepSpeed
deepspeed --num_gpus 8 src/train.py examples/train_lora/qwen2_5vl_lora_sft.yaml
```

### Key HeatTok Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `use_semantic_patches` | Enable HeatTok semantic tokenization | `true` |
| `global_downsample` | Enable global branch for scene-level context | `true` |
| `semantic_patch_size` | Target size for semantic patches | `28` |
| `sam_checkpoint` | Path to FastSAM weights | Required |

---

## 📊 Evaluation

Evaluation code will be uploaded soon.

```bash
# Coming soon
# python evaluate.py --model_path <checkpoint> --dataset <vrsbench|earthvqa>
```

---

## 📌 Citation

If you find HeatTok useful in your research, please cite our paper:

```bibtex
@inproceedings{yan2026heatok,
    title={HeatTok: Enhancing Remote Sensing Image Understanding via Thermodiffusion-based Tokenization},
    author={Yingying Yan and Jiaqi Tang and Wei Wei and Qianzhou Wang and Jinjian Wu and Botong Geng and Jianmin Chen and Yuyang Xia and Lei Zhang},
    booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
    year={2026},
    publisher={ACM},
    address={Rio de Janeiro, Brazil},
    doi={10.1145/3767308.3835640}
}
```


---

## 🤝 Acknowledgements

We thank the authors of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for their excellent unified fine-tuning framework, [FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) for efficient segment anything, and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) for the powerful multimodal foundation model.

---

