# LFINet：面向农机轨迹影像的乡村专题路网拓扑构建

**Laplacian Frequency Interaction Network（拉普拉斯频率交互网络）** · 中国农业大学 · 陈柏艳等 · IEEE IJCNN 2026

## 论文方法总览（Overview）

![论文方法总览 / Overview of the proposed framework](overview.png)

## 论文全文（PDF）


[**LFINet · IJCNN 2026 论文全文（arXiv）**](http://arxiv.org/abs/2605.02866)

## Abstract

Rural thematic road network construction aims to extract topological road structures from movement trajectory images of agricultural machinery. However, this task faces challenges where downsampling methods commonly used in existing studies tend to blur the sparse high-frequency road structures, and the heavy noise from dense field operations often leads to fragmented or redundant topologies in the extracted networks. To address these challenges, we propose **LFINet**, a **Laplacian Frequency Interaction Network**. The network begins with a **Laplacian Multi-scale Separator (LMS)** to decouple the image into low-frequency semantic contexts and high-frequency structural details. These components are then processed by the **Cross-Frequency Interaction Block (CFIB)** through a dual-pathway architecture in which a **High-Frequency Block (HFB)** refines local structures while a **Spatial Transformer (ST)** captures global semantics. Subsequently, a **Frequency Gated Modulation (FGM)** mechanism integrates the features from pathways by leveraging semantic contexts to calibrate the structural details. Finally, a **Progressive Reconstruction Decoder** iteratively fuses multi-scale features to ensure topological consistency. Experiments conducted on a real-world agricultural trajectories dataset from **Henan Province, China**, show that LFINet establishes a new state-of-the-art. Specifically, it achieves an **F1-score of 92.54%** and an **IoU of 86.12%**, surpassing the second-ranked method by **0.64%** and **1.13%**, respectively. This confirms its capability to effectively construct topological road networks from noisy and sparse field data.

## 中文摘要要点

- **任务**：从**农业机械作业轨迹影像**中恢复**拓扑化道路结构**（乡村专题路网构建）。  
- **难点**：常用下采样易模糊**稀疏高频**道路结构；密集田间作业带来的强噪声易使提取路网**碎片化或冗余**。  
- **方法（LFINet）**：**LMS** 将输入解耦为低频语义与高频结构细节；**CFIB** 双路结构中 **HFB** 强化局部结构、**ST** 捕获全局语义；**FGM** 用语义上下文校准结构细节；**渐进式重建解码器**多尺度迭代融合以兼顾拓扑一致性。  
- **实验**：中国**河南省**真实农机轨迹数据集；**F1 = 92.54%**，**IoU = 86.12%**，相对第二名方法分别提升 **0.64%** 与 **1.13%**。

**Index Terms** — Road network extraction, Laplacian pyramid, Agricultural machinery trajectory, Frequency-aware learning, Multi-scale feature fusion.

## 论文模块与代码对应关系

| 论文模块 | 作用简述 | 本仓库代码位置 |
|----------|----------|----------------|
| **LMS**（Laplacian Multi-scale Separator） | 拉普拉斯多尺度分解，解耦低频语义与高频结构细节 | `PPB.py`：`Lap_Pyramid_Conv.pyramid_decom`；由 `Encoder.py` 调用 |
| **CFIB**（Cross-Frequency Interaction Block） | 双路交互：高频路 + 低频路并行处理 | `Encoder.py`：`HighFrequencyBlock` 三路金字塔高频 + `Spatial_Transformer` 处理最低频近似 |
| **HFB**（High-Frequency Block） | 细化局部道路结构 | `Encoder.py`：`HighFrequencyBlock`、`MultiOrderDWConv` |
| **ST**（Spatial Transformer） | 捕获全局语义 | `Spatial_Transformer.py`：`Spatial_Transformer` |
| **FGM**（Frequency Gated Modulation） | 用语义上下文门控与校准高频结构细节 | `Encoder.py`：`FrequencyGatedModulation` |
| **Progressive Reconstruction Decoder** | 渐进融合多尺度特征，输出与输入对齐的道路表征 | `model.py`：`LFINet`（跳连 + 转置卷积上采样 + 多尺度融合） |
| 数据与参考拓扑 | 轨迹/专题图与 GT 路网 | `dataset_agri.py`：`SatMapDataset`、`GraphLabelGenerator`；`dataset_agri/new_agri/` |


## 方法概述（结合代码）

### LFINet 数据流

1. **LMS**：`Lap_Pyramid_Conv` 将输入分解为多层高频残差与一层低频近似，缓解下采样对稀疏高频道路结构的模糊问题（`PPB.py`）。  
2. **CFIB / 双路编码**：  
   - **ST** 支路：对金字塔最底层低频近似做窗口注意力编码—解码，建模长程语义（`Spatial_Transformer.py`）。  
   - **HFB** 支路：各尺度高频经 `HighFrequencyBlock` 强化局部几何（`Encoder.py`）。  
3. **FGM**：`FrequencyGatedModulation` 将低频语义上采样后与高频特征交互，自适应加权融合，抑制田块与作业噪声引起的伪结构（`Encoder.py`）。  
4. **渐进式重建解码器**：`LFINet` 对多尺度特征投影、池化—上采样与 U-Net 式跳连融合，输出单通道道路概率图（`model.py`）。

### 训练目标与指标（当前实现）

- **损失**：`DiceBCELoss`（Dice 与 BCE 各 0.5 权重），见 `pl_wrapper_new.py`。  
- **输入归一化**：`ComparisonModelPL` 对 `0–255` 输入做 ImageNet 均值方差归一化。  
- **验证监控**：`val_iou`、`val_f1`（TorchMetrics，`threshold=0.5`）；可选 Weights & Biases 可视化。  
- 仓库内若存在与论文表一致的 checkpoint，可与 `metrics.py` 在验证集上复现 IoU/F1；论文报告数值以上述摘要为准。

### 路网拓扑与「当前训练行为说明」

- 样本除 `rgb`、`road_mask`、`keypoint_mask` 外，还包含 `GraphLabelGenerator` 生成的 `graph_points`、`pairs`、`connected`、`valid` 等拓扑监督相关字段（依赖 `graph_utils` 与 `igraph`，见 `dataset_agri.py`）。  
- **`ComparisonModelPL` 当前仅对 `road_mask` 优化分割损失与 IoU/F1**；若全文含拓扑联合损失，需在 `pl_wrapper_new.py` 中扩展；配置中 `TOPO_SAMPLE_NUM`、`NEIGHBOR_RADIUS` 等已为拓扑项预留（`config/256_agri_epoch100.yaml`）。

## 运行环境

### 硬件建议

- Windows 10/11 或 Linux，x86_64  
- 内存 ≥ 16 GB；NVIDIA GPU 显存 ≥ 8 GB（训练推荐）  
- 安装与 PyTorch 版本匹配的 CUDA 驱动  

### 软件依赖

- Python 3.10 或 3.11  
- PyTorch 2.x、CUDA 11.8+（按本机选择官方 wheel）  
- 主要 Python 包：`torch`、`torchvision`、`torchaudio`、`pytorch-lightning`、`torchmetrics`、`timm`、`numpy`、`scipy`、`opencv-python`、`pillow`、`tifffile`、`pyyaml`、`addict`、`rtree`、`igraph`、`wandb`、`tqdm`  

将 **`graph_utils.py`** 置于与 `dataset_agri.py` 同级或加入 `PYTHONPATH`（本仓库未内置）。

## 数据目录约定

在 `dataset_agri/new_agri/` 下按 `train` / `val` / `test` 划分，每个 basename 需包含：

| 相对路径模式 | 说明 |
|--------------|------|
| `{split}/rgb/{basename}_sat.png` | 三通道输入（可与轨迹/专题图渲染一致） |
| `{split}/mask/{basename}_mask.png` | 道路二值掩膜（主分割标签） |
| `{split}/keypoint/{basename}_keypoint.png` | 关键点/交叉口等辅助图 |
| `{split}/graph/{basename}_refine_gt_graph.p` | 参考路网 pickle，供 `GraphLabelGenerator` |

## 项目文件说明

| 文件 | 作用 |
|------|------|
| `train_comparison.py` | 训练入口：配置、`DataLoader`（`SatMapDataset`）、`ComparisonModelPL`、`Trainer` |
| `pl_wrapper_new.py` | Lightning 封装、损失与指标 |
| `model.py` | `LFINet`、渐进式重建解码器 |
| `Encoder.py` | LMS 调用、CFIB（HFB+ST）、FGM |
| `PPB.py` | 拉普拉斯卷积金字塔（LMS 核心） |
| `Spatial_Transformer.py` | ST 低频支路 |
| `layers.py` | Transformer 子层 |
| `dataset_agri.py` | 数据加载与拓扑采样 |
| `metrics.py` | 离线评估：Precision / Recall / F1 / IoU / PSNR / CRR 等 |
| `utils.py` | YAML → `addict.Dict` |
| `config/256_agri_epoch100.yaml` | 训练与 `model` 超参 |
| `*.ckpt` | Lightning 检查点 |

## 安装步骤

```bash
conda create -n lfinet python=3.10 -y
conda activate lfinet
```

按官网选择 CUDA 版本安装 PyTorch，例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install pytorch-lightning torchmetrics timm numpy scipy opencv-python pillow tifffile pyyaml addict rtree igraph wandb tqdm notebook jupyterlab
```

## 训练

```bash
python train_comparison.py --model LFINet --config config/256_agri_epoch100.yaml
```

可选：`--resume <ckpt>`、`--precision 16`、`--fast_dev_run`。

检查点与日志：

```text
lightning_logs/comparison/<model_name>/<train_id>/
├── all_ckpt/
└── best_ckpt/    # 监控 val_iou
```

## 模型测评

```bash
python metrics.py
```

在脚本中设置 `LFINet_ckpt` 与 `config/256_agri_epoch100.yaml` 以加载权重并在验证集上汇总指标。

## 开源与许可

本项目使用 PyTorch、PyTorch Lightning、TorchMetrics、timm 等开源组件，使用与分发时请遵守各自许可证并保留版权声明。
