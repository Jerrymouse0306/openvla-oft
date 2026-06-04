# README_self

## 项目内容总结

这个仓库是 **OpenVLA-OFT** 的代码实现，用于对 Vision-Language-Action (VLA) 模型做机器人任务微调、推理部署和基准评测。它的核心思想是：输入机器人观测图像、任务语言指令，以及可选的机器人本体状态 proprioception，输出一段未来机器人动作 chunk。项目支持原始 OpenVLA 的离散动作 token 预测，也支持 OpenVLA-OFT 中更高效的连续动作头，例如 L1 regression action head 和 diffusion action head。

从整体工作流看，这个库可以分为四条主线：

1. 模型定义：视觉 backbone、语言模型 backbone、VLM/VLA wrapper、动作 token 化、连续动作头。
2. 数据处理：RLDS / Open-X Embodiment 数据加载、轨迹转换、图像和动作归一化、batch collator。
3. 训练微调：LoRA/OFT 微调入口、DDP/FSDP 训练策略、loss 和 metric 计算、checkpoint 保存。
4. 推理评测：FastAPI 动作服务、LIBERO 仿真评测、ALOHA 机器人评测、模型权重转换和验证工具。

## 核心代码

### 训练入口

- `vla-scripts/finetune.py`
  - OpenVLA-OFT 的主要微调脚本。
  - 定义 `FinetuneConfig`，配置数据集、checkpoint、LoRA、动作头、图像数量、proprio、训练步数、WandB 等参数。
  - 负责加载 base VLA、processor、action head、proprio projector、RLDSDataset，并执行训练循环。
  - 支持 L1 regression、diffusion、FiLM、多图像输入、proprio 输入和 LoRA 微调。

### 推理/部署入口

- `vla-scripts/deploy.py`
  - 启动一个 FastAPI 服务，对外暴露 `/act` 接口。
  - 接收 observation 和 instruction，调用 `get_vla_action()` 返回机器人动作。
  - 主要类是 `OpenVLAServer`，加载 VLA、processor、action head、proprio projector，并负责请求解析和动作预测。

- `experiments/robot/openvla_utils.py`
  - 推理和评测时最常用的工具函数集合。
  - 包括 `get_vla()`、`get_processor()`、`get_action_head()`、`get_proprio_projector()`、`get_vla_action()` 等。
  - 也处理 Hugging Face checkpoint 兼容、模型逻辑文件同步、LoRA/附加模块加载等细节。

### 模型主体

- `prismatic/models/vlms/prismatic.py`
  - 定义通用的 `PrismaticVLM`。
  - 负责把 vision backbone 的图像 patch embedding 通过 projector 接到 LLM embedding 空间。
  - 支持不同训练阶段的冻结/解冻策略，例如 align、finetune、full-finetune、VLA train 等。

- `prismatic/models/vlas/openvla.py`
  - 定义 `OpenVLA`，继承自 `PrismaticVLM`。
  - 增加 VLA 特有逻辑：把任务 prompt 和图像输入模型，生成动作 token，再用 `ActionTokenizer` 解码成连续动作。
  - `predict_action()` 是传统 token-based OpenVLA 推理的核心函数。

- `prismatic/extern/hf/modeling_prismatic.py`
  - Hugging Face 风格的模型实现，用于 `AutoModelForVision2Seq` 加载。
  - 包含 HF-compatible 的视觉 backbone、projector、forward/generate/action prediction 逻辑。
  - 实际部署和 fine-tune 很多时候会通过这里的 `OpenVLAForActionPrediction` 走 Hugging Face 接口。

- `prismatic/models/action_heads.py`
  - 定义连续动作头。
  - `L1RegressionActionHead`：使用 MLP 从动作 token hidden states 直接回归连续动作。
  - `DiffusionActionHead`：使用 DDIM diffusion 方式生成动作。

- `prismatic/models/projectors.py`
  - 定义额外 projector。
  - `ProprioProjector` 用于把机器人 proprio state 投影到语言模型 hidden space。
  - `NoisyActionProjector` 用于 diffusion action head 中的 noisy action 条件输入。

- `prismatic/models/film_vit_wrapper.py`
  - FiLM 视觉 backbone wrapper。
  - 用语言信息调制视觉特征，适用于 `use_film=True` 的模型设置。

### 动作表示和数据

- `prismatic/vla/action_tokenizer.py`
  - 把连续动作离散化到固定 bins，并映射到 tokenizer 词表尾部 token。
  - 也支持把动作 token ids 解码回归一化连续动作。

- `prismatic/vla/constants.py`
  - VLA 动作维度、proprio 维度、action chunk 长度、特殊 token index、归一化方式等关键常量。

- `prismatic/vla/datasets/datasets.py`
  - PyTorch Dataset wrapper。
  - `RLDSBatchTransform` 把 RLDS batch 转成 OpenVLA 训练需要的 `pixel_values`、`input_ids`、`labels`、`actions`、可选 wrist image 和 proprio。
  - `RLDSDataset` 基于 TensorFlow/RLDS pipeline 构建可迭代训练数据。

- `prismatic/vla/datasets/rlds/`
  - RLDS 数据管线实现。
  - 包括轨迹切片、图像变换、goal relabeling、任务增强、数据统计、Open-X Embodiment 数据集 mix 配置等。

### 训练支持

- `prismatic/training/train_utils.py`
  - 训练 loss 和 mask 工具。
  - 包括 action token accuracy、当前动作 mask、未来动作 mask、L1 loss 等。

- `prismatic/training/strategies/`
  - 分布式训练策略。
  - `ddp.py` 是 DistributedDataParallel 训练策略。
  - `fsdp.py` 是 Fully Sharded Data Parallel 训练策略。
  - `base_strategy.py` 定义共同接口和保存/加载逻辑。

- `prismatic/util/data_utils.py`
  - 数据 collator 和通用数据结构工具。
  - `PaddedCollatorForActionPrediction` 负责 padding token、labels、图像和动作 batch。

## 文件夹作用

### 根目录

- `README.md`
  - 官方项目简介、快速开始、安装、训练评测入口和引用信息。

- `SETUP.md`
  - 环境安装说明。

- `LIBERO.md`
  - LIBERO 仿真 benchmark 的微调和评测说明。

- `ALOHA.md`
  - ALOHA 真实机器人任务的微调和评测说明。

- `pyproject.toml`
  - Python 包元信息和依赖声明。
  - 关键依赖包括 PyTorch、Transformers fork、PEFT、TensorFlow、TFDS、dlimp、diffusers、FastAPI、uvicorn 等。

### `vla-scripts/`

面向用户直接运行的 VLA 脚本目录。

- `finetune.py`：主微调入口。
- `deploy.py`：启动推理服务。
- `merge_lora_weights_and_save.py`：合并 LoRA 权重并保存完整模型。
- `extern/convert_openvla_weights_to_hf.py`：把 OpenVLA 权重转成 Hugging Face 格式。
- `extern/verify_openvla.py`：验证 OpenVLA 权重/模型加载是否正确。

### `experiments/robot/`

机器人实验、评测和任务相关工具。

- `openvla_utils.py`：加载模型、processor、动作头和推理动作的核心工具。
- `robot_utils.py`：图像尺寸、观测预处理等机器人通用工具。
- `libero/`
  - LIBERO 仿真任务相关代码。
  - `run_libero_eval.py` 是 LIBERO 评测入口。
  - `libero_utils.py` 放 LIBERO 环境和任务辅助函数。
  - `regenerate_libero_dataset.py` 用于重新生成 LIBERO 数据。
  - `sample_libero_spatial_observation.pkl` 是 README 快速开始里的样例 observation。
- `aloha/`
  - ALOHA 真实机器人任务相关代码。
  - `run_aloha_eval.py` 是 ALOHA 评测入口。
  - `real_env.py`、`robot_utils.py`、`aloha_utils.py` 处理真实机器人环境、观测和动作。
  - `preprocess_split_aloha_data.py` 用于预处理和划分 ALOHA 数据。

### `prismatic/`

项目的核心 Python package。模型、数据、训练和工具基本都在这里。

- `prismatic/conf/`
  - Draccus 配置定义。
  - `vla.py` 定义 VLA 训练/微调配置 registry。
  - `models.py` 定义 VLM/backbone 模型配置。
  - `datasets.py` 定义数据集配置。

- `prismatic/models/`
  - 模型核心实现。
  - `backbones/vision/`：CLIP、SigLIP、DINOv2、DINO+SigLIP 等视觉 backbone。
  - `backbones/llm/`：LLaMA2、Mistral、Phi 等语言模型 backbone。
  - `backbones/llm/prompting/`：不同 LLM 对应的 prompt builder。
  - `vlms/`：通用 Vision-Language Model 抽象和 `PrismaticVLM`。
  - `vlas/`：VLA wrapper，目前核心是 `OpenVLA`。
  - `action_heads.py`：L1 regression 和 diffusion 连续动作头。
  - `projectors.py`：视觉/动作/proprio 到 LLM hidden space 的投影模块。
  - `materialize.py`：根据 registry 创建 vision backbone、LLM backbone 和 VLM。
  - `load.py`：模型加载工具。

- `prismatic/vla/`
  - VLA 专用逻辑。
  - `action_tokenizer.py`：连续动作和 token 的互相转换。
  - `constants.py`：动作、proprio、chunk、特殊 token 等常量。
  - `materialize.py`：VLA 构建入口。
  - `datasets/`：RLDS/Open-X 数据加载和转换。

- `prismatic/extern/hf/`
  - Hugging Face 兼容层。
  - `configuration_prismatic.py`：HF config。
  - `modeling_prismatic.py`：HF model。
  - `processing_prismatic.py`：HF processor/image processor。
  - 这部分对于 checkpoint 上传、`trust_remote_code` 加载、部署推理很重要。

- `prismatic/training/`
  - 训练框架和分布式策略。
  - `metrics.py`：metric 计算。
  - `train_utils.py`：loss、mask、accuracy 等训练辅助函数。
  - `strategies/`：DDP/FSDP/base strategy。

- `prismatic/preprocessing/`
  - 预处理和下载相关代码。
  - 包括数据下载、预处理数据集 registry 等。

- `prismatic/util/`
  - 通用工具。
  - 包括 batch padding、tensor/device 工具、神经网络 helper、tree map 等。

- `prismatic/overwatch/`
  - 日志封装。
  - `initialize_overwatch()` 返回统一 logger，用于训练和模型初始化过程中的结构化日志。

### `scripts/`

通用外部工具脚本。

- `scripts/extern/convert_prismatic_weights_to_hf.py`
  - 把 Prismatic 权重转换为 Hugging Face 格式。
- `scripts/extern/verify_prismatic.py`
  - 验证 Prismatic 权重和模型逻辑。

## 建议阅读顺序

如果目标是快速理解这个仓库，可以按下面顺序读：

1. `README.md`、`SETUP.md`、`LIBERO.md` 或 `ALOHA.md`
2. `vla-scripts/deploy.py`
3. `experiments/robot/openvla_utils.py`
4. `vla-scripts/finetune.py`
5. `prismatic/models/action_heads.py`
6. `prismatic/extern/hf/modeling_prismatic.py`
7. `prismatic/vla/datasets/datasets.py`
8. `prismatic/vla/datasets/rlds/oxe/configs.py` 和 `mixtures.py`

## 一句话版架构

`vla-scripts/finetune.py` 读取 RLDS 数据，经 `prismatic/vla/datasets` 转成图像、语言、动作训练 batch，送入 `prismatic/extern/hf/modeling_prismatic.py` 或 `PrismaticVLM/OpenVLA`，再通过 token prediction、L1 regression action head 或 diffusion action head 学习动作；推理时 `vla-scripts/deploy.py` 调用 `experiments/robot/openvla_utils.py` 加载模型并返回动作。

## debug记录

## 代码学习
