---
title: "Qwen 多模态模型系列：从 Qwen-VL 到 Qwen3-Omni"
date: 2026-10-02 00:00:00 +0800
permalink: /posts/qwen-multimodal-models/
description: "从视觉编码器、视觉 token、位置编码、视频建模和 Thinker-Talker 架构出发，梳理 Qwen 多模态模型系列的演进。"
categories: [Qwen]
tags: [Multimodal, Vision-Language, Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Omni]
math: true
toc: true
image:
  path: /assets/img/qwen/qwen-vl-training.png
  alt: "Qwen-VL 多阶段训练流程"
---

<div class="qwen-note" markdown="1">

这篇笔记梳理 Qwen 多模态模型的主线演进，重点关注**视觉语言模型（Vision-Language Model, VLM）**和**全模态模型（Omni Model）**：图像或视频如何变成语言模型能处理的 token，视觉信息如何和文本对齐，视频的时间信息如何建模，以及模型如何从“看图回答问题”走向视觉 Agent 和实时语音交互。

本文不把所有带有 Qwen 名字的视觉和语音项目混在一起：

- **Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL**是视觉语言理解主线；
- **Qwen2.5-Omni → Qwen3-Omni**是文本、图像、音频、视频统一输入，并可输出文本和语音的 Omni 主线；
- Qwen-Audio、Qwen2-Audio、Qwen3-ASR、Qwen3-TTS 属于专门音频/语音方向；
- Qwen-Image 属于图像生成与编辑方向，不是传统意义上的视觉语言理解模型。

多模态模型和文本模型最大的区别，是模型输入端多了一条视觉或音频信息通路。它最终仍然可以使用语言模型的自回归生成机制，但在进入语言模型之前，需要先解决一个关键问题：**如何把连续的图像、视频或音频信号转换成语言模型能够理解的离散序列或隐状态。**

## 1. 先看一条演进主线

| 系列 | 模型/报告时间 | 主要输入与输出 | 核心架构变化 | 主要能力 |
| --- | --- | --- | --- | --- |
| Qwen-VL | 2023-08/09；报告 2023-08 | 图像 + 文本 → 文本 | OpenCLIP ViT-bigG + Cross-Attention Adapter + Qwen-7B | 图像问答、OCR、定位、文本识别 |
| Qwen2-VL | 2024-08/09；报告 2024-09 | 图像/视频 + 文本 → 文本 | 675M ViT、动态分辨率、M-RoPE | 任意分辨率、视频理解、视觉定位、视觉 Agent |
| Qwen2.5-VL | 2025-01；报告 2025-02 | 图像/视频 + 文本 → 文本 | 从头训练 ViT、窗口注意力、动态 FPS、绝对时间 M-RoPE | 长视频、文档、GUI、手机/电脑 Agent |
| Qwen2.5-Omni | 2025-03；报告 2025-03 | 文本/图像/音频/视频 → 文本 + 语音 | Thinker-Talker、TMRoPE、分块流式编码 | 实时多模态对话、端到端语音输出 |
| Qwen3-VL | 2025-09 发布；报告 2025-11 | 图像/视频 + 文本 → 文本 | SigLIP2、DeepStack、Interleaved-MRoPE、文本时间戳 | 256K 长上下文、视觉思考、电脑/手机 Agent |
| Qwen3-Omni | 2025-09-22；报告 2025-09 | 文本/图像/音频/视频 → 文本 + 自然语音 | Thinker/Talker MoE、多码本语音、Code2Wav | 统一全模态理解、实时语音、音频描述 |

可以把这条路线压缩成一句话：

> **Qwen-VL 先解决“视觉信息接入语言模型”，Qwen2-VL 解决“任意分辨率和视频时空信息”，Qwen2.5-VL 解决“更高效、更真实时间对齐和视觉 Agent”，Qwen3-VL 开始把视觉理解、长上下文、思考和行动统一起来；Omni 主线则在此基础上增加音频理解和实时语音生成。**

## 2. 多模态模型到底多了什么？

一个纯文本 Qwen 大致可以写成：

```text
文本 → Tokenizer → 文本 token → Decoder Transformer → 文本输出
```

视觉语言模型则多了一条视觉编码路径：

```text
图像/视频
   ↓
视觉编码器（ViT）
   ↓
Projector / Merger / Cross-Attention Adapter
   ↓
视觉 token 或视觉 hidden states
   ↓
与文本 token 拼接或交错
   ↓
语言模型 Decoder
   ↓
文本回答、坐标、结构化结果或工具调用
```

如果图像被切成 $N$ 个 patch，视觉编码器输出可以抽象为：

$$
X_v\in\mathbb{R}^{N\times d_v},
$$

其中 $N$ 是视觉 token 数量，$d_v$ 是视觉编码器的 hidden size。连接器将它映射到语言模型的 hidden size $d_l$：

$$
X_v\rightarrow X'_v\in\mathbb{R}^{N'\times d_l}.
$$

真正的工程问题集中在三个地方：

1. **视觉编码器如何提取信息**：是使用已有 CLIP，还是从头训练 ViT；
2. **连接器如何压缩和对齐信息**：视觉 token 太多会让语言模型上下文迅速膨胀；
3. **位置和时间如何表示**：图片有二维空间，视频还有时间轴，不能只把它们当作普通文本 token 排列。

## 3. Qwen-VL：用 Adapter 把视觉接入 Qwen

Qwen-VL 的初始版本在 2023 年发布，技术报告为 [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)。需要注意，Qwen 官方后来发布的 [Qwen-VL 博客](https://qwenlm.github.io/blog/qwen-vl/)主要介绍的是 Qwen-VL-Plus 和 Qwen-VL-Max 的升级能力，不等同于最初开源版本的发布时间。

### 3.1 模型架构

Qwen-VL 的整体结构可以概括为：

```text
OpenCLIP ViT-bigG → Position-aware Cross-Attention Adapter → Qwen-7B
```

![Qwen-VL 的三阶段训练流程：视觉预训练、多任务预训练和监督微调](/assets/img/qwen/qwen-vl-training.png){: .qwen-figure width="1330" height="590" }

图 1 · Qwen-VL 原论文中的三阶段训练流程。冻结和解冻状态、视觉分辨率变化以及数据类型都画在同一张方法图中。来源：[Qwen-VL Technical Report，Figure 3](https://arxiv.org/abs/2308.12966)。
{: .qwen-caption }

- **视觉编码器**：使用 OpenCLIP 的 ViT-bigG，先把图像 patch 编码成视觉特征。它并不是直接让 Qwen-7B 读取像素，而是先使用一个独立的视觉模型提取图像表示。

- **Position-aware Cross-Attention Adapter**：视觉编码器和语言模型之间增加一个单层 cross-attention adapter。Adapter 使用一组可学习 query 从视觉特征中读取信息，并把视觉序列压缩成固定数量的视觉 token。

- **固定长度视觉 token**：Qwen-VL 使用 256 个 learnable queries，将不同数量的视觉特征压缩到固定长度。这样做可以控制视觉输入占用的语言模型上下文，代价是细粒度空间信息会受到压缩限制。

- **二维位置**：Adapter 的 Q/K 中加入二维绝对位置信息，使模型知道视觉特征来自图像的哪个位置。此时的位置建模还比较接近“图像编码器 + 连接器”的传统 VLM 方案。

- **文本化的定位结果**：除了回答问题，模型还可以输出 `<|box|>`、`<|ref|>` 等特殊标记和坐标文本，用统一的语言生成接口完成目标定位和 grounding。

### 3.2 三阶段训练

Qwen-VL 的训练过程体现了早期多模态模型常见的渐进式路线：

1. **视觉预训练**：冻结 Qwen-7B，只训练视觉编码器和 Adapter，让视觉特征先和语言空间对齐；
2. **多任务预训练**：提高图像分辨率，解冻整个多模态模型，加入图像描述、VQA、OCR、定位、引用和图文交错数据；
3. **多模态 SFT**：冻结视觉编码器，使用约 350K 多模态指令数据，训练模型成为可对话的视觉助手。

这个设计的核心思想是：先让视觉模块“说得上话”，再让完整模型学习多模态任务，最后用指令数据调整交互方式。

### 3.3 这一代的局限

Qwen-VL 的视觉 token 数量经过 Adapter 压缩并固定下来，推理成本比较可控，但对于高分辨率文档、复杂布局和长视频并不理想。后续 Qwen2-VL 的主要变化，就是不再把所有图像压成固定数量的视觉 token，而是让 token 数量随图像分辨率变化。

## 4. Qwen2-VL：动态分辨率和多维位置编码

Qwen2-VL 在 2024 年 8 月发布，72B 版本和论文在 9 月进一步公开。技术报告为 [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)，官方介绍见 [Qwen2-VL 博客](https://qwenlm.github.io/blog/qwen2-vl/)。

### 4.1 模型架构

Qwen2-VL 的主线变化是：**不再强行把每张图片压成固定数量的视觉 token，而是让视觉 token 数量适应输入内容。**

- **视觉编码器与语言模型**：使用约 675M 参数的视觉编码器，连接 Qwen2 的 2B、7B 或 72B 语言模型。视觉编码器负责感知，语言模型负责跨模态理解和生成。

- **Naive Dynamic Resolution**：模型按照输入图像的实际分辨率动态生成视觉 token。高分辨率图像可以保留更多细节，低分辨率图像则使用更少 token。它不再用固定长度 Adapter 把所有图片压缩到同一个长度，因此更适合 OCR、表格和细粒度文档理解。

- **视觉 token 与文本 token 统一输入**：视觉特征经过投影后，可以和文本 token 放在同一条因果序列中处理。模型训练目标仍然主要落在文本输出上，所以视觉理解最终通过语言生成体现出来。

- **M-RoPE**：Qwen2-VL 将位置编码拆成 temporal、height、width 三个方向。图片主要使用 height/width 两个维度，视频额外使用 temporal 维度，从而把图像空间和视频时间统一到一个位置编码框架中。

![Qwen2-VL 的多模态旋转位置编码 M-RoPE](/assets/img/qwen/qwen2-vl-mrope.png){: .qwen-figure width="3059" height="613" }

图 2 · M-RoPE 将旋转位置编码拆成时间、高度和宽度三个坐标轴。图像使用二维空间坐标，视频再增加时间坐标。来源：[Qwen2-VL Technical Report，Figure 3](https://arxiv.org/abs/2409.12191)。
{: .qwen-caption }

### 4.2 图像和视频的统一建模

Qwen2-VL 不再把视频简单理解成“抽几张图片再分别问答”。视频帧经过视觉编码后会保留时间维度，语言模型可以在图像空间和视频时间之间建立联系。

- 单张图片可以表示为空间 token；
- 多帧视频可以表示为带有时间坐标的视觉 token；
- 文本问题和视频帧在同一上下文中交错输入；
- 视觉 grounding 可以输出框、点等结构化坐标。

这使得模型能够处理长视频、事件定位、动作理解和视频问答，也为 GUI、机器人、游戏和导航 Agent 提供了视觉输入。

### 4.3 训练与能力重点

Qwen2-VL 的技术报告重点放在视觉编码器、动态分辨率、视频时空建模和视觉 Agent 数据上。相比 Qwen-VL，公开报告没有像 Qwen-1 那样完整展开一个 PPO/RLHF 式的统一后训练配方，因此不应擅自把它概括成某一种具体 RL 流程。

Qwen2-VL 的关键贡献可以概括为：

> **把视觉 token 从“固定长度的压缩接口”升级成“随输入分辨率和视频时间变化的动态序列”。**

## 5. Qwen2.5-VL：重新训练视觉编码器，理解真实时间和视觉 Agent

Qwen2.5-VL 于 2025 年 1 月发布，技术报告提交于 2025 年 2 月，报告为 [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)，官方介绍见 [Qwen2.5-VL 博客](https://qwenlm.github.io/blog/qwen2.5-vl/)。

它不是简单的“Qwen2-VL 加更多数据”。这一代真正重要的变化集中在视觉编码器、视频采样、时间编码和后训练。

### 5.1 模型架构

- **从头训练的视觉编码器**：Qwen2.5-VL 重新设计并从头训练 ViT，而不是直接沿用 Qwen2-VL 的 675M ViT。视觉编码器使用 2D-RoPE、RMSNorm、SwiGLU 和窗口注意力，在保留局部视觉建模效率的同时，间隔插入 full-attention 层。

- **窗口注意力与少量全局注意力**：大多数视觉编码器层只在局部窗口内计算注意力，降低高分辨率图像的二次复杂度；少数层使用 full attention，让全局区域之间仍然可以交换信息。

- **Patch Merger**：相邻视觉 patch 经过合并和 MLP 投影，减少进入语言模型的视觉 token 数量。它在保留动态分辨率的同时，控制跨模态序列长度。

- **Dynamic Resolution + Dynamic FPS**：图片仍然按实际尺寸产生不同数量的视觉 token；视频则不再使用固定帧率盲目抽帧，而是根据视频内容和时长动态采样 FPS，在细节、事件覆盖和计算成本之间做平衡。

- **Absolute-time M-RoPE**：Qwen2-VL 的时间位置主要按帧序号组织；Qwen2.5-VL 进一步把视频时间编码和真实时间间隔对齐，使模型知道事件发生在视频的第几秒，而不仅仅是第几帧。

### 5.2 三阶段预训练

报告将训练分为三个阶段：

1. **Visual Pre-training**：约 1.5T token，主要训练视觉编码器，数据覆盖图像描述、知识和 OCR；
2. **Multimodal Pre-training**：约 2T token，加入纯文本、图文交错、VQA、视频、grounding 和 Agent 数据，并联合训练视觉模块与语言模型；
3. **Long-context Pre-training**：约 0.6T token，将序列长度扩展到 32K，重点加入长视频、长文档和 Agent 数据。

### 5.3 后训练：SFT + DPO

Qwen2.5-VL 的报告明确写的是 **SFT + DPO**，不要把它写成 Qwen2.5 文本模型那套 GRPO 流程。

![Qwen2.5-VL 的视觉编码器、动态分辨率和绝对时间建模](/assets/img/qwen/qwen25-vl-architecture.png){: .qwen-figure width="4579" height="2999" }

图 3 · Qwen2.5-VL 原论文中的整体架构图：图像和视频经过动态分辨率视觉编码器后，进入 Qwen2.5 语言模型；右侧展示窗口注意力与少量 full attention，底部展示动态 FPS 和绝对时间 M-RoPE。来源：[Qwen2.5-VL Technical Report，Figure 1](https://arxiv.org/abs/2502.13923)。
{: .qwen-caption }

- **SFT**：约 2M 条样本，覆盖纯文本和多模态任务，重点训练文档、图表、OCR、空间定位、视频理解、GUI 和 Agent 能力；
- **DPO**：进一步优化回答质量和人类偏好，报告中视觉编码器在这一阶段保持冻结。

### 5.4 能力变化

Qwen2.5-VL 重点强化了：

- 文档、表格、图表和复杂布局理解；
- 长视频事件定位；
- 视觉 grounding，输出框和点坐标；
- 电脑和手机使用；
- 结构化 JSON 输出；
- 数学、图表推理和视觉 Agent。

这一代的核心转变是：

> **模型不再只是“看懂图片并回答”，而是开始根据视觉信息定位对象、读取结构、理解时间并执行操作。**

## 6. Qwen2.5-Omni：从视觉语言模型走向全模态实时交互

Qwen2.5-Omni 于 2025 年 3 月发布，报告为 [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)，官方介绍见 [Qwen2.5 Omni 博客](https://qwenlm.github.io/blog/qwen2.5-omni/)。

它和 Qwen2.5-VL 的关系不是简单的“大一点的 VL 模型”：Qwen2.5-VL 主要解决视觉理解，Qwen2.5-Omni 试图把文本、图像、音频和视频放进一个端到端模型，并同时生成文本和自然语音。

### 6.1 Thinker-Talker 架构

- **Thinker**：负责理解文本、图像、音频和视频，并生成文本 hidden states 与文本回答。它可以理解为负责“思考和组织答案”的主干语言模型。

- **Talker**：读取 Thinker 的高层表示，生成离散语音 token。它不是简单外挂一个 TTS API，而是和 Thinker 一起端到端训练，使文本回答和语音回答共享上下文。

- **TMRoPE**：音频和视频具有不同的时间轴。Qwen2.5-Omni 使用 Time-aligned Multimodal RoPE，把视频时间戳和音频时间对齐，使模型能够理解“某个声音在画面中什么时候发生”。

- **分块流式编码**：音频和视觉编码器以 block-wise 方式处理输入，模型不必等整段音视频结束后才开始回答，因此可以实现实时交互。

- **滑窗 DiT 语音解码**：Talker 生成离散语音 token 后，使用受限感受野的解码器逐块重建语音，降低首包延迟。

![Qwen2.5-Omni 的 Thinker-Talker 架构](/assets/img/qwen/qwen25-omni-thinker-talker.png){: .qwen-figure width="860" height="750" }

图 4 · Qwen2.5-Omni 原论文中的 Thinker-Talker 总览：Thinker 处理文本、视觉和音频输入并生成文本，Talker 读取 Thinker 的高层表示并流式生成语音。来源：[Qwen2.5-Omni Technical Report，Figure 2](https://arxiv.org/abs/2503.20215)。
{: .qwen-caption }

### 6.2 训练与交互形式

Qwen2.5-Omni 的训练目标不仅是音频理解准确，还包括端到端语音指令跟随、文本回答、图像/视频理解和语音生成自然度。它支持：

- 只输入文本；
- 输入图像或视频并用文本提问；
- 输入音频，让模型分析说话内容、环境声或音乐；
- 输入音视频并进行跨模态问答；
- 以文本和实时语音两种形式输出。

因此，Omni 模型的核心难点从“视觉 token 如何对齐文本”扩展成了“多个连续模态如何共享时间轴，并在输出端同时生成文字和语音”。

## 7. Qwen3-VL：视觉理解、思考与行动统一

Qwen3-VL 的模型权重约在 2025 年 9 月 23 日发布，技术报告则在 2025 年 11 月提交，二者日期不要混写。报告为 [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)，可查看 [HTML 全文](https://arxiv.org/html/2511.21631v1)和官方发布说明。

### 7.1 模型架构

- **SigLIP2 视觉编码器**：Qwen3-VL 使用更强的视觉编码器处理图像和视频输入，再通过视觉到语言的 merger 接入语言模型。报告中视觉编码器、merger 和语言模型是一个整体设计，而不是单纯把图像描述文本拼接到 prompt 中。

- **DeepStack**：不只使用视觉编码器最后一层的特征，而是把多个层级的视觉特征注入语言模型。浅层特征保留局部细节，深层特征提供更强的语义信息，这有助于 OCR、细粒度识别和空间推理。

- **Interleaved-MRoPE**：文本、图像和视频可以交错出现在同一个长上下文中，位置编码同时表达文本顺序、图像空间和视频时间。

- **文本时间戳对齐**：视频事件不仅由帧位置表示，还可以通过文本化的时间戳描述事件发生时间，从而改善视频 grounding 和事件定位。

- **原生 256K 上下文**：Qwen3-VL 技术报告和模型发布说明宣称支持 256K token 的长文本和长多模态上下文，目标是让文档、视频、截图和文本可以在同一个任务中被交叉引用。

![Qwen3-VL 的视觉编码器、DeepStack 和多模态序列](/assets/img/qwen/qwen3-vl-architecture.png){: .qwen-figure width="5908" height="3413" }

图 5 · Qwen3-VL 原论文中的整体架构图：视觉编码器产生图像/视频 token，DeepStack 将多层视觉特征注入语言模型的中间层，同时保留文本时间戳和交错多模态上下文。来源：[Qwen3-VL Technical Report，Figure 1](https://arxiv.org/abs/2511.21631)。
{: .qwen-caption }

### 7.2 Thinking 与 Non-thinking

Qwen3-VL 延续 Qwen3 的思考模式设计，同时让视觉输入参与推理：

- **Instruct / Non-thinking** 更适合快速视觉问答、文档处理和工具调用；
- **Thinking** 更适合视觉数学、复杂图表、多图关系、空间推理和长视频分析；
- 视觉 Agent 可以先理解截图或界面，再决定点击、输入、滚动或调用工具。

这和 Qwen2.5-VL 的主要区别，不只是视觉编码器更强，而是模型开始把视觉理解、推理预算和 Agent 行动放进同一套后训练框架。

### 7.3 主要能力

Qwen3-VL 的目标从“视觉问答”扩展到了：

- 长文档和长视频理解；
- 视觉数学和复杂图表推理；
- 电脑、网页和手机操作；
- 视觉代码生成，例如从界面或设计图生成 HTML/CSS/JavaScript；
- 空间理解、3D grounding 和多模态 Agent。

## 8. Qwen3-Omni：端到端全模态与低延迟语音

Qwen3-Omni 于 2025 年 9 月 22 日发布，报告为 [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)，可查看 [HTML 全文](https://arxiv.org/html/2509.17765v1)和 [官方仓库](https://github.com/QwenLM/Qwen3-Omni)。

### 8.1 Thinker/Talker 都进入 MoE 化

- **多模态 Thinker**：统一处理文本、图像、音频和视频，负责跨模态理解、推理和文本生成。

- **语音 Talker**：根据 Thinker 的 hidden states 生成自然语音。和 Qwen2.5-Omni 相比，Qwen3-Omni 更强调 Talker 的低延迟和多码本语音建模。

- **Thinker/Talker MoE**：两个模块都采用稀疏 MoE 设计，在维持较大模型容量的同时，降低每个 token 的激活计算。

- **多码本自回归语音生成**：Talker 先预测语音 codec token，并使用 MTP 等机制提高后续 codebook 的生成效率；然后通过轻量级 causal ConvNet 和 Code2Wav 过程重建音频。

- **异步流式处理**：输入音频、图像和视频可以分块进入模型，文本和语音输出也可以流式返回，从而减少实时对话的首包延迟。

![Qwen3-Omni 的 MoE Thinker-Talker、多码本和 MTP 架构](/assets/img/qwen/qwen3-omni-architecture.png){: .qwen-figure width="870" height="850" }

图 6 · Qwen3-Omni 原论文中的架构图：Thinker 和 Talker 都采用 MoE，Talker 使用多码本语音建模和 MTP 模块，最后通过流式 codec decoder 重建语音。来源：[Qwen3-Omni Technical Report，Figure 2](https://arxiv.org/abs/2509.17765)。
{: .qwen-caption }

### 8.2 Captioner 与语音能力

Qwen3-Omni 还提供基于 Omni 模型微调的 Captioner，用于生成详细的音频描述。它不只识别语音内容，也可以描述环境声、音乐、说话人和复杂音频事件。

报告强调，Qwen3-Omni 在音频和音视频基准上尤其突出，同时保持文本、图像和视频任务的能力。这说明 Omni 模型的目标不是把多个专用模型简单串联，而是让多个模态共享一个统一的理解和生成系统。

## 9. 视觉语言主线之外的相关方向

### 9.1 Qwen-Audio 与 Qwen2-Audio

Qwen-Audio（2023）和 Qwen2-Audio（2024）主要是音频理解模型，输入语音、环境声、音乐或歌曲，通常输出文本。它们不是 Qwen-VL 的视觉分支，也没有 Thinker-Talker 那种同时生成语音的 Omni 结构。

- [Qwen-Audio Technical Report](https://arxiv.org/abs/2311.07919)
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)

### 9.2 Qwen3-ASR 与 Qwen3-TTS

Qwen3-ASR 是专门的自动语音识别和 forced alignment 系列；Qwen3-TTS 是文本转语音、音色克隆和声音设计系列。它们可以复用 Qwen3-Omni 的音频能力，但不应和 Omni 主模型混为一谈。

- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)

### 9.3 Qwen-Image 与 Qwen VLo

Qwen-Image 是图像生成和编辑模型，主要使用扩散式视觉生成架构；Qwen VLo 则是“视觉理解 + 图像生成”的统一预览模型。它们与 Qwen-VL 的关系更接近“视觉理解能力被用于生成”，而不是传统的视觉语言问答模型。

- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [Qwen VLo 官方介绍](https://qwen.ai/blog?from=research.research-list&id=5a32ae43df17771e727c3c30112ae2ab8233e399)

## 10. 架构演进总结

| 组件 | Qwen-VL | Qwen2-VL | Qwen2.5-VL | Qwen2.5-Omni | Qwen3-VL | Qwen3-Omni |
| --- | --- | --- | --- | --- | --- | --- |
| 视觉/音频输入 | 图像 | 图像、视频 | 图像、视频 | 文本、图像、音频、视频 | 图像、视频 | 文本、图像、音频、视频 |
| 视觉编码器 | OpenCLIP ViT-bigG | 675M ViT | 从头训练 ViT | 图像/音频编码器 | SigLIP2 等升级视觉编码器 | 多模态编码器 + Thinker |
| 连接方式 | Cross-Attention Adapter | 动态视觉 token | Merger + 动态视觉 token | Thinker-Talker | Merger + DeepStack | Thinker-Talker MoE |
| 视觉 token | 256 个 learnable queries | 动态数量 | 动态数量并合并 patch | 分块视觉/音频表示 | 多层视觉特征 | 跨模态 hidden states |
| 位置/时间 | 2D 绝对位置 | M-RoPE：时间/高/宽 | 绝对时间 M-RoPE | TMRoPE | Interleaved-MRoPE + 文本时间戳 | 跨模态时间对齐 |
| 输出 | 文本、坐标 | 文本、坐标 | 文本、JSON、工具调用 | 文本、语音 | 文本、Thinking、工具调用 | 文本、自然语音、音频描述 |
| 后训练重点 | 多模态 SFT | 多任务与视觉 Agent | SFT + DPO | Thinker/Talker 语音训练 | Instruct/Thinking、视觉 Agent | 流式语音、多模态推理 |

这张表背后有三条清晰的技术路线：

1. **视觉 token 从固定压缩变成动态分辨率**：Qwen-VL 的 256 个 query 控制成本，Qwen2-VL 开始让 token 数量适应图像，Qwen2.5-VL 和 Qwen3-VL 继续优化 token 合并和多层特征。
2. **位置编码从二维空间走向时空统一**：Qwen2-VL 用 M-RoPE 表示时间、高度和宽度，Qwen2.5-VL 引入真实时间，Qwen3-VL 进一步处理图文交错和文本时间戳。
3. **模型目标从理解走向行动和生成**：Qwen-VL 主要回答问题，Qwen2.5-VL 开始操作 GUI，Qwen2.5-Omni 能实时说话，Qwen3-VL 和 Qwen3-Omni 则把思考、工具和长程交互纳入统一模型。

## 11. 阅读顺序与技术报告

推荐按下面的顺序阅读：

1. **Qwen-VL**：先理解视觉编码器、Adapter 和三阶段训练；
2. **Qwen2-VL**：重点看动态分辨率、M-RoPE 和视频 token；
3. **Qwen2.5-VL**：重点看 ViT 重构、窗口注意力、动态 FPS 和真实时间；
4. **Qwen2.5-Omni**：重点看 Thinker-Talker 和 TMRoPE；
5. **Qwen3-VL**：重点看 DeepStack、256K 上下文和视觉 Thinking/Agent；
6. **Qwen3-Omni**：重点看 MoE Thinker/Talker、多码本语音和低延迟流式生成。

完整报告链接：

- [Qwen-VL Technical Report（2023）](https://arxiv.org/abs/2308.12966)
- [Qwen2-VL Technical Report（2024）](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL Technical Report（2025）](https://arxiv.org/abs/2502.13923)
- [Qwen2.5-Omni Technical Report（2025）](https://arxiv.org/abs/2503.20215)
- [Qwen3-VL Technical Report（2025）](https://arxiv.org/abs/2511.21631)
- [Qwen3-Omni Technical Report（2025）](https://arxiv.org/abs/2509.17765)

## 结语

Qwen 多模态模型的演进，不是简单地给文本模型接一块视觉编码器，而是逐步解决了四个问题：

1. **视觉信息如何进入语言模型**：从 Qwen-VL 的 Adapter 到 Qwen2-VL 的动态视觉 token；
2. **高分辨率和长视频如何处理**：通过动态分辨率、patch 合并、窗口注意力和动态 FPS 控制 token 与计算量；
3. **图像、视频和文本如何共享位置**：通过 M-RoPE、真实时间对齐、Interleaved-MRoPE 和文本时间戳表达空间与时间；
4. **模型如何从看懂走向行动和说话**：通过视觉 Agent、Thinker-Talker、MoE 和多码本流式语音生成完成闭环。

如果文本模型的核心问题是“如何预测下一个 token”，多模态模型的核心问题就是：**如何把不同模态的信息变成同一个模型可以理解、推理和行动的上下文。**

</div>
