---
title: "Qwen 多模态模型系列：从 Qwen-VL 到 Qwen3-VL"
date: 2026-10-10 00:00:00 +0800
permalink: /posts/qwen-multimodal-models/
description: "从视觉编码器、跨模态接口、位置编码和训练流程出发，梳理 Qwen-VL 系列的多模态设计。"
categories: [Qwen]
tags: [Multimodal, Vision-Language, Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL]
math: true
toc: true
image:
  path: /assets/img/qwen/qwen-vl-training.png
  alt: "Qwen-VL 的三阶段训练流程"
---

<div class="qwen-note" markdown="1">

本文按 Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL 的顺序，梳理语言模型如何处理视频和图像，以及视觉信息与文本的对齐。文本模型系列的架构与后训练演进见 [Qwen 文本模型系列](/posts/qwen-text-models/)。

## 1. 速看版总结

| 系列 | 时间 | 架构 | 训练 | 主要能力 |
| --- | --- | --- | --- | --- |
| Qwen-VL | 2023-09 | OpenCLIP ViT-bigG、单层 Cross-Attention Adapter、256 个 learnable queries | 两阶段预训练 + 多模态 SFT | 图像描述、VQA、OCR、grounding、视觉对话 |
| Qwen2-VL | 2024-08 | 675M ViT、动态分辨率、2×2 Patch Merger、M-RoPE | 视觉预训练、多模态联合预训练、指令微调 | 任意分辨率、视频理解、空间定位、视觉 Agent |
| Qwen2.5-VL | 2025-01 | 从头训练的动态分辨率 ViT、窗口注意力、Dynamic FPS、Absolute-time M-RoPE | 1.5T + 2T + 0.6T 三阶段预训练；SFT + DPO | 文档/OCR、图表、grounding、长视频、电脑/手机 Agent |
| Qwen3-VL | 2025-09 | SigLIP-2、MLP Merger、DeepStack、Interleaved-MRoPE、时间戳对齐 | 四阶段预训练；SFT、蒸馏、RL | 多模态推理、长文档/视频、空间理解、视觉 Agent |

## 2. Qwen-VL：把视觉信息接入 Qwen-7B

Qwen-VL  [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)。模型提供图像描述、视觉问答、 OCR、细粒度定位和多轮视觉对话等多种任务。

### 2.1 模型架构

Qwen-VL 由三个部分组成，总参数量约 9.6B：

| 组件 | 初始化/规模 | 作用 |
| --- | --- | --- |
| 视觉编码器 | OpenCLIP ViT-bigG，约 1.9B | 将图像 patch 编码成视觉特征 |
| VL Adapter | 随机初始化，约 0.08B | 压缩视觉特征并接入语言模型 |
| LLM | Qwen-7B 初始化，约 7.7B | 融合视觉特征和文本指令，生成回答 |

**视觉编码器：ViT-bigG 与 patch 特征。** Qwen-VL 使用 ViT 架构，并从 OpenCLIP 的 ViT-bigG 初始化。输入图像按固定分辨率缩放后，以 stride 14 切成 patch，再由 ViT 编码。训练时第一阶段用 224×224，第二阶段升到 448×448。

**单层 Cross-Attention Adapter：** ViT 输出的是一长串视觉特征，维度也和语言模型不同。若把所有 patch 特征原样送进 LLM，会增加上下文长度和计算成本。Qwen-VL 在 ViT 和 Qwen-7B 之间加入随机初始化的单层 Cross-Attention Adapter，用 **earnable query** 从视觉特征中读取并汇总信息。

**256 个 learnable queries：将 patch 特征压缩为固定长度。** Adapter 将图像特征压缩为固定长度 256 的视觉向量序列。它们是从整张图的视觉特征中注意力汇总出的表示，**不是从原图中挑选 256 个 patch**。固定长度便于控制 LLM 的视觉输入成本；代价是压缩可能损失细节。

**Position-aware：压缩时保留图像位置。** 如果只汇总内容，模型可能知道“图里有一只猫”，却难以判断猫在哪里。Adapter 在 query-key 交互中加入二维绝对位置编码，让视觉特征保留空间位置，这对 OCR 和 grounding 尤其重要。

### 2.2 输入输出接口：统一图像、坐标和文字
图像经 visual encoder 和 adapter 后得到固定长度（256）的视觉特征序列，它们与文本 token 一起进入 LLM。需要回答或定位的内容，则由 LLM 自回归生成，将所有任务统一到自回归文本生成框架下。

**图像输入。** 视觉特征序列前后使用 <code>&lt;img&gt;</code> 与 <code>&lt;/img&gt;</code> 标记：

> <code>&lt;img&gt; [256 个视觉特征向量] &lt;/img&gt; 请描述这张图。</code>



**Bounding box 文本化。** 框坐标归一化到 [0, 1000)，序列化为坐标字符串，并由 <code>&lt;box&gt;</code>、<code>&lt;/box&gt;</code> 包裹，例如：

> <code>&lt;box&gt;(120, 80), (760, 920)&lt;/box&gt;</code>

坐标作为文本 token 被生成和理解，而不是新添加坐标特殊 token，captioning、VQA、grounding、OCR 等所有任务共享同一个自回归生成框架，无需为不同任务设计不同的输出头或解码策略。

**Reference 标记。** <code>&lt;ref&gt;</code> 与 <code>&lt;/ref&gt;</code> 标记被定位的文字片段。它与 box 配合，将语言中的对象和图像区域对应起来：

> <code>&lt;ref&gt;红色汽车&lt;/ref&gt;&lt;box&gt;(120, 300), (640, 760)&lt;/box&gt; 停在路边。</code>

这让同一个生成接口可以同时输出描述和定位结果，但 grounding 仍然需要专门的区域标注和任务训练。

### 2.3 训练流程：两阶段预训练 + 最终 SFT

![Qwen-VL 的三阶段训练流程：视觉预训练、多任务预训练和监督微调](/assets/img/qwen/qwen-vl-training.png){: .qwen-figure width="1090" height="510" }

图 1 · 原论文 Figure 3。图中雪花表示冻结，火焰表示训练；可对照三个阶段中哪些模块更新。来源：[Qwen-VL Technical Report](https://arxiv.org/abs/2308.12966)。
{: .qwen-caption }

论文里的流程是**两阶段预训练，再进行一阶段监督微调**：

| 阶段 | 训练什么 | 数据与分辨率 | 目标 |
| --- | --- | --- | --- |
| 预训练 Stage 1 | 冻结 LLM；训练 ViT + Adapter | 5B 原始图文对清洗后保留 1.4B；224×224 | 建立视觉表征与语言空间的基本对齐 |
| 预训练 Stage 2 | 解冻并训练全模型 | 高质量多任务数据；448×448 | 学习细粒度视觉任务并联合适配 LLM |
| SFT Stage 3 | 冻结 ViT；训练 LLM + Adapter | 约 350K 多模态指令样本，并混合纯文本对话 | 学习遵循指令、多轮对话和任务输出格式 |

**Stage 1：先对齐视觉特征与语言空间。** 第一阶段使用网页抓取的弱标注图文对。清洗后约 1.4B 对，其中英文约 77.3%，中文约 22.7%。训练时冻结 Qwen-7B，只更新视觉编码器和 Adapter；图像缩放到 224×224，训练 50,000 步、batch size 为 30,720。论文称训练消耗约 1.5B 个图文样本。

先固定 LLM、训练视觉前端，能够让视觉特征先适配一个稳定的语言表示空间。

**Stage 2：七类任务联合预训练。** 第二阶段将分辨率提高到 448×448，解冻 LLM，训练整个模型。数据覆盖七类任务：

| 任务 | 样本数 | 代表内容 |
| --- | ---: | --- |
| Captioning | 19.7M | 图像描述 |
| VQA | 3.6M | 通用问答、图表/文档问答 |
| Grounding | 3.5M | 根据文本定位区域 |
| Ref Grounding | 8.7M | 理解文字指代并定位对象 |
| Grounded Captioning | 8.7M | 描述图像并对齐区域 |
| OCR | 24.8M | 场景文字、PDF/HTML 与合成 OCR |
| Pure-text Autoregression | 7.8M | 保持纯文本生成能力 |

论文还构造了长度为 2048 的交错图文序列。模型不只学“图片对应一句 caption”，也在同一阶段学习问答、文字阅读、区域定位和纯文本生成。

**Stage 3：SFT 冻结视觉编码器。** 最终指令微调数据约 350K，包含模型生成、人工标注、定位和多图对话数据，并混入纯文本对话。
从训练分工看，前两阶段已经用大规模图文及细粒度任务训练视觉表示；Stage 3 SFT 的重点转向指令遵循、对话和输出格式，这时的重点在语言模型，因此固定视觉前端，让 LLM 学会按指令调用视觉信息、Adapter 微调跨模态接口。

## 3. Qwen2-VL：动态分辨率、视频与 M-RoPE

![Qwen2-VL 的整体架构与动态视觉 token](/assets/img/qwen/qwen2-vl-architecture.png){: .qwen-figure width="1441" height="991" }

图 2 · 原论文 Figure 2：不同分辨率的图片和视频经过 Vision Encoder 后，形成不同长度的视觉 token，再与文本一起送入 Qwen2-VL Decoder。来源：[Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191)。
{: .qwen-caption }

Qwen2-VL 于 2024 年 8 月发布：[Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191)。相比Qwen-VL，它的关键变化是让视觉 token 数随输入分辨率变化，并把空间位置和视频时间纳入同一套位置编码。

### 3.1 架构设计

Qwen2-VL 仍由视觉编码器、连接模块和语言模型组成：视觉端采用约 675M 参数的 ViT，语言端换成 Qwen2（2B/7B/72B）。

**创新一：Naive Dynamic Resolution。** 这是 Qwen2-VL 最核心的架构变化。

**问题：固定分辨率会损失细节。** 初代 Qwen-VL 先把图像 resize 到固定分辨率，再压缩成固定 256 个视觉 token。无论原图是 200×200 还是 4000×3000，最后进入 LLM 的视觉 token 数都一样。高分辨率图片中的小字、表格和布局细节，可能在缩放和压缩时丢失。

**解法：不再把所有图片统一缩放到同一尺寸。** Qwen2-VL 移除 ViT 的绝对位置编码，改用 **2D-RoPE**，让视觉编码器适应不同高宽和长宽比。图片仍按 **14×14** 的 patch 切分，但图越大，patch 就越多；再通过下面的 Merger 合并为视觉 token。

**效果：视觉 token 数量随分辨率变化。** 不再让大小图片都挤进固定 256 个 token；大图可以保留更多信息，小图则占用更少上下文。实际推理时仍会受到视觉 token 预算限制，但模型不再一开始就把所有图片压成同样长度。

**创新二：2×2 MLP Patch Merger。** 替代了初代的 Cross-Attention Adapter。

**问题：patch 特征太多，维度也和 LLM 不同。** 如果全部原样送入语言模型，既占用很长的上下文，也无法直接匹配 LLM 的 hidden size。

**解法：相邻四个 patch 合成一个视觉 token。** Merger 把 **2×2** 个相邻 patch 特征拼接，再通过两层 MLP 投影到 LLM 的隐藏维度。它不像初代 Adapter 那样，用固定 256 个 query 从整张图汇总信息，而是逐个合并局部网格。因此序列长度缩短为原来的 **1/4**，同时仍保留“这个 token 来自哪个网格位置”的对应关系。

以预处理后的 H×W 图片为例，视觉 token 数为 <code>(H/14/2) × (W/14/2)</code>，不计首尾边界标记：

| 输入分辨率 | ViT 的 patch 数 | 2×2 合并后的视觉 token 数 |
| --- | --- | --- |
| 224×224 | 16×16 = 256 | 8×8 = **64** |
| 896×896 | 64×64 = 4096 | 32×32 = **1024** |

论文中 224×224 图片对应的 **66 个 token**，是 64 个视觉 token 加上两个边界标记。

**创新三：Multimodal Rotary Position Embedding（M-RoPE）。** 上面的 2D-RoPE 用在 ViT 内部；这里的 M-RoPE 用在 LLM 中，负责统一文本、图像和视频的位置表示。

**问题：一维编号会把二维网格“压扁”。** 文本只需要知道先后顺序，一个位置 ID 就够。但图像是二维的，视频还要再加一个时间维度，一维编号更不够用。

**解法：把 RoPE 的通道分成三组。** RoPE 的做法是给每一对通道分配一个频率 $\theta_r$，再用位置 $m$ 旋转角度 $m\theta_r$。M-RoPE 把参与旋转的通道拆成时间（**t**）、高度（**h**）、宽度（**w**）三组，每组用**自己的位置 ID** 旋转：

- **文本**：三组共用同一个 ID。比如序号为 5 的文本 token 是 <code>(5, 5, 5)</code>，等价于普通 1D-RoPE。
- **图片**：t 固定，h 是该 token 在合并后网格里的行号，w 是列号。
- **视频**：h、w 与图片相同，t 随时间块递增。这里的时间块指相邻两帧经 3D 卷积合并后的结果，因此 t 每加 1 代表前进一个时间块，而不是前进一帧。

**例子：2×2 的视觉 token 网格。** 为方便理解，先忽略前面的文本和边界标记，从 0 开始编号：

| 网格位置 | 图片 / 视频第一个时间块 | 视频第二个时间块 |
| --- | --- | --- |
| 左上 | <code>(0, 0, 0)</code> | <code>(1, 0, 0)</code> |
| 右上 | <code>(0, 0, 1)</code> | <code>(1, 0, 1)</code> |
| 左下 | <code>(0, 1, 0)</code> | <code>(1, 1, 0)</code> |
| 右下 | <code>(0, 1, 1)</code> | <code>(1, 1, 1)</code> |

从左上角 <code>(0, 0, 0)</code> 出发，三个方向各自只改一个坐标：

- **向右一格** → 只改 w：<code>(0, 0, 1)</code>
- **向下一格** → 只改 h：<code>(0, 1, 0)</code>
- **跳到下一个时间块的同一位置** → 只改 t：<code>(1, 0, 0)</code>

这样“左右相邻”“上下相邻”“时间先后”就落在三组互不干扰的通道上，模型能分别学习，不会像一维编号那样互相混淆。

**长度外推。** 长视频按一维展开时，位置 ID 会被帧数乘以每帧 token 数不断推高；M-RoPE 里 t、h、w 各自独立增长，最大值小得多，推理时可以帮助长度外推。

**多段输入。** 混合输入时，每一段的位置编号都从“前一段的最大 ID 加 1”开始，而不是每张图都重新从 0 数。论文 Figure 3 就是这个例子：视频网格的 t、h、w 最大值分别是 2、2、3，取全局最大 3 再加 1，后面的文字就从 <code>(4, 4, 4)</code>、<code>(5, 5, 5)</code> 继续。

![Qwen2-VL 的多模态旋转位置编码 M-RoPE](/assets/img/qwen/qwen2-vl-mrope.png){: .qwen-figure width="3059" height="613" }

图 3 · 原论文 Figure 3：M-RoPE 将时间、高度和宽度位置纳入同一套旋转位置表示。左边是视频的位置 ID 网格，右边是文字沿用 1D 的连续编号。来源：[Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191)。
{: .qwen-caption }

### 3.2 输入输出与训练

ViT 输出的 patch 特征经过 2×2 MLP merger 后，序列长度减为原来的 1/4，特征维度投影到 LLM 的隐藏维度。这些视觉 token 与文本 token 在序列中直接拼接，用特殊的 token 标记边界。LLM 以自回归 next-token prediction 的方式统一处理所有任务，与 Qwen-VL 的设计一脉相承。

图像和视频视觉特征以 <code>&lt;|vision_start|&gt;</code>、<code>&lt;|vision_end|&gt;</code> 标记。定位任务使用 <code>&lt;|box_start|&gt;</code> / <code>&lt;|box_end|&gt;</code> 与 <code>&lt;|object_ref_start|&gt;</code> / <code>&lt;|object_ref_end|&gt;</code> 标记坐标和对应文本。这里的标记和Qwen-VL还不一样。

训练仍是三阶段：首先主要训练 ViT；其次解冻全模型，用图文、OCR、VQA、视频和纯文本数据联合预训练；最后锁住 ViT，以指令数据微调 LLM。论文所述预训练累计约 1.4T token，其中初始阶段约 600B、第二阶段再加入约 800B 图像相关 token。第三阶段包含文本对话、多图比较、文档解析、视频理解和 Agent 交互。



## 4. Qwen2.5-VL：原生分辨率、真实时间与视觉 Agent

Qwen2.5-VL 于 2025 年 1 月发布：[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)。重要变化包括重做 ViT、降低高分辨率视觉注意力成本、视频动态采样和更精确的时间对齐。

### 4.1 架构设计

Qwen2.5-VL 的四个改动都围绕同一个问题：**分辨率越高、视频越长，视觉 token 就越多，而注意力的计算量随 token 数平方增长**。

**从头训练的 ViT。** 论文的说法是：原生分辨率输入带来的**计算量随图像尺寸平方增长**，这会造成训练和推理的负载不均衡，所以它们重新设计了 ViT 架构，并用 DataComp 加内部数据从零训练（而不是沿用某个现成的视觉预训练权重）。结构上用 2D-RoPE 支持任意分辨率，用 SwiGLU 和 RMSNorm 与 Qwen2.5 LLM 保持结构一致。视觉编码器为 32 层、hidden size 1280、16 个注意力头；图像尺寸调整为 28 的倍数，再用 patch size 14 切分。

**窗口注意力。** 这是上一条要解决的核心问题。高分辨率文档图动辄产生上万个 patch，如果每层都做全局注意力，成本无法接受。Qwen2.5-VL 让 32 层中的大多数层只在局部窗口内做注意力（窗口最大 112×112 像素，即 8×8 个 patch），只保留第 7、15、23、31 层做 full attention。

> 局部层负责把细节看细，成本近似线性；少数全局层负责把远处信息连起来。论文特别提到，因为 ViT 的参数比 LLM 少得多、又用窗口注意力进一步压低了开销，省下的显存和算力就能留给语言模型——训练时的负载均衡也因此主要围绕 LLM 来做。

**MLP Merger。** 和 Qwen2-VL 一样，把空间相邻的 2×2 个 patch 特征拼接，经两层 MLP 投影到 LLM 的 hidden size，序列长度减为 1/4。不同规模的 Qwen2.5-VL 共用同一套 ViT，只靠 Merger 的输出维度去匹配各自语言模型（3B 到 2048、7B 到 3584、72B 到 8192），这也是"灵活压缩"的工程意义。

**Absolute-time M-RoPE。** Qwen2-VL 的 temporal ID 跟着**帧序号**走，问题在于同一个 ID 在不同 FPS 下代表的时间长度完全不同：2 FPS 时 t=10 是第 5 秒，4 FPS 时 t=10 却是第 2.5 秒。模型因此学不到"这两件事隔了多久"。Qwen2.5-VL 直接把 temporal ID 和**真实时间戳**对齐，让 ID 之间的间隔本身就编码了时间跨度。这样不同采样率的视频可以互相比较，也支持"第 30 秒发生了什么"这类按秒定位的问题。论文强调这个做法不需要额外的文本时间戳或预测头，没有额外计算开销。

![Qwen2.5-VL 的视觉编码器、动态分辨率和绝对时间建模](/assets/img/qwen/qwen25-vl-architecture.png){: .qwen-figure width="4579" height="2999" }

图 4 · 原论文 Figure 1：整体架构、动态分辨率视觉 token、Dynamic FPS 和窗口/全局注意力。来源：[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)。
{: .qwen-caption }

### 4.2 预训练：约 4.1T token 的三阶段配方

Qwen2.5-VL 的 LLM 从 Qwen2.5 初始化，视觉编码器从头训练。预训练分三段：

| 阶段 | 数据量 | 序列长度 | 训练模块与主要目标 |
| --- | ---: | ---: | --- |
| Visual Pre-training | 1.5T | 8K | 主要训练 ViT；caption、视觉知识、OCR |
| Multimodal Pre-training | 2T | 8K | 解冻 ViT 与 LLM；交错图文、VQA、视频、grounding、Agent、纯文本 |
| Long-context Pre-training | 0.6T | 32K | 联合训练；提高长视频、长文档和 Agent 数据比例 |

总预训练量约 4.1T token。定位数据使用原始图像尺寸下的坐标，使框和点携带尺度信息；文档数据则将段落、表格、图表、公式等元素与布局信息组织为结构化表示。

### 4.3 后训练：SFT + DPO

SFT 数据约 2M 条，纯文本和多模态样本约各占一半；多模态部分包含图文和视频，覆盖 VQA、数学、代码、文档/OCR、grounding、视频和 Agent。复杂推理样本使用 rejection sampling 和质量过滤。

在 SFT 和 DPO 阶段，报告都冻结 ViT，主要更新语言模型相关参数。工程上可理解为：预训练阶段已经建立通用视觉表征，后训练侧重指令行为、答案质量和偏好对齐；固定视觉前端也能控制训练成本。

## 5. Qwen3-VL：DeepStack、多模态长上下文与推理

Qwen3-VL 报告：[Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)，[HTML 全文](https://arxiv.org/html/2511.21631v1)。原生支持 256K token 的文本及交错多模态输入。

### 5.1 架构设计

Qwen3-VL 延续三模块结构（视觉编码器、Merger、Qwen3 LLM）。它的几个架构改动都可以看作**对 Qwen2.5-VL 某处不足的直接回应**。

**换用 SigLIP-2：要一个更强的视觉底座。** Qwen2.5-VL 的 ViT 是从头训练的，虽然摆脱了分辨率限制，但也放弃了大规模视觉预训练积累的表征能力。Qwen3-VL 改用 SigLIP-2（旗舰用 SigLIP2-SO-400M，小模型用 SigLIP2-Large 300M），在它的预训练权重上继续做动态分辨率训练——既保留强视觉表征，又支持任意分辨率。为了兼容动态分辨率，它同样使用 2D-RoPE，并按输入尺寸插值绝对位置编码。

**DeepStack：只把 ViT 最后一层送给 LLM，信息损失太大。** 这是 Qwen3-VL 最实质的改动。此前所有视觉信息都被压到 ViT 的**最后一层**输出里，再经 Merger 变成视觉 token。问题是 ViT 的浅层保留纹理、边缘、小字这类局部细节，深层才是语义；只取最后一层，浅层信息已经被逐层抽象掉了。

DeepStack 的做法是**从三个不同层抽取视觉特征**，各用一个专用 Merger 投影成视觉 token，然后**直接加到 LLM 前三个层的 hidden state 上**。

> 关键在于它不增加上下文长度。这些特征不是作为新 token 拼进序列，而是**加**到已有的 hidden state 上。所以模型既拿到了浅层的局部细节和深层的语义，又不用为多层特征付出更长的序列和更多的注意力计算——这正好补上了 Qwen2.5-VL 在细粒度视觉感知上的短板。

**Interleaved-MRoPE：原版 M-RoPE 的频谱分配不均衡。** 原版把通道分成 t、h、w 三块连续区间，每块各自分配频率。但 RoPE 的频率是按通道对从高到低排布的，这样切分后，三个轴拿到的**频率分布并不相同**——有的轴集中在高频（只能分辨近距离），有的集中在低频。论文指出这种不均衡的频谱会拖累长视频理解。Interleaved-MRoPE 改成把 t、h、w **交错**分配到各个维度上，让每个轴都同时拥有低频和高频分量，频谱更均衡。

**文本时间戳：取代 Qwen2.5-VL 的绝对时间对齐。** Qwen2.5-VL 把 temporal ID 对齐到真实秒数，Qwen3-VL 发现了两个副作用：

1. 长视频里时间 ID 会变得**又大又稀疏**——ID 直接按秒数增长，视频越长数值越大，且相邻帧之间的间隔很不均匀，反而损害长时序建模；
2. 要让模型学好，就得在训练时**覆盖足够多、分布均匀的不同帧率**，数据构造成本很高。

Qwen3-VL 因此改用**文本时间戳**：在每个视频时间块前面直接插入一段文字，比如 <code>&lt;3.0 seconds&gt;</code>，训练时还同时提供秒和 `时:分:秒` 两种格式。代价是上下文略微变长，换来的是时间感知更直接、更精确，也不需要为帧率分布操心。

![Qwen3-VL 的视觉编码器、DeepStack 和多模态序列](/assets/img/qwen/qwen3-vl-architecture.png){: .qwen-figure width="5908" height="3413" }

图 5 · 原论文 Figure 1：视觉编码器、Merger、DeepStack 多层注入和交错多模态序列。来源：[Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)。
{: .qwen-caption }

### 5.2 预训练：四阶段由对齐到超长上下文

| 阶段 | 目标 | 更新模块 | 数据量 / 序列长度 |
| --- | --- | --- | --- |
| S0 | Vision-Language Alignment | 只训练 Merger；ViT、LLM 冻结 | 67B token / 8K |
| S1 | Multimodal Pre-training | 全模型 | 约 1T / 8K |
| S2 | Long-context Pre-training | 全模型 | 约 1T / 32K |
| S3 | Ultra-Long-Context Adaptation | 全模型 | 100B / 262K |

S0 先用图像描述、视觉知识和 OCR 数据建立视觉语言连接；后续阶段再逐步解冻全模型，加入图文交错、VQA、grounding、STEM、视频、Agent、长文档和超长上下文数据。

### 5.3 后训练：SFT、蒸馏、RL 与视觉 Agent

Qwen3-VL 的后训练包括以下几部分：

1. **SFT**：分 32K 与 256K 长上下文阶段训练；Instruct 与 Thinking 版本使用不同回答格式，后者包含 CoT 数据。
2. **Strong-to-Weak Distillation**：用强教师模型向较小模型迁移能力。报告中的蒸馏阶段主要用文本数据微调 LLM backbone，以较低成本提升小模型的文本和多模态能力。
3. **Reasoning RL / General RL**：覆盖数学、OCR、grounding、指令遵循等任务，结合规则/代码验证奖励和模型偏好奖励。
4. **Thinking with Images / Visual Agent**：先用冷启动 SFT 学习“思考→调用视觉工具→读取反馈→回答”，再用多轮工具交互 RL 训练视觉 Agent。

这几类训练目标各自解决不同问题：SFT 学习指令格式，蒸馏迁移大模型能力，RL 优化可验证推理和偏好，而视觉 Agent 训练让模型学会依据环境反馈连续行动。


## 6. 其余多模态方向

视觉语言模型之外，Qwen 家族还有专用音频与图像生成模型。

| 方向 | 代表系列 | 主要任务 | 技术报告 |
| --- | --- | --- | --- |
| 音频理解 | Qwen-Audio、Qwen2-Audio | 语音、环境声、音乐理解，通常输出文本 | [Qwen-Audio](https://arxiv.org/abs/2311.07919)、[Qwen2-Audio](https://arxiv.org/abs/2407.10759) |
| 语音识别/合成 | Qwen3-ASR、Qwen3-TTS | ASR、时间对齐、TTS、音色克隆 | [Qwen3-ASR](https://arxiv.org/abs/2601.21337)、[Qwen3-TTS](https://arxiv.org/abs/2601.15621) |
| 图像生成/编辑 | Qwen-Image | 扩散式图像生成和编辑，不是输入图像后由 VLM 回答 | [Qwen-Image](https://arxiv.org/abs/2508.02324) |



</div>
