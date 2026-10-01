---
title: "Qwen 文本模型系列：从 Qwen-1 到 Qwen3 的架构与后训练演进"
date: 2026-09-26 00:00:00 +0800
permalink: /posts/qwen-text-models/
description: "本文梳理了 Qwen 文本模型系列的架构、Tokenizer、上下文扩展和后训练流程，包括Qwen、Qwen1.5、Qwen2、Qwen2.5 和 Qwen3。"
categories: [Qwen]
tags: [LLM, RLHF, PPO, DPO, GRPO, MoE]
math: true
toc: true
image:
  path: /assets/img/qwen/qwen-cover.webp
  alt: "Qwen3 文本模型的后训练与蒸馏流程"
---

<div class="qwen-note" markdown="1">

本文以 Qwen 文本模型为对象，对比 Qwen-1、Qwen1.5、Qwen2、Qwen2.5 与 Qwen3 的设计演进，重点分析各代在模型架构、预训练数据、后训练策略与推理能力上的改进与取舍。

> 本文主要讨论公开技术报告较完整的 Qwen、Qwen1.5、Qwen2、Qwen2.5 和 Qwen3。后续的 Qwen3.5、Qwen3.6、Qwen3.8 只简单提及。

## 1. 速看版总结

<!-- > **先抓住这条演进主线**
>
> - **[Qwen-1](#2-qwen-1)**：<span class="qwen-term">RoPE / RMSNorm / SwiGLU</span> 建立主干，<span class="qwen-change">QKV bias</span> 改善外推，<span class="qwen-data">SFT → PPO</span> 完成对齐。
> - **[Qwen1.5](#3-qwen-15)**：扩大尺寸和上下文，在大模型上尝试 <span class="qwen-term">GQA</span>，引入 <span class="qwen-term">MoE</span>。
> - **[Qwen2](#4-qwen-2)**：全尺寸 <span class="qwen-term">GQA</span>，<span class="qwen-term">DCA + YaRN</span> 扩展上下文，离线与在线 <span class="qwen-data">DPO</span> 优化偏好。
> - **[Qwen2.5](#5-qwen-25)**：重点在数据与后训练，<span class="qwen-data">18T token、百万级 SFT、DPO + GRPO</span>。
> - **[Qwen3](#6-qwen3)**：<span class="qwen-change">移除 QKV bias、加入 QK-Norm</span>，融合思考与非思考模式，并通过 <span class="qwen-data">Strong-to-Weak Distillation</span> 训练轻量模型。
{: .qwen-summary } -->

### 1.1 模型列表

| 系列 | 发布阶段 | 典型规模 | 架构 | 上下文 | 后训练 |
| --- | --- | --- | --- | --- | --- |
| Qwen-1 | 2023 | 1.8B / 7B / 14B | RoPE、QKV bias、Pre-Norm、RMSNorm、SwiGLU、untied embedding | NTK-aware interpolation、LogN-Scaling、窗口注意力；统一到 8192 | SFT、PPO |
| Qwen1.5 | 2024-02～04 | 0.5B～110B，另有 MoE | 延续 Qwen-1；大尺寸 32B/110B 引入 GQA；推出 Qwen1.5-MoE | 统一到 32K | 主要延续上一代，重点改善对齐和开发体验 |
| Qwen2 | 2024-06 | 0.5B～72B，另有 57B-A14B | 全尺寸 GQA、DCA、YARN、细粒度 MoE、共享专家 | 训练 32K；推理可到 128K | SFT + 离线/在线 DPO；更多自动化数据 |
| Qwen2.5 | 2024-09 | 0.5B～72B；API 有 Turbo/Plus | 基本沿用 Qwen2；控制 token 扩展；MoE API | 开源模型 32K/128K；Turbo 训练到 262K、推理到 1M | 百万级 SFT、DPO、GRPO、长输出和结构化数据 |
| Qwen3 | 2025-04 | 0.6B～32B；30B-A3B / 235B-A22B | 移除 QKV bias、加入 QK-Norm；128 专家、Top-8；无共享专家 | 训练 32K，YARN/DCA 外推 | 长 CoT 冷启动、推理 GRPO、Thinking/Non-thinking 融合、蒸馏 |

Qwen 官方报告：

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

>Qwen1.5 没有单独发表完整的技术报告，主要信息来自官方发布博客、模型卡和后续报告中的回顾。

<!-- ## 2. Qwen 的共同骨架：一个持续演进的 Decoder-only Transformer

Qwen 文本模型的基本数据流一直没有改变：

```text
文本
  ↓
Byte-level BPE Tokenizer
  ↓
Token Embedding
  ↓
多层 Decoder Transformer
  ├─ RMSNorm（Pre-Norm）
  ├─ Causal Self-Attention（MHA 或 GQA）
  ├─ RoPE / QK-Norm 等位置与稳定性机制
  ├─ 残差连接
  ├─ RMSNorm
  └─ SwiGLU FFN 或 MoE FFN
  ↓
LM Head
  ↓
下一个 token 的概率分布
```

### 2.1 Tokenizer：从 Qwen-1 开始，Qwen 一直坚持大词表 byte-level BPE

这里先统一一个容易混淆的术语：**BPE（Byte Pair Encoding）是合并算法，BBPE（Byte-level BPE）是以 byte 为基础符号的 BPE 实现**。因此，BBPE 属于 BPE；“BPE”和“BBPE”不是两种互斥的模型架构。Qwen 的官方报告在不同版本中有时只写 BPE，有时写 byte-level BPE/BBPE。本文后面统一写成 **byte-level BPE（BBPE，官方报告有时简称 BPE）**。

Qwen-1 的技术报告把它简称为 **BPE**：实现基于 `tiktoken`，以 `cl100k_base` 作为起点，再加入常用中文字符、中文词和其他语言内容，最终词表约 **152K**。从 tokenizer 的实际实现和后续 Qwen2/Qwen3 报告的明确表述看，它属于 byte-level BPE，也就是 BBPE。Qwen 团队还将数字拆成单个数字，以改善数学和数字相关任务的编码。

从 Qwen2 开始，报告明确写作 **byte-level BPE**，常规词表为 **151,643** 个 token，并配有控制 token。Qwen2.5 延续了这个 tokenizer，同时把控制 token 从 3 个扩展到 22 个，加入工具调用等场景需要的特殊 token。Qwen3 继续使用同一类 Qwen BBPE tokenizer；技术报告给出的词表规模为 **151,669**，而 Hugging Face 配置中有时会看到 **151,936**，后者通常还包含为了并行训练和硬件对齐而补齐的 embedding 尺寸。

这里最重要的并不是词表数字本身，而是两个工程选择：

- **byte-level** 让几乎所有输入都能被编码，不需要依赖有限的“未知词”集合；
- 较大的、对中文和多语言更友好的词表，减少了同一段文本需要的 token 数，从而降低训练和推理成本。

因此，Qwen 的中文能力不能只归因于“模型参数更多”。Tokenizer 的编码效率本身就是模型系统的一部分。

### 2.2 Pre-Norm、RMSNorm 和 SwiGLU：Qwen-1 的稳定基线

Qwen-1 参考 LLaMA 的 decoder-only Transformer，但做了几处明确修改：

- **Pre-Norm**：在注意力和 FFN 之前归一化，而不是采用传统的 Post-Norm。深层模型训练时，Pre-Norm 通常更稳定；
- **RMSNorm**：用均方根归一化替代 LayerNorm，省去均值中心化，计算更简单；
- **SwiGLU**：用门控线性单元替代普通 GeLU/ReLU。可以粗略写成：

  $$
  \operatorname{SwiGLU}(x) = \bigl(\operatorname{Swish}(xW_1) \odot xW_2\bigr)W_3.
  $$

  为了控制参数量，Qwen-1 将 FFN 中间维度从传统的 $4d$ 调小到约 $\frac{8}{3}d$；
- **Untied embedding**：输入 embedding 和输出投影不共享权重。这样会多占一些参数和显存，但 Qwen-1 的实验认为可以换来更好的效果。

这些选择后来基本都保留了下来。Qwen 的几代升级，更多发生在注意力形式、长上下文方法、MoE 和后训练，而不是反复更换整个 Transformer 骨架。

### 2.3 QKV bias：Qwen-1 的一个反直觉选择

Qwen-1 遵循“多数线性层不使用 bias”的趋势，但**保留了注意力 Q、K、V 投影中的 bias**。技术报告给出的动机是改善长度外推能力。

这个选择在当时并不是主流的简单复刻。它体现了一个很重要的工程判断：某个结构是否“标准”，不如它在目标任务和训练设置下是否有效。Qwen2 和 Qwen2.5 延续了 QKV bias；到了 Qwen3，团队为了训练稳定性重新做了取舍，移除 QKV bias，同时在 Q 和 K 上加入 QK-Norm。-->

### 1.2 参数规模、宽度和深度

模型名中的 `7B`、`32B`、`235B` 是总参数量，但它们并不能直接告诉我们网络的宽度和深度。下面列出几个有代表性的配置。`Q/KV heads` 中的 KV 头数越少，通常意味着 KV cache 越小。

| 模型 | 参数量 | hidden size | 层数 | Q / KV heads | FFN intermediate size | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen-7B | 7B | 4096 | 32 | 32 / 32 | 11008 | Qwen-1，MHA |
| Qwen-14B | 14B | 5120 | 40 | 40 / 40 | - | Qwen-1，MHA |
| Qwen1.5-7B | 7B | 4096 | 32 | 32 / 32 | 11008 | 主要延续 Qwen-1 |
| Qwen1.5-32B | 32B | 5120 | 64 | 40 / 8 | 27392 | 较早引入 GQA |
| Qwen2-7B | 7B | 3584 | 28 | 28 / 4 | 18944 | GQA |
| Qwen2-72B | 72B | 8192 | 80 | 64 / 8 | 29568 | GQA |
| Qwen2.5-7B | 7B | 3584 | 28 | 28 / 4 | 18944 | 主要延续 Qwen2 |
| Qwen3-8B | 8B | 4096 | 36 | 32 / 8 | 12288 | QK-Norm、无 QKV bias |
| Qwen3-32B | 32B | 5120 | 64 | 64 / 8 | 25600 | 纯 Dense 主力型号 |
| Qwen3-30B-A3B | 30B 总参数 | 2048 | 48 | 32 / 4 | 128 专家、每 token 激活 8 个 | MoE，约 3B 激活参数 |

在实际中真正影响部署的通常是：

1. **<span class="qwen-term">hidden size 和层数</span>**决定主干计算量；
2. **<span class="qwen-term">Q/KV heads 比例</span>**决定 KV cache 大小；
3. **<span class="qwen-change">总参数和激活参数的差异</span>**决定 MoE 的存储成本与单 token 计算成本。

## 2. Qwen-1

Qwen-1 的公开权重最早在 2023 年发布，技术报告覆盖 Qwen、Qwen-Chat，以及 Code-Qwen、Math-Qwen 等衍生方向。

### 2.1 架构设计

Qwen-1 采用了 LLaMA 的模型架构，主要设计在于：

- **位置编码**：<a class="note-link" href="{{ '/posts/positional-encoding/' | relative_url }}"><i class="far fa-file-alt fa-fw"></i>大模型中的位置编码：从绝对位置到 RoPE</a>，在实现中使用 FP32 保存，以优先考虑模型性能；
- **<span class="qwen-change">QKV bias</span>**：只在 Q、K、V 投影保留 bias，改善长度外推；
- **<span class="qwen-term">FFN 降维 / SwiGLU</span>**：SwiGLU 的表达力更强，但会增加中间层参数，所以把 FFN 中间维度从 $4d$ 减小至 $8d/3$；
- **<span class="qwen-change">不共享输入输出 embedding</span>**：输入 embedding 与输出投影不共享权重。

  假设词表大小为 $V$，模型 hidden size 为 $d$，输入 embedding 矩阵 $E\in\mathbb{R}^{V\times d}$ 通过查表将 token ID 转换为 $d$ 维向量。以往都会采用权重共享（tied embedding），将输出投影使用它的转置 $E^\top\in\mathbb{R}^{d\times V}$，将最终 hidden state 映射为词表中各 token 的分数。

  但 Qwen-1 则使用独立的输出矩阵 $W_{\mathrm{out}}\in\mathbb{R}^{d\times V}$，让输入表示与输出预测分别学习，以提升性能，代价是更多参数和显存。

### 2.2 上下文扩展

Qwen-1 的原始训练序列长度是 2048。为了在推理时扩展模型的上下文，使用了：

- <span class="qwen-term">NTK-aware interpolation</span>；
- <span class="qwen-term">dynamic NTK-aware interpolation</span>；
- <span class="qwen-term">LogN-Scaling</span>；
- 分层分配的窗口注意力。

这些方法主要是在不重新训练模型的情况下延长上下文，使模型在超过训练长度后仍能保持相对稳定的性能。

### 2.3 后训练：SFT → 奖励模型 → PPO/RLHF

Qwen-1 的后训练并不复杂，主要流程是：

![Qwen 原始报告中的模型谱系，含预训练、奖励模型、SFT 和 RLHF 路径](/assets/img/qwen/qwen1-lineage.webp){: .qwen-figure width="1113" height="477" }

图 1 · 从预训练模型到对话、奖励和专用模型的训练路径。来源：[Qwen Technical Report，Figure 1](https://arxiv.org/abs/2309.16609)，PDF 第 3 页。原图也包含 Qwen-VL 分支。
{: .qwen-caption }

```text
预训练 Qwen
   ↓
SFT：学习对话、指令、工具和安全数据
   ↓
PMP / Reward Model：学习人类偏好
   ↓
PPO：用奖励模型优化策略
   ↓
Qwen-Chat / Qwen-Chat-RLHF
```

<span class="qwen-data">SFT</span> 阶段使用 <span class="qwen-term">ChatML</span> 风格格式区分 system、user 和 assistant，并对 system/user 部分做 loss mask，主要让模型学习 assistant 的目标输出。数据不仅包括普通问答，也包括工具调用、代码解释器、Agent 和安全拒答。

<span class="qwen-data">RLHF</span> 阶段先训练偏好模型，再使用 <span class="qwen-data">PPO</span>。Qwen 报告中还提到：

- 使用不同大小的 Qwen 和不同采样策略生成多样回答；
- 通过约 6600 个细粒度标签控制提示词的覆盖范围和难度；
- 使用 KL 惩罚限制策略模型偏离参考模型；
- 使用预训练梯度缓解 alignment tax，也就是对齐后通用能力下降的问题。



## 3. Qwen-1.5

Qwen1.5 是 Qwen-1 的一次系统升级，但没有完全换代。官方在发布时提供了 0.5B、1.8B、4B、7B、14B、32B、72B、110B，以及 Qwen1.5-MoE-A2.7B 等规模。

### 3.1 架构变化：在大尺寸模型上尝试 GQA

Qwen1.5 仍然保留 Qwen-1 的大部分设计：RoPE、QKV bias、SwiGLU、RMSNorm、Pre-Norm 和 BBPE。比较明显的变化是：

- **<span class="qwen-change">Qwen1.5-32B 和 Qwen1.5-110B 使用 GQA</span>**，减少 KV cache 和推理带宽；其他小尺寸在早期版本中仍主要使用 MHA；
- 首次发布了 MoE 模型即 **<span class="qwen-term">Qwen1.5-MoE-A2.7B</span>**，用较少的激活参数获得接近 7B Dense 模型的效果；
- 所有系列统一支持最长约 32K 的上下文，并将代码合入 Hugging Face Transformers，推理和微调门槛明显降低。

![MHA、GQA 与 MQA 中 Query、Key、Value 头的对应关系](/assets/img/qwen/gqa-heads.webp){: .qwen-figure width="1230" height="417" }

图 2 · 中间的 GQA 让一组 Query 共享 Key / Value；左右分别是 MHA 和 MQA。来源：[GQA 原论文，Figure 2](https://arxiv.org/abs/2305.13245)，PDF 第 2 页。
{: .qwen-caption }

### 3.2 后训练

Qwen1.5 110B 发布说明中明确表示，预训练和后训练配方没有发生剧烈变化，主要收益来自模型规模扩大。

总的来说，Qwen-1.5 相较于Qwen-1 而言，除了开始验证 GQA 之外，基本没有太大的变化，主要是进行了数据规模和模型尺寸的扩大。


## 4. Qwen-2

Qwen2 在 2024 年 6 月发布，模型规模包括 0.5B、1.5B、7B、57B-A14B 和 72B。

### 4.1 模型架构

Qwen2 延续了 RoPE、QKV bias、SwiGLU 和 Pre-Norm + RMSNorm，主要改进在于：

- **<span class="qwen-term">分组查询注意力（GQA）</span>**：所有尺寸模型统一使用 GQA，降低 KV cache 的开销。

  传统 MHA 为每个 Q 头分配独立的 K/V 头，而 GQA 让多个 Q 头共享一组 K/V 头。例如，Qwen2-7B 有 28 个 Q 头、4 个 KV 头，即每 7 个 Q 头共享一组 K/V；Qwen2-72B 则是 64 个 Q 头、8 个 KV 头。由于自回归生成需要缓存历史 K/V，减少 KV 头数可以降低缓存显存和带宽开销，改善长上下文推理的吞吐。

- **<span class="qwen-term">长上下文扩展</span>**：<span class="qwen-term">DCA + YARN</span>。

  Qwen2 在预训练末期将序列长度从 4K 扩展到 32K，并把 RoPE 基数从 10,000 提高到 1,000,000。推理时，DCA 将长序列分成 chunk，处理块内与块间的相对位置；YARN 调整 RoPE 频率和注意力缩放，缓解超过训练长度后的性能下降。通过这些方法，Qwen2 上下文扩展至 **32k**，最高支持 **128K** 的上下文。

- **<span class="qwen-change">细粒度 MoE + 共享专家</span>**：在 Qwen2-57B-A14B 中，用多个专家 FFN 替代普通的 Dense FFN。

  该模型包含 64 个路由专家，每个 token 根据路由分数选择其中 8 个参与计算，另有 8 个共享专家参与所有 token 的计算。相比少量大专家，细粒度设计使用更小的专家，在相近的总参数和激活参数预算下提供更多组合；共享专家用于学习通用特征，路由专家则可以形成更有区分度的能力。模型总参数约 57B，但每个 token 只激活约 14B 参数，从而兼顾模型容量和计算成本。

![DCA 中块内、跨块和相邻块三种注意力的相对位置矩阵](/assets/img/qwen/dca-attention.webp){: .qwen-figure width="1194" height="417" }

图 3 · DCA 分别处理块内、跨块与相邻块边界的相对位置。三幅子图的 Query / Key 位置索引均保留完整。来源：[DCA 原论文，Figure 2](https://arxiv.org/abs/2402.17463)，PDF 第 4 页。
{: .qwen-caption }

> <span class="qwen-term">GQA</span> 主要减少 KV cache 的存储和带宽开销；<span class="qwen-term">DCA / YaRN</span> 主要解决长上下文中的位置与外推问题。二者改善的是不同环节。
{: .qwen-insight }

### 4.2 预训练：从 3T 扩大到 7T

Qwen2 的数据规模从 Qwen1.5 的约 <span class="qwen-data">3T</span> 增加到超过 <span class="qwen-data">7T</span>。报告还专门强调了对数据的选择和清洗；Qwen2 的 MoE 模型还额外进行了约 4.5T token 的 upcycling 式训练。

### 4.3 后训练： SFT & DPO

Qwen2 的后训练仍然分为 SFT 和偏好优化。

<!-- #### SFT -->
1. SFT 阶段构造了超过 50 万条 SFT 数据，并使用 32K 序列长度训练两个 epoch。

<!-- #### RLHF -->

2. RLHF 被拆成两个阶段：

- **<span class="qwen-data">Off-Policy 阶段</span>**：使用预先构造的偏好对，进行 DPO；
- **<span class="qwen-data">On-Policy 阶段</span>**：当前策略模型生成多条回答，由奖励模型选出较好和较差的回答，再把新偏好对继续用于 DPO。



## 5. Qwen-2.5

Qwen2.5  延续了 Qwen2 的架构设计，开源尺寸包括 0.5B、1.5B、3B、7B、14B、32B 和 72B。7B 以上模型支持 128K 上下文，0.5B/1.5B/3B 主要是 32K。

Qwen2.5 还扩展了控制 token，并将高质量预训练数据扩展到 **<span class="qwen-data">18T token</span>**，对于更大的数据规模进行了更细致的数据配比，这里很明显的突出了高质量的数据对于模型训练的重要性。


### 5.1 预训练：长上下文扩展

- 采用两阶段预训练：初始阶段使用 4K token 的上下文长度，第二扩展阶段将上下文扩展至32K token（Qwen2.5-Turbo通过四阶段逐步扩展至256K token）。

- 使用 ABF 技术将 RoPE 的基础频率从 10,000 提升至 1,000,000（Qwen2.5-Turbo 模型 10,000,000），并通过 YARN 和 DCA 实现推理阶段的4倍上下文扩展能力（1M）。


### 5.2 后训练：百万级 SFT & Off-Policy DPO & On-Policy GRPO

这是 Qwen2.5 最重要的部分。

#### 第一步：扩大 SFT 覆盖面

Qwen2.5 的 <span class="qwen-data">SFT</span> 数据超过 <span class="qwen-data">100 万条</span>，并专门补齐 Qwen2 的短板（依旧数据工程）：

- 输出长度从常见的 2K 提高到最高 8K；
- 数学中加入 Qwen2.5-Math 的 CoT 数据；
- 增加 70,000 条逻辑推理问题；
- 使用大量 system prompt，降低模型对系统提示变化的敏感性。
- ......


#### 第二步：Off-Policy RL / DPO

Qwen2.5 构造约 <span class="qwen-data">15 万对偏好数据</span>。正确或满足约束的回答作为正样本，错误回答作为负样本，再用 <span class="qwen-data">DPO</span> 训练。

这类任务的共同特点是：答案虽然难生成，但相对容易验证。因此，先用程序或规则建立可靠的偏好信号，再做 DPO，比单纯依赖人工偏好更可扩展。

#### 第三步：在线 RL / GRPO

在线阶段 Qwen2.5 采用 <span class="qwen-data">GRPO</span>，每个问题采样多条回答，用奖励模型和规则奖励比较它们的相对质量，再更新模型。

可以把 Qwen2.5 的后训练总结成：

```text
百万级、多技能 SFT
        ↓
离线 DPO：数学 / 代码 / 指令 / 逻辑
        ↓
在线 GRPO：真实性 / 帮助性 / 安全 / 风格
        ↓
长文本、结构化输出、工具调用能力
```

## 6. Qwen3

Qwen3 是文本主线中最重要的一次后训练范式变化。它不再把“普通对话模型”和“长思维链推理模型”完全拆成两个产品，而是让同一个模型在 Thinking 和 Non-thinking 两种模式之间切换。

### 6.1 架构：移除 QKV bias，加入 QK-Norm

Qwen3 的 Dense 主干仍然使用 GQA、SwiGLU、RoPE 和 RMSNorm + Pre-Norm，但做了两个关键改变：

1. **<span class="qwen-change">移除 QKV bias</span>**：与 Qwen-1/Qwen2/Qwen2.5 的做法相反；
2. **<span class="qwen-change">加入 QK-Norm</span>**：对注意力中的 Q 和 K 单独归一化，改善训练稳定性（与 LLaMA4 保持一致）。

### 6.2 MoE：128 个专家并支持 Top-8 加权输出，无共享专家

Qwen3-30B-A3B 和 Qwen3-235B-A22B 都采用 <span class="qwen-term">128 个总专家，每个 token 激活 8 个专家</span>；和 Qwen2 的 MoE 不同，Qwen3 **<span class="qwen-change">不再设置 shared experts</span>**，并引入 global-batch load balancing loss，让路由器在全局 batch 范围内更均衡地使用专家。

### 6.3 预训练：36T token、119 种语言、三阶段训练

Qwen3 的预训练数据约 <span class="qwen-data">36T token</span>，覆盖 <span class="qwen-data">119 种语言和方言</span>。它还使用多模态模型辅助文本数据构造：

- 用 Qwen2.5-VL 从 PDF 类文档中抽取文字；
- 用 Qwen2.5-Math 合成数学数据；
- 用 Qwen2.5-Coder 合成代码和代码指令数据。

预训练分为三个阶段：

1. **General Stage**：超过 30T token，序列长度 4K，建立语言和世界知识；
2. **Reasoning Stage**：约 5T 高质量 token，提高 STEM、代码和推理数据比例；
3. **Long Context Stage**：数千亿 token，序列长度 32K，并继续使用 ABF、YARN 和 DCA。

这是一种很典型的“先学通用能力，再提高知识和推理密度，最后扩展上下文”的训练安排。

### 6.4 四阶段后训练

Qwen3 的后训练可以分成四个阶段。

![Qwen3 旗舰模型四阶段后训练和轻量模型蒸馏的完整流程](/assets/img/qwen/qwen3-post-training.webp){: .qwen-figure width="1365" height="555" }

图 4 · 上半部分是旗舰模型的四阶段后训练，下半部分是轻量模型的蒸馏路径。来源：[Qwen3 Technical Report，Figure 1](https://arxiv.org/abs/2505.09388)，PDF 第 9 页。
{: .qwen-caption }

- <span class="qwen-data">Stage 1：Long-CoT Cold Start</span>：构造数学、代码、逻辑推理和 STEM 问题。Qwen2.5-72B-Instruct 用来过滤问题，QwQ-32B 用来生成候选长思维链。

- <span class="qwen-data">Stage 2：Reasoning RL</span>：收集了 3,995 个 query-verifier pair，使用 GRPO 做推理强化学习，主要覆盖数学和代码等可验证任务。

- <span class="qwen-data">Stage 3：Thinking Mode Fusion</span>：如果只做长 CoT，模型可能会在简单问题上也输出很长的思考。Qwen3 因此把“思考”和“不思考”的数据合并进行持续 SFT，并设计了 `/think` 和 `/no_think` 控制标记：


>用户问题 /think       → 生成 `<think>...</think>` 后回答
>
>用户问题 /no_think    → 跳过实质推理，直接回答


默认情况下模型可以进入 Thinking 模式，但开发者可以通过 chat template 或参数关闭它。更重要的是，模型还可以根据一个 thinking budget 截断思考，在有限 token 预算下直接利用已经生成的中间推理给出答案。

- <span class="qwen-data">Stage 4：General RL</span>：最后用覆盖 20 多类任务的奖励系统，奖励由三类信号组成：规则奖励、带参考答案的模型奖励、不带参考答案的偏好奖励。

<!-- ![Qwen3-235B-A22B 在四项基准中随思考预算变化的成绩曲线](/assets/img/qwen/qwen3-thinking-budget.webp){: .qwen-figure width="1368" height="891" }

图 5 · 思考预算与效果的关系。四个基准、两类模式的图例以及坐标轴均保留；横轴是思考 token 预算，而非耗时。来源：[Qwen3 Technical Report，Figure 2](https://arxiv.org/abs/2505.09388)，PDF 第 20 页。
{: .qwen-caption } -->

### 6.5 Strong-to-Weak Distillation(感觉是和DeepSeek R1 蒸馏模型一个模式)

对于 0.6B～14B Dense 模型和 30B-A3B 这类轻量模型，Qwen3 采用 Strong-to-Weak Distillation：

1. **<span class="qwen-data">Off-policy distillation</span>**：用大教师模型在 `/think` 和 `/no_think` 两种模式下生成数据，先教学生模型基本的推理和模式切换；
2. **<span class="qwen-data">On-policy distillation</span>**：让学生模型自己生成序列，再让学生的 logits 对齐 Qwen3-32B 或 Qwen3-235B-A22B 教师 logits，最小化 KL 散度。

报告中的实验显示，蒸馏相比直接 RL 只需约十分之一的 GPU 小时，同时在部分任务上取得更好的结果。

<!-- ## 7. 后训练流程的总演进

把几代模型放在同一张图里，可以看到后训练的重点变化：

```text
Qwen-1
  SFT → Reward Model / PMP → PPO / RLHF
  目标：让基础模型成为可对话、可调用工具的助手

Qwen2
  自动化数据构造 → SFT → 离线 DPO → 在线 DPO
  目标：减少人工标注，覆盖代码、数学、多语言和安全

Qwen2.5
  百万级多技能 SFT → 离线 DPO → 在线 GRPO
  目标：长输出、结构化数据、指令验证、数学和代码

Qwen3
  Long-CoT 冷启动 → 推理 GRPO → Thinking/Non-thinking 融合
       → General RL → Strong-to-Weak Distillation
  目标：同一个模型可控制地“深思”或快速回答，并把能力高效迁移到小模型
```

可以把这条路线概括成四个阶段：

1. **从人类示范中学习格式和行为**：Qwen-1 的 SFT；
2. **从偏好数据中学习选答案**：Qwen-1 的 RLHF，Qwen2 的 DPO；
3. **从验证器和奖励中学习可检查能力**：Qwen2.5 的数学/代码 DPO 和 GRPO；
4. **把推理过程本身变成可控能力**：Qwen3 的长 CoT、推理 RL、模式融合和蒸馏。

这里的一个重要趋势是：后训练逐渐从“人工写答案”转向“构造可验证环境”。数学有答案验证，代码有编译和单元测试，指令跟随有自动检查器，Agent 有真实工具反馈。**奖励信号越可验证，后训练就越容易规模化。** -->

## 7. 架构演进的总结

| 组件 | Qwen-1 | Qwen1.5 | Qwen2 | Qwen2.5 | Qwen3 |
| --- | --- | --- | --- | --- | --- |
| 主干 | Decoder-only Transformer | 延续 | 延续 | 延续 | 延续 |
| 位置编码 | RoPE；FP32 inverse frequency | RoPE | RoPE + YARN + DCA | RoPE + ABF + YARN + DCA | RoPE + ABF + YARN + DCA |
| 注意力 | MHA | 大尺寸开始 GQA |  GQA | GQA | GQA + QK-Norm |
| QKV bias | 保留 | 保留 | 保留 | 保留 | 移除 |
| 归一化 | Pre-Norm + RMSNorm | 延续 | 延续 | 延续 | 延续 |
| 激活 | SwiGLU | 延续 | 延续 | 延续 | 延续 |
| FFN | Dense FFN | Dense；MoE 试验 | Dense + 细粒度 MoE + shared experts | Dense；API 有 MoE | Dense + 128-expert MoE，无 shared experts |
| Tokenizer | 约 152K BBPE | 延续 | 151,643 BBPE + 控制 token | 增加控制 token | 151,669 BBPE |
<!-- | 主要目标 | 训练稳定、中文多语言、基础对齐 | 规模和部署体验 | KV cache、长上下文、自动化对齐 | 数据规模、结构化输出、可验证后训练 | 推理模式控制、MoE 效率、蒸馏 | -->

<!-- 最值得注意的不是某个孤立组件，而是这些组件之间的配合：

- GQA 降低 KV cache，YARN/DCA 扩大上下文，二者一起决定长文本推理是否真的可部署；
- QKV bias、QK-Norm 影响训练稳定和长度外推，但必须放在具体训练配方里评价；
- MoE 让总参数和激活参数解耦，但也引入路由负载均衡和跨卡通信问题；
- 后训练从人工偏好转向规则、执行器和真实环境反馈，决定了模型能否从“会说”变成“会做”。 -->

## 8. Qwen3 之后：文本模型正在变成原生多模态 Agent

如果把范围扩展到 2026 年，Qwen3.5、Qwen3.6 和 Qwen3.8 已经不再是简单的“文本模型下一代”。从 [Qwen3.5/Qwen3.6/Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8) 和 [Qwen3.5 发布说明](https://qwen.ai/blog?email_hash=23463b99b62a72f26ed677cc556c44e8&id=qwen3.5) 看，它们开始采用：

- 视觉和文本的 early fusion；
- Gated DeltaNet + Gated Attention 的混合注意力；
- 更高稀疏度的 MoE；
- 面向 Agent 长程执行的强化学习；
- 更广的语言和方言覆盖。

这条路线说明 Qwen 的主线正在从“更强的文本生成器”转向“能理解文本、图像、工具和环境，并把任务执行完的基础模型”。不过，如果只想学习 Qwen 文本模型的核心方法，读到 Qwen3 已经足够建立完整的架构和后训练框架；后续版本可以作为混合注意力和原生多模态的下一篇笔记。

## 完整链接：

- [Qwen Technical Report（2023）](https://arxiv.org/abs/2309.16609)
- [Qwen2 Technical Report（2024）](https://arxiv.org/abs/2407.10671)
- [Qwen2.5 Technical Report（2024）](https://arxiv.org/abs/2412.15115)
- [Qwen2.5-1M Technical Report（2025）](https://arxiv.org/abs/2501.15383)
- [Qwen3 Technical Report（2025）](https://arxiv.org/abs/2505.09388)
- [Qwen1.5 官方介绍](https://qwenlm.github.io/blog/qwen1.5/)
- [Qwen2 官方介绍](https://qwenlm.github.io/blog/qwen2/)
- [Qwen3 官方介绍](https://qwenlm.github.io/blog/qwen3/)



<!-- 1. **Attention / RoPE / GQA / MoE 如何让模型更高效地处理更长上下文；**
2. **SFT / DPO / GRPO / Distillation 如何把基础模型变成可控、可验证、可执行的 Agent。** -->

</div>
