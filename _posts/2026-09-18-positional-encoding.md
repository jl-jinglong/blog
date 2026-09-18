---
title: "大模型中的位置编码：从绝对位置到 M-RoPE"
date: 2026-09-18 00:00:00 +0800
categories: [研究笔记]
tags: [Transformer, LLM, Positional-Encoding, RoPE, M-RoPE]
math: true
toc: true
---

Transformer 可以借助自注意力机制，让任意两个 token 直接交互；但它本身既没有 RNN 按时间步递推的结构，也没有 CNN 局部滑动的结构。因此，若不额外提供顺序，`我 喜欢 你` 与 `你 喜欢 我` 对 Transformer 来说只是同一组 token 的不同排列，模型并不知道谁在前、谁在后。

**位置编码（Positional Encoding, PE）**的任务，就是把“第几个 token”“两个 token 相隔多远”“图像 patch 位于哪里”等位置信息交给模型。它经历了从绝对位置编码，到相对位置编码，再到为长上下文和多模态服务的 RoPE、M-RoPE 的演进。

本文按下面这条线索整理：

$$
\text{绝对位置} \longrightarrow \text{相对位置} \longrightarrow \text{ALiBi / RoPE} \longrightarrow \text{M-RoPE}
$$

> 本文默认序列位置从 \(0\) 开始编号；不同论文可能从 \(1\) 开始，这只会让公式整体平移，不改变核心结论。

## 0. 为什么没有位置编码的 Transformer 不知道顺序？

设一段长度为 \(L\) 的序列经过词表查表后得到输入矩阵：

$$
X =
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{L-1}
\end{bmatrix}
\in \mathbb{R}^{L \times d_{\text{model}}},
$$

其中 \(x_i\) 是第 \(i\) 个 token 的 embedding，\(d_{\text{model}}\) 是隐藏维度。单头自注意力先计算：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

这里 \(d_h\) 是单个注意力头的维度。现在用一个置换矩阵 \(P\) 打乱 token 的行顺序，例如把 `[我, 喜欢, 你]` 变成 `[你, 喜欢, 我]`。新的输入为 \(X'=PX\)，于是：

$$
Q'=PQ,\qquad K'=PK,\qquad V'=PV.
$$

代入注意力公式可得：

$$
\operatorname{Attention}(Q',K',V')
=P\operatorname{Attention}(Q,K,V).
$$

也就是说，**自注意力对输入排列是置换等变（permutation equivariant）的**：输入行怎样重排，输出行也只是相同地重排。它会判断 token 内容是否相关，却没有任何依据判断“它原本在第几个位置”。

位置编码就是用来打破这条对称性的。

### 0.1 记号约定

| 记号 | 含义 |
| --- | --- |
| \(L\) | 当前序列长度 |
| \(i,j\) | token 的位置下标，\(0\le i,j<L\) |
| \(x_i\in\mathbb{R}^{d_{\text{model}}}\) | 第 \(i\) 个 token 的词向量 / hidden state |
| \(p_i\) | 位置 \(i\) 的绝对位置向量 |
| \(r_{i-j}\) | 位置差 \(i-j\) 对应的相对位置表示 |
| \(q_i,k_i,v_i\) | 位置 \(i\) 在某个注意力头中的 query、key、value |
| \(d_h\) | 单个注意力头的维度，通常 \(d_h=d_{\text{model}}/\text{num\_heads}\) |
| \(s_{ij}\) | token \(i\) 对 token \(j\) 的 softmax 前注意力分数（logit） |

## 1. 绝对位置编码

绝对位置编码直接回答“当前 token 是序列中的第几个”。最常见的做法是在进入第一层 Transformer 前，将词向量与同维的位置向量相加：

$$
h_i^{(0)}=x_i+p_i.
$$

其中 \(h_i^{(0)}\) 是第一层的输入。加法不增加序列长度，也不改变隐藏维度，因此实现极其简单。

### 1.1 Sinusoidal Absolute Positional Encoding：固定正弦位置编码

《Attention Is All You Need》采用的固定位置编码为：

$$
\begin{aligned}
p_{i,2k} &= \sin\left(\frac{i}{B^{2k/d_{\text{model}}}}\right),\\
p_{i,2k+1} &= \cos\left(\frac{i}{B^{2k/d_{\text{model}}}}\right),
\end{aligned}
\qquad
k=0,1,\ldots,\frac{d_{\text{model}}}{2}-1.
$$

原论文取 \(B=10000\)。\(2k\) 与 \(2k+1\) 是一对相邻维度：偶数维使用 \(\sin\)，奇数维使用 \(\cos\)。因此每个位置由多组、不同频率的圆周坐标共同描述。

令第 \(k\) 对维度的角频率为：

$$
\omega_k=B^{-2k/d_{\text{model}}},
$$

那么该二维位置子向量可写成 \([\sin(i\omega_k),\cos(i\omega_k)]\)。\(k\) 较小时频率高、变化快，适合区分近邻位置；\(k\) 较大时频率低、变化慢，提供更长尺度的位置变化。

#### 一个小例子

设 \(d_{\text{model}}=4\)，\(B=10000\)，则位置 \(i\) 的位置向量是：

$$
p_i=
\left[
\sin(i),\ \cos(i),\ \sin\left(\frac{i}{100}\right),\ \cos\left(\frac{i}{100}\right)
\right].
$$

当 \(i=0\) 时，\(p_0=[0,1,0,1]\)；当 \(i=1\) 时，前两维已经有明显变化，而后两维只轻微变化。多频率叠加后，每个位置都会得到不同的“指纹”。

#### 为什么正弦和余弦有利于表示相对距离？

对同一频率 \(\omega\)，有恒等式：

$$
\sin(i\omega)\sin(j\omega)+\cos(i\omega)\cos(j\omega)
=\cos((i-j)\omega).
$$

也就是说，一对正余弦位置向量的内积只与位置差 \(i-j\) 有关。这个性质让模型有机会从绝对位置向量中推断相对距离；但请注意，模型看到的是 \(x_i+p_i\) 经过投影后的结果，而不是单独的位置向量，因此这种“相对性”是**间接获得**的。

**优点**：无可学习参数；任意位置都能按公式算出向量；训练长度以外的位置在形式上仍然可计算。

**局限**：

1. 位置和语义从输入起就相加并混合，模型需要自己把两者拆开。
2. 注意力模块并未直接接收 \(i-j\)，相对距离关系需要从混合表示中间接学习。
3. 虽然可以计算未见过的位置，训练时未接触过的频率组合和长度分布仍会带来泛化问题；“有公式”不等于“长文本效果一定好”。
4. 固定频率无法针对具体数据分布调整。

**代表**：原始 Transformer；一些早期机器翻译 Transformer 实现。

### 1.2 Learned Absolute Positional Embedding：可学习绝对位置编码

可学习版本不再手工指定正余弦，而是建立一个可训练的位置表：

$$
P\in\mathbb{R}^{L_{\max}\times d_{\text{model}}},
\qquad p_i=P[i].
$$

输入仍为：

$$
h_i^{(0)}=x_i+P[i].
$$

与词表 embedding 一样，反向传播会直接更新 \(P\) 中被访问的位置行。模型可以自由学习适合训练数据的位置模式，而不是被正弦频率约束。

这里的 \(L_{\max}\) 是预先设定的最大位置数。例如 \(L_{\max}=512\) 时，位置表只有 \(0\) 到 \(511\) 行；推理时第 \(512\) 个 token 没有对应的 \(P[512]\)。因此它的核心短板是：**原生地无法处理超过位置表长度的序列，也几乎没有长度外推能力**。插值、扩表后微调等技巧可以缓解这个问题，但不是这一方法的原生能力。

**优点**：灵活，模型可学习训练集里真正有用的位置模式。

**局限**：参数量随 \(L_{\max}\) 线性增长；最大长度固定；长度外推弱。

**代表**：BERT、GPT-2 等早期预训练模型。

> 绝对位置编码并非“错误的方法”。当上下文窗口固定、任务长度稳定时，它简单有效；只是长上下文 LLM 更需要直接、可扩展地表示相对距离。

## 2. 相对位置编码

语言中的很多关系更依赖相对距离，而不是绝对下标。例如，在“我今天去学校上课”中，“去”与“学校”的距离通常比“去”是第 \(3\) 个还是第 \(300\) 个 token 更重要。相对位置编码的核心目标是让注意力分数显式依赖 \(i-j\)：

$$
s_{ij}=f(q_i,k_j,i-j).
$$

这样，模型能直接知道 token \(j\) 位于 token \(i\) 的左边还是右边、相隔多远。下面的方案主要区别在于：它们将 \(i-j\) 以“向量”“可学习偏置”“旋转相位”中的哪一种形式注入注意力。

### 2.1 Relative Position Representation

#### Shaw et al.：给 Key 和 Value 加相对位置向量

一种经典做法是为每个相对距离 \(r=i-j\) 准备两组可学习向量 \(a_r^K\)、\(a_r^V\)。注意力分数和聚合结果变为：

$$
s_{ij}=\frac{q_i^\top(k_j+a_{i-j}^K)}{\sqrt{d_h}},
$$

$$
\alpha_{ij}=\operatorname{Softmax}_j(s_{ij}),
\qquad
z_i=\sum_j\alpha_{ij}(v_j+a_{i-j}^V).
$$

其中 \(\alpha_{ij}\) 是位置 \(i\) 对 \(j\) 的注意力权重。展开第一式可见额外项：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}
+\frac{q_i^\top a_{i-j}^K}{\sqrt{d_h}}.
$$

第二项使 query 可以根据相对距离，偏好或抑制不同的 key。实际实现中常把距离裁剪到 \([-K,K]\)：

$$
\operatorname{clip}(i-j,-K,K).
$$

这意味着“距离大于 \(K\)”的 token 共用一类表示，避免位置表无限增长。

**优点**：相对位置信息直接参与注意力计算，且不再在输入端和词向量硬相加。

**代价**：需要维护相对位置向量，并为大量 \((i,j)\) 关系计算额外交互；超长序列下的效率与泛化仍需仔细设计。

#### Transformer-XL：将相对项拆进注意力分数

Transformer-XL 将相对位置注意力写得更细。略去缩放系数后，一种常见记法是：

$$
\begin{aligned}
s_{ij}={}&q_i^\top k_j
+q_i^\top W_{K,R}r_{i-j}\\
&+u^\top k_j
+v^\top W_{K,R}r_{i-j}.
\end{aligned}
$$

四项可分别理解为：内容到内容、内容到相对位置、全局内容偏置、全局位置偏置。Transformer-XL 还使用 **relative shift** 技巧高效对齐 \(i-j\) 的索引，并配合跨段记忆处理更长的上下文。

#### T5：把相对位置压缩成 attention bias

T5 使用更轻量的做法：不向 key/value 加向量，只为每个注意力头和距离桶学习一个标量偏置：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}
+b_{h,\operatorname{bucket}(i-j)}.
$$

其中 \(h\) 是注意力头编号，\(\operatorname{bucket}(\cdot)\) 会把较近距离分得更细、较远距离合并得更粗。它保留了相对位置的直接性，又避免为每个距离维护完整向量。

**代表**：Shaw et al. 的相对位置 Transformer、Transformer-XL、T5（相对位置 bias）。

### 2.2 ALiBi：Attention with Linear Biases

ALiBi 不再学习一张位置向量表，而是直接在 attention logit 上减去一个与距离成正比的惩罚。对于因果语言模型，token \(i\) 只能看 \(j\le i\)，定义距离 \(d=i-j\ge0\)，则：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}-m_h(i-j),
\qquad j\le i.
$$

\(m_h>0\) 是第 \(h\) 个注意力头的斜率。距离越远，减去的值越大，注意力权重通常越小。不同头采用不同大小的斜率：有的头更关注近邻，有的头可以更容易看向远处。实现中这些斜率通常按几何级数生成，而不是依赖训练得到的位置 embedding。

例如某个头取 \(m_h=0.25\)，当 query 位于 \(i=10\) 时：

$$
\begin{aligned}
j=9 &: \quad \text{bias}=-0.25,\\
j=6 &: \quad \text{bias}=-1.00,\\
j=0 &: \quad \text{bias}=-2.50.
\end{aligned}
$$

即使三个 key 的内容相似，较近的 key 也会天然更占优势。注意这里的 bias 只改变 softmax 前的分数，**不会改变输入 embedding，也不会旋转或修改 value**。

**优点**：

1. 几乎不增加参数和显存。
2. 没有固定长度的位置表；在比训练时更长的长度上仍能计算相同的线性偏置。
3. 对自回归模型很自然，代码改动小。

**局限**：偏置形式是单调线性的距离惩罚，表达力相对受限；它倾向“越近越好”，但有些任务需要周期性、特定跨度或二维空间结构等更复杂的关系。

**代表**：BLOOM、MPT 等模型采用或提供了 ALiBi 配置。

### 2.3 RoPE：Rotary Position Embedding

RoPE（旋转位置编码）是当前 LLM 中最常见的位置编码方案之一。它不把位置向量加到输入上，也不只加一个标量 bias；它把 \(q\) 和 \(k\) 的相邻维度两两看成二维平面中的向量，再按位置旋转不同角度。

#### 2.3.1 二维旋转：RoPE 的最小单元

二维向量 \([a,b]^\top\) 逆时针旋转 \(\phi\) 后为：

$$
R(\phi)
\begin{bmatrix}a\\b\end{bmatrix}
=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}.
$$

设单头维度 \(d_h\) 为偶数，第 \(k\) 个二维平面的频率为：

$$
\theta_k=B^{-2k/d_h},
\qquad k=0,1,\ldots,\frac{d_h}{2}-1,
$$

通常 \(B=10000\)。位置 \(i\) 在这个平面中的旋转角就是 \(i\theta_k\)。将所有二维旋转拼成一个块对角矩阵，记为 \(R_i\)：

$$
R_i=\operatorname{diag}\bigl(R(i\theta_0),R(i\theta_1),\ldots,R(i\theta_{d_h/2-1})\bigr).
$$

RoPE 只对 query 和 key 应用旋转：

$$
\widetilde q_i=R_iq_i,
\qquad
\widetilde k_j=R_jk_j,
$$

$$
s_{ij}=\frac{\widetilde q_i^\top\widetilde k_j}{\sqrt{d_h}}.
$$

value \(v_j\) 一般不做旋转，最终仍是 \(z_i=\sum_j\alpha_{ij}v_j\)。

#### 2.3.2 为什么旋转后自然得到相对位置？

旋转矩阵满足 \(R_i^\top R_j=R_{j-i}\)，因此：

$$
\widetilde q_i^\top\widetilde k_j
=q_i^\top R_i^\top R_jk_j
=q_i^\top R_{j-i}k_j.
$$

位置 \(i\) 和 \(j\) 不再分别以“两个绝对位置”出现，而是通过 \(j-i\) 这个相对距离影响内积。这是 RoPE 的关键：**它在 QK 内积中以乘性方式注入相对位置，同时保留 token 内容向量。**

也可以把第 \(k\) 个二维平面写成复数。若 \(q_{i,2k}+\mathrm{i}q_{i,2k+1}\) 是 query 的一对分量，则 RoPE 相当于：

$$
\bigl(q_{i,2k}+\mathrm{i}q_{i,2k+1}\bigr)
\mapsto
\bigl(q_{i,2k}+\mathrm{i}q_{i,2k+1}\bigr)e^{\mathrm{i}i\theta_k}.
$$

乘上 \(e^{\mathrm{i}i\theta_k}\) 就是旋转 \(i\theta_k\)；两个位置做内积时，相位之差自然只剩 \((j-i)\theta_k\)。

#### 2.3.3 一个二维数值例子

只看一对维度，取 \(q=[1,0]^\top\)、\(k=[1,0]^\top\)，并令 \(\theta=\pi/6\)。当 query 在位置 \(i=3\)、key 在位置 \(j=1\) 时：

$$
\widetilde q_3=R\left(\frac{\pi}{2}\right)q=[0,1]^\top,
$$

$$
\widetilde k_1=R\left(\frac{\pi}{6}\right)k=
\left[\frac{\sqrt{3}}{2},\frac{1}{2}\right]^\top.
$$

其内积为：

$$
\widetilde q_3^\top\widetilde k_1=\frac12
=\cos\left((j-i)\theta\right)
=\cos\left(-\frac{\pi}{3}\right).
$$

把两个 token 同时向右平移 \(5\) 位，即使用位置 \((8,6)\)，差 \(j-i=-2\) 不变，结果也不变。这正是相对位置不变性的直观体现。

#### 2.3.4 工程实现

实现时通常不显式构造 \(d_h\times d_h\) 的矩阵。把向量的偶数、奇数通道分开即可。对一个二维对 \((a,b)\)，有：

$$
\operatorname{RoPE}(a,b;\phi)
=\left(a\cos\phi-b\sin\phi,\ a\sin\phi+b\cos\phi\right).
$$

伪代码如下，其中 `cos`、`sin` 会预先缓存为形状 `[seq_len, d_h / 2]` 的表：

```python
def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return -x_odd, x_even

def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return interleave(out_even, out_odd)
```

不同开源实现可能将通道按“相邻两两配对”或“前半段与后半段配对”，也可能在旋转方向上使用相反的符号。只要 Q、K 使用一致的约定，且最终相位差由 \(i-j\) 决定，本质是等价的。

#### 2.3.5 RoPE 的优点与长上下文问题

**优点**：

1. 位置只作用于 Q、K 的相关性计算，语义内容与位置信息不必在输入端直接相加。
2. 注意力分数天然含有相对位置 \(i-j\)。
3. 没有长度为 \(L_{\max}\) 的可学习位置表，参数开销很小。
4. 可直接用于自回归 KV cache：缓存的 key 已经带有其所在位置的旋转。

**挑战**：旋转角 \(i\theta_k\) 会随位置增长。若模型仅在较短序列上训练，直接把位置编号推到很大时，远距离的相位模式可能落到训练中未见过的区域，性能会下降。因而 RoPE 虽具备形式上的长度外推能力，但长上下文通常仍需要专门训练或缩放策略。

常见的扩窗思路包括：

| 方法 | 核心想法 |
| --- | --- |
| Position Interpolation (PI) | 将推理位置按比例压缩，例如从 \(i\) 改为 \(i/s\)，让更长上下文映射回训练时较熟悉的位置范围。 |
| NTK-aware / Dynamic NTK scaling | 调整不同频率对应的基底或缩放方式，尽量保留高频局部信息，同时扩展低频尺度。 |
| YaRN、LongRoPE 等 | 对不同频段采用更细的插值/外推策略，并通过少量长文本继续训练校正。 |

这些方法解决的是“如何把 RoPE 用得更长”，而不是改变 RoPE 的基本相对位置原理。

**代表**：RoFormer、LLaMA 系列、Qwen 系列等大量现代 LLM。

### 2.4 三种主流相对方案的对比

| 方法 | 注入位置的位置 | 核心形式 | 长度外推直觉 | 典型特点 |
| --- | --- | --- | --- | --- |
| Relative Position | QK 分数和/或 V 聚合 | \(a_{i-j}^K,a_{i-j}^V\) | 依赖裁剪、bucket 或训练设计 | 直接、表达力强，但实现与计算更复杂 |
| ALiBi | QK 的 logit | \(-m_h(i-j)\) | 线性公式可直接延长 | 极轻量，偏好近邻，表达形式较简单 |
| RoPE | Q 和 K | \(R_iq_i,R_jk_j\) | 可算到任意位置，但长窗常需缩放 | 相位差编码相对距离，现代 LLM 的常用选择 |

## 3. M-RoPE：面向多模态的旋转位置编码

RoPE 的位置 \(i\) 是一维序列下标，适合纯文本。但图像和视频并不是天然的一维：图像 patch 有高度、宽度坐标，视频还多了时间坐标。如果仅把图像 patch 按行展开成一维序列，二维中相邻的 patch 可能在序列中相隔很远；模型也难以区分“向右一格”和“向下一格”。

M-RoPE（Multimodal Rotary Position Embedding）将位置从一个标量扩展为多维坐标。对视频/图像 token，可为第 \(n\) 个视觉 token 分配：

$$
\mathbf{p}_n=(t_n,h_n,w_n),
$$

其中 \(t_n\) 是时间或帧坐标，\(h_n\) 是高度坐标，\(w_n\) 是宽度坐标。对于图像，可将 \(t_n\) 视为固定值；对于纯文本，可让三组位置 ID 都随文本位置递增，以维持一维文本顺序。

### 3.1 将注意力通道分给时间、高度和宽度

将一个注意力头的通道拆为三组：

$$
q_n=\left[q_n^{(t)};q_n^{(h)};q_n^{(w)}\right],
\qquad
k_n=\left[k_n^{(t)};k_n^{(h)};k_n^{(w)}\right].
$$

分别使用对应坐标旋转：

$$
\widetilde q_n=
\left[
R(t_n)q_n^{(t)};
R(h_n)q_n^{(h)};
R(w_n)q_n^{(w)}
\right],
$$

$$
\widetilde k_m=
\left[
R(t_m)k_m^{(t)};
R(h_m)k_m^{(h)};
R(w_m)k_m^{(w)}
\right].
$$

于是 QK 内积同时含有三种相对位移：

$$
\widetilde q_n^\top\widetilde k_m
=g_t(t_m-t_n)+g_h(h_m-h_n)+g_w(w_m-w_n),
$$

其中 \(g_t,g_h,g_w\) 由各自通道中的内容向量与旋转频率共同决定。这里的加法来自通道分组后的内积展开，不表示三种关系一定同等重要；模型可以通过训练学习每一组通道的使用方式。

对于一张 \(H\times W\) patch 网格，位置 \((h,w)=(2,5)\) 与 \((2,6)\) 的宽度差是 \(1\)，而与 \((3,5)\) 的高度差是 \(1\)。M-RoPE 让模型在不同子空间中分别感知这两种“相邻”，无需依赖一维 flatten 后的偶然编号。

### 3.2 文字、图像、视频如何共用一套位置系统？

以文本-图像混合序列为例，M-RoPE 会在输入阶段为每个 token 准备三行 position IDs：

$$
\operatorname{position\_ids}
\in\mathbb{Z}^{3\times L}.
$$

文本 token 使用沿文本顺序增长的位置；视觉 token 使用其时间、高度、宽度坐标。实现还需要处理一个工程细节：不同模态 token 在拼接后的序列里不能发生位置冲突。因此，视觉坐标通常会与此前文本的累计位置进行适当偏移，后续文本的位置也会接在视觉内容之后。具体偏移规则会随模型实现而变，但目的相同：既保留视觉内部的二维/三维结构，又保持多模态序列整体的顺序关系。

**代表**：Qwen2-VL 提出了 M-RoPE，用它统一建模文本的一维顺序、图像的二维空间与视频的三维时空位置；后续 Qwen-VL 系列也沿用这一思路。

## 4. 如何建立自己的理解框架？

学习各种位置编码时，可以反复问四个问题：

1. **位置是什么？** 是绝对下标 \(i\)，相对距离 \(i-j\)，还是 \((t,h,w)\) 这样的多维坐标？
2. **位置注入到哪里？** 是输入 \(x_i\)，注意力 logit \(s_{ij}\)，还是 Q/K 的几何变换？
3. **模型能直接获得什么关系？** 是“第 128 个 token”，还是“向左 3 个 token”，还是“同一行右边 1 格”？
4. **长度和模态变化时会怎样？** 是否受 \(L_{\max}\) 限制？能否外推？是否保留图像/视频的空间结构？

用这四个问题回看全文，可以得到一条清晰主线：

$$
\begin{array}{c}
\text{绝对位置编码：告诉模型“我在第几位”}\\
\Downarrow\\
\text{相对位置编码：告诉模型“你离我多远、在哪个方向”}\\
\Downarrow\\
\text{RoPE：把相对距离写入 QK 内积的相位差}\\
\Downarrow\\
\text{M-RoPE：把一维相位差扩展到时间、高度、宽度}
\end{array}
$$

## 5. 进一步阅读

### 原始论文

1. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
2. Shaw, Uszkoreit and Vaswani. [Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/), 2018.
3. Dai et al. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860), 2019.
4. Press, Smith and Lewis. [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409), 2021.
5. Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
6. Wang et al. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191), 2024.

### 本次学习材料

1. [Bilibili 视频 1](https://www.bilibili.com/video/BV113Kp6RECF/)
2. [Bilibili 视频 2](https://www.bilibili.com/video/BV1aLCsBhE7i/)
3. [Bilibili 视频 3](https://www.bilibili.com/video/BV1FjrCBdESo/)
4. [知乎：大模型的位置编码](https://zhuanlan.zhihu.com/p/650469278)

> 建议下一步用一个极小的 \(d_h=4\) 示例，手算一次普通 attention、ALiBi attention 与 RoPE attention 的 \(s_{ij}\)。能亲手看到“加 bias”和“旋转 Q/K”分别改变了哪里，位置编码就不再只是一组公式了。
