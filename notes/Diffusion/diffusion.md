---
title: "Diffusion 模型笔记：从 DDPM 到 v-prediction 与 Flow Matching"
date: 2026-10-18 00:00:00 +0800
permalink: /posts/diffusion-ddpm-v-prediction-flow-matching/
description: "从 DDPM 的前向加噪与方差保持讲起，解释 ε、x₀、v 三种预测参数化的训练目标、统计难度、变量转换与数值性质，并连接到 Flow Matching。"
categories: [Generation]
tags: [Diffusion, DDPM, Flow-Matching, v-prediction]
math: true
toc: true
image:
  path: /assets/img/diffusion/ddpm-cover.webp
  alt: "DDPM 在 CelebA-HQ 与 CIFAR-10 上的生成样本"
---

<div class="qwen-note" markdown="1">

这篇笔记先交代 DDPM 的加噪/去噪框架，随后比较 $\epsilon$-、$x_0$-、$v$-prediction 三种参数化，最后从 $v$-prediction 自然过渡到 Flow Matching的讲解。

## 1. DDPM：前向扩散与反向去噪

首先是 DDPM，它把一张干净图像 $x_0$ 经过 $T$ 步加噪变成 $x_T$，再学习一个反向过程，把它一步步还原。前向过程是固定的，没有可学习参数；训练只发生在反向过程。

### 1.1 前向过程

单步前向加噪定义为

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right),
$$

写成采样形式：

$$
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

这里 $\beta_t\in(0,1)$ 是第 $t$ 步加入的噪声方差。

我们令 $\alpha_t=1-\beta_t$，利用重参数化，可以从 $x_0$ 直接跳到任意 $t$：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

其中 $\bar\alpha_t=\prod_{s=1}^t(1-\beta_s)$。这条公式是后面所有预测目标共同的出发点。

### 1.2 反向去噪与训练

反向过程学习

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(\mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t)\right),
$$

网络输入是含噪图 $x_t$ 和时间 $t$，输出用于估计这一步的去噪均值；方差通常固定，也可以学习。DDPM 把训练目标简化成对噪声的 MSE：

$$
L_{\text{simple}}
=\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2.
$$

训练时，$x_0$ 来自数据，$t$ 均匀采样，$\epsilon$ 是前向加噪时实际采样的噪声；网络只要从 $x_t$ 中把它猜出来。实际实现中常用 U-Net 作为 $\epsilon_\theta$。采样时从 $x_T\sim\mathcal N(0,I)$ 出发，逐步应用反向分布，最后得到 $x_0$。

这里写的是 $\epsilon$-prediction，也就是 DDPM 原始论文常用的参数化。网络也可以预测 $x_0$，或预测两者的线性组合 $v$。

![DDPM 在 CIFAR-10 上的渐进生成：同一批样本从噪声逐步变为清晰图像](/assets/img/diffusion/ddpm-progressive.webp){: .qwen-figure width="1200" height="246" }

图 1 · DDPM 的渐进生成。每一行是同一个 CIFAR-10 样本在反向过程中的 $\hat{x}_0$，从左到右噪声逐渐减少。来源：[Ho et al. (2020)](https://arxiv.org/abs/2006.11239), Figure 6。
{: .qwen-caption }

### 1.3 α 与 β

$\beta_t$ 是第 $t$ 步的噪声方差，$\alpha_t=1-\beta_t$ 是单步保留信号的系数，$\bar\alpha_t=\prod_{s=1}^t\alpha_s$ 是累积保留系数。三者一起决定信号被噪声淹没的速度。

| 符号 | 含义 |
| --- | --- |
| $\beta_t$ | 第 $t$ 步加入多少噪声；越大，加噪越猛 |
| $\alpha_t=1-\beta_t$ | 第 $t$ 步保留多少上一步的信号 |
| $\bar\alpha_t=\prod_{s=1}^t\alpha_s$ | 到第 $t$ 步还剩多少原始信号；$\sqrt{\bar\alpha_t}$ 是 $x_t$ 中 $x_0$ 的系数 |

随着反向过程，$t$ 逐渐减小，$\beta_t$ 不断变小，$\bar\alpha_t$ 因此不断增大。低噪声端 $\bar\alpha_t\approx1$，$x_t\approx x_0$；高噪声端 $\bar\alpha_t\approx0$，$x_t\approx\epsilon$。这就是后面反复用到的两个端点。

$\beta_t$ 的序列叫 **noise schedule**。原始 DDPM 用线性调度，但它在低噪声端让 $\bar\alpha_t$ 下降偏慢、在高噪声端下降偏快；Improved DDPM 的余弦调度让 $\bar\alpha_t$ 整体下降更均匀。

### 1.4 系数带根号：保持方差

前向加噪的系数满足

$$
\left(\sqrt{1-\beta_t}\right)^2+\left(\sqrt{\beta_t}\right)^2
=(1-\beta_t)+\beta_t=1.
$$

所以当 $x_{t-1}$ 和 $\epsilon_t$ 独立、方差都为 1 时，

$$
\operatorname{Var}(x_t)=(1-\beta_t)\cdot1+\beta_t\cdot1=1.
$$

累积形式同样有

$$
\left(\sqrt{\bar\alpha_t}\right)^2
+\left(\sqrt{1-\bar\alpha_t}\right)^2
=\bar\alpha_t+(1-\bar\alpha_t)=1,
$$

因此任意时刻的信号方差与噪声方差都按比例拼成 1，输入尺度保持稳定；这也是 variance preserving（保持方差）这个名字的来源。

如果去掉根号，写成 $x_t=\bar\alpha_t x_0+(1-\bar\alpha_t)\epsilon$，方差会变成 $\bar\alpha_t^2+(1-\bar\alpha_t)^2$。当 $\bar\alpha_t=0和1$ 时，方差为1，但当 $\bar\alpha_t=0.5$ 时，这个值只有 $0.5$，输入尺度缩小到 $\sqrt{0.5}\approx0.707$ 倍，中间时刻会塌缩。根号就是让系数的平方等于方差占比，从而避免这个问题；反向高斯后验的闭式解也更整洁。

## 2. 三种预测目标：ε、x₀、v

三种参数化都从同一个前向公式出发：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

DDPM 中的 $x_0$ 表示干净图像，$\epsilon$ 表示前向加噪时实际采样的高斯噪声。模型接收 $x_t,t$，可以选择预测 $\epsilon$、$x_0$，或者它们的线性组合 $v$。


### 2.1 ε-prediction：预测被加进去的噪声

网络 $\epsilon_\theta(x_t,t)$ 拟合前向过程中实际采样的噪声 $\epsilon$，训练损失为

$$
L_\epsilon=\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2.
$$

**目标本身的预测难度。** 高噪声端 $x_t\approx\epsilon$，输入中主要就是这次噪声，因此从 $x_t$ 预测 $\epsilon$ 通常比较直接。低噪声端 $x_t\approx x_0$，而 $\epsilon$ 与干净图像独立，近乎干净的输入无法告诉模型这次具体采样了哪一份噪声，所以对具体 $\epsilon$ 的条件不确定性较大。

**变量转换：如果需要从噪声估计得到干净图像。** 前向公式可解出

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}.
$$

把真实噪声换成网络预测 $\hat\epsilon$，得到

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.
$$

令噪声预测误差为 $\Delta\epsilon=\hat\epsilon-\epsilon$。两式相减：

$$
\begin{aligned}
\hat x_0-x_0
&=-\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}
  (\hat\epsilon-\epsilon)\\
&=-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\Delta\epsilon.
\end{aligned}
$$

转换系数 $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ 在低噪声端趋于 $0$，在高噪声端趋于无穷大。因此，**如果预测噪声的误差大小相同，从噪声估计换算成 $x_0$ 时，高噪声端的换算会更敏感。**


### 2.2 x₀-prediction：直接预测干净图像

网络 $x_{0,\theta}(x_t,t)$ 直接拟合干净图像 $x_0$，训练损失为

$$
L_{x_0}=\left\|x_0-x_{0,\theta}(x_t,t)\right\|^2.
$$

**目标本身的预测难度。** 低噪声端 $x_t\approx x_0$，输入已经接近答案，直接预测干净图像通常比较容易。高噪声端 $x_t\approx\epsilon$，输入几乎是独立噪声，不含这张具体图像的足够信息，因此准确恢复该图像更困难；在均方误差下，最优预测会趋向给定 $x_t$ 后的条件均值。

**变量转换：如果采样公式需要噪声估计。** 前向公式可解出

$$
\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}.
$$

把 $x_0$ 换成网络预测 $\hat x_0$，得到

$$
\hat\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}}.
$$

令图像预测误差为 $\Delta x_0=\hat x_0-x_0$。两式相减：

$$
\begin{aligned}
\hat\epsilon-\epsilon
&=-\frac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}
  (\hat x_0-x_0)\\
&=-\sqrt{\frac{\bar\alpha_t}{1-\bar\alpha_t}}\,\Delta x_0.
\end{aligned}
$$

转换系数 $\sqrt{\bar\alpha_t/(1-\bar\alpha_t)}$ 在高噪声端趋于 $0$，在低噪声端趋于无穷大。因此，**如果 $x_0$ 预测误差大小相同，从图像估计换算成噪声时，低噪声端的换算会更敏感。**

### 2.3 v-prediction：旋转后的预测参数化

前两种参数化在换算成另一变量时，各有一个端点会出现趋于无穷大的系数。$v$-prediction 提供了另一种参数化：预测 $x_0$ 与 $\epsilon$ 的一个正交线性组合：

$$
v\equiv\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0.
$$

训练损失为

$$
L_v=\left\|v-v_\theta(x_t,t)\right\|^2.
$$

把 $x_t$ 和 $v$ 放在一起：

$$
\begin{pmatrix}x_t\\ v\end{pmatrix}
=
\begin{pmatrix}
\sqrt{\bar\alpha_t} & \sqrt{1-\bar\alpha_t}\\
-\sqrt{1-\bar\alpha_t} & \sqrt{\bar\alpha_t}
\end{pmatrix}
\begin{pmatrix}x_0\\ \epsilon\end{pmatrix}.
$$

中间的矩阵是正交矩阵，因此可以把它理解为对 $(x_0,\epsilon)$ 做了一个随时间变化的旋转。逆变换为

$$
x_0=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,v,
\qquad
\epsilon=\sqrt{1-\bar\alpha_t}\,x_t+\sqrt{\bar\alpha_t}\,v.
$$

这两条式子没有除以 $\sqrt{\bar\alpha_t}$ 或 $\sqrt{1-\bar\alpha_t}$ 的项。若 $v$ 的预测误差为 $\Delta v=\hat v-v$，则

$$
\hat x_0-x_0=-\sqrt{1-\bar\alpha_t}\,\Delta v,
\qquad
\hat\epsilon-\epsilon=\sqrt{\bar\alpha_t}\,\Delta v.
$$

所以

$$
\|\hat x_0-x_0\|\leq\|\Delta v\|,
\qquad
\|\hat\epsilon-\epsilon\|\leq\|\Delta v\|.
$$

**这说明什么？** 对于给定大小的 $v$ 预测误差，将其换算成 $x_0$ 或 $\epsilon$ 时，不会因为时间走到端点而被这些系数显式放大。这是 $v$-prediction 的一个数值优势（虽然这个数值优势并不能保证生成质量更好）。

端点上，目标的形式为

$$
\bar\alpha_t\to0:\ v\approx-x_0,
\qquad
\bar\alpha_t\to1:\ v\approx\epsilon.
$$

因此，高噪声端的 $v$ 目标接近负的干净图像，低噪声端接近噪声。两端仍可能有较大的条件不确定性：高噪声输入不能确定对应的具体干净图像，低噪声输入也不能确定独立采样的具体噪声。

这也是为什么不能把 $v$-prediction 的优势称为“它解决了反推误差”：因为反变换有界只是一个可分析的性质而已。Salimans 与 Ho 在[渐进蒸馏论文的附录 D](https://arxiv.org/abs/2202.00512)中指出，在他们讨论的角度参数化下，$v$-prediction 具有使 DDIM 步长表达式不直接依赖信噪比（SNR）的独特性质，这有利于少步采样的稳定性；但生成具体性能仍取决于模型、损失、调度和采样方法。

![v-prediction 的几何解释：干净图像、噪声与速度构成一组旋转坐标](/assets/img/diffusion/v-prediction-geometry.webp){: .qwen-figure width="790" height="799" }

图 2 · v-prediction 的几何解释。图中 $\mathbf{x}$ 对应干净图像 $x_0$，$\mathbf{z}$ 对应含噪图 $x_t$，$\mathbf{v}$ 是旋转后的速度方向；$\alpha,\sigma$ 对应本笔记的 $\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t}$。来源：[Salimans & Ho (2022)](https://arxiv.org/abs/2202.00512), Figure 5。
{: .qwen-caption }

### 为什么要把预测结果“反推”成另一个变量去分析误差放大？

**反推不是训练的必经步骤，也不是每种采样算法都必须做的事。** 训练 $\epsilon$-prediction 时，损失只比较预测噪声与真实噪声；训练 $x_0$-prediction 时，损失只比较预测图像与真实图像。采样时，也可以直接使用模型原本预测的变量。

这里分析变量转换，主要是为了回答一个不同的问题：**如果我们把模型输出换算成另一种变量，原来的预测误差会怎样传播？** 这有助于比较不同参数化的数值性质，了解这个性质，可以帮助我们深入了解 Diffusion Model 的机制。

### 2.4 三种参数化对比

三种参数化在给定 $x_t,t$ 时可以代数换算，但它们各自的 MSE 训练目标并不完全相同：转换关系包含随时间变化的系数，因此也对应不同的隐式损失权重。关键问题是：**预测目标本身训练难不难**，以及**反推另一个变量时会不会放大误差**。

| 预测目标 | 目标本身较难的端点（统计层面） | 换算成另一变量时的敏感端点（代数层面） | 应如何理解 |
| --- | --- | --- | --- |
| $\epsilon$-prediction | 低噪声端：近乎干净的输入难以确定本次独立采样的具体噪声 | 若由 $\hat\epsilon$ 换算 $\hat x_0$，高噪声端系数 $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ 很大 | 不是说训练必然不稳定；敏感性是否影响采样，要看采样式是否使用该转换 |
| $x_0$-prediction | 高噪声端：近乎纯噪声的输入难以确定具体干净图像 | 若由 $\hat x_0$ 换算 $\hat\epsilon$，低噪声端系数 $\sqrt{\bar\alpha_t/(1-\bar\alpha_t)}$ 很大 | 不是说直接预测 $x_0$ 在低噪声端困难；若直接使用 $\hat x_0$，该转换不是必需步骤 |
| $v$-prediction | 两端仍有条件不确定性：高噪声端 $v\approx-x_0$，低噪声端 $v\approx\epsilon$ | 换算成 $x_0$ 或 $\epsilon$ 时没有上述小分母系数 | 反变换系数有界是优势之一，但不等于目标一定更容易或所有设置下效果更好 |

误差转换公式是分析工具，不是训练目标，也不能判断最终图像的质量。要评价采样时的误差，需要把模型输出代入实际使用的采样更新公式；要评价训练难度，则要看目标的条件不确定性、MSE 的时间权重和模型的实际拟合能力。

接下来，Flow Matching 会把“预测目标可以怎样定义”这个想法进一步推广：直接定义一条从噪声到数据的路径，让网络预测路径速度。
## 3. Flow Matching：从 v-prediction 到直线路径

$v$-prediction 说明预测目标不一定非得是 $x_0$ 或 $\epsilon$，也可以是它们的线性组合。Flow Matching（FM）把这件事定义的更直接：直接定义一条从噪声到数据的路径，并让网络预测沿路径的速度。这里只讨论最直观的直线版本。

> **记号切换**：在 FM 这一节，$x_0$ 表示噪声起点，$x_1$ 表示干净数据；这和 DDPM 中 $x_0$ 表示干净图像相反。

### 3.1 直线路径与条件速度

FM 的直线版本在噪声 $x_0$ 与数据 $x_1$ 之间做线性插值：

$$
x_t=(1-t)x_0+t x_1,\qquad t\in[0,1].
$$

对固定的 $(x_0,x_1)$ 配对，路径对时间的导数是

$$
v_{\mathrm{FM}}(x_t\mid x_0,x_1)
=\frac{dx_t}{dt}=x_1-x_0.
$$

这个**条件速度**只由起点和终点决定，与 $t$ 无关。训练时，网络 $v_\theta(x_t,t)$ 拟合它，损失为

$$
L_{\mathrm{FM}}
=\left\|(x_1-x_0)-v_\theta(x_t,t)\right\|^2.
$$

直观地说，DDPM 的 $v$-prediction 像随时变化的指南针，FM 的直线条件路径像一张固定的路线图；但实际训练后 FM 并无法做到真正的直线路径，因为实际轨迹仍可能是弯曲的。

和 DDPM 的 $v$-prediction 不同，这里的 $v$ 来自路径的导数，而不是 $x_0,\epsilon$ 的正交参数化。

![Flow Matching 中扩散路径与 OT 路径的条件向量场对比](/assets/img/diffusion/fm-vector-fields.webp){: .qwen-figure width="1400" height="223" }

图 3 · 扩散路径与 OT 路径的条件向量场对比。左半部分是扩散路径的条件 score，右半部分是 OT 路径的条件速度场；后者的方向更稳定，对应更直的传输路径。来源：[Lipman et al. (2023)](https://arxiv.org/abs/2210.02747), Figure 2。
{: .qwen-caption }

### 3.2 训练与生成

训练流程可以写成四步：

1. 从数据取一张图 $x_1$，从标准高斯采一个噪声图 $x_0$。
2. 均匀采一个 $t\in[0,1]$，构造 $x_t=(1-t)x_0+t x_1$。
3. 计算条件速度 $v=x_1-x_0$。
4. 让网络 $v_\theta(x_t,t)$ 回归 $v$，用 MSE 更新参数。

生成时解 ODE：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t),
$$

从 $t=0$ 的噪声 $x_0$ 出发，用欧拉法或其他数值积分器逐步走到 $t=1$：

$$
x_{t+\Delta t}=x_t+v_\theta(x_t,t)\,\Delta t.
$$

所以 FM 仍然是多步迭代生成；“训练目标与 $t$ 无关”不等于“推理可以一步到位”。

### 3.3 条件直线的端点反解：一个代数性质

FM 的速度预测还有一个与 DDPM 的 $v$-prediction 相似的数值性质：从预测速度反推端点估计时，不需要除以会在端点趋近于零的系数。

对固定的 $(x_0,x_1)$ 配对，由 $x_t=(1-t)x_0+t x_1$ 和 $v_{\mathrm{FM}}=x_1-x_0$ 可直接反解出

$$
x_0=x_t-t\,v_{\mathrm{FM}},\qquad
x_1=x_t+(1-t)\,v_{\mathrm{FM}}.
$$

若速度估计存在误差 $\Delta v$，即 $\hat v_{\mathrm{FM}}=v_{\mathrm{FM}}+\Delta v$。把 $\hat v_{\mathrm{FM}}$ 代入上面的逆变换，再减去精确式，得

$$
\hat x_0-x_0=-t\,\Delta v,\qquad
\hat x_1-x_1=(1-t)\,\Delta v.
$$

因此

$$
\left\|\hat x_0-x_0\right\|\le\|\Delta v\|,\qquad
\left\|\hat x_1-x_1\right\|\le\|\Delta v\|.
$$

因为 $t\in[0,1]$，两个系数 $t$ 和 $1-t$ 都不超过 1。时间趋近 0 时，$x_0$ 的反变换系数趋于 0；时间趋近 1 时，$x_1$ 的反变换系数趋于 0。也就是说，FM 的速度误差转换到任一端点的估计时，不会仅仅因为时间到达端点而被显式放大。

在这一点上，FM 与 DDPM 的 $v$-prediction 共享一项数值优势：DDPM 中 $v$ 的误差传到 $x_0$ 和 $\epsilon$ 的系数是 $\sqrt{1-\bar\alpha_t}$ 和 $\sqrt{\bar\alpha_t}$，FM 中对应的是 $t$ 和 $1-t$，它们都落在 $[0,1]$ 内，都不会在端点产生除零式放大。两者都避免了像 $\epsilon$-prediction 反推 $x_0$、或 $x_0$-prediction 反推 $\epsilon$ 那样除以趋近于零的系数。
### 3.4 条件速度 vs. 边缘速度

上面的 $v_{\mathrm{FM}}=x_1-x_0$ 是**条件速度**：给定具体（噪音、图像）对配对时，它就是常数。但训练时网络只看到 $x_t$，不知道这个 $x_t$ 来自哪一对 $(x_0,x_1)$。因此网络实际拟合的是**边缘速度**

$$
v(x_t,t)=\mathbb E[x_1-x_0\mid x_t].
$$

这个期望是依赖 $t$ 的，因为 $x_t$ 的分布随 $t$ 变化。给定同一个位置，可能有很多条直线穿过它，网络输出的是这些条件速度的平均。所以网络仍然需要接收 $t$ 作为输入，训练标签的构造简单，不等于边缘速度场简单。

简单来说就是，因为训练时不止有一个具体的图像数据对，所以模型无法去拟合一条从噪音到纯净图像的直线（条件速度），只能预测数据分布的平均，实际轨迹仍然是弯曲的（边缘速度）。

### 3.5 为什么不能一步到位？

理论上，如果边缘速度场恰好是常数，所有配对的速度完全一致（模型拟合出一条从噪音到纯净图像的直线），那确实可以一步：

$$
x_1=x_0+v\cdot1.
$$

但现实中边缘速度场通常不是常数，原因有两个：

1. **边缘路径通常是弯的**：条件路径是直线，但对所有配对取期望后，边缘轨迹不一定直。
2. **数值积分有误差**：网络预测的速度是近似值，而欧拉法等求解器也会产生离散化误差。步长过大时，通常会增加积分误差；但误差大小取决于速度场和求解器，不能笼统说每一步都会把误差放大。

所以，FM 的直线条件路径为少步采样提供了有利条件，但实际需要多少步，还取决于学到的边缘路径有多直、网络误差和数值求解器。不能因为 FM “条件路径是直线”就直接认为“推理一步到位”或“任何 FM 都必然比 DDPM 步数少”。蒸馏可以把步数进一步压到 1～4 步，但那是额外训练的结果，不是 FM 本身的性质。

### 3.6 和 DDPM 的关系

| 维度 | DDPM | Flow Matching（直线版） |
| --- | --- | --- |
| 预测目标 | $\epsilon$、$x_0$ 或 $v$，由前向混合定义 | 路径速度 $v_{\mathrm{FM}}=x_1-x_0$ |
| 训练标签 | 由 $t$ 和前向混合构造 | 固定配对下与 $t$ 无关 |
| 推理 | 迭代反向去噪 | 迭代积分 ODE |
| 路径 | 由噪声调度决定，通常弯曲 | 条件路径是直线；边缘路径由模型学到的速度场决定 |
| 变量转换的代数性质 | 若转换到另一变量，$\epsilon$/$x_0$ 参数化各有一端出现小分母；$v$ 参数化的转换系数有界 | 固定条件线段的端点反解系数 $t,1-t$ 有界；这不直接保证边缘 ODE 的整体采样误差有界 |

两者都是迭代生成，差别主要在**路径形状、训练目标和数值条件**。FM 与 DDPM 的 $v$-prediction 共享“反变换不放大误差”的数值性质，但 FM 的速度来自路径导数，DDPM 的 $v$ 是 $x_0,\epsilon$ 的正交参数化；二者的结构并不完全相同。

## 4. 总结

1. **DDPM**：前向逐步加噪，重参数化可一步得到 $x_t$；反向网络用 MSE 学习去噪。$\alpha,\beta$ 控制加噪节奏，根号让信号/噪声系数的平方和为 1，保持 $x_t$ 的方差。
2. **三种预测目标**：$\epsilon$、$x_0$、$v$ 数学等价，但误差反推的数值性质不同。$\epsilon$-prediction 在高噪声端反推 $x_0$ 时放大，$x_0$-prediction 在低噪声端反推 $\epsilon$ 时放大；$v$-prediction 通过正交旋转缓解了两端的小分母问题。
3. **Flow Matching**：把“预测速度”推广到路径导数，直线条件路径的速度是 $x_1-x_0$；反推端点时系数是 $t$ 和 $1-t$，同样不会在端点产生除零式放大。但边缘速度仍依赖 $t$，生成仍需迭代积分。


## 5. 参考

1. Ho, Jain, Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
2. Nichol, Dhariwal. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), 2021.
3. Salimans, Ho. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512), 2022.
4. Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023.

</div>
