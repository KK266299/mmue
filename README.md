# 面向分类与语义分割的多模态 Unlearnable Examples：基于 Fusion Attack 的统一研究方案

## 1. 研究背景与核心动机

Unlearnable Examples（UE）的目标是在尽量不影响人类感知的前提下，对训练样本加入微小扰动，使未经授权的模型在这些受保护数据上训练后难以获得良好的泛化能力。

现有 UE 方法主要集中在以下方向：

- 单模态分类任务中的输入级扰动；
- 图文对比学习中的多模态联合扰动；
- 以损失函数设计为主的保护方式。

但是，对于多模态分类与多模态语义分割而言，真正决定性能的关键并不只是输入本身，而是：

- 模态之间如何对齐；
- 信息如何跨模态路由；
- 哪个模态在融合时占主导；
- 哪些共享语义被保留；
- 哪些互补协同信息被利用；
- 哪些伪相关捷径被模型当作"有用模式"学走。

因此，一个更强的多模态 UE 方向不应被简单写成"降低模态间互信息"或"叠加跨模态损失"，而应被统一表述为：

> 通过攻击多模态模型内部的融合机制（fusion mechanism），诱导模型学习错误的跨模态信息流、错误的模态权重分配以及错误的共享表示结构，从而在分类或语义分割任务中失去泛化能力。

基于这一思路，本文将多模态 UE 统一建模为三类互补的 fusion attack：

1. **Fusion Routing Attack**  
   攻击 cross-attention、token routing、共享信息流；

2. **Fusion Weight Attack**  
   攻击 gate、uncertainty weighting、modality contribution；

3. **Fusion Interface Attack**  
   攻击 bottleneck、adapter、prompt、shared latent 等共享融合接口。

这三类攻击并不是彼此割裂的，而是可以统一到一个面向 fusion mechanism 的保护框架中。

---

## 2. 问题定义

### 2.1 多模态学习基本形式

设多模态数据集为

$$

\mathcal{D} = \{(x_i^{(1)}, x_i^{(2)}, y_i)\}_{i=1}^{N},

$$

其中：

- $x_i^{(1)}$ 表示模态 1；
- $x_i^{(2)}$ 表示模态 2；
- $y_i$ 表示标签。

对于分类任务，有

$$

y_i \in \{1,2,\dots,C\}.

$$

对于语义分割任务，有

$$

y_i \in \{1,2,\dots,C\}^{H \times W}.

$$

一个典型的多模态模型可写为

$$

f_{\theta}(x^{(1)}, x^{(2)})
=
h_{\theta}\left(
F_{\theta}\left(
E_{\theta}^{(1)}(x^{(1)}),
E_{\theta}^{(2)}(x^{(2)})
\right)
\right),

$$

其中：

- $E_{\theta}^{(1)}$ 和 $E_{\theta}^{(2)}$ 分别为两个模态的编码器；
- $F_{\theta}$ 为融合模块；
- $h_{\theta}$ 为任务头。

标准训练目标为

$$

\min_{\theta}
\;
\mathbb{E}_{(x^{(1)},x^{(2)},y)\sim \mathcal{D}}
\left[
\mathcal{L}_{\text{task}}\left(
f_{\theta}(x^{(1)},x^{(2)}), y
\right)
\right].

$$

---

### 2.2 受保护样本与目标

受保护数据集记为

$$

\tilde{\mathcal{D}}
=
\{(\tilde{x}_i^{(1)}, \tilde{x}_i^{(2)}, y_i)\}_{i=1}^{N},

$$

其中

$$

\tilde{x}_i^{(1)} = x_i^{(1)} + \delta_i^{(1)}, \qquad
\tilde{x}_i^{(2)} = x_i^{(2)} + \delta_i^{(2)}.

$$

扰动满足约束

$$

\|\delta_i^{(1)}\|_{p} \le \epsilon_1, \qquad
\|\delta_i^{(2)}\|_{p} \le \epsilon_2.

$$

多模态 UE 的目标不是简单让训练误差升高，而是：

> 让模型在受保护数据上训练时，学习到错误的融合规则、错误的交互结构和错误的模态依赖关系，从而在干净测试分布上泛化失败。

---

## 3. 总体思想：将多模态 UE 统一建模为 Fusion Attack

### 3.1 总体原则

本文不把多模态 UE 仅仅看作输入扰动问题，而是把它看作一个对融合机制进行干预的过程。

设保护器为 $G_{\phi}$，则其生成双模态扰动：

$$

(\delta^{(1)}, \delta^{(2)}) = G_{\phi}(x^{(1)}, x^{(2)}, y).

$$

得到受保护样本：

$$

\tilde{x}^{(1)} = x^{(1)} + \delta^{(1)}, \qquad
\tilde{x}^{(2)} = x^{(2)} + \delta^{(2)}.

$$

相应的特征表示为

$$

z^{(1)} = E_{\theta}^{(1)}(\tilde{x}^{(1)}), \qquad
z^{(2)} = E_{\theta}^{(2)}(\tilde{x}^{(2)}).

$$

然后，保护器不只是希望输出预测变差，而是希望影响以下三个层面：

1. 跨模态信息如何流动；
2. 融合时哪个模态更被信任；
3. 哪个共享接口承载了伪共享语义。

因此，本文提出一个统一形式：

$$

z_{f}
=
F_{\theta}\left(
z^{(1)}, z^{(2)};\,
R_{\phi}, W_{\phi}, I_{\phi}
\right),

$$

其中：

- $R_{\phi}$ 表示 routing attack 相关控制变量；
- $W_{\phi}$ 表示 weight attack 相关控制变量；
- $I_{\phi}$ 表示 interface attack 相关控制变量。

最终输出为

$$

\hat{y} = h_{\theta}(z_f).

$$

---

## 4. 模块一：Fusion Routing Attack

## 4.1 核心思想

Fusion Routing Attack 的目标是攻击跨模态信息的流动路径，而不是只攻击最终特征值。

在很多多模态架构中，跨模态交互通过以下机制完成：

- cross-attention；
- token-to-token routing；
- shared token exchange；
- feature matching and message passing。

因此，如果能让模型在训练时形成**错误的信息路由模式**，就可能使其把错误的跨模态对应关系当成"稳定规律"学走。

---

## 4.2 一般化建模

设两个模态特征分别为

$$

z^{(1)} \in \mathbb{R}^{n_1 \times d}, \qquad
z^{(2)} \in \mathbb{R}^{n_2 \times d}.

$$

标准 cross-attention 融合可以写为

$$

A = \operatorname{Softmax}\left(
\frac{Q(z^{(1)})K(z^{(2)})^{\top}}{\sqrt{d}}
\right),

$$

$$

u^{(1 \leftarrow 2)} = A \, V(z^{(2)}).

$$

Routing Attack 的目标是对路由矩阵或注意力先验施加干预，得到

$$

\tilde{A} = \operatorname{Softmax}\left(
\frac{Q(z^{(1)})K(z^{(2)})^{\top}}{\sqrt{d}} + B_{\phi}
\right),

$$

其中 $B_{\phi}$ 表示由保护器诱导的路由偏置。

于是受保护的跨模态消息变为

$$

\tilde{u}^{(1 \leftarrow 2)} = \tilde{A} \, V(z^{(2)}).

$$

---

## 4.3 设计目标

Routing Attack 的核心目标包括：

1. 让正确的 token 对应关系减弱；
2. 让错误的 token 匹配增强；
3. 让共享信息流优先通过错误的局部区域；
4. 让模型在训练中稳定依赖这种错误路由。

可以定义一个路由错配损失：

$$

\mathcal{L}_{\text{route}}
=
\mathbb{E}
\left[
D\left(\tilde{A}, A^{\star}\right)
\right],

$$

其中：

- $D(\cdot,\cdot)$ 表示距离函数；
- $A^{\star}$ 表示预设的错误路由模式或目标偏置路由。

也可以从一致性破坏的角度定义：

$$

\mathcal{L}_{\text{route}}
=
-
\mathbb{E}
\left[
D\left(\tilde{A}, A_{\text{clean}}\right)
\right].

$$

---

## 4.4 对分类与分割的适配

### 分类任务

对于分类，Routing Attack 更偏向于样本级或 token 级：

- 错误跨模态匹配；
- 错误语义 token 对齐；
- 错误全局注意力聚合。

### 分割任务

对于分割，Routing Attack 更适合局部空间级：

- 边界区域的错误跨模态对应；
- 小目标区域的错误信息流；
- patch-to-patch 的错配；
- 多尺度特征图间的错误融合路径。

---

## 5. 模块二：Fusion Weight Attack

## 5.1 核心思想

Fusion Weight Attack 的目标是攻击"哪个模态在融合时更重要"。

在很多多模态模型中，融合并不是简单平均，而是通过以下方式动态决定模态贡献：

- gate；
- uncertainty weighting；
- reliability score；
- confidence-aware fusion；
- modality contribution reweighting。

因此，如果保护器能让一个错误模态在训练时"看起来更可靠"，模型就会逐步偏向该模态，形成错误的优化动力学。

---

## 5.2 一般化建模

设两个模态特征分别为 $z^{(1)}$ 和 $z^{(2)}$。  
一种基本的加权融合形式为

$$

z_f = \alpha_1 z^{(1)} + \alpha_2 z^{(2)},

$$

其中

$$

\alpha_1, \alpha_2 = g_{\theta}(z^{(1)}, z^{(2)}), \qquad
\alpha_1 + \alpha_2 = 1.

$$

当模型引入不确定性时，可进一步写为

$$

q^{(1)}(z \mid x^{(1)}) = \mathcal{N}(\mu_1, \Sigma_1), \qquad
q^{(2)}(z \mid x^{(2)}) = \mathcal{N}(\mu_2, \Sigma_2).

$$

门控权重依赖于不确定性估计：

$$

\alpha_1, \alpha_2 = g_{\theta}(\Sigma_1, \Sigma_2).

$$

于是融合表示可写为

$$

z_f = \alpha_1 \mu_1 + \alpha_2 \mu_2.

$$

---

## 5.3 保护目标

Fusion Weight Attack 希望诱导：

$$

\tilde{\alpha}_{\text{wrong}} \uparrow, \qquad
\tilde{\alpha}_{\text{correct}} \downarrow.

$$

也就是说：

- 错误模态的权重被抬高；
- 正确模态的权重被压低。

可以定义权重错配损失：

$$

\mathcal{L}_{\text{weight}}
=
\mathbb{E}
\left[
\|\tilde{\alpha} - \alpha^{\star}\|_2^2
\right],

$$

其中 $\alpha^{\star}$ 表示预设的错误模态权重分布。

如果希望直接从不确定性角度控制，也可以定义：

$$

\mathcal{L}_{\text{uncert}}
=
\mathbb{E}
\left[
D\left(
g(\tilde{\Sigma}_1, \tilde{\Sigma}_2),
g(\Sigma_1, \Sigma_2)
\right)
\right].

$$

---

## 5.4 训练动力学解释

设训练损失为 $\mathcal{L}$，则模态 1 编码器的梯度更新近似满足

$$

\nabla_{\theta_1}\mathcal{L}
\propto
\alpha_1 \cdot \nabla_{\theta_1}\ell_1.

$$

如果保护器系统性提高错误模态的权重，则该模态在训练中会持续获得更多梯度，从而产生如下自增强过程：

1. 错误模态获得更大梯度；
2. 其更快拟合受保护样本；
3. 融合层进一步信任该模态；
4. 最终模型形成稳定的错误模态依赖。

因此，Fusion Weight Attack 攻击的不是单次输出，而是**训练中的模态主导关系**。

---

## 6. 模块三：Fusion Interface Attack

## 6.1 核心思想

Fusion Interface Attack 的目标是攻击模态之间实际发生交互的共享接口。

这里的"interface"包括但不限于：

- bottleneck；
- adapter；
- prompt；
- shared latent；
- shared token bank；
- fusion memory unit。

这些结构的共同特点是：

> 它们不是单个模态的内部层，而是模态之间共享信息、交换语义、形成联合表征的中间接口。

因此，它们是非常自然的攻击面。

---

## 6.2 为什么要强调 interface

本文的主线是 fusion attack，而不是 bottleneck attack。  
但是在具体实现时，如果没有一个明确的"共享交互位置"，就很难：

- 画清楚方法图；
- 写清楚信息流；
- 做理论分析；
- 解释为什么攻击会迁移。

所以，Fusion Interface Attack 的价值在于：

1. 它为 fusion attack 提供了具体落点；
2. 它让模型内部"共享语义被污染"这个说法可被显式建模；
3. 它便于与现有 prompt / adapter / bottleneck 文献对接。

---

## 6.3 以 bottleneck 为例的形式化建模

设两个模态特征为

$$

z^{(1)} \in \mathbb{R}^{n_1 \times d}, \qquad
z^{(2)} \in \mathbb{R}^{n_2 \times d}.

$$

引入 $K$ 个共享 bottleneck token：

$$

B = \{b_k\}_{k=1}^{K}, \qquad b_k \in \mathbb{R}^{d}.

$$

标准 bottleneck 融合可写为

$$

B'
=
\operatorname{CrossAttn}(B, z^{(1)})
+
\operatorname{CrossAttn}(B, z^{(2)}).

$$

在受保护条件下，加入 interface 控制变量：

$$

B'
=
\operatorname{CrossAttn}(B, z^{(1)}; \Pi^{(1)})
+
\operatorname{CrossAttn}(B, z^{(2)}; \Pi^{(2)}),

$$

其中：

- $\Pi^{(1)}$、$\Pi^{(2)}$ 表示保护器引入的共享接口偏置；
- 可对应路由优先级、token mask 或共享通道选择。

---

## 6.4 更一般的 interface 表达

为了不把方法限制死，可以把共享接口统一记为 $S_{\text{int}}$：

$$

s_{\text{int}}
=
\mathcal{I}_{\theta}(z^{(1)}, z^{(2)}; I_{\phi}),

$$

其中：

- $\mathcal{I}_{\theta}$ 表示任意共享融合接口；
- $I_{\phi}$ 表示由保护器控制的接口污染变量。

然后下游表示可写为

$$

z_f = \mathcal{F}_{\theta}(z^{(1)}, z^{(2)}, s_{\text{int}}).

$$

这一定义可以统一覆盖：

- bottleneck token；
- adapter feature；
- prompt embedding；
- shared latent memory。

---

## 6.5 设计目标

Fusion Interface Attack 的目标不是简单让共享表示"更噪"，而是让其保存错误共享语义。

设：

- $T$ 表示真实任务语义；
- $S$ 表示伪相关捷径因子；
- $H$ 表示共享融合接口中的表示。

我们希望诱导：

$$

I(H;S) \uparrow, \qquad I(H;T) \downarrow.

$$

定义共享接口捷径主导度：

$$

\Delta_{\text{int}} = I(H;S) - I(H;T).

$$

当 $\Delta_{\text{int}}$ 较大时，说明模型学到的共享融合表示更偏向伪相关模式，而不是真实跨模态语义。

---

## 7. 三类 Fusion Attack 的统一框架

## 7.1 统一表示

三类攻击可以统一到如下融合表达中：

$$

z_f
=
F_{\theta}\left(
z^{(1)}, z^{(2)};
R_{\phi}, W_{\phi}, I_{\phi}
\right),

$$

其中：

- $R_{\phi}$ 控制信息路由；
- $W_{\phi}$ 控制模态权重；
- $I_{\phi}$ 控制共享接口。

最终预测为

$$

\hat{y} = h_{\theta}(z_f).

$$

---

## 7.2 三类攻击的关系

三类攻击分别回答三个不同问题：

### Fusion Routing Attack
回答：

> 信息应该从哪里流向哪里？

### Fusion Weight Attack
回答：

> 哪个模态更应该被相信？

### Fusion Interface Attack
回答：

> 共享语义到底存放在哪里，又该如何被污染？

三者联合起来后，攻击的不是单一 loss，而是整个融合机制的三个关键层面：

- 路径；
- 权重；
- 接口。

因此，这种设计比"再加一个互信息损失"更像真正的架构创新。

---

## 8. 交互结构视角：为什么不能只谈互信息

在多模态任务中，预测信息可以分解为：

$$

I(Y;X^{(1)},X^{(2)}) = R + U_1 + U_2 + S,

$$

其中：

- $R$ 表示冗余信息；
- $U_1$ 表示模态 1 的独有信息；
- $U_2$ 表示模态 2 的独有信息；
- $S$ 表示协同信息。

如果简单最小化模态间互信息，可能会把：

- 有害的伪共享信息；
- 有用的任务协同信息；

一起削弱掉，这并不精细。

因此，更合理的目标是：

- 压制任务相关协同信息；
- 抬高伪冗余信息；
- 放大导致模态偷懒的伪独有信息。

定义交互扭曲目标：

$$

\mathcal{L}_{\text{int}}
=
- \lambda_R \hat{R}_{\text{spurious}}
+ \lambda_S \hat{S}_{\text{task}}
- \lambda_U \hat{U}_{\text{shortcut}}.

$$

从最大化角度写为

$$

\max_{\phi}
\;
\lambda_R \hat{R}_{\text{spurious}}
-
\lambda_S \hat{S}_{\text{task}}
+
\lambda_U \hat{U}_{\text{shortcut}}.

$$

这样，多模态 UE 的重点就从"整体一致性破坏"变成了"有选择地污染交互结构"。

---

## 9. 理论分析

## 9.1 基于共享接口的信息瓶颈解释

设共享融合接口表示为 $H$。  
标准信息瓶颈希望 $H$ 尽量保留与任务 $Y$ 有关的信息，并压缩无关细节。

对于受保护数据，我们希望让 $H$ 更偏向捷径信息而非真实语义：

$$

I(H;S) \uparrow, \qquad I(H;T) \downarrow.

$$

定义

$$

\Delta_{\text{IB}} = I(H;S) - I(H;T).

$$

当 $\Delta_{\text{IB}}$ 较大时，说明共享接口内部的表示更依赖捷径因素。

于是可以给出一种启发式风险解释：

$$

\mathcal{R}_{\text{test}}(f)
\le
\mathcal{R}_{\text{train}}(f)
+
C \cdot \Delta_{\text{IB}},

$$

其中 $C > 0$ 为任务相关常数。

这说明：

- 模型在训练集上可能仍然可以拟合；
- 但其共享表示已经被伪相关模式主导；
- 因而测试泛化会显著恶化。

---

## 9.2 基于交互分解的解释

设

$$

I(Y;X^{(1)},X^{(2)}) = R + U_1 + U_2 + S.

$$

对于真正依赖多模态融合的任务，协同信息 $S$ 往往非常重要。

若受保护数据使模型估计到：

$$

\hat{R}_{\text{spurious}} \uparrow, \qquad
\hat{S}_{\text{task}} \downarrow,

$$

则模型会逐渐退化成"伪共享驱动"的学习方式。

定义交互扭曲度：

$$

\Delta_{\text{PID}} = \hat{R}_{\text{spurious}} - \hat{S}_{\text{task}}.

$$

$\Delta_{\text{PID}}$ 越大，说明模型越依赖错误的共享结构，而越少使用真正有用的协同信息。

---

## 9.3 基于训练动力学的解释

对于加权融合：

$$

z_f = \alpha_1 z^{(1)} + \alpha_2 z^{(2)},

$$

模态 1 编码器的梯度更新近似满足：

$$

\nabla_{\theta_1}\mathcal{L}
\propto
\alpha_1 \cdot \nabla_{\theta_1}\ell_1.

$$

如果受保护数据持续诱导错误模态具有更高的权重，则：

1. 错误模态更快拟合训练数据；
2. 融合层更依赖该模态；
3. 梯度进一步偏向该模态；
4. 最终形成稳定的捷径依赖。

因此，Fusion Weight Attack 影响的是整个训练路径，而不只是某个时刻的预测结果。

---

## 10. 双层优化

### 10.1 内层问题

受害模型在受保护数据上训练：

$$

\theta^{\star}(\phi)
=
\arg\min_{\theta}
\mathcal{L}_{\text{train}}(\theta,\phi),

$$

其中

$$

\mathcal{L}_{\text{train}}(\theta,\phi)
=
\mathbb{E}_{(x^{(1)},x^{(2)},y)}
\left[
\mathcal{L}_{\text{task}}
\left(
f_{\theta}\left(
x^{(1)}+\delta_{\phi}^{(1)},
x^{(2)}+\delta_{\phi}^{(2)}
\right),
y
\right)
\right].

$$

---

### 10.2 外层问题

保护器更新目标为：

$$

\max_{\phi}
\;
\mathcal{L}_{\text{outer}}(\theta^{\star}(\phi), \phi).

$$

可定义

$$

\mathcal{L}_{\text{outer}}
=
\lambda_{\text{task}} \mathcal{L}_{\text{val}}
+
\lambda_{\text{route}} \mathcal{L}_{\text{route}}
+
\lambda_{\text{weight}} \mathcal{L}_{\text{weight}}
+
\lambda_{\text{intf}} \Delta_{\text{int}}
+
\lambda_{\text{pid}} \Delta_{\text{PID}}
-
\lambda_{\text{per}} \mathcal{L}_{\text{per}}.

$$

其中：

- $\mathcal{L}_{\text{val}}$ 表示干净验证集上的性能退化；
- $\mathcal{L}_{\text{route}}$ 表示路由攻击目标；
- $\mathcal{L}_{\text{weight}}$ 表示模态权重攻击目标；
- $\Delta_{\text{int}}$ 表示共享接口捷径主导度；
- $\Delta_{\text{PID}}$ 表示交互结构扭曲度；
- $\mathcal{L}_{\text{per}}$ 表示感知约束。

---

## 11. 面向分类与语义分割的任务适配

## 11.1 多模态分类

分类任务中，更适合重点攻击：

- 全局 token 路由；
- 样本级模态权重；
- 全局共享语义接口。

输出形式为

$$

\hat{y} = \operatorname{Softmax}(W z_f).

$$

分类损失为

$$

\mathcal{L}_{\text{cls}}
=
-
\sum_{c=1}^{C}
y_c \log \hat{y}_c.

$$

---

## 11.2 多模态语义分割

语义分割更强调局部空间融合，可写为

$$

z_f(h,w)
=
F_{\theta}\left(
z^{(1)}(h,w), z^{(2)}(h,w)
\right).

$$

预测输出为

$$

\hat{Y}(h,w) = \operatorname{Softmax}(W z_f(h,w)).

$$

分割损失可写为

$$

\mathcal{L}_{\text{seg}}
=
\mathcal{L}_{\text{CE}}
+
\lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}.

$$

对于分割任务，三类 fusion attack 分别可落到：

### Routing Attack
- 边界区域错误跨模态对应；
- 小目标 patch 错误对齐；
- 多尺度特征流错误路由。

### Weight Attack
- 像素级门控错配；
- 边界区域权重偏置；
- 局部不确定性误导。

### Interface Attack
- 局部共享 latent 污染；
- patch-level adapter 污染；
- 区域级 bottleneck / prompt 污染。

---

## 12. 架构创新点总结

本文的架构创新不应表述为"多加几个 loss"，而应表述为：

### 创新点 1：面向融合路径的结构化攻击
通过对 cross-attention、token routing、共享信息流进行显式建模，把 UE 从输入级扰动扩展到跨模态信息路径级干预。

### 创新点 2：面向融合权重的动态攻击
通过攻击门控、置信度和不确定性估计，改变模态贡献比例和训练中的梯度分配。

### 创新点 3：面向共享融合接口的表示污染
通过攻击 bottleneck、adapter、prompt、shared latent 等共享接口，诱导模型学习错误共享语义。

### 创新点 4：三类 Fusion Attack 的统一框架
把路径、权重、接口统一到一个融合攻击框架中，形成真正的架构型多模态 UE，而不是简单损失堆叠。

---

## 13. 实验设计建议

## 13.1 任务

建议至少覆盖以下任务之一：

- 多模态分类；
- 多模态语义分割。

---

## 13.2 数据模态

可优先考虑以下组合：

- RGB + Depth；
- Optical + SAR；
- MRI T1 + T2 / FLAIR；
- 图像 + 文本分类变体。

---

## 13.3 融合架构

应覆盖不同 fusion 范式，以证明方法不仅对单一结构有效：

1. early fusion；
2. middle fusion；
3. cross-attention fusion；
4. bottleneck fusion；
5. prompt / adapter fusion；
6. uncertainty-aware fusion。

---

## 13.4 评估指标

### 分类任务
- Accuracy；
- Macro-F1；
- clean / protected train-test gap。

### 分割任务
- Dice；
- mIoU；
- HD95（若任务适用）。

### 保护效果
- 未授权模型在干净测试集上的性能下降；
- 不同 fusion 架构上的迁移性；
- 扰动可感知性评价。

### 机制验证
- 注意力路由偏移；
- 模态权重变化；
- 共享接口表示偏移；
- 协同信息代理量变化；
- 冗余信息代理量变化；
- 不确定性校准误差变化。

---

## 13.5 消融实验

至少包括：

1. 仅使用 Routing Attack；
2. 仅使用 Weight Attack；
3. 仅使用 Interface Attack；
4. Routing + Weight；
5. Routing + Interface；
6. Weight + Interface；
7. 三者全部使用；
8. 仅使用简单互信息损失；
9. 仅使用单模态 UE。

这些消融非常关键，因为它们能证明：

> 方法的优势来自 fusion mechanism attack，而不只是因为目标函数更多。

---

## 14. 论文写法建议

## 14.1 推荐题目风格

可以考虑以下方向：

- 面向融合机制攻击的多模态 Unlearnable Examples
- 基于 Fusion Attack 的多模态 Unlearnable Examples
- 面向分类与语义分割的多模态融合污染方法
- 通过信息路由、模态权重与共享接口污染实现多模态保护

---

## 14.2 贡献点建议写法

本文的主要贡献可以概括为：

1. 提出一个面向多模态分类与语义分割的统一 Fusion Attack 型 UE 框架；
2. 从信息路由、模态权重和共享接口三个层面系统建模多模态融合攻击；
3. 提出可统一覆盖 bottleneck、adapter、prompt 和 shared latent 的共享接口污染机制；
4. 从信息瓶颈、交互分解和训练动力学三个角度解释为何受保护数据会诱导模型学习错误融合规则；
5. 在多种多模态融合架构上验证方法的有效性与迁移性。

---

## 14.3 核心 novelty 句式

可以直接写：

> 本文并非仅通过扰动多模态输入来降低模型性能，而是通过系统性操控融合路径、模态权重和共享接口，使模型在训练过程中学习到错误的跨模态融合规则，从而在分类与语义分割任务中失去泛化能力。

---

## 15. 最小可行实现路线

如果当前资源有限，建议按以下顺序推进：

### 第一步
先确定一个明确任务：
- 多模态分类；或
- 多模态语义分割。

### 第二步
选择一个具有显式 cross-attention 或共享接口的 fusion backbone。

### 第三步
先实现三个模块中的两个：
- Routing Attack；
- Weight Attack；
或
- Routing Attack；
- Interface Attack。

### 第四步
用基础双层优化训练保护器。

### 第五步
先验证以下三件事：
1. 干净测试集性能明显下降；
2. 注意力路由发生偏移；
3. 模态贡献比例发生改变。

### 第六步
再补充第三类攻击模块，并完善理论部分。

---

## 16. 最终总结

本文主张将多模态 UE 统一理解为一个 **fusion mechanism attack** 问题，而不是单纯的输入扰动问题或互信息最小化问题。

一个强的多模态 UE 框架应同时攻击以下三个层面：

1. **Fusion Routing Attack**  
   攻击跨模态信息路径，破坏正确的信息流；

2. **Fusion Weight Attack**  
   攻击模态权重与不确定性估计，诱导错误的模态主导关系；

3. **Fusion Interface Attack**  
   攻击 bottleneck、adapter、prompt、shared latent 等共享接口，污染共享语义表示。

三者结合后，模型在受保护数据上训练时，不再是简单地"难学"，而是会被系统性引导去学习错误的跨模态融合规则。这种思路更适合：

- 做架构创新；
- 做数学推导；
- 扩展到分类与语义分割；
- 构成 AAAI 风格的完整论文叙事。
