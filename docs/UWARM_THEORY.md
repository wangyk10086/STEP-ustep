# U-STEP 理论：warm start 的分布误差界与闭式自适应起点

本文档替换 STEP 附录 D 的"局部收缩性"论证，给出一个**正确、与分布无关**的误差界，并由此导出自适应起点 t\*(c) 的闭式解。所有量都在归一化动作空间中，A ∈ ℝ^d 为整个 action chunk（d = H·D_a）。

---

## 0. 为什么原收缩性证明需要替换

STEP 附录 D.10 要求

$$\frac{1}{\sqrt{\alpha_k}} + \Big|\sqrt{1-\bar\alpha_{k-1}} - \sqrt{\tfrac{1-\bar\alpha_k}{\alpha_k}}\Big| L < 1 .$$

由于 α_k ∈ (0,1)，第一项 1/√α_k > 1，**该不等式对任何 L ≥ 0 都不可能成立**。根本原因是 DDIM 每一步都把信号缩放 1/√α_k，原始坐标下反向映射本来就是"扩张"的，收缩性不是 warm start 起作用的机制。

此外，定理 D.13 中的"最优轨迹 A\*_k"没有定义；表 11 实测的 c_k 是两个误差范数之比，并不是 D.10 中的 Lipschitz 常数。

正确的机制是**分布层面的**：反向过程对任意 f-散度都是非扩张的（数据处理不等式），warm start 的作用是减小**初始分布**与真实边缘分布之间的散度。下面把这一点严格化。

---

## 1. 设定

- 条件 c（观测，及可选的历史动作）；目标条件分布 p(A | c)。
- VP 前向过程：x_t = √ᾱ_t A + √(1−ᾱ_t) ε，ε ~ N(0, I)。
- 信噪比 SNR(t) = ᾱ_t / (1−ᾱ_t)，关于 t 严格单调递减。
- 真实边缘分布 q_t(· | c) = ∫ N(√ᾱ_t A, (1−ᾱ_t) I) p(dA | c)。
- 预测器给出 μ(c)，其条件均方误差记为 **m(c) = E_{A∼p(·|c)} ‖A − μ(c)‖²**。
- warm start 分布（即正确的前向边缘，与代码中 `scheduler.add_noise` 一致）：
  w_t(· | c) = N(√ᾱ_t μ(c), (1−ᾱ_t) I)。
- P_θ^{t→0}：学到的反向采样器从 t 积分到 0 的马尔可夫核（DDPM），或确定性映射（DDIM）。

---

## 2. 主要结果

**引理 1（warm start 失配，与分布无关）。** 对任意 p(·|c) 和任意 t：

$$\mathrm{KL}(w_t \,\|\, q_t) \;\le\; \tfrac12\,\mathrm{SNR}(t)\, m(c), \qquad \mathrm{KL}(q_t \,\|\, w_t) \;\le\; \tfrac12\,\mathrm{SNR}(t)\, m(c).$$

*证明。* 记 Q_A = N(√ᾱ_t A, (1−ᾱ_t)I)，则 q_t = ∫ Q_A p(dA)。KL 关于两个参数联合凸，故

KL(w_t ‖ ∫Q_A p(dA)) ≤ ∫ KL(w_t ‖ Q_A) p(dA) = ∫ ᾱ_t ‖μ − A‖² / (2(1−ᾱ_t)) p(dA) = ½ SNR(t) m(c)。

另一个方向同理（对第一个参数用凸性）。∎

*紧性。*
- 对点质量目标 p = δ_{A₀}，等号成立：此时误差全部来自预测器的偏差。
- 对无偏的高斯目标 p = N(μ, Σ)，精确值为 ½ Σ_i [log(1+ρ_i) − ρ_i/(1+ρ_i)]，其中 ρ_i = SNR·λ_i。小 ρ 时约为 ρ²/4，比线性界小一阶。

因此线性界在"有偏的预测"时紧，在"纯随机散布"时偏保守。`experiments/toy_bimodal.py` 对双峰目标做了 Monte Carlo 验证：所有 t 上 KL_mc ≤ bound，比值在 0.1–0.7 之间。

**定理 1（端到端误差）。** 令 p̂_t := q_t P_θ^{t→0}，即从**精确**边缘出发、用学到的采样器得到的分布，并记 ε_samp(t, N) := TV(p̂_t, p(·|c))。则 warm-started 输出 w_t P_θ^{t→0} 满足

$$\mathrm{TV}\big(w_t P_\theta^{t\to 0},\; p(\cdot|c)\big) \;\le\; \tfrac12\sqrt{\mathrm{SNR}(t)\, m(c)} \;+\; \varepsilon_{\text{samp}}(t, N).$$

*证明。* 由三角不等式，TV(w_t P, p) ≤ TV(w_t P, q_t P) + TV(q_t P, p)。

- 第一项：TV 在马尔可夫核（含确定性映射）下不增，所以 TV(w_t P, q_t P) ≤ TV(w_t, q_t)；再用 Pinsker 不等式和引理 1，得 TV(w_t, q_t) ≤ √(½ KL) ≤ ½√(SNR·m)。
- 第二项按定义等于 ε_samp(t, N)。∎

关于 ε_samp：它是"从 t 时刻精确边缘出发"的标准采样误差，包含 score 估计误差在 [0, t] 上的积分，以及步长为 t/N 的离散化误差（参见 Chen et al., ICLR 2023；Benton et al., ICLR 2024）。它满足两点：
- 固定 N 时，ε_samp 随 t 不减：要走的路更长。
- 固定 t 时，ε_samp 随 N 递减。

**两项的权衡就是 warm start 的全部本质。**
- t 小：warm start 失配项大，但采样项小、步数少。
- t → T：SNR → 0，失配项消失，退化为原始扩散策略，多模态完全保留，但代价最大。

**推论 1（闭式自适应起点）。** 给定 warm start 项的 KL 预算 δ，满足引理 1 界的**最晚（噪声最少、最便宜）**起点为

$$\boxed{\;\mathrm{SNR}(t^*(c)) = \frac{2\delta}{m(c)}\;}\qquad\Longleftrightarrow\qquad t^*(c) = \mathrm{SNR}^{-1}\!\Big(\frac{2\delta}{m(c)}\Big).$$

对 squaredcos 调度，SNR⁻¹ 也有闭式：令 ᾱ\* = s\*/(1+s\*)（s\* 为目标 SNR），则

t\* = T[(1+s)(2/π) arccos(√(ᾱ\* f(0))) − s]，其中 f 为 cosine 调度函数，s 为其偏移量。

代码中直接在 `alphas_cumprod` 上查表（`select_start_timesteps`），适用于任意调度。

步数取 N(c) = ⌈(t\*+1)/Δ⌉，即保持步长 Δ 不变，使离散化误差在不同 c 之间大致相同。于是：

- **易状态**（m 小）：t\* ≈ 0，**1 次 NFE**，结果接近直接预测。
- **歧义/多模态状态**（m 大）：t\* → T，**退化为原始扩散策略**，模态被重新打开。

固定起点的 STEP 是 t\* ≡ const 的特例。它在易状态上浪费噪声（加完噪声再去掉，反而引入误差），在难状态上噪声又不够（样本落到两个模态之间）。

---

## 3. m(c) 的估计：异方差头是 proper scoring rule

用逐元素高斯 NLL 训练方差头，并对残差做 **stop-gradient**：

$$\mathcal L_{\text{var}} = \tfrac12\,\mathbb E\Big[\frac{\mathrm{sg}(A-\mu)^2}{\sigma^2(c)} + \log\sigma^2(c)\Big].$$

它对 σ² 的逐点最优解是 σ²(c) = E[(A−μ)² | c]（逐元素），因此 tr Σ̂(c) 恰好是 m(c) 的插件估计。

由于对 μ 做了 stop-gradient，均值预测器的训练与 STEP 完全相同（纯 MSE）。这意味着可以在**已有的 STEP 检查点上冻结均值、只训练方差头**（`policy.freeze_mean=True`），无需重训。

剩余的整体尺度误差用验证集上的标量 κ 校准：κ = E‖e‖² / E tr Σ̂（`calibrate_uwarm.py`）。

**多模态的关键点。** 按全方差公式，MSE 最优的 μ 在两个模态之间时，其误差 m(c) 包含**模态间方差**。所以预测器"不知道走哪边"的状态会被自动识别为高 m 状态，从而分到更大的 t\*。这正是 STEP 在 Push-T 上表现最弱的状态类型。

**关于历史条件。** 预测器以 (o, h) 为条件，扩散策略建模的是 p(A|o)。我们用 σ̂²(o, h) 作为 m 的插件估计。当历史能消除歧义（A ⊥̸ h | o）时，它小于 E_{p(·|o)}‖A−μ‖²，这是**有意的**：此时希望模型沿历史已经选定的模态继续执行，而不是重新随机选择。论文中应把这一点作为假设或讨论写明。

---

## 4. 附：STEP 停滞的一个具体机制（copycat）

STEP 训练工作区使用的 `ActionPredictorDatasetWrapper` 设置 `prev_action = action[:8]`，取自**同一窗口**，而该窗口同时也是回归目标。预测器因此可以靠复制输入来降低前 8 步的 MSE。

推理时，`prev_action` 是**上一次调用**的 chunk，时间上整体错后 n_action_steps 步。模型于是把"过去的动作"复制到新 chunk 的开头，机器人会重复已经执行过的动作，表现为原地徘徊或停滞。这与 STEP 需要 velocity-aware perturbation 来打破停滞的现象一致。

`AlignedPrevActionWrapper` 修正了这一对齐问题：用上一次调用实际对应的真值 chunk 作为 prev_action，并在 episode 开头置零（与推理时一致）。结合历史噪声和 dropout 增强，可以作为第二个贡献，用"去掉 VPI 后不再停滞"来验证。

---

## 5. 论文中可直接使用的表述

- **Lemma 1**：KL(w_t ‖ q_t) ≤ ½ SNR(t) m(c)，与分布无关，并给出何时紧。
- **Theorem 1**：TV 端到端界，把 warm start 失配与采样误差分开。
- **Corollary 1**：t\*(c) = SNR⁻¹(2δ / m(c))，并说明 STEP 是其常数特例。
- **Proposition**：NLL 方差头的最优解给出 m(c)，且模态间方差会被计入 m(c)。
- **Remark**：说明原收缩性条件为什么不可能成立，作为 related-work 中对 STEP 的修正。
