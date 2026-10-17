# U-STEP：不确定性校准的自适应 warm start

在 STEP 代码库上的增量实现。**所有新功能默认关闭**，不加新参数时行为与 STEP 完全一致，原有结果可复现。

理论推导见 [`docs/UWARM_THEORY.md`](docs/UWARM_THEORY.md)。

## 核心思想（一句话）

预测器在给出 warm start 均值 μ 的同时，预测自身误差 m(c) = E‖A−μ‖²。每个样本按闭式规则

**SNR(t\*) = 2δ / m̂(c)**

选择起点，从精确前向边缘 √ᾱ μ + √(1−ᾱ) ε 出发，步数随 t\* 增长。效果是：易状态只用 1 次 NFE，歧义/多模态状态自动退化为原始扩散策略。

## 改动清单

| 文件 | 内容 |
|---|---|
| `diffusion_policy/policy/adaptive_warmstart.py` | **核心**：闭式 t\* 规则、逐样本 DDIM 调度、批量自适应采样器 |
| `diffusion_policy/model/action_predictor/action_predictor_transformer.py` | `predict_var` 方差头（MLP，可后挂） |
| `diffusion_policy/model/action_predictor/uncertainty.py` | stop-grad NLL 损失、历史增强、从 STEP 检查点加载、冻结均值的优化器 |
| `diffusion_policy/dataset/aligned_prev_action_wrapper.py` | 修正 prev_action 泄漏/错位（copycat 停滞） |
| `enhanced_diffusion_unet_{lowdim,image}_policy.py` | `set_adaptive_warmstart()`，`predict_action(..., init_var=)` |
| `combined_inference_policy.py` | 透传方差，记录 t\*/NFE 统计 |
| `eval_combined_inference.py` | `--ustep_deltas` 扫描；图像 wrapper 同步支持 |
| `calibrate_uwarm.py` | 拟合 κ，输出可靠性图和隐含 NFE 分布 |
| `eval_distribution_fidelity.py` | 离线多模态指标：MMD / precision / recall / 多样性 / minADE |
| `experiments/toy_bimodal.py` | 双峰玩具实验 + 引理 1 数值验证（论文 Fig.1 候选） |
| `tests/test_ustep.py` | 7 个单元/集成测试 |

## 运行步骤

以下以 Push-T lowdim 为例；图像任务把 `lowdim` 换成 `image`，并使用对应的配置文件。

### 1. 在已有 STEP 预测器上后挂方差头（不重训均值，最快）

```bash
python train.py --config-name=train_action_predictor_lowdim_workspace \
  task=pusht_lowdim \
  init_predictor_ckpt=/path/to/STEP_predictor.ckpt \
  policy.model.predict_var=True policy.freeze_mean=True \
  task.dataset.val_ratio=0.02 \
  training.num_epochs=300 exp_name=ustep_varhead
```

> 方差头只有约 2 万参数，几百个 epoch 足够。日志中重点看两项：`train_calib_ratio` 应接近 1，`train_err_var_corr` 越高越好。

### 2. 校准 κ

```bash
python calibrate_uwarm.py --action_predictor_ckpt PRED_VAR.ckpt --modality lowdim \
  --out data/ustep/calib_pusht.json
```

输出包括 κ、误差与预测方差的 Spearman 相关、可靠性图（`*_reliability.png`），以及不同 δ 下隐含的平均 NFE。

### 3. 闭环成功率 vs NFE 的 Pareto 曲线（主实验）

```bash
python eval_combined_inference.py \
  --action_predictor_ckpt PRED_VAR.ckpt --diffusion_policy_ckpt DP.ckpt \
  --modality lowdim --steps_list 2 4 --use_ddim \
  --ustep_deltas 0.02 0.05 0.1 0.3 1 3 --ustep_kappa <κ> \
  --eval_seeds 100000 100100 100200 100300 100400 \
  --output_dir data/outputs/ustep_pusht
```

结果 JSON 中，每个 `ustep_delta*` 条目包含成功率，以及 `ustep_avg_nfe`、`ustep_p90_nfe`、`ustep_frac_1nfe`。把它们与 `steps_k/combined`（即 STEP）画在同一张"成功率 vs 平均 NFE"图上。

### 4. 多模态证据（STEP 原文缺失的部分）

```bash
python eval_distribution_fidelity.py \
  --action_predictor_ckpt PRED_VAR.ckpt --diffusion_policy_ckpt DP.ckpt \
  --modality lowdim --history none --deltas 0.03 0.1 0.3 \
  --step_steps 2 4 --ddim_steps 2 4 --kappa <κ> --out data/ustep/fid_pusht.json
```

判读要点：**precision ≈ 1 而 recall 明显偏低**，就是模式坍缩的标志。预期 STEP 在 Push-T 上出现这一现象，而 U-STEP 的 recall 接近参考。

### 5.（第二贡献）去掉 VPI 也不停滞

```bash
python train.py --config-name=train_action_predictor_lowdim_workspace task=pusht_lowdim \
  aligned_prev_action=True prev_action_horizon=16 \
  policy.model.predict_var=True \
  policy.prev_action_noise_std=0.05 policy.prev_action_dropout=0.1 \
  exp_name=ustep_aligned
```

在真机上比较以下四组的停滞率和 episode 时长：{原 wrapper, aligned} × {有 VPI, 无 VPI}。

### 测试与玩具实验

```bash
python -m pytest tests/test_ustep.py -q          # 7 passed
python experiments/toy_bimodal.py --out data/ustep/toy
```

## 玩具实验结果（CPU，已跑通）

50% 状态单峰、50% 状态双峰：

| 方法 | 易状态误差 / NFE | 双峰：落在模态上 / 两侧平衡 / NFE | 平均 NFE |
|---|---|---|---|
| STEP K=2 | 0.157 / 2 | 0.482 / 0.804 / 2 | 2.0 |
| STEP K=4 | 0.162 / 4 | 0.817 / 0.880 / 4 | 4.0 |
| **U-STEP δ=1** | **0.075 / 1** | 0.826 / 0.906 / 6.4 | **3.7** |
| U-STEP δ=0.1 | 0.122 / 3 | 0.906 / 0.918 / 9 | 6.0 |
| DDPM-100（参考） | — | 0.967 / 0.930 / 100 | 100 |

在更少的平均 NFE 下，U-STEP 同时提升了易状态精度和双峰覆盖，计算量被自动分配到歧义状态上。引理 1 在所有 t 上都通过了 Monte Carlo 验证。可视化见 `docs/toy_bimodal.png`。

注意：玩具中易状态的误差随 NFE 增加而上升，是因为玩具扩散模型训练不足（DDPM-100 在易状态上误差 0.21），不代表真实任务中的趋势。

## 超参数建议

- **δ（总 KL 预算，单位 nats）**：主旋钮，扫描 {0.02 … 3} 即可得到 Pareto 曲线。δ 越小，起点越噪、NFE 越多。
- **n_max**：最大 NFE，默认 10；stride 默认为 T // n_max。
- **κ**：用 `calibrate_uwarm.py` 拟合，不要手调。
- **exec_slice**：可设为执行段，例如 `(1, 9)`，只依据实际会被执行的动作的不确定性来选起点。
