# 易腐库存的订货与动态定价

对易腐商品的日常补货与分年龄定价做联合优化。当前实验是基于 Nomura、Liu 和 Nishi（2025）模型的**合成 Case 1 扩展**。论文见 [出版方 PDF](https://mdpi-res.com/d_attachment/applsci/applsci-15-02421/article_deploy/applsci-15-02421.pdf)。

## 实验设计

- 环境为 3 天保质期、1 天到货提前期，单次最多订 10 件。每天到货后观察三个年龄档的库存，再决定订货量与各档报价。
- 潜在需求服从 Gamma 分布，均值 4、变异系数 0.5。MNL 效用为 `2 − bⱼ·pⱼ/10`，敏感度为 `(3, 5, 7)`。只对有库存的档位分配需求，保留不购买选项。实际售价为 10、7 或 5，收入按实际售价计算。
- 单位持有、订货、缺货和报废处置成本分别为 1、3、1、1。观察状态和销量可取连续值。
- 完整 DP 在 918,731 个网格状态上考虑 297 个动作。ar-DP 按论文第 4.2 节的文字规则约束新鲜档报价 ≥ 中间档 ≥ 临期档，允许同价，动作减至 110 个。最优固定价 DP 枚举 27 套固定报价，并为每套报价重新求解随库存变化的订货策略，以空库存的无限期折扣价值选优。三种 DP 的折扣因子均为 0.99。
- PPO 用 Stable-Baselines3 2.4.1，每个种子训练 73,000 步。五个训练种子为 142–146。所有策略用相同的 100 个需求种子 42–141，从空库存和 `[3, 2.3, 1.7]` 各做 365 天模拟。表中报告不折扣的年度现金利润。

## 结果

| 方法 | 空库存年均利润 | 有初始库存年均利润 | 记录耗时 |
| --- | ---: | ---: | --- |
| 完整动态价 DP | 2,286.86 | 2,299.85 | 154.33 秒，构建 124.58、求解 29.75 |
| ar-DP | 2,282.89 | 2,296.07 | 56.92 秒，构建 45.36、求解 11.57 |
| 最优固定价 DP（10 / 7 / 5） | 2,130.78 | 2,139.91 | — |
| 确定性 PPO | 2,003.61 | 2,013.18 | 平均 68.82 秒/种子 |
| 随机 PPO | 1,961.10 | 1,971.37 | 同上 |

- 空库存下，动态价 DP 比固定价 DP 多赚 156.08（7.33%），配对 95% 区间为 [149.29, 162.87]。
- ar-DP 节省 63.12% 的构建与求解时间。空库存下，完整 DP 比 ar-DP 多赚 3.97，配对 95% 区间为 [1.24, 6.71]。网格上的最大折扣价值差为 5.14。
- 确定性 PPO 的临期库存位于 (0, 0.5] 时，报 5 元比例为 77.6%；库存位于 (0.5, 1] 时，比例为 99.3%。PPO 的利润仍低于固定价 DP。
- 三种算法均用 CPU。DP 记构建与求解时间，PPO 记五个并行进程各自的训练时间。耗时均不含策略评估。

数据：[完整 DP、ar-DP 与计时](published_results/alpha2/ar_dp.json) · [27 种固定价排名](published_results/alpha2/fixed_price_ranking.json) · [固定价配对评估](published_results/alpha2/fixed_price.json) · [PPO 逐种子结果与报价分布](published_results/alpha2/ppo.json)。

## 复运行

建议用 Python 3.10，在仓库根目录执行。为对齐计时，可将 OMP、OpenBLAS 和 MKL 线程数设为 1。

```bash
python -m pip install -r requirements-ppo.txt
python -m pytest -q tests
python -m algorithms.dp
python -m experiments.ppo_study dp
for seed in 142 143 144 145 146; do
  python -m experiments.ppo_study train --seed "$seed"
  python -m experiments.ppo_study eval --seed "$seed"
done
python -m experiments.ppo_study summary
python -m experiments.fixed_price_study
python -m experiments.ar_dp_study
```

复跑生成的模型、checkpoint 和日志写到 `checkpoints/` 或 `results/`。用 [DP 计算说明](docs/dp_computation_scheme.md) 可查看 Gamma 积分、网格和收敛口径。

## 季节需求扩展

[季节配置](configs/seasonal.yaml) 可编辑各季天数和需求均值倍率。默认示例为 91、91、91、92 天，倍率 0.8、1.0、1.2、1.0。变异系数保持不变，观察中附加归一化周期日，策略可以看到旺淡季以及距离切换还有多久。

```bash
python -m experiments.ppo_study train --seed 142 --seasonal-schedule configs/seasonal.yaml
```

季节模型和训练记录写入 `results/seasonal/`。上面的结果表使用恒定潜在需求分布。当前 DP 入口求解该恒定需求模型。若要加入季节性 DP，需在价值表中增加日历状态。
