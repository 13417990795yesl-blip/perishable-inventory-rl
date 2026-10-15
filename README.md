# 易腐库存定价与订货论文复现

目标：复现 Nomura、Liu 与 Nishi（2025）的 baseline，分阶段开发 Environment、DP、ar-DP、PPO，再运行 Cases 1、2、4。

论文：https://doi.org/10.3390/app15052421

原文：https://ousar.lib.okayama-u.ac.jp/files/public/6/68433/20250319104940670083/fulltext.pdf

服务器项目：`/data1/yesiliang/perishable_inventory_rl`

## 快速开始

使用Python 3.10，在自己的计算机或服务器上克隆并安装依赖：

```bash
git clone https://github.com/13417990795yesl-blip/perishable-inventory-rl.git
cd perishable-inventory-rl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m experiments.validate environment
```

Windows PowerShell把激活命令替换为 `.\.venv\Scripts\Activate.ps1`。以上操作不需要访问作者的服务器路径。

单次DP求解运行 `python -m algorithms.dp`；完整网格与策略审计运行 `python -m experiments.validate dp`。运行完整审计前建议把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为1，以保持与已记录计时一致。

Git仓库包含代码、配置、测试和计算说明。实验原始数据、日志、已训练策略及恢复归档保存在原服务器，未随代码上传；在新机器运行时会重新生成所需输出。当前结果和建模约定见下文及 `docs/relative_price_revision.md`。代码版本标签为 `relative-price-dp-v1`。

## 当前模型版本

2026-10-14 按用户明确要求切换为相对价格需求：效用使用实际售价/原价，收入继续使用实际售价。配置显式记录`demand_price_basis: relative`。修改前的实际价格版本已完整归档，新结果输出到各目录下的`relative_price/`。本次修订、兼容性与重算说明见`docs/relative_price_revision.md`。下文历史Phase4数据属于此前实际价格版本，不应与新结果混用。

相对价格DP已重新求解并验收：68项测试通过，600次年度策略模拟通过网格稳定性检查。空库存100次365步模拟平均利润为**2916.30**，标准误9.47；无限期折扣价值为766.31。checkpoint为`checkpoints/relative_price/dp_grid_0p0078125.npz`，报告为`results/logs/relative_price/phase4_validation.json`，当前模型阶段归档为`checkpoints/relative_price/model_revision_project.tar.gz`。尚未完全对齐论文Case1的利润数值。

## 阶段状态

- 环境检查：完成，版本与硬件记录在 `results/logs/server_environment.json`。
- Phase 1 项目目录：完成；依赖检查、目录检查、配置读取及导入检查已通过。归档见 `checkpoints/phase1_project.tar.gz`。
- Phase 2 Inventory Environment：完成；reset/step、一期交货与奖励分解的基本检查通过，已保存 `checkpoints/phase2_environment.json`。
- Phase 3 Environment 测试：完成；27 项测试通过，15 个完整 episode、5,475 步审计通过。详见 `results/logs/phase3_validation.json`，阶段归档为 `checkpoints/phase3_project.tar.gz`。
- Phase 4 DP：实际价格版本完成历史数值验收；相对价格版本已重算并通过68项测试及细化检查。当前结果以`docs/relative_price_revision.md`与`results/logs/relative_price/phase4_validation.json`为准。
- Phase 5–8：待实施。

`env/` 和 `algorithms/dp.py` 已实现；ar-DP、PPO、正式 Cases 实验及绘图仍待实施。空占位脚本已清理，后续功能在实现时再加入。PPO 训练尚未启动。

## 运行环境

服务器系统 Python 为 3.8.2；conda 23.3.1 位于 `/opt/anaconda3`。已有环境涵盖 Python 3.8.20、3.9.25、3.10.9、3.11.16，但均未安装 Gymnasium 或 Stable-Baselines3。

本项目使用 `/opt/anaconda3/bin/python`（3.10.9）创建独立 `.venv`，不继承其他环境的包。论文使用 Python 3.8.10；选择 3.10.9 是为支持 Gymnasium，并记录此运行环境差异。当前仅安装 NumPy、SciPy、Gymnasium、PyYAML、pytest 及其依赖；完整版本冻结在 `requirements.txt`。PPO 阶段再选择兼容的 Stable-Baselines3/PyTorch。

```bash
cd /data1/yesiliang/perishable_inventory_rl
source .venv/bin/activate
python -m pip check
```

## 建模核对

已核对论文第 3 节和 Tables 3–5，详细未决点见 `docs/modeling_decisions.md`。保留易腐性、库存年龄、正交货时间、动态价格和随机需求。

`configs/base_config.yaml` 保存 baseline 参数；`configs/ppo_config.yaml` 保存后续 PPO 参数，不代表 PPO 已实现。正式实验优先级为 Cases 1、2、4。

默认随机种子为 42，属于复现自选值。每阶段完成后保存检查记录、阶段归档并更新本文件。

## Environment 约定

用户已选择通常的一期交货：`q_t` 在 `t+lead_time` 首次可售。状态在本期到货之后观察，按 `[age_1,...,age_M,transit_due_in_1,...,transit_due_in_L-1]` 排列。因此 baseline 的 `M=3,L=1` 状态维度为 3；`L=2` 时为 4，并显式保存尚未到货的订单。此处按用户选择调整协议示例中的四维状态，不添加始终为零的冗余槽位。

一个 step：决定订货和折扣→Gamma 总需求及 MNL 分配→各年龄独立销售→奖励分解→最老库存报废→库存老化→接收下一期到货→下一期到货后的状态。

当前MNL效用使用相对价格1/.7/.5，销售收入使用实际售价10/7/5；两套价格在Environment与DP中明确分开。旧checkpoint缺少价格解释字段时按历史absolute模式恢复。

动作 `[q,discount_index_1,discount_index_2,discount_index_3]` 为 MultiDiscrete，baseline 共 297 种动作。折扣索引 0/1/2 对应价格 10/7/5。

Gamma 总需求为连续值；年龄需求为 `d*lambda`，不取整。MNL 分母包含不购买选项，缺货时不重新分配需求。持有成本仅收取销售后、报废后留下的在架库存；365 步触发 truncation。

Environment 提供含配置、库存、时间与两个随机数生成器状态的 JSON checkpoint，支持精确续跑。阶段 tar 归档包含源文件、配置和检查日志，不包含 `.venv`。

## 验证结果和复运行

```bash
cd /data1/yesiliang/perishable_inventory_rl
source .venv/bin/activate
python -m pytest -q tests
python -m experiments.validate environment
```

测试涵盖 Gymnasium `check_env`、手算收入和四类成本、交货时间 1/2/3 期、三期销售后报废、缺货不替代、MNL 不购买选项、Gamma 均值/CV、全部 297 个 baseline 动作的守恒、checkpoint 精确恢复、时间上限及非法输入。

长期审计使用随机策略、种子 42/123/2026、不同保质期和交货期；日志中的利润仅用于诊断，不是 DP/PPO 实验结果。环境验证通过代表所记录约定下的实现正确，尚未证实与作者实际实现或论文数值完全一致。

## 历史 Phase 4 实际价格版本计算方案与结果

按用户授权补充 DP 数值细节，保留连续 Environment，不对真实销量和库存取整。DP 使用价值插值诱导的有限状态模型，Gamma 完整需求范围通过分段不完全 Gamma 矩积分计算；全部 297 个动作保留。详细公式、近似含义和自选参数依据见 `docs/dp_computation_scheme.md`。

最终 `configs/dp_config.yaml`：库存 [0,2] 间距 1/128，(2,10] 间距 0.25；918,731 个状态，gamma=0.99，Bellman 残差目标 1e−7，最多 5,000 次迭代。使用保持原 Bellman 方程的常数平移加速。

- 最后两层密集网格公共状态最大价值变化：0.041739，相对规模 0.0159%；外侧网格再细化的最大变化约 1.18e−6。
- 最后两层策略使用相同 100 个种子、两种初始库存，年度利润成对平均变化 −0.08614，近似 95% 区间 [−0.48231,0.31003]，通过预设 ±0.5 标准。
- 最终 Bellman 残差 4.777e−8，对投影模型的价值误差上界 4.777e−6。这个上界不含连续模型网格误差。
- 最终构造耗时约 58.55 秒、迭代约 6.02 秒、22 次迭代；CPU 单线程相关设置已记录。不同算法计算时间以后需统一计时口径。
- NPZ 保存完整价值表、策略表、延续价值与配置：`checkpoints/dp_grid_0p0078125.npz`；阶段归档：`checkpoints/phase4_project.tar.gz`。

最终三层策略在连续 Environment 上共运行 600 个年度 episode、219,000 步。这是数值方案审计，还不是论文 DP/ar-DP/PPO 比较实验。粗网格失败记录和普通 VI 结果也保留在 logs 中。

当前模型的空库存无限期折扣价值约 −45.257；100 次模拟的 365 步不折扣平均利润约 −143.676（标准误 5.201）。二者是不同指标。这里的负利润属于已记录建模解释下的结果；尚未对齐作者实际实现或确认论文数值复现。

负利润的收入／成本分解、需求量级和论文解释核对点见 `docs/profit_diagnosis.md`；原始 100 次诊断账目保存为 `results/logs/profit_diagnosis.json`。

2026-10-14 已直接核对原文 Figure 2–3：作者七个案例的最终结果均为正利润，Case 1 DP 约 3,600（图上估读）。当前需求解释的 365 步期望收入上界仅约 1,833.52，故与原文有重要模型／计量差异。数值验收通过不代表论文结果对齐通过；详见 `docs/paper_alignment_diagnosis.md`。代码与模拟确实在服务器执行，本次图形核对在本机完成。

```bash
cd /data1/yesiliang/perishable_inventory_rl
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python -m algorithms.dp
python -m experiments.validate dp
```

下一步先核对原文需求／价格尺度及计量口径；结果对齐后再进入 ar-DP。后续方法需使用同一 Environment 和同样的评估口径。

## 仓库和统一入口

代码、配置、单元测试和计算说明纳入 Git。`.venv/`、缓存、`results/`、`checkpoints/` 和下载的参考PDF保留在磁盘，但不纳入代码提交。新检出的代码可以重新生成结果；既有策略与历史报告仍位于上述目录。

```bash
python -m experiments.validate environment
python -m experiments.validate dp
python -m experiments.validate all --reuse-solved
```

统一审计入口接受 `--config` 和 `--dp-config`；`--reuse-solved` 仅适用于DP，要求checkpoint与当前配置匹配，并继续执行网格细化和连续环境策略模拟。单次DP求解仍使用 `python -m algorithms.dp`，单元测试仍使用 `python -m pytest -q tests`。

目录整理、验证和版本保存说明见 `docs/repository_maintenance.md`。
