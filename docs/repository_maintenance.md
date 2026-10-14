# 仓库整理记录

本次整理保留相对价格模型、DP数学算法、参数和已有实验结果。

## 入口合并

原 `tests/validate_environment.py` 和 `tests/validate_dp.py` 合并到 `experiments/validate.py`，分别使用 `environment`、`dp`、`all` 子命令。环境审计读取同一基础配置；DP审计保留网格求解、外侧细化、相同种子的策略评估、成本账目和验收阈值。支持自定义配置路径和复用匹配的DP checkpoint；自定义库存上限时，将非网格初始库存限定在合法范围内，正式baseline的初始库存保持不变。

`tests/` 只保留单元测试；`algorithms/dp.py` 保留单次求解入口。README及计算文档中的复运行命令已更新。

## 清理范围

删除10个仅包含“Implementation pending”的空占位文件：四个 `experiments/run_case*.py`、`algorithms/ar_dp.py`、`algorithms/ppo_train.py`、`algorithms/evaluate.py`、三个 `utils/` 文件。它们没有运行行为或计算结果；ar-DP、PPO和正式Cases仍为后续待实现功能。运行缓存可以重新生成。

没有发现额外的项目临时脚本。服务器用户目录中的其他脚本不属于本项目，未改动。历史checkpoint、日志、表格、参考PDF和恢复tar归档均保留。

## Git范围

Git仓库根目录为 `perishable_inventory_rl/`，与只读项目资料 `sources/` 分开。跟踪代码、配置、依赖冻结、单元测试和Markdown说明。`.gitignore` 排除虚拟环境、缓存、生成结果、checkpoint及下载的参考PDF；`.gitattributes` 为文本源文件统一LF换行。

仓库身份仅用于本项目：`yesiliang <yesiliang@localhost>`，不修改全局Git配置。保存本地提交和版本标签，不向外部托管平台发布。

相对价格模型重算前后的恢复归档仍可用于查阅整理前的完整目录。Git只保存代码版本，恢复既有实验数据时需同时保留 `results/` 和 `checkpoints/`。

## 整理后验证

服务器实际执行：68项单元测试通过；新环境审计入口运行15个完整年度、5,475步，通过守恒和奖励核对；DP新求解及checkpoint复用入口均通过端到端检查。

DP入口检查在临时隔离目录中采用qmax=2、每episode30步、两层网格1/32与1/64、3个种子，并保留原验收阈值。此项验证用于检查整理后的命令流程，不替代此前正式baseline的600个年度episode审计。测试结束后临时目录自动清理，正式参数和结果均未覆盖。

本次检查日志与报告保存在 `results/logs/repository_cleanup/`。版本标签为 `relative-price-dp-v1`，指向整理后的相对价格DP代码。
