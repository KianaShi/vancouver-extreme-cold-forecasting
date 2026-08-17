# 项目审计与面试叙事

## 一句话定位

这是一个从“能跑的毕业作业”重构为“可信、可复现、可审计的数据科学项目”的案例。重点不是模型多复杂，而是我发现并修复了会让结论失真的数据实体、缺失值、时间窗口和评估设计问题。

## 原项目的关键问题

1. **实体污染**：工作簿含 3 个气象站，原代码未按站点分组，滑窗会把不同站点的记录拼在一起。
2. **时间污染**：全表日期未排序且存在 9,307 个跨站点重复日期；“最后 20%”不是论文描述的 2019-2025 测试集。
3. **缺失值被当成天气**：GSOD 用 `9999.9`/`999.9` 表示缺失。原暴雪条件把缺失雪深和阵风判成真，导致 27,569/29,016 行成为“暴雪”。
4. **非连续窗口**：原代码只看相邻行，不验证是否为连续自然日，也不阻止跨站点。
5. **评估泄漏风险**：无独立验证阶段、无阈值选择过程；LR 和 RBF-SVM 未标准化；只报告测试集单次结果。
6. **工程不可复现**：硬编码个人 OneDrive 路径、无版本约束、无测试、无 CLI、无 CI、无数据契约。

## 重构后的设计

- 固定并验证目标站点 `71892099999`，按日期排序并去重。
- 将 GSOD 缺失哨兵转为真正缺失值，在模型 Pipeline 内拟合中位数填补，避免训练/测试信息混用。
- 仅接受过去 5 天连续自然日的窗口，并加入 lag、5 日均值、温度趋势和年周期特征。
- 2019 年前训练，2019 年后最终测试；训练期尾部仅用于阈值选择。
- 用 Average Precision 选择模型，同时报告 ROC AUC、Brier、Precision、Recall、F1 和混淆矩阵。
- `uv.lock`、pytest、Ruff、GitHub Actions 和命令行入口让别人能复现与审查。

## 经验证的数据与结果

- 目标站点：17,708 行，1957-01-06 至 2025-04-01。
- 清洗后极寒标签：398；阵风缺失 10,327 行，雪深缺失 17,056 行。
- 2019-2025 留出集：2,253 个可预测日期，50 个正例，正例率 2.22%。
- Random Forest：Average Precision 0.715，ROC AUC 0.990，Brier 0.011；验证集选择的阈值为 0.661。
- 测试集混淆矩阵：TN 2,190、FP 13、FN 24、TP 26。

## 面试时应主动说明的边界

这不是业务可部署的 6 日天气预报。输入是截至前一天的观测值，本质上是次日风险识别；真正的多日提前预报需要在出报时可用的 NWP 预报场。单站点、长期气候漂移、仪器/站点变化和标签定义也限制了外推。主动说清这些边界，比把高 AUC 包装成“可上线模型”更专业。

## 适合简历的表达

> Re-engineered a 68-year rare-weather forecasting pipeline by identifying cross-station window contamination and GSOD sentinel-value bugs; implemented gap-safe temporal features, leakage-aware holdout evaluation, validation-tuned decision thresholds, calibration diagnostics, tests, CI, and reproducible `uv` environments. Achieved 0.721 average precision with the Random Forest baseline on an untouched 2019-2025 holdout (2.22% event rate).

## LightGBM 受控升级

项目只新增 LightGBM 作为 Random Forest 的现代树模型 challenger，并保留 Logistic Regression 参考模型。三个模型共享相同特征以及 Train（1957-2010）、Validation（2010-2018）和 2019-2025 OOT Test。阈值只在 Validation 上按最大 F1 选择；RF 与 LightGBM 的主模型选择只看 Validation AP，之后才用全部 2019 年前数据重训并一次性评估 Test。

LightGBM 的 Validation AP 为 0.680，高于 RF 的 0.617，因此按预先规则被选中。但 Test AP 仅从 RF 的 0.721 提升至 0.725，不构成实质超越。LightGBM recall 更高且训练更快；RF precision、F1 和 Brier 更好。这个结果支持“现代 boosting 提供不同 trade-off，但严格 OOT 验证下经典 RF 仍然很强”的克制结论。

本次还修复了一个新的评估泄漏：旧实现用 test AP 选择要保存的 winner。现在 winner 只能由 Validation AP 决定。Reliability diagram 仍是 test-only 诊断；项目没有拟合概率校准器，也没有使用 test 标签训练任何后处理步骤。

## Multi-horizon 与历史回测升级

现在每个样本都有明确的 issue date `t` 和 target date `t+h`。输入只包含 `t` 当日及之前四天的观测、trailing-only 统计和 issue-date seasonality；标签等价于 `extreme_cold.shift(-h)`。1/3/7-day 都要求历史窗口和未来目标日期在日历上连续，不允许跨缺测日期。时间划分和年度归属按 target date 执行。

严格 OOT 中，RF 的 AP 从 1-day 0.711 降至 3-day 0.361 和 7-day 0.110；LightGBM 从 0.704 降至 0.305 和 0.098。延长 lead time 后预测难度明显上升。7-day RF 的 recall 虽为 0.74，但 precision 只有 0.107，是 validation threshold 更宽松造成的，不应解读为更好的预测。

2011-2025 expanding-window backtest 完整执行了 90 个 model-year fold。每个评估年只使用此前数据训练，并使用此前 trailing 3 年选择阈值。2025 只有截至 4 月 1 日的 91 个 target day，因此保留在结果中但标记为 partial year，并从年度 drift slope 中排除。

完整年度没有出现明显的 1-day 性能退化；RF 和 LightGBM 的 AP trend 都略为正。年度指标仍有较高方差，尤其 3/7-day 和只有少量正例的年份。2023 年 3-day 只有 3 个正例，RF/LightGBM AP 分别为 0.013/0.027；这类年份必须保留并结合事件数量解释。

项目现在可以合理称为 historical multi-horizon forecasting system，因为它具有 issue-time contract、未来 target、temporal-safe threshold 和 walk-forward simulation。但它仍不是 operational weather service；真正的 3/7-day 部署需要 NWP forecast fields 等在出报时可用的未来天气预测输入。
