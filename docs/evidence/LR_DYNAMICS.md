<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# LR-Switch Dynamics v1 primary analysis

本报告范围严格为 **historical A1 LR-switch trajectory across 36 states**。唯一问题：已由 endpoint experiment 建立的 causal LR contrast 在 epoch18 内何时形成。

## A. Integrity status

36 states（1–36），每 state 有 Switch/.05 与 Continue/.10；72/72 COMPLETED_PASS，72/72 identity-bound endpoint gates PASS；missing=0、duplicate=0。只使用 production，preflight 仅作为执行身份绑定校验，未计入分析。

先调用现有 integrity()，核对 protocol ID、协议/实现哈希、runtime、A1、canonical state/source/checkpoint/future-stream mapping、10-point probes、79 updates、全部 snapshot 哈希、execution records 和 gate checks，PASS 后才开始 offline validation。分析后同一 integrity() 再核查 PASS。这里复核已有 exact endpoint gates，没有重跑 gate 或训练。

720/720 snapshots 复用现有 offline_report/evaluate，full 5000 validation examples、CrossEntropyLoss、原 sample-weighted reduction；无 proxy、smoothing、interpolation。144/144 arm-boundary loss（t0/t79）逐 state 精确匹配历史；36/36 d0 精确为零。Switch 与 frozen historical Wait3 的 360 个 state/probe loss 全部精确一致。历史 A1 endpoint D79=0.044738770161486335，与本轮一致至浮点汇总精度，不存在近零分母冲突。

**Verdict: PASS。** production artifacts 和仓库代码未修改，无 training/replay/A2/m4。

## B. Primary LR trajectory table

D_t=mean_s(L_.10,t−L_.05,t)；G_t=D_t/D79，为未截断的描述性 ratio of means。

| t | L_.05 | L_.10 | D_t | pointwise 95% CI for D_t | G_t |
|---:|---:|---:|---:|:---|---:|
| 0 | 0.461481395 | 0.461481395 | 0.000000000 | [0.000000000, 0.000000000] | 0.000000 |
| 1 | 0.471675282 | 0.491794942 | 0.020119660 | [0.011583179, 0.029490580] | 0.449714 |
| 3 | 0.458282920 | 0.482479419 | 0.024196499 | [0.012607315, 0.039620149] | 0.540840 |
| 8 | 0.439578242 | 0.474659692 | 0.035081450 | [0.020952611, 0.052817605] | 0.784140 |
| 16 | 0.426924991 | 0.459339640 | 0.032414649 | [0.022805483, 0.042033934] | 0.724532 |
| 32 | 0.414817968 | 0.450521786 | 0.035703818 | [0.027745915, 0.043713851] | 0.798051 |
| 48 | 0.410135992 | 0.444265849 | 0.034129858 | [0.025739754, 0.043300697] | 0.762870 |
| 64 | 0.409442331 | 0.448461088 | 0.039018756 | [0.033111670, 0.045131467] | 0.872146 |
| 78 | 0.409980610 | 0.440427938 | 0.030447327 | [0.022063242, 0.039614713] | 0.680558 |
| 79 | 0.414494191 | 0.459232961 | 0.044738770 | [0.033531138, 0.056076629] | 1.000000 |

NumPy Generator(PCG64)，seed=10698172564239836591，B=20000；每 draw 抽 36 个完整 state trajectories，保留两臂及全部 probes；percentile 95% CI，linear quantile。设置在首次 interior evaluation 前已固定于协议及现有分析实现。没有 bootstrap steps、独立 arms 或 simultaneous bands；G 无 CI，百分比差异仅描述。

!Primary contrast (internal archive; excluded)

## C. Main numerical findings

全部 D1、D3、D8、D16、D32、D48、D64、D78、D79 及 G 已列于上表。第一步 D1=0.020119660，约 endpoint 的44.97%；t3=54.08%，t8=78.41%。t16/t32 为72.45%/79.81%，t48=76.29%，t64=87.21%，t78回落至68.06%，t79为100%。所有非零 probes 的 D pointwise CI 均高于零，但不能据此声称所有 probes 同时显著或相邻变化显著。

## D. Arm decomposition

以下均相对于各自 t0，正号为 worsening，负号为 improvement：

| t | ΔL_.05 | ΔL_.10 |
|---:|---:|---:|
| 1 | +0.010193887 | +0.030313547 |
| 3 | -0.003198475 | +0.020998024 |
| 8 | -0.021903153 | +0.013178297 |
| 16 | -0.034556404 | -0.002141755 |
| 32 | -0.046663427 | -0.010959609 |
| 79 | -0.046987204 | -0.002248434 |

t1：两臂均 worsening，.10 上升0.030313547，大于 .05 的0.010193887，形成正 contrast。t3：.10 worsening 为主，.05 已轻微改善；t8：both，.05 improvement 0.021903153 与 .10 worsening 0.013178297 共同贡献。t16/t32：两臂均改善，但 .05 改善更多。

整个 epoch，.05 改善0.046987204，.10 改善0.002248434；endpoint contrast 主要来自 .05 的更大 improvement，不能描述成全 epoch .10 worsening。这里的 loss movement 是描述性算术分解，不是 causal mediation。

!Arm means (internal archive; excluded)

## E. Temporal morphology

**明显的 immediate component + front-loaded formation，叠加 non-monotone observed transient 与显著的末步增量。** 不能把完整 endpoint effect 称为 near-immediate：t1/t3 只有约45%/54%，到t8才约78%。这不是 late-only formation，也不符合全 epoch 平稳累积；t8→16、t32→48、t64→78 的 observed means 有回撤。未对这些回撤做配对显著性检验，不把它们的统计稳定性视为 established；未做 change-point、slope tests 或 adjacent-contrast fishing。

## F. Historical collapse alignment

历史 C=Wait3−Now；closure fraction F=(C0−Ct)/(C0−C79)。LR D=Continue−Switch，G=D/D79；这是不同 contrasts 的各自终点归一化，不是可互换的 estimand。历史 C0=0.052862267，C79=−0.000730415；LR D0=0，D79=0.044738770。

| t | historical C_t | historical closure F_t | LR D_t | LR G_t |
|---:|---:|---:|---:|---:|
| 8 | 0.022847827 | 0.560047 | 0.035081450 | 0.784140 |
| 16 | 0.011317949 | 0.775186 | 0.032414649 | 0.724532 |
| 32 | 0.002702873 | 0.935937 | 0.035703818 | 0.798051 |

来源：[frozen Collapse Dynamics report](HISTORICAL_DYNAMICS.md) 与 aggregate_summary.csv；已核对 artifact_hashes，精确引用记在 source_references.json。

**早期形成的宽泛时间尺度 broadly aligned，但完整 temporal shapes materially different。** LR 在t8达到78.4%，早于 collapse 的56.0%；t16较接近（72.5% vs77.5%）；t32 collapse 已93.6%，LR仍约79.8%。因此不能给出全程“LR更早”或“LR更晚”的单一排序。历史初始t1 closure为−6.38%，LR已形成44.97%；历史final batch仅贡献closure的2.84%，LR末步为31.94%。LR有更明显的即时分量与末端贡献，历史collapse则更接近前32步内完成的reconvergence。

两者共享同一条 Switch/Wait3 臂，不能视为两条独立 replication。G的分母包含末步增量，归一化比较也受其影响。Temporal similarity不证明LR uniquely causes historical collapse；shape差异不否定已有endpoint causality。

## G. t78→79

D79−D78=**0.014291442884**，paired whole-state bootstrap 95% CI=**[0.005958286263, 0.023164936010]**，占D79的**31.944201%**（描述性比例）。

末步 .05 loss 上升0.004513580164；.10上升0.018805023048；contrast增量来自.10更大的loss上升。它是量值上重要、CI高于零的末步贡献；不是endpoint的多数来源，t78之前仍有68.06%。未预先定义“异常”阈值或普通单步reference distribution，不能声称已经建立statistical anomaly，更不能从该观察断言batch size=16本身造成该增量。该区别与historical collapse的末步结论明确不同。

## H. Scientific interpretation

### Established

在本36-state historical A1样本与冻结probes上，causal LR contrast第一步即为正，主要部分在前8步已经形成；随后observed trajectory非单调，末步追加正contrast且paired CI高于零。完整轨迹不支持单一、平滑、前32步几乎完成的形成故事。

### Weakened / ruled out

在本描述分辨率下，“仅晚期才形成”“第1步已完成几乎全部”“末16-example batch可忽略”“LR与collapse具有相同归一化形状”均不符合数据。不能据此排除probe间excursions、future-randomness差异或把未检验的回撤称显著。

### Still unresolved

精确形成步数、未记录probe间的形状、回撤稳定性、跨future realizations的timing以及末步贡献为何较大均未解决。没有估计完整future-randomness population expectation；pointwise state CI不能代替该层估计。Gradient noise、deterministic dynamics、curvature、momentum、basin等机制均未由本轮识别。

## I. A2 decision

**A2 needed now: NO。** 对指定historical A1的broad timing已经清楚：即时分量、前段主体、非单调observed path、重要末步贡献。这里的NO不意味着已建立future-randomness稳定性；A2也不能提高10-point grid的时间分辨率。

## J. Cheapest next scientific step

**m=4 timing now: NO。** 本轮已提供temporal target，但LR与collapse的明显末端差异尚不能支持把m4排成最便宜、最有辨识力的下一步；m4会同时改变batch construction/exposure，不能直接隔离这里的末步来源。

只推荐一个下一步：**先冻结一个针对t78→79的最小配对反事实实验设计**，从各臂各自的t78完整状态出发，围绕末步batch构造提出可识别的单步对照，明确batch-size与sample-composition不能混同。现存snapshot只有t78 model，没有t78 optimizer；设计需先只读确认恢复完整状态所需工作与成本，不能直接拿model冒充checkpoint。设计优先于现在启动新的整epoch m4 timing；这是基于本结果的研究判断，不是已验证的计算成本最优性。尚未设计、执行或授权任何新replay、training或diagnostics。

## K. Files produced

- per_state_trajectory.csv：720行，现有evaluator输出，附source/stream身份；per_state/下36组原始offline表和manifest。
- per_state_paired_trajectory.csv：360行，成对L_.05、L_.10、d_t。
- primary_table.csv / aggregate_summary.csv：10-point结果及pointwise CI。
- arm_decomposition.csv、historical_alignment.csv：描述性分解及历史对照。
- bootstrap_metadata.json、analysis_summary.json、supplementary_summary.json、scientific_interpretation.json。
- primary_gap.png/pdf、arm_trajectories.png/pdf：两张图，probe points，无拟合/连线插值；主图含同数据的early放大视图。
- integrity_check.json、analysis_manifest.json、source_references.json、report_finalization_manifest.json、artifact_hashes.json。
- analyze_lr_switch_dynamics.py、finalize_lr_dynamics.py、write_lr_scientific_report.py：复现脚本副本。
- scientific_report.md：本报告。

执行命令：`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 analyze_lr_switch_dynamics.py --production internal-artifacts/reflexml-lr-switch-dynamics-v1-production-20260928-001 --output internal-artifacts/reflexml-lr-switch-dynamics-analysis-v1-20260928-001`。

验证：720条键唯一且有限；从tidy CSV独立重建paired means、全10点CI和G，与现有primary_table一致；36个d0为零；360个Switch/Wait3点精确相同；144个historical boundaries精确相同；production完整性前后PASS。图已render并目视检查；bootstrap只增加本次明确要求的d79−d78，不做其他post-hoc contrasts。
