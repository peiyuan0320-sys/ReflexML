<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# Collapse Dynamics Primary Trajectory Analysis

范围：**historical A1 trajectory across 36 states**。本轮不估计 Phase-5 population-expected trajectory over future randomness。

## A. Integrity status

- 36 states（base_run_id=1..36）；每个 state 均有 Wait3 + Now 配对。
- 72 replay，72 exact historical endpoint gates PASS；missing=0，duplicate=0；未混入 preflight。
- protocol/source/state/A1/stream/code/snapshot identities 和 frozen 10-point probe grid 均通过核查。
- 先只读核查完整 dataset，PASS 后才运行 offline validation；分析后再次核查 production identity/hash，PASS。
- 720/720 snapshots 使用历史完整 5,000-example validation objective；144/144 t0/t79 loss 精确匹配历史 h3/h4。没有修改 production artifacts，没有训练或新 replay。
- **Verdict: PASS。**

## B. Primary trajectory table

W、N 是各臂 full-validation loss 的 across-state mean；C 是逐 state 的 Wait3−Now paired gap 均值；F=(C0−Ct)/(C0−C79)，未截断。

| t | W_t | N_t | C_t | pointwise 95% CI for C_t | F_t |
|---:|---:|---:|---:|:---|---:|
| 0 | 0.461481395 | 0.408619128 | 0.052862267 | [0.043095691, 0.063483984] | 0.000000000 |
| 1 | 0.471675282 | 0.415395254 | 0.056280028 | [0.045129998, 0.068346144] | -0.063772911 |
| 3 | 0.458282920 | 0.425502503 | 0.032780417 | [0.022233110, 0.044636593] | 0.374712545 |
| 8 | 0.439578242 | 0.416730415 | 0.022847827 | [0.017164086, 0.028258082] | 0.560047355 |
| 16 | 0.426924991 | 0.415607042 | 0.011317949 | [0.004915567, 0.017481668] | 0.775186393 |
| 32 | 0.414817968 | 0.412115095 | 0.002702873 | [-0.000868559, 0.006191991] | 0.935937378 |
| 48 | 0.410135992 | 0.408487326 | 0.001648666 | [-0.001177077, 0.004378944] | 0.955608096 |
| 64 | 0.409442331 | 0.408893479 | 0.000548852 | [-0.001738145, 0.002740055] | 0.976129812 |
| 78 | 0.409980610 | 0.409190852 | 0.000789759 | [-0.001708326, 0.003286587] | 0.971634680 |
| 79 | 0.414494191 | 0.415224606 | -0.000730415 | [-0.003868380, 0.002262361] | 1.000000000 |

CI：NumPy Generator(PCG64)，seed=10698172564239836591，20,000 draws；按 state 有放回抽样，每次保留 Wait3/Now pair 及所有 probes；95% percentile pointwise intervals（linear quantile）。复用 Phase5 primary RNG/seed/draw-count convention，采用本轮要求的 state-only bootstrap，无 inner future resampling。设置在首次 offline validation 前记录。F 是描述性点估计，未给 F 的 CI。

## C. Main numerical findings

上表列出全部 10 个 C_t、pointwise CI 和对应 F_t。最终 closure C0−C79=0.053592681914，配对 state-bootstrap 95% CI=[0.042864124178, 0.065055701261]。

t1：gap 从 0.052862266865 增至 0.056280028181，F1=−0.063772910679；没有立即 closure。t3、t8、t16、t32 的 F 分别为 0.374712545371、0.560047354709、0.775186392766、0.935937377930。t48 以后 observed mean gap 已很小；从 t32 起 C 的 pointwise CI 跨零，这不证明等价或精确为零。

## D. Arm decomposition

全 epoch：Wait loss 0.461481395094→0.414494190650，ΔW=−0.046987204444（paired-change 95% CI=[−0.059490607602, −0.035162778684]）；Now loss 0.408619128229→0.415224605699，ΔN=+0.006605477470（95% CI=[0.001016107240, 0.012191271882]）。

描述性 identity：closure=(W0−W79)+(N79−N0)。Wait catch-up 贡献 87.674665%，Now deterioration 贡献 12.325335%。因此 **both，Wait catch-up dominant**；这些比例是终点算术分解，不是 causal mediation。

来源随时间变化：t0→t3，Wait improvement=0.003198475132、Now deterioration=0.016883375121，早期 closure 主要来自 Now deterioration；到 t8/t16，Wait 累计 improvement=0.021903152834/0.034556404265，主要后续 closure 转为 Wait catch-up。Now 在 t3 升至 0.425502503350，之后回落，到 t48 为 0.408487325564。

## E. Temporal morphology

**前段逐步累积 closure，伴随初始 non-monotone transient。** t1 先扩大，t3→8→16→32 继续缩小；t8 已完成约 56.0%，t16 约 77.5%，t32 约 93.6%。这支持 early/front-loaded formation，但不是第一步立即完成；也不是整个 epoch 均匀持续下降或 late-only formation。

Observed means 在 t64→78 小幅反弹（0.000548852356→0.000789758522），t79 略低于零。初始反弹和这些晚期小变化的稳定性不能从 C_t pointwise CI 的重叠推断。没有对单个 state 定义 timing，也没有阈值分类、change-point 或插值。

## F. t78→79 check

C78=0.000789758522→C79=−0.000730415049；additional closure=0.001520173572，占最终 closure **2.836532%**，paired contrast 95% CI=[−0.000589063695, 0.004106318015]。

这一边界 Wait loss 上升 0.004513580164，Now loss 上升 0.006033753736。其 observed gap closing 来自 Now 的 loss 上升更多。最后 16-example batch 有可见的臂 loss 跳动，但绝大部分 closure 早已发生；不支持“最终 short batch 形成主要 collapse”，其 additional gap change 的 CI 跨零。未冻结异常阈值，也未逐 step 对照，不能把该步定性为已建立的异常或机制效应。

## G. Scientific interpretation

**Established / Observed：** 在这组 historical A1、36-state mean trajectory 的冻结 probes 上，原本 epoch-level 的 abrupt h3→h4 collapse 实际主要分布在 epoch18 的前 32 updates；先扩张，再逐步 closing。终点变化由 Wait catch-up 主导，Now deterioration 也贡献，且来源随时间改变。

**Weakened / ruled out at this descriptive resolution：** “第 1 步已经完成 collapse”、“只有 epoch 后段才开始形成”、“最后 short batch 贡献主要 closure”以及严格逐 probe 单调下降，均与 observed means 不符。不能据此排除 probe 间短暂变化，也不把 observed morphology 当成所有 future realizations 的定律。

**Still unresolved：**

- Temporal：exact completion step、probe 间 excursion、初始小反弹与晚期小反转是否稳定；C79 的 CI 跨零，未建立终点反转或精确 zero gap。
- Future-realization：只有 A1，未估计 future expectation 或 future-realization timing variability；state CI 不包含该层完整不确定性。
- Mechanism：这条 loss trajectory 无法区分 optimizer、gradient noise、curvature 或其他机制；不做 D/N、mediation 或机制故事。

## H. A2 decision

**A2 needed now: NO。** A1 已足以区分 broad morphology：明显前段集中、逐步 closure，带初始非单调形态；主要 collapse 不在 late epoch 或 final short batch。A2 不能改善 frozen probe grid 的时间分辨率。若另立 future-randomness expectation 目标，那属于后续独立问题。本轮没有执行 A2。

## I. Cheapest next step

只推荐一个：**利用已有 A1 表，对 C1−C0 做单个 paired state-bootstrap temporal contrast，核查初始反弹是否稳定。** 当前 point estimate 为 +0.003417761317；pointwise C 的 CI 不能代替这个 paired difference 的 CI。这是看过 primary trajectory 后提出的描述性跟进，必须如实标注并先固定该单一 contrast；不需要新增 snapshots 或 replay。本轮未执行该跟进。

## J. Files produced

- per_state_trajectory.csv：720 行 tidy arm observations，附 source/stream identity。
- per_state_paired_trajectory.csv：360 行 paired W_t、N_t、c_t。
- aggregate_summary.csv：完整 10-point primary table 与 C/W/N pointwise CI。
- bootstrap_metadata.json；bootstrap_gap_means.npy：固定方法和 bootstrap C means。
- primary_gap.png / .pdf；arm_trajectories.png / .pdf：只显示冻结点及 pointwise CI，无 smoothing、spline 或插值。
- analysis_summary.json；scientific_interpretation.json：machine-readable numerical results 与 bounded interpretation。
- scientific_report.md：本报告。
- integrity_check.json；analysis_manifest.json；report_finalization_manifest.json；artifact_hashes.json：provenance / integrity。
- per_state_reports/state_001..036：复用 offline_report 的 36 份源表和 manifests。
- descriptive_means/historical_a1_summary.csv：复用 summarize_a1 的原始均值汇总。
- analyze_collapse_dynamics_primary.py；finalize_collapse_primary_report.py：分析及报告 materialization 脚本副本。

执行：`python3 analyze_collapse_dynamics_primary.py --production internal-artifacts/reflexml-collapse-dynamics-replay-v1-production-20260928-001 --output internal-artifacts/reflexml-collapse-dynamics-analysis-v1-20260928-001`。

验证：配对 resampling 的 deterministic synthetic check；从 tidy CSV 独立重建全部 C/W/N/F；逐条历史边界 equality；全部 production hashes 的前后核查；两张 PNG 的 rendered visual inspection。已有 protected kernel / launcher / endpoint gate / training loop 未修改。本轮仅新增 narrow analysis script；未读取 m=4 trajectory、执行新 replay/training 或进行机制分析。
