<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# Phase 5B 冻结数据二次分析与 precision planning

日期：2026-09-28。范围：复用已完成的 `Post-Phase5-EIV-v3.1` 正式输出，核验冻结输入，机械汇总已有 precision 模拟；另补充明确标注的解析 planning benchmark。**本轮新增训练、states、future replicas、EIV 拟合、bootstrap、Monte Carlo 模拟均为零。**

结论：**当前数据不能排除有实际意义的 association。主要问题是有限副本测量噪声较大、ρ 的状态间异质性未被可靠识别，同时 N=36 带来采样不确定性。瓶颈分类为 mixed，新增计算若以 covariance precision 为目标，应优先增加 K_A。唯一推荐的首个候选是 36 / 30 / 10；这是后续设计建议，不是执行授权，也不保证识别 latent correlation。**

## A. Data integrity

冻结数据完整可用，本次 input-contract 检查 PASS：

- `base_run_id=1..36`，逐 state 恰好 15 个 A 和 10 个 B；合计 540 A、360 B。无缺失、重复或非有限主要观测。
- ρ 只来自 A 的 `R=(ΔL1+ΔL2+ΔL3)/3`；τ 只来自 B 的 `G=(mean_W_28:30−mean_N_28:30)/max(mean_W_28:30,1e-12)`。逐副本重算公式与 CSV 的最大误差均为 0；是每个副本的 ratio 先算、再跨副本求平均。
- 用冻结 validator 从逐副本 CSV 重算 state 均值、SD、SE、horizon summaries，全部在 `rtol=1e-10, atol=1e-12` 内匹配。没有重新生成原始结果。
- 窄范围流式读取 frozen ledger 的身份字段，核对 2,700 事件、连续版本及 predecessor head、900 个完整成功 attempt、CSV 与 ledger 的 attempt/hash 900/900 匹配。36 个 checkpoint hash 唯一。900 个 future seed 和 900 个 future stream identity 全局唯一，A/B overlap 都为 0。
- 协议、分析实现、两个输入 CSV 的哈希匹配正式 manifest；当前 repository HEAD 为 `82387119ec558c719d51199e613b9e8c4f65a4f3`。当前存在其他工作改动，但 EIV 实现与协议精确匹配已执行版本。
- 主分析 JSON、原脚本、bootstrap NPZ 的哈希匹配 `RESULTS_PHASE5.md`。本次前后核验的 20 个 source files 全部字节不变。没有重新遍历/审计 900 个原始 artifact 内容；原始 artifact/pair-order 检查复用冻结 audit，而本次独立检查 CSV、ledger 身份绑定和统计输入。

精确身份：

| 文件 | SHA-256 |
| --- | --- |
| `POST_PHASE5_EIV_PLAN.md` | `91cac4d95bf8d4606ef2c790efc22724d4723db6be413ecf2c147beeaad7369b` |
| `reflexml/post_phase5_eiv.py` | `6e49f34470889743f1c44dec576d989f066a117875967ed2ef71a6c5c9e734a0` |
| `phase5_replica_estimates.csv` | `1868b85cf9ac9f8ac6c158a8af7fa057f92a46ad492bf6a30942b098cace34d1` |
| `phase5_state_estimates.csv` | `d119d62176a780fe683ea7b38c630a21a4f0121e350420e7372fd536b1f0a056` |

最终 ledger head：`3222eb22d21b2b0bb81aa9062e5444be8c038ffacdf4000dd0639504f45639ab`。

主分析 bootstrap：20,000 nested state draws，NumPy `Generator(PCG64)`，seed `10698172564239836591`；每个 sampled outer-state occurrence 独立重采 A/B whole records。EIV outer-only bootstrap：20,000/20,000 有效，seed `12479654370683736347`；hierarchical EIV bootstrap：2,000/2,000 成功，seed `3757477645905739108`。正式 runtime 为 Python 3.14.3 / NumPy 2.4.4 / SciPy 1.17.1。完整 RNG namespace 与哈希保留在原始 JSON 和本次 `verified_evidence.json`。

**误差独立性的边界：**检查实现和身份记录支持在给定 S 下采用独立 A/B measurement error 的工作假设；不同 seed/stream identity 本身不能数学证明误差独立。每个 Now/Wait pair 内的共同随机数配对，与 A/B blocks 之间的分离是两个不同层次。没有用任意 A/B replica 编号配对来估计 error covariance。

## B. Naive Phase 5B result

冻结 primary estimand 始终是 `ψ=Cov_S(ρ,τ)`，符号始终 Wait minus Now，outcome window 始终 epoch 28–30。

| 量 | 点估计 | 冻结 95% percentile CI |
| --- | ---: | --- |
| Phase 5A μρ | 0.0484693 | [0.0461314, 0.0509002] |
| Phase 5B ψ | −6.95000744×10⁻⁶ | [−2.51969340×10⁻⁵, 1.02023749×10⁻⁵] |
| observed Pearson r，描述性 | −0.230142 | 不替代 primary covariance CI |

结论保持：**5A signature supported；5B primary association insufficiently precise。**负点估计不是负关联已建立；跨零不是没有关联。

## C. Measurement-error decomposition

**post-outcome exploratory。**令每个 state 的副本样本方差用 `K−1` 分母计算，均值的 measurement variance 为 `v_si=s²_w,si/K`。Across-state observed variance 用 `N−1` 分母。raw between-state component 为：

`σ²_x,raw = S²(x_hat) − mean_s(v_x,s)`。

这包含有限 K 的 sample-variance correction，允许每个 state 的 noise variance 不同。需要：states 独立；给定 state 的副本可视作来自相同条件分布且独立；conditional mean error 为零且有有限二阶矩。不能默认 homoskedastic，也不能把副本当 independent states。

| 量 | ρ，A / K=15 | τ，B / K=10 |
| --- | ---: | ---: |
| Observed across-state variance | 2.70778677×10⁻⁵ | 3.36792901×10⁻⁵ |
| Mean within-state **replica** variance | 4.29013233×10⁻⁴ | 2.45718255×10⁻⁴ |
| Mean **state-estimator** measurement variance | 2.86008822×10⁻⁵ | 2.45718255×10⁻⁵ |
| Raw between-state component | **−1.52301445×10⁻⁶** | 9.10746451×10⁻⁶ |
| Raw signal fraction | **−0.05625；不是合法 reliability** | 0.27042，reliability-style estimate |
| RMS state-estimator SE | 0.005348 | 0.004957 |
| State-specific measurement variance min–max | 7.25750×10⁻⁶–8.83639×10⁻⁵ | 1.09540×10⁻⁵–4.96504×10⁻⁵ |
| Outer-only q025–q975 sensitivity range，raw between variance | [−1.22777×10⁻⁵, 8.68114×10⁻⁶] | [−5.05646×10⁻⁶, 2.31622×10⁻⁵] |

ρ 的 measurement variance 约为 observed variance 的 106%；τ 约为 73%。这是**噪声很大、ρ 的异质性未分辨**的情况，接近用户提出的 Scenario B，并伴有 Scenario C 的可能性。不能确认 ρ 恰好无异质性，也不能说“只是 N 不够”。

ρ 的 overall mean 明确为正与其 across-state signal 很弱可以同时成立：前者问平均干预反应，后者问 state ranking 是否携带信息。ρ 的 observed across-state SD≈0.005204，与 RMS measurement SE≈0.005348 相当。

Outer-only percentiles 是 sensitivity ranges，latent-variance coverage 未建立。原始 nested τ-variance bootstrap 的 upward shift 已在冻结记录中说明，不能用其正下界宣称真实 heterogeneity 已被可靠识别。

## D. Classical attenuation / EIV sensitivity

**post-outcome exploratory。**独立、conditional-unbiased 的 A/B errors 下：

`E[sample Cov(rho_hat,tau_hat)] = ψ`。

Measurement error 增加 covariance 的采样方差，但不产生需要用 reliability 除掉的 covariance attenuation。Correlation 的 population attenuation benchmark 为 `r_observed≈r_latent sqrt(Rel_ρ Rel_τ)`，前提包括正 latent variances。

本数据 raw ρ signal fraction 为负，因此 `r_observed/sqrt(Rel_ρ Rel_τ)` **不可用**。把负值截为零再除，会制造无穷/未定义值；用另一种 variance 点估计代入则高度依赖模型。故 classical corrected correlation 标记 **NA / unstable**，没有给出 confirmatory corrected r。

复用冻结的 heteroskedastic Gaussian EIV specification：latent state effects 为 bivariate Gaussian；observed means 的协方差为 `Σ+diag(v_ρ,s,v_τ,s)`；state-specific estimated measurement variances 在 plug-in likelihood 内视为给定量。latent effects 的 Gaussian 形状、mean-error approximation、noise specification 及 estimated variances 的不确定性均属假设。

| EIV 量 | Conditional plug-in 点估计 | Descriptive 95%-reference likelihood support |
| --- | ---: | --- |
| σ²ρ | 5.37399757×10⁻⁶ | **[0, 2.19116043×10⁻⁵]** |
| σ²τ | 1.14838278×10⁻⁵ | [1.15844506×10⁻⁶, 3.18626352×10⁻⁵] |
| ψlatent | −2.83179177×10⁻⁶ | [−1.53605448×10⁻⁵, 8.17728703×10⁻⁶] |
| rlatent，弱识别 | 数值点 −0.360470，仅记录 | **[−1,1]** |

Full fit、global boundary reference 和 709 profile points 均 PASS。ρ variance support 含 exact zero boundary；该边界下 covariance 必须为零，而 correlation **undefined**。支持集覆盖 [−1,1] 的意思是正方差 conditional profiles 没有排除整个相关域，不是给零方差状态定义了 correlation。

这些 reference support sets **不是具有已验证 nominal 95% coverage 的 CI**，尤其存在 variance boundary。不能把范围变窄解读为 Phase 5B 已被“修正”。

Measurement-variance uncertainty：已有 hierarchical bootstrap 2,000/2,000 成功，0 failed/unstable，9 boundary fits。成功拟合的 q025–q975 为 ψ `[−2.11106×10⁻⁵,1.16690×10⁻⁵]`，r `[−0.740810,0.467011]`。它们是 conditional sensitivity percentiles，**无 calibrated coverage claim**；nested resampling 的 variance components 也明显上移。因此不能因 +0.5 略超这个 percentile range 就宣称已排除 +0.5。

Pooled-noise sensitivity 用同一模型、统一 A/B noise：ψ=−4.68253×10⁻⁶，σ²ρ=2.41211×10⁻⁶，σ²τ=9.08999×10⁻⁶，r 点落在 −1 boundary。和 heteroskedastic fit 的 r≈−0.36 不同，进一步说明 latent r 对 noise specification 敏感。没有选择较强的点估计作为最终结果。

LOO 36/36 完成、无失败/不稳定或 ψ sign reversal；ψ范围 `[−5.18532×10⁻⁶,−9.44261×10⁻⁷]`。这只是负点估计对单 state 的稳定性，不是总体负 association established。A residual skewness≈0.736、excess kurtosis≈1.716，Gaussian 假设有可见限制。

## E. What EIV changes

**modestly。**它改善了对“不确定为何发生”的诊断：state effects 尤其 ρ 测得很吵，latent variance 与 correlation 有 boundary/identifiability 问题。它没有改变 frozen primary classification，也没有支持 surrogacy、证明负关联或证明独立。不能给“measurement error 是唯一原因”的定论。

## F. Current uncertainty / practical exclusion

未制定“有实用价值”的阈值；只采用用户指定的 reference points。直接读取已有 profile grid，不新增 profile fitting：

| latent r reference | −|r| 是否仍兼容 | +|r| 是否仍兼容 |
| --- | --- | --- |
| 0.2 | 是，Δ2loglik=0.0586 | 是，0.6718 |
| 0.3 | 是，0.0084 | 是，0.9149 |
| 0.5 | 是，0.0456 | 是，1.4710 |

r=0 也兼容，Δ2loglik=0.2872；冻结 reference cutoff 为 3.84。**±0.2、±0.3、±0.5 均未被这项 EIV sensitivity 排除。**没有 robust practical-exclusion 结论，更不能据此宣布 short-term response 无信息。

## G. Precision bottleneck

**mixed，优先 K_A。**两边 mean-effect measurement noise 都大；ρ raw positive heterogeneity 无法分辨，τ reliability-style estimate 也只有约 0.27。N=36 是另一层限制，但不是高-reliability 状态下的纯 N bottleneck。现有数据不能确认瓶颈是 fundamentally low heterogeneity，因为 σ²ρ=0 与 positive small variance 都在支持范围内。

增加 K_A 能购买更精确的 ρ measurement，且 A pair 只需 8 branch-epochs，B pair 需 32。若真实 σ²ρ 极小，更多 K_A 仍可能只更清楚地识别“异质性很低”；它不能制造 ranking signal。这也是 future expansion 的科学风险。

## H. Precision planning table

**post-outcome exploratory。**复用全部 **19 scenarios × 2 latent families × 2 noise engines × 36 designs = 2,736 cells**；每 cell 3,000 synthetic datasets，所有 attempted=valid=3,000，failed=0。没有重跑。

L1 Gaussian；L2 standardized bivariate t5。M1 Gaussian state-specific noise templates；M2 empirical centered residual pools，后者无法生成经验支持之外的 tails。冻结情景含 zero variances、small/large positive variance pairs，并在每个 positive pair 上含 `r=−.3,0,+.3`。

指标为 ordinary covariance estimator 的 state-jackknife interval `ψ_hat±1.96 SE_J`，不是 primary nested bootstrap，也不是 latent r CI。下面对全部 76 matched scenario/family/engine combinations 等权汇总；“中位数”无 prior probability 或 expected-utility 含义。Precision gain 先在每个 combination 相对同情景 baseline 计算，再汇总，避免挑情景。

基线成本 `C0=36(14+8×15+32×10)=16,344` epoch-equivalents；增量成本：

`Cinc=36[8(K_A−15)+32(K_B−10)]+(N−36)[14+8K_A+32K_B]`。

Relative compute=`1+Cinc/C0`，是**最终设计总成本**与 baseline 的比值，不是“本轮已用算力”。成本不含 wall-clock、存储、checkpoint restoration 等 overhead。

| Design | N | K_A | K_B | Relative compute | 增量 epochs | Covariance width 减少，中位数 [范围] | Gain / 1000 epochs | Coverage，中位数 [范围] |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Current / do nothing | 36 | 15 | 10 | 1.000 | 0 | 0% | NA | .9505 [.9380,.9593] |
| Double N | 72 | 15 | 10 | 2.000 | 16,344 | 28.0% [26.8,29.0] | .01713 | .9503 [.9403,.9570] |
| Double K_A | 36 | 30 | 10 | **1.264** | **4,320** | **24.7% [19.2,30.1]** | **.05720** | .9500 [.9380,.9600] |
| Double K_B | 36 | 15 | 20 | 1.705 | 11,520 | 20.3% [12.0,29.7] | .01766 | .9503 [.9383,.9597] |
| Double both K | 36 | 30 | 20 | 1.969 | 15,840 | 38.9% [29.3,50.2] | .02456 | .9492 [.9380,.9573] |
| Balanced alternative | 48 | 30 | 15 | 2.156 | 18,888 | 41.5% [34.7,49.8] | .02198 | .9498 [.9310,.9627] |
| Larger A expansion | 36 | 60 | 10 | 1.793 | 12,960 | 40.3% [31.3,50.5] | .03106 | .9488 [.9323,.9610] |
| Modest N expansion | 48 | 15 | 10 | 1.333 | 5,448 | 12.7% [11.3,13.5] | .02326 | .9503 [.9413,.9600] |

Baseline absolute median covariance interval width across 76 combinations = **2.03167×10⁻⁵**；double K_A = **1.61497×10⁻⁵**；double N = **1.46373×10⁻⁵**。这些不是从现有 primary CI 按比例保证缩小的预测；模拟估计的是各 final design 的 unconditional planning precision。

Coverage ranges、Monte Carlo SD、RMSE 和 q90 width 全部保留在 `design_comparison.csv`；完整 608 个 displayed-design cells 保留在 `selected_design_cells.csv`。例如 baseline median MC SD / RMSE≈5.642×10⁻⁶、median q90 width≈2.88154×10⁻⁵；double K_A 相应约 4.372×10⁻⁶ / 4.373×10⁻⁶ / 2.23238×10⁻⁵。不能仅按窄度忽略 coverage；每 cell coverage 的 MC SE 在 .95 附近约 .004。

Pure K_A expansion 的 36/30/10 与 36/60/10 在 76/76 combinations 均 Pareto-nondominated；pure N 或 K_B expansions 在 0/76。Frontier 是 frozen grid 内的 cost-width 比较，不是全局 optimum，也不是统计/科学结论。

### 用户要求的 |r|=.2/.3/.5：单独解析 supplement

**post-outcome exploratory；不是冻结模拟结果。**冻结模拟没有 .2/.5 cells，未擅自扩展。为补足这些 reference points，使用透明 iid Gaussian / pooled-noise moment benchmark。令 positive latent variances `a,b` 是已知固定情景量，measurement variance 用 observed mean within-replica variances `w_A/K_A,w_B/K_B`，则：

`SE(ψ_hat)≈sqrt(((a+w_A/K_A)(b+w_B/K_B)+r²ab)/(N−1))`，

`planning width≈3.92 SE`。

这是 homoskedastic Gaussian approximation 的解析基准，不能替代前面的 heteroskedastic/heavy-tail 模拟。使用冻结 V4–V8 全部五对 positive variance 情景，完整结果在 `gaussian_reference_benchmark.csv`；正负 r 因 r² 在此近似内对称。

下面展示 V5 `(a,b)=(5×10⁻⁶,9.107×10⁻⁶)` 的直观尺度例子，不用它替代全情景推荐：

| True |r| scenario | Current covariance width | 36/30/10 width | 减少 | Current→36/30/10 的 correlation halfwidth，**已知 a,b 时的换算** |
| --- | ---: | ---: | ---: | --- |
| 0 | 2.22898×10⁻⁵ | 1.68933×10⁻⁵ | 24.21% | 1.652→1.252 |
| .2 | 2.23077×10⁻⁵ | 1.69169×10⁻⁵ | 24.17% | 1.653→1.253 |
| .3 | 2.23301×10⁻⁵ | 1.69465×10⁻⁵ | 24.11% | 1.655→1.256 |
| .5 | 2.24016×10⁻⁵ | 1.70406×10⁻⁵ | 23.93% | 1.660→1.263 |

Correlation 换算为 `width_ψ/(2 sqrt(ab))`，假定 latent variances 已知，未包含实际 variance-estimation uncertainty，**不是已校准的 latent correlation CI，也不截断到 [−1,1] 来假装获得 precision**。即使在这个理想化尺度换算中 halfwidth 也大于 1，说明首个 K_A expansion 的 covariance gain 不代表足以区分 |r|=.2/.3/.5。实际 zero-variance support 使 correlation 更难识别。

## I. Best compute allocation

**唯一推荐：将 36 / 30 / 10 作为未来独立冻结 continuation design 的首个候选，优先 K_A:15→30。**

理由：在完整冻结情景集合内，最小的 A expansion 以 4,320 incremental epoch-equivalents 换得 median 24.7% covariance-width reduction；displayed alternatives 中 gain per incremental compute 最高。它直接对准当前 unresolved ρ measurement/signal separation。无需通过挑 states 或修改 outcomes 获得该优势。

这不是承诺 latent r 可识别，不是已证明最优，不是在原 Phase 5 primary 中追加观测，更不是启动计算的许可。不推荐直接倍增 N、K_B 或大规模双侧 expansion；更大的 precision gain 会花更多成本，而当前缺少具体 calibrated exclusion target 和未来 budget。这里不输出“DO NOT EXPAND PHASE 5B YET”作为低收益判断，因为现有规划确实指出了可购买的信息；本轮仍严格停止于建议。

## J. Scientific conclusion

1. **当前数据是否支持 short-term response 作为 long-term surrogate？不支持验证性 surrogate 主张。**ρ 的平均短期反应已建立，但 association 和 state ranking precision 不足；即使 association 将来得到支持，也不等于 causal surrogacy 或 pre-intervention observability。
2. **当前数据是否反驳它？没有。**covariance CI 跨零，latent correlation 弱识别，±.2/.3/.5 均仍兼容；没有排除实用 association。
3. **measurement error 是否足以解释 weak observed association？是合理且重要的解释，但没有证明它是唯一或主要因果原因。**两边 noise 大，ρ positive heterogeneity 未分辨；独立 error 可 attenuation population correlation 并增大 covariance uncertainty。Low latent signal / N=36 sampling noise 仍与数据兼容。不能把 weak covariance 的负点估计用 attenuation formula“救回”。
4. **下一次最小动作是什么？先将 36/30/10 的 continuation 目标、固定 roster、独立追加 A streams、与 immutable Phase 5 primary 的分离、variance/covariance uncertainty procedure 及 resource budget 冻结成独立方案。**若随后获得新增计算授权，最小已比较执行动作是对全部 36 states 各增加 15 个 A pairs，保持 B=10。原 Phase 5 estimands、outputs 和 claims 保留不动；不按 outcomes 选 state，也不自动决定再扩样本。

## K. Stop / reproducibility

本轮结束，不启动新实验。已有工作区改动保留；本次只新增此目录的报告、提取脚本与 evidence/CSV。

来源：

- [Frozen primary results](../../RESULTS_PHASE5.md)
- Frozen EIV protocol (internal archive; excluded)
- [Original EIV results](EIV_RESULTS.md)
- Original complete precision grid (internal archive; excluded)
- Verified evidence and source hashes (internal archive; excluded)
- Eight-design comparison (internal archive; excluded)
- All displayed-design cells (internal archive; excluded)
- Labelled analytic reference benchmark (internal archive; excluded)
- Read-only extraction script (internal archive; excluded)

脚本仅有 `load_inputs()` 验证、source hashing、ledger identity 汇总、既有表格汇总和闭式公式；不调用 formal analysis、fit、bootstrap、simulation 或 training。输出使用 exclusive creation，拒绝覆盖。复现时应在脚本副本中把 OUT 指向新的空目录，再从 repo root 以 `PYTHONPATH=. python3 <script-copy>` 运行，并保留 ROOT 为本 repository 路径。完整文件 identities 保存在 evidence JSON；本次没有新的 calibration 或独立科学 audit。
