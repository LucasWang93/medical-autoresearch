# Baseline Methods & Results

## Method: GRU + REINFORCE History Attention

所有baseline采用同一个模型：**GRU序列编码器 + REINFORCE策略梯度的历史注意力选择机制**。

这不是一个标准的clinical ML baseline（如LR、XGBoost、LSTM），而是一个**RL-augmented sequential model**，设计目标是让autoresearch agent后续可以自主修改和优化。

### 为什么用RL而不是标准的supervised model

传统clinical prediction pipeline是纯监督学习：特征提取 → 分类器 → loss。我们在此基础上加入了一个RL组件，让模型**学习选择关注哪些历史visit**，而不是简单地用最后一个hidden state或mean pooling。这为后续autoresearch提供了更多可优化的维度（策略网络、reward shaping、探索策略等）。

### 方法详解

#### 1. 输入表示：Medical Code Embedding

每个患者的每次就诊（visit/admission）包含一组医疗编码：
- **Conditions**: ICD诊断码（MIMIC-IV取top 500，SUPPORT2取离散化的临床特征）
- **Procedures**: ICD手术码（MIMIC-IV取top 200，SUPPORT2取人口统计学特征）
- **Drugs** (仅drug_rec任务): 处方药物名（top 300）

每个编码通过 `nn.Embedding(dim=128)` 映射为向量，同一次visit内的所有编码**求和池化**（sum pooling）得到visit-level表示。

```
visit_codes = [ICD-401.1, ICD-250.00, ICD-272.4]
              → embed each → sum → visit_embedding ∈ R^128
```

#### 2. 序列编码：GRU

多次visit的embedding序列输入单层GRU（hidden_dim=128），建模患者的纵向就诊历史：

```
[visit_1, visit_2, ..., visit_T] → GRU → [h_1, h_2, ..., h_T]
```

#### 3. RL历史选择：REINFORCE PolicyAgent

核心创新点。在每个时间步t，模型不是简单取h_t作为最终表示，而是：

1. **观察**当前状态 h_t
2. **PolicyAgent**（2层MLP）输出在过去N=10个visit中选择哪一个的概率分布
3. **采样**一个action（选中某个历史visit的hidden state h_k）
4. **融合**当前状态与选中的历史状态：`fusion([h_t; h_k])` → 128维

训练方式：
- **Policy gradient**: REINFORCE算法，reward = 预测是否正确（binary/multiclass）或Jaccard similarity（multilabel）
- **Baseline**: 学习的value network用于方差减小（advantage = reward - V(s)）
- **Entropy bonus**: 鼓励探索（coef=0.01）
- **Discount**: γ=0.95，奖励从最终预测反向传播到每个时间步的action

```python
# 伪代码
for t in range(1, T):
    action = PolicyAgent.sample(h_t)           # 选择关注哪个历史visit
    h_selected = history[action]                # 取出选中的hidden state
    fused = MLP([h_t; h_selected])             # 融合当前+历史
    
# RL loss
advantage = reward - baseline(h_t)
policy_loss = -log_prob(action) * advantage     # REINFORCE
```

#### 4. 输出头：Task-Specific Prediction

融合后的最后一个有效时间步的表示送入线性层：
- **Binary** (mortality, readmission, survival): Linear(128→2) + CrossEntropy
- **Multiclass** (dzclass, los): Linear(128→K) + CrossEntropy
- **Multilabel** (drugrec): Linear(128→300) + BCEWithLogits

#### 5. 总损失

```
L_total = L_task (CE/BCE) + L_rl (REINFORCE policy gradient)
```

两个loss联合优化，task loss驱动预测准确性，RL loss驱动历史选择策略。

### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| embedding_dim | 128 | 医疗编码嵌入维度 |
| hidden_dim | 128 | GRU隐藏层维度 |
| n_rnn_layers | 1 | GRU层数 |
| dropout | 0.3 | 全局dropout |
| n_actions | 10 | RL agent的历史窗口大小 |
| gamma | 0.95 | RL折扣因子 |
| entropy_coef | 0.01 | 熵正则化系数 |
| lr | 1e-3 | Adam学习率 |
| weight_decay | 1e-5 | L2正则 |
| batch_size | 32 | 批大小 |
| time_budget | 120s | 训练时间预算 |

### 参数量

- Binary/multiclass任务 (2个feature key): ~289K
- Multilabel drug_rec任务 (3个feature key + 大输出): ~440K

---

## 数据处理

### MIMIC-IV 3.1

- **来源**: Beth Israel Deaconess Medical Center真实EHR数据
- **患者**: 223,452名（>=2次入院的患者）
- **样本**: 322,576个（每个非首次入院=1个样本）
- **特征构建**: 每次入院(hadm_id)=一个visit，visit内包含该次入院的所有ICD诊断码和手术码
- **词表**: Top 500 ICD诊断 + Top 200 ICD手术 + Top 300 药物（drug_rec）
- **读取方式**: 直接用pandas读CSV.gz，不依赖PyHealth

### SUPPORT2

- **来源**: 5家美国医院的9,105名重症患者（公开数据集）
- **特征构建**: 表格数据，数值特征（生命体征、实验室检查、评分）做quantile binning（10 bins），类别特征做one-hot编码，特征组映射为"pseudo-visits"以适配GRU pipeline
- **注意**: 已修复信息泄露（dzclass任务排除dzgroup/dzclass，survival任务排除sps/surv/prg）

### Synthetic EHR

- **生成方式**: 10个疾病archetype，每个archetype有特定的diagnosis→drug映射
- **用途**: 开发调试，不需要真实数据权限

---

## Baseline Results

### MIMIC-IV (223K patients)

| Task | 任务描述 | Type | Metric | Baseline | 
|------|---------|------|--------|----------|
| `mimic4_mortality` | 院内死亡率预测 | binary | AUROC | **0.961** |
| `mimic4_readmission` | 30天再入院预测 | binary | AUROC | **0.669** |
| `mimic4_los` | 住院时长分类 (<3d/3-7d/7-14d/>14d) | 4-class | F1_macro | **0.481** |
| `mimic4_drugrec` | 药物推荐 (300种药物) | multilabel | Jaccard | **0.166** |
| `mimic4_phenotyping` | 下次入院 25 类 Harutyunyan 表型预测 | multilabel | AUROC_macro | **0.816** |

### SUPPORT2 (9K patients)

| Task | 任务描述 | Type | Metric | Baseline |
|------|---------|------|--------|----------|
| `support2_mortality` | 院内死亡率预测 | binary | AUROC | **0.916** |
| `support2_dzclass` | 疾病分类 (4类) | 4-class | F1_macro | **0.779** |
| `support2_survival` | 2个月生存预测 | binary | AUROC | **0.936** |

### Synthetic (2K patients)

| Task | 任务描述 | Type | Metric | Baseline |
|------|---------|------|--------|----------|
| `drug_recommendation` | 合成数据药物推荐 | multilabel | Jaccard | **0.654** |

---

## 结果分析

### 为什么mortality AUROC这么高 (0.961)

MIMIC-IV的diagnosis codes是**出院时编码**（discharge diagnoses），包含了整个住院过程中发生的所有诊断（如cardiac arrest）。用这些编码来预测院内死亡存在一定程度的circularity。这是clinical ML benchmark的标准做法（PyHealth、MIMIC-Extract等都是如此），但在实际临床部署时应使用入院时（admission-time）诊断。

### 为什么readmission只有0.669

30天再入院受很多非临床因素影响（保险、社会支持、患者依从性等），仅靠ICD码很难预测。文献中MIMIC上readmission的AUROC通常在0.65-0.75之间，我们的baseline在合理范围内。

### 为什么drugrec只有0.166

Drug recommendation是300类的多标签问题，Jaccard similarity要求预测的药物集合与真实处方高度重叠。0.166意味着平均约16.6%的交集率。这个任务的state-of-the-art（如GAMENet、SafeDrug）在MIMIC-III上约0.45-0.52，但它们使用了完整的DDI矩阵、drug molecule graph等额外信息。

### 与文献对比

| Task | 我们的Baseline | 文献参考范围 | 说明 |
|------|--------------|-------------|------|
| Mortality (MIMIC) | 0.961 | 0.85-0.96 | 偏高，discharge diagnosis优势 |
| Readmission (MIMIC) | 0.669 | 0.65-0.75 | 合理 |
| LOS (MIMIC) | 0.481 | 0.40-0.60 | 合理 |
| Drug Rec (MIMIC) | 0.166 | 0.45-0.52* | 低，需要更多特征和更好的架构 |

*Drug rec文献数字基于MIMIC-III + DDI + molecular features，非直接可比。

### Phenotyping 任务说明

`mimic4_phenotyping` 复刻 Harutyunyan et al. (2019) 的 25 类急性期表型任务,原任务是在 MIMIC-III 上基于 HCUP CCS 分组定义的。我们做了两处适配:

1. **平台迁移**:MIMIC-IV 混用 ICD-9 和 ICD-10,我们用 `icd_version` 列消歧后应用前缀规则近似 CCS 映射(未使用 HCUP 官方映射文件),因此与 MIMIC-III 文献数值**不可直接比较**,但任务语义与评估协议与原论文一致。
2. **时序设定**:原任务是 ICU 单次入住末的表型分类,容易与输入特征泄漏。我们改为**下次入院的 25 类表型预测**——特征只用 visit[0..t-1],标签是 visit t 的全部诊断码映射得到的 25 类多标签向量。这是一个非平凡的纵向预测任务。

Harutyunyan 在 MIMIC-III 报告 macro-AUROC ≈ 0.77 (LSTM);MIMIC-IV 上的数值要首次跑完才能得出。

---

## 改进方向

按提升潜力排序：

1. **mimic4_drugrec (0.166 → target 0.40+)**: 加入DDI矩阵约束、drug molecule embedding、transformer替代GRU、更长训练时间
2. **mimic4_los (0.481 → target 0.55+)**: ordinal regression（LOS是有序的）、加入admission_type和demographic特征
3. **mimic4_readmission (0.669 → target 0.72+)**: time-aware encoding（入院间隔）、加入lab/vital特征
4. **support2_dzclass (0.779 → target 0.85+)**: 特征交互、class-balanced sampling
5. **mimic4_mortality (0.961)**: 空间有限，可关注calibration和prospective-only setting

---

## 复现

```bash
# MIMIC-IV (需要本地MIMIC-IV 3.1数据)
for task in mimic4_mortality mimic4_readmission mimic4_los mimic4_drugrec mimic4_phenotyping; do
  sbatch --partition=gpu --time=00:20:00 scripts/run.sh --task $task --time-budget 120
done

# SUPPORT2 (自动从HuggingFace下载)
for task in support2_mortality support2_dzclass support2_survival; do
  sbatch --partition=gpu_devel --time=00:10:00 scripts/run.sh --task $task --time-budget 120
done

# Synthetic
sbatch scripts/run.sh --task drug_recommendation --time-budget 60 --n-patients 2000
```

## 数据泄露记录

| 任务 | 泄露原因 | 泄露前 | 修复后 | 修复方式 |
|------|---------|--------|--------|---------|
| support2_dzclass | dzgroup/dzclass在输入特征中 | 1.000 | 0.779 | `_SUPPORT2_EXCLUDE`排除 |
| support2_survival | sps/surv/prg在输入特征中 | 0.975 | 0.936 | `_SUPPORT2_EXCLUDE`排除 |
| mimic4_mortality | 出院诊断含临终事件 | — | 0.961 | 已知局限，标准benchmark做法 |
