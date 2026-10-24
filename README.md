# MillScreen: Paper Mills Are Not Outliers
NYCU Data Mining 2026, HW2 (Anomaly Detection).


- 主要發現：
  - **論文工廠不是離群點**，而是彼此很像的密集群聚，所以 one-class 方法全部接近隨機。
  - 改用「離正常遠、離已知造假近」的半監督 kNN（**MillScreen**）才有效。
  - 依出版年份切分後，**所有方法的 PR-AUC 都掉了約 70%**。
  - 撤稿資料庫的標題帶有「RETRACTED」標記，會造成**標籤洩漏**，讓監督式模型嚴重高估。

---

## 目錄結構

```
MillScreen/
├── README.md
├── requirements.txt
├── src/
│   ├── common.py        # 路徑、OpenAlex 抓取（有快取）、摘要重建、tortured phrase、標題清理
│   └── methods.py       # 切分載入、kNN / MillScreen / Mahalanobis / IForest、τ、metrics
├── scripts/             # 00 → 09 依序執行
├── data/
│   ├── raw/             # Retraction Watch CSV（不進 git，自己下載）
│   ├── cache/           # OpenAlex API 回應快取（不進 git，約 710 MB）
│   └── processed/
│       ├── splits/      # 進 git：5 seeds × {temporal, random} 的 DOI 切分
│       ├── *_log.csv    # 進 git：資料清理每一步剩幾筆
│       └── *.pkl / *.npy / *.npz   # 不進 git，可重建
├── results/             # 所有表格（csv / json）
│   ├── scores/          # 每個方法、每個 seed 在 val / test 的分數
│   └── leaky_titles/    # 修正標籤洩漏「之前」的結果（對照用）
└── figures/             # EDA 圖（跑 05_eda.py 產生）
```

---

## 環境

測試環境：Windows 11、Python 3.12、NVIDIA RTX 4050 Laptop（6 GB VRAM）、CUDA 12.1。

```bash
git clone <repo URL> MillScreen && cd MillScreen
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

- 需要 GPU 的步驟：04（編碼）、06b（SciBERT fine-tune）、09（分段編碼）。其他步驟用 CPU 就能跑，但 kNN 在 GPU 上快很多。
- Windows 的終端機如果遇到 `UnicodeEncodeError: 'cp950'`，先設定 `PYTHONIOENCODING=utf-8`（PowerShell：`$env:PYTHONIOENCODING="utf-8"`）。
- 以下指令都在 repo 根目錄下執行。

---

## 資料

| 來源 | 用途 | 取得方式 |
|---|---|---|
| [Retraction Watch Database](https://gitlab.com/crossref/retraction-watch-data)（Crossref） | 異常標籤（撤稿原因含 Paper Mill） | 下載 `retraction_watch.csv`，放到 `data/raw/retraction_watch.csv`。 |
| [OpenAlex API](https://openalex.org)（CC0） | 摘要、正常論文抽樣、出版社 / 國家 / topic | 腳本會自動抓，不需要帳號 |

---

## 重跑步驟

| 步驟 | 指令 | 做什麼 | 輸出 | 時間（RTX 4050） |
|---|---|---|---|---|
| 0 | `python scripts/00_rw_overview.py` | 描述兩波論文工廠（只讀 Retraction Watch CSV，不呼叫 API）：篇數、Hindawi / 中國佔比、學科、電腦生成內容、撤稿時間 | `results/rw_waves.csv`、`rw_waves_subject_labels.csv` | < 1 分鐘 |
| 1 | `python scripts/01_build_anomalies.py` | 篩出 paper-mill 撤稿，到 OpenAlex 抓摘要 | `data/processed/anomalies.pkl`、`anomalies_log.csv` | 約 5 分鐘（114 個請求） |
| 2 | `python scripts/02_build_normals.py` | 依每年 paper mill 的領域比例，抽同年份、同領域的正常論文 | `normals.pkl`、`normals_log.csv`、`normals_plan.csv` | 約 20 分鐘（382 個請求） |
| 3 | `python scripts/03_make_splits.py` | 合併、**清除標題裡的撤稿標記**、產生 5 seeds 的時間切分和隨機切分 | `docs.pkl`、`splits/*.json`、`split_summary.csv` | < 1 分鐘 |
| 4 | `python scripts/04_embed.py minilm`<br>`python scripts/04_embed.py specter2` | 用凍結的預訓練模型把標題 + 摘要編碼 | `emb_minilm*.npy`、`emb_specter2*.npy` | 約 2 分鐘 + 5 分鐘 |
| 5 | `python scripts/05_eda.py minilm` | EDA（只用 train）：特徵分離度、漂移、密度 | `results/eda_*.csv`、`figures/eda_*` | 約 3 分鐘（t-SNE） |
| 6a | `python scripts/06a_tier12.py` | Tier 1（tortured phrase、always normal）、Tier 2（MillScreen、one-class kNN）；**只在 val 上**選 encoder 和 k | `results/scores/*.npz`、`t2_selection_val.csv`、`t2_config.json` | 約 2 分鐘 |
| 6b | 見下方 | Tier 3：SciBERT fine-tune | `results/scores/scibert_*.npz` | **每次約 7.5 分鐘** |
| 6c | `python scripts/06c_main_table.py 2e-05` | 主結果表、顯著性檢定（paired t + 期刊 cluster bootstrap）、審稿量 1 / 2 / 5% 敏感度 | `main_table.csv`、`main_per_seed.csv`、`main_significance.csv`、`budget_sensitivity.csv` | 約 3 分鐘 |
| 7 | `python scripts/07_ablations.py 2e-05` | Ablation A1–A6、test 異常比例敏感度 R | `ablations.csv`、`ablations_per_seed.csv` | 約 3 分鐘 |
| 8 | `python scripts/08_analysis.py 2e-05` | 錯誤案例、FN / FP 共同模式、句子 leave-one-out interpretability、實際影響、Hindawi 子集 | `error_*.csv`、`interpret_loo.csv`、`impact_per_seed.csv`、`analysis_report.json` | 約 2 分鐘 |
| 8b | `python scripts/08b_topic_check.py 2e-05` | 混淆檢查：同一個 OpenAlex topic 內部的 ROC-AUC | `topic_check_per_seed.csv` | < 1 分鐘（讀快取） |
| 9 | `python scripts/09_fix.py` | 修正：長摘要分段編碼，只在 val 上選合併方式，test 只跑一次 | `fix_per_seed.csv`、`scores/millscreen_chunkmax_*.npz` | 約 5 分鐘 |

步驟 6b（SciBERT）：

```bash
# 選學習率（只看 val，不會算 test）
python scripts/06b_scibert.py temporal 0 2e-5 --tune
python scripts/06b_scibert.py temporal 0 5e-5 --tune

# 正式訓練：temporal 5 seeds（主結果）+ random 3 seeds（只給 A1 用）
for s in 0 1 2 3 4; do python scripts/06b_scibert.py temporal $s 2e-5; done
for s in 0 1 2;     do python scripts/06b_scibert.py random   $s 2e-5; done
```

一次跑完全部（約 1.5 小時，其中 SciBERT 約 1 小時）：

```bash
export PYTHONIOENCODING=utf-8
python scripts/00_rw_overview.py
python scripts/01_build_anomalies.py && python scripts/02_build_normals.py && python scripts/03_make_splits.py
python scripts/04_embed.py minilm && python scripts/04_embed.py specter2
python scripts/05_eda.py minilm && python scripts/06a_tier12.py
for s in 0 1 2 3 4; do python scripts/06b_scibert.py temporal $s 2e-5; done
for s in 0 1 2;     do python scripts/06b_scibert.py random   $s 2e-5; done
for x in 06c_main_table 07_ablations 08_analysis 08b_topic_check; do python scripts/$x.py 2e-05; done
python scripts/09_fix.py
```

`results/` 裡已經有所有表格和每個 seed 的分數，可以直接用。

---

## 實驗設定

- **切分**（依出版年份，模擬「用過去偵測未來」）：

  | Split | 出版年 | 正常 | 異常 | 用途 |
  |---|---|---|---|---|
  | Train | 2015–2019 | 15,952 | 1,420（已知造假參考 / Tier 3 訓練） | 建參考集合 |
  | Val | 2020 | 6,564 | 66（1%） | 只用來選超參數和定 τ |
  | Test | 2021–2023 | 25,402 | 257（1%） | 只跑一次 |

  每個 seed 從正常樣本池重新抽 80%；三個 Tier 共用同一組 DOI 切分。
- **MillScreen**：$s(x)=d_k(x,\text{正常})-d_k(x,\text{已知造假})$，all-MiniLM-L6-v2，k = 10（val 選出）。
- **τ**：編輯每 1,000 篇深審 20 篇（review budget 2%），τ 設在 val 分數的第 98 百分位。
- **Metrics**：PR-AUC（主要）、Recall@τ、Precision@τ、ROC-AUC、每篇推論時間。
- **顯著性**：5 seeds paired t-test，加上以期刊為單位的 cluster bootstrap 95% CI。

---

## 結果（test，5 seeds 平均 ± 標準差）

### 兩波論文工廠（`00_rw_overview.py`，所有 paper-mill 撤稿）
| | 第一波 2015–19（train） | 2020（val） | 第二波 2021–23（test） |
|---|---|---|---|
| 篇數 | 1,789 | 953 | 8,706 |
| Hindawi 佔比 | 0.6% | 10.7% | 83.6% |
| 生命科學（BLS）標籤佔比 | 64.3% | 32.2% | 12.1% |
| 商業與科技（B/T）標籤佔比 | 12.2% | 30.2% | 34.7% |
| 撤稿原因含電腦生成內容 | 13.2% | 23.7% | 67.4% |
| 出版到撤稿的中位數 | 3.9 年 | 2.5 年 | 1.5 年 |

### 資料規模
- 異常：Retraction Watch 73,346 筆 → 真正撤稿 67,906 → 原因含 Paper Mill 11,714 → 有 DOI 11,662 → 2015–23 年 11,396 → OpenAlex 查得到 11,386 → 摘要 ≥ 30 字 **10,890 篇**
- 正常：抽樣 71,004 → 去重複 62,510（排除任何 Retraction Watch 紀錄）→ 摘要 ≥ 30 字 **59,898 篇**

### 主結果
| 方法 | PR-AUC | Recall@τ | ROC-AUC | ms / 篇 |
|---|---|---|---|---|
| T1 Always normal | 0.010 | 0 | 0.500 | 0 |
| T1 Tortured phrases | 0.010 | 0 | 0.500 | 0.26 |
| T2 變體：one-class kNN（只用正常） | 0.012 ± 0.001 | 0.016 ± 0.006 | 0.577 ± 0.015 | 1.23 |
| **T2 MillScreen（半監督 kNN）** | **0.050 ± 0.008** | **0.104 ± 0.028** | **0.761 ± 0.011** | 1.24 |
| T3 SciBERT（監督式） | 0.117 ± 0.017 | 0.242 ± 0.029 | 0.875 ± 0.012 | 11.4 |

- MillScreen vs SciBERT：PR-AUC 差 −0.067，p = 0.0003；cluster bootstrap 95% CI [−0.113, −0.028]
- MillScreen vs Tortured：PR-AUC 差 +0.040，p = 0.0004

### 審稿量敏感度（Recall@τ，τ 設在 val 分數的第 99 / 98 / 95 百分位）
| 方法 | 1% | 2% | 5% |
|---|---|---|---|
| Tortured phrases | 0 | 0 | 0 |
| One-class kNN | 0.004 | 0.016 | 0.054 |
| MillScreen | 0.057 | 0.104 | 0.230 |
| SciBERT | 0.132 | 0.242 | 0.497 |

### 標籤洩漏（修正前 → 修正後，test PR-AUC）
| 方法 | 時間切 | 隨機切 |
|---|---|---|
| MillScreen | 0.049 → 0.050 | 0.182 → 0.181 |
| SciBERT | 0.422 → **0.117** | 0.718 → **0.427** |

### Ablation（test PR-AUC）
| # | 改什麼 | 結果 |
|---|---|---|
| A1 | 隨機切 → 時間切 | MillScreen 0.181 → 0.050（−72%）；SciBERT 0.427 → 0.117（−73%） |
| A2 | 異常分數算法 | MillScreen 0.050；只看離已知造假多近 0.030；one-class kNN 0.011；Mahalanobis 0.011；Isolation Forest 0.011 |
| A3 | 正常參考混入 0 / 1 / 3% 造假 | 0.047 / 0.044 / 0.043 |
| A4 | 加入出版社 / 國家（logistic regression） | 時間切 0.048 → 0.145；隨機切 0.161 → 0.479 |
| A5 | τ：審稿量 2% / val best-F1 / train 3σ | Recall 0.104 / 0.093 / 0.002 |
| A6 | 已知造假 25 / 100 / 400 / 1,420 篇 | 0.020 / 0.030 / 0.045 / 0.050 |
| R | test 異常比例 0.5 / 1 / 5% | MillScreen 0.026 / 0.050 / 0.188；SciBERT 0.070 / 0.117 / 0.368；排名不變 |

### 混淆檢查（同 topic 內 ROC-AUC）
MillScreen 0.761 → 0.672；SciBERT 0.875 → 0.682；one-class 0.577 → 0.526。兩個方法在同主題內都仍高於 0.5（但只有約 28 篇 paper mill 符合條件）。

### 修正：長摘要分段編碼（`09_fix.py`）
47% 的論文超過 MiniLM 的 320 token 上限。切段後每段各自算分數、取最高（val 選出）：
- Recall@τ 0.104 → **0.119**（p = 0.034）；長論文的 Recall 0.086 → 0.109
- PR-AUC 0.050 → 0.050（p = 0.94），整體排序沒變。截斷只是 FN 的次要原因，主因是第二波的模板在第一波沒出現過。

---

## 引用的資料與模型
- Retraction Watch Database（Crossref），2026-10-22 版本，GitLab commit `13d5439`
- OpenAlex（Priem et al., 2022），CC0
- all-MiniLM-L6-v2（Reimers & Gurevych, 2019；Wang et al., 2020）、SPECTER2（Singh et al., 2023）、SciBERT（Beltagy et al., 2019）
- Tortured phrases：Cabanac, Labbé & Magazinov (2021), arXiv:2107.06751，Table 1 的 29 組
- Tier 3 參考：Scancar et al. (2026), *BMJ*, doi:10.1136/bmj-2025-087581

錯誤案例和 interpretability 的原始資料見 `results/error_cases.csv`、`results/interpret_loo.csv`。
