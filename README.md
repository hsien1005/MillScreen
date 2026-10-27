# MillScreen: Paper Mills Are Not Outliers
NYCU Data Mining 2026, HW2 (Anomaly Detection).


- 主要發現：
  - **論文工廠不是離群點**，而是彼此很像的密集群聚，所以 one-class 方法全部接近隨機。
  - 改用「離正常遠、離已知造假近」的半監督 kNN（**MillScreen**）才有效，不用訓練，表現和同一個 embedding 上的監督式 logistic regression 打平。
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

測試過的環境（套件版本相同，都照 `requirements.txt`）：
- Windows 11、Python 3.12、NVIDIA RTX 4050 Laptop（6 GB VRAM）、CUDA 12.1
- Ubuntu 22.04、Python 3.12、NVIDIA RTX 6000 Ada（48 GB）、CUDA 12.1 wheel。**`results/` 目前的檔案是這台跑的。**

```bash
git clone <repo URL> MillScreen && cd MillScreen
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

- 需要 GPU 的步驟：04（編碼）、06b（SciBERT fine-tune）、09（分段編碼）。其他步驟用 CPU 就能跑，但 kNN 在 GPU 上快很多。
- SciBERT 在不同 GPU 上的分數不會逐位元相同（例如 test PR-AUC 0.117 vs 0.119），差距在 seed 之間的變異內；其他方法的結果完全一致。
- Windows 的終端機如果遇到 `UnicodeEncodeError: 'cp950'`，先設定 `PYTHONIOENCODING=utf-8`（PowerShell：`$env:PYTHONIOENCODING="utf-8"`）。
- 以下指令都在 repo 根目錄下執行。

---

## 資料

| 來源 | 用途 | 取得方式 |
|---|---|---|
| [Retraction Watch Database](https://gitlab.com/crossref/retraction-watch-data)（Crossref） | 異常標籤（撤稿原因含 Paper Mill） | 下載 `retraction_watch.csv`，放到 `data/raw/retraction_watch.csv`。我們用 2026-10-22 的版本：`https://gitlab.com/crossref/retraction-watch-data/-/raw/13d54397/retraction_watch.csv` |
| [OpenAlex API](https://openalex.org)（CC0） | 摘要、正常論文抽樣、出版社 / 國家 / topic | 腳本會自動抓，不需要帳號 |

⚠ OpenAlex 的資料和使用限制：
- **不帶 API key 時每天只有 1,000 個請求**（2026-10 起），超過會回 429，隔天才重置。步驟 1 約 114 個請求、步驟 2 約 382 個、步驟 3b 約 600 個，所以建議到 [openalex.org/signup](https://openalex.org/signup) 申請**免費 API key**（每天 10,000 個請求），再設成環境變數：`export OPENALEX_API_KEY=你的key`（PowerShell：`$env:OPENALEX_API_KEY="你的key"`）。key 只放在請求標頭，不會進網址或快取檔名；不要把 key commit 進 repo。每個回應都存在 `data/cache/`，重跑不會再花額度。
- **OpenAlex 一直在更新**，所以步驟 2 用同一個 seed 重新抽樣，抽到的正常論文已經不完全一樣（2026-10-25 重抽得到 59,944 篇，不是 59,898 篇）。要重現 `results/`，請用**步驟 3b**：它照 `data/processed/normals_openalex_ids.csv` 記錄的 OpenAlex 作品 ID，把切分檔裡同一批正常論文抓回來，切分檔保持不變。用 ID 而不用 DOI，是因為有些 DOI 在 OpenAlex 對應到好幾筆作品（例如整本會議摘要集共用一個 DOI），有些 DOI 還含有 `|`、`,` 這類會破壞查詢條件的字元。
- 少數論文的內容也會變：重抓時有 75 篇正常論文的摘要已被 OpenAlex 移除（變成空的），另有十幾篇論文工廠的標題新加上了撤稿標記。這只讓 EDA 的數字在小數第 3 位後有差異。

---

## 重跑步驟

**重現 `results/`（建議）**：0 → 1 → 3b → 4 → … → 9。**從頭重建資料集**：0 → 1 → 2 → 3 → 4 → … → 9（正常樣本會跟我們的不同）。

| 步驟 | 指令 | 做什麼 | 輸出 | 時間（RTX 4050） |
|---|---|---|---|---|
| 0 | `python scripts/00_rw_overview.py` | 描述兩波論文工廠（只讀 Retraction Watch CSV，不呼叫 API）：篇數、Hindawi / 中國佔比、學科、電腦生成內容、撤稿時間 | `results/rw_waves.csv`、`rw_waves_subject_labels.csv` | < 1 分鐘 |
| 1 | `python scripts/01_build_anomalies.py` | 篩出 paper-mill 撤稿，到 OpenAlex 抓摘要 | `data/processed/anomalies.pkl`、`anomalies_log.csv` | 約 5 分鐘（114 個請求） |
| 2 | `python scripts/02_build_normals.py` | 依每年 paper mill 的領域比例，抽同年份、同領域的正常論文 | `normals.pkl`、`normals_log.csv`、`normals_plan.csv` | 約 20 分鐘（382 個請求） |
| 3 | `python scripts/03_make_splits.py` | 合併、**清除標題裡的撤稿標記**、產生 5 seeds 的時間切分和隨機切分 | `docs.pkl`、`splits/*.json`、`split_summary.csv` | < 1 分鐘 |
| 3b | `python scripts/03b_rebuild_from_splits.py` | **取代 2 + 3**：照 `data/processed/normals_openalex_ids.csv` 記錄的 OpenAlex 作品 ID，抓回切分檔裡同一批正常論文，清除撤稿標記（加 `--keep-markers` 則保留，用來重現 `results/leaky_titles/`） | `normals.pkl`、`docs.pkl` | 約 15 分鐘（約 600 個請求） |
| 4 | `python scripts/04_embed.py minilm`<br>`python scripts/04_embed.py specter2` | 用凍結的預訓練模型把標題 + 摘要編碼 | `emb_minilm*.npy`、`emb_specter2*.npy` | 約 2 分鐘 + 5 分鐘 |
| 5 | `python scripts/05_eda.py minilm` | EDA（只用 train）：特徵分離度、漂移、密度 | `results/eda_*.csv`、`figures/eda_*` | 約 3 分鐘（t-SNE） |
| 6a | `python scripts/06a_tier12.py` | Tier 1（tortured phrase、type-token ratio、always normal）、Tier 2（MillScreen、one-class kNN）、同 embedding 的 Tier 3 對照（logistic regression，C 在 val 上選）；**只在 val 上**選 encoder、k、C | `results/scores/*.npz`、`t2_selection_val.csv`、`t2_config.json` | 約 2 分鐘 |
| 6b | 見下方 | Tier 3：SciBERT fine-tune | `results/scores/scibert_*.npz` | **每次約 7.5 分鐘** |
| 6c | `python scripts/06c_main_table.py 2e-05` | 主結果表；MillScreen 對每個方法的顯著性檢定（paired t + 期刊 cluster bootstrap）；審稿量 1 / 2 / 5% 敏感度 | `main_table.csv`、`main_per_seed.csv`、`main_significance.csv`、`budget_sensitivity.csv` | 約 3 分鐘 |
| 7 | `python scripts/07_ablations.py 2e-05` | Ablation A1–A6（A6 也用同一批已知造假訓練 LR）、test 異常比例敏感度 R | `ablations.csv`、`ablations_per_seed.csv` | 約 3 分鐘 |
| 8 | `python scripts/08_analysis.py 2e-05` | 錯誤案例、FN / FP 共同模式、句子 leave-one-out interpretability、實際影響、Hindawi / 非 Hindawi 子集與其顯著性 | `error_*.csv`、`interpret_loo.csv`、`impact_per_seed.csv`、`slice_significance.csv`、`analysis_report.json` | 約 2 分鐘 |
| 8b | `python scripts/08b_topic_check.py 2e-05` | 混淆檢查：同一個 OpenAlex topic 內部的 ROC-AUC | `topic_check_per_seed.csv` | < 1 分鐘（讀快取） |
| 9 | `python scripts/09_fix.py` | 修正：長摘要分段編碼，只在 val 上選合併方式，test 只跑一次；另外在相同送審比例下比較召回率 | `fix_per_seed.csv`、`fix_equal_rate.csv`、`scores/millscreen_chunkmax_*.npz` | 約 5 分鐘 |
| 10 | `python scripts/10_wording_check.py 2e-05` | SciBERT 為什麼贏：同樣的切分上，用標題 + 摘要的 word 1–3-gram（TF-IDF）做 logistic regression，並和只用前兩句、只用其餘部分比較 | `wording_check_per_seed.csv`、`wording_check_significance.csv`、`wording_check_top_ngrams.csv` | 約 5 分鐘 |
| 11 | `python scripts/11_flag_confidence.py` | 旗標可信度：把 MillScreen 的旗標依 s(x)−τ、依最像的已知造假相似度各分三等份，看各組的 precision；並報告 test 實際送審比例 | `flag_confidence.csv`、`flag_confidence_summary.json` | 約 10 分鐘 |

步驟 6b（SciBERT）：學習率在 val 上選定為 2e-5，正式結果都用這個值。

```bash
# 選學習率（只看 val，不會算 test；結果寫到 results/scibert_tune_val.csv）
python scripts/06b_scibert.py temporal 0 2e-5 --tune
python scripts/06b_scibert.py temporal 0 5e-5 --tune

# 正式訓練：temporal 5 seeds（主結果）+ random 5 seeds（只給 A1 用）
for s in 0 1 2 3 4; do python scripts/06b_scibert.py temporal $s 2e-5; done
for s in 0 1 2 3 4; do python scripts/06b_scibert.py random   $s 2e-5; done
```

一次跑完全部（約 1.5 小時，其中 SciBERT 約 1 小時）：

```bash
export PYTHONIOENCODING=utf-8
python scripts/00_rw_overview.py
python scripts/01_build_anomalies.py && python scripts/03b_rebuild_from_splits.py
python scripts/04_embed.py minilm && python scripts/04_embed.py specter2
python scripts/05_eda.py minilm && python scripts/06a_tier12.py
for s in 0 1 2 3 4; do python scripts/06b_scibert.py temporal $s 2e-5; done
for s in 0 1 2 3 4; do python scripts/06b_scibert.py random   $s 2e-5; done
for x in 06c_main_table 07_ablations 08_analysis 08b_topic_check; do python scripts/$x.py 2e-05; done
python scripts/09_fix.py
python scripts/10_wording_check.py 2e-05
python scripts/11_flag_confidence.py
```

重現 `results/leaky_titles/`（標題保留撤稿標記，其他設定完全相同，SciBERT 一樣用 2e-5）：在另一份 repo 副本裡跑 `03b_rebuild_from_splits.py --keep-markers`，接著跑 4、5、6a、6b（temporal 5 seeds、random 5 seeds）、6c、7、8、8b，再把產生的表格和 SciBERT 分數複製到 `results/leaky_titles/`。

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
- 正常：抽樣 71,004 → 去重複 62,510（排除任何 Retraction Watch 紀錄）→ 摘要 ≥ 30 字 **59,898 篇**（2026-10-24 抽樣；記錄在 `splits/*.json`）

### 主結果
| 方法 | PR-AUC | Recall@τ | ROC-AUC | ms / 篇 | 訓練（秒） |
|---|---|---|---|---|---|
| T1 Always normal | 0.010 | 0 | 0.500 | 0 | 0 |
| T1 Tortured phrases | 0.010 | 0 | 0.500 | 0.15 | 0 |
| T1 Type-token ratio | 0.021 ± 0.001 | 0.058 ± 0.007 | 0.704 ± 0.007 | 0.01 | 0 |
| T2 變體：one-class kNN（只用正常） | 0.012 ± 0.001 | 0.016 ± 0.006 | 0.577 ± 0.015 | 0.24 | 0 |
| **T2 MillScreen（半監督 kNN）** | **0.050 ± 0.008** | **0.104 ± 0.028** | **0.761 ± 0.011** | 0.24 | 0 |
| T3 LR（同一個 MiniLM embedding） | 0.047 ± 0.008 | 0.126 ± 0.023 | 0.780 ± 0.012 | 0.24 | 0.1 |
| T3 SciBERT（監督式） | 0.119 ± 0.021 | 0.234 ± 0.017 | 0.878 ± 0.013 | 0.70 | 76 |

（ms / 篇和訓練時間在 RTX 6000 Ada 上測；MillScreen、one-class 和 LR 的時間幾乎都花在 MiniLM 編碼。）

- MillScreen vs SciBERT：PR-AUC 差 −0.069，p = 0.0004；cluster bootstrap 95% CI [−0.120, −0.034]
- MillScreen vs LR（同 embedding）：PR-AUC 差 +0.003，p = 0.21，CI [−0.014, +0.024]，**打平**；但 Recall@τ 是 LR 較高（0.126 vs 0.104，p = 0.04）
- MillScreen vs type-token ratio：PR-AUC 差 +0.029，p = 0.002；vs Tortured：+0.040，p = 0.0004

### 審稿量敏感度（Recall@τ，τ 設在 val 分數的第 99 / 98 / 95 百分位）
| 方法 | 1% | 2% | 5% |
|---|---|---|---|
| Tortured phrases | 0 | 0 | 0 |
| Type-token ratio | 0.021 | 0.058 | 0.163 |
| One-class kNN | 0.004 | 0.016 | 0.054 |
| MillScreen | 0.057 | 0.104 | 0.230 |
| LR（同 embedding） | 0.086 | 0.126 | 0.272 |
| SciBERT | 0.120 | 0.234 | 0.500 |

### 標籤洩漏（修正前 → 修正後，test PR-AUC）
| 方法 | 時間切 | 隨機切 |
|---|---|---|
| MillScreen | 0.049 → 0.050 | 0.182 → 0.181 |
| LR（同 embedding） | 0.051 → 0.047 | |
| SciBERT | 0.427 → **0.119** | 0.715 → **0.416** |

### Ablation（test PR-AUC）
| # | 改什麼 | 結果 |
|---|---|---|
| A1 | 隨機切 → 時間切 | MillScreen 0.181 → 0.050（−72%）；SciBERT 0.416 → 0.119（−71%） |
| A2 | 異常分數算法 | MillScreen 0.050；只看離已知造假多近 0.030；one-class kNN 0.011；Mahalanobis 0.011；Isolation Forest 0.011 |
| A3 | 正常參考混入 0 / 1 / 3% 造假 | 0.047 / 0.044 / 0.043 |
| A4 | 加入出版社 / 國家（logistic regression） | 時間切 0.047 → 0.145；隨機切 0.161 → 0.478 |
| A5 | τ：審稿量 2% / val best-F1 / train 3σ | Recall 0.104 / 0.096 / 0.002 |
| A6 | 已知造假 25 / 100 / 400 / 1,420 篇 | MillScreen 0.020 / 0.030 / 0.045 / 0.050；LR（同一批造假）0.032 / 0.035 / 0.043 / 0.047 |
| R | test 異常比例 0.5 / 1 / 5% | MillScreen 0.026 / 0.050 / 0.188；SciBERT 0.072 / 0.119 / 0.373；排名不變 |

### Hindawi / 非 Hindawi 子集（`slice_significance.csv`）
- Hindawi 內部（約 430 篇裡 197 篇 paper mill，隨機基準 0.46）：MillScreen 0.508、LR 0.500、SciBERT 0.528，都接近隨機。
- 非 Hindawi（隨機基準 0.0024）：MillScreen 0.042（18 倍）、LR 0.026（p = 0.07，CI 含 0）、SciBERT 0.085。

### SciBERT 為什麼贏（`10_wording_check.py`）
- word 1–3-gram 的 logistic regression：PR-AUC 0.104 ± 0.025，和 SciBERT（0.119）沒有顯著差異（p = 0.08），比 MillScreen（0.050）高（p = 0.002）。
- 所以 SciBERT 的優勢來自用詞：凍結的整篇摘要 embedding 會把這些字詞訊號平均掉。最像 paper mill 的 n-gram 混合了主題詞（apoptosis、intuitionistic fuzzy）和模板用語（this paper、in conclusion、moreover）。
- 只用前兩句 0.059、只用其餘部分 0.067，都比全文低：訊號分散在整篇摘要。

### 旗標可信度（`11_flag_confidence.py`）
- test 實際送審比例 1.04%（val 設定 2%；第二波的分數分布和 2020 年不同）。
- 被 flag 的論文裡，依 s(x)−τ 分三等份的 precision 是 9.2% / 8.6% / 12.4%，依最像的已知造假相似度分是 11.4% / 7.4% / 11.3%：每一份都大約是 1% 基準比例的 10 倍，所以整份旗標清單都值得深審。

### 混淆檢查（同 topic 內 ROC-AUC）
MillScreen 0.761 → 0.672；SciBERT 0.878 → 0.686；one-class 0.577 → 0.526。兩個方法在同主題內都仍高於 0.5（但只有約 28 篇 paper mill 符合條件）。

### 修正：長摘要分段編碼（`09_fix.py`）
47% 的論文超過 MiniLM 的 320 token 上限。切段後每段各自算分數、取最高（val 選出）：
- Recall@τ 0.104 → 0.119（p = 0.034），但實際送審比例也從 1.04% 變成 1.16%。
- **在相同送審比例下比較**（`fix_equal_rate.csv`）：1% 0.106 → 0.102、2% 0.160 → 0.172、5% 0.299 → 0.298，都不顯著（p ≥ 0.19）。PR-AUC 0.050 → 0.050。
- 所以截斷不是 FN 的主因；主因是第二波的模板在第一波沒出現過。

---

## 引用的資料與模型
- Retraction Watch Database（Crossref），2026-10-22 版本，GitLab commit `13d54397`
- OpenAlex（Priem et al., 2022），CC0
- all-MiniLM-L6-v2（Reimers & Gurevych, 2019；Wang et al., 2020）、SPECTER2（Singh et al., 2023）、SciBERT（Beltagy et al., 2019）
- Tortured phrases：Cabanac, Labbé & Magazinov (2021), arXiv:2107.06751，Table 1 的 29 組
- Tier 3 參考：Scancar et al. (2026), *BMJ*, doi:10.1136/bmj-2025-087581

錯誤案例和 interpretability 的原始資料見 `results/error_cases.csv`、`results/interpret_loo.csv`。
