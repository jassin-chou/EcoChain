# EcoChain — 快速開始 & 協作指南

> 這份 README 的目標：**組員 clone 下來照著做就能跑起來**。
> 本地只需要跑後端 + 前端 HTML，不需要設定 Azure / Firebase。

---

## 目錄

1. [專案架構](#架構總覽)
2. [本地跑起來（5分鐘版）](#本地跑起來5分鐘版)
3. [完整環境設定](#完整環境設定)
4. [Git 分支協作指南](#git-分支協作指南)
5. [部署到 Azure（管理員用）](#部署到-azure管理員用)
6. [檔案說明](#檔案說明)
7. [常見錯誤排除](#常見錯誤排除)

---

## 架構總覽

```
前端 HTML (瀏覽器)
    ↕  HTTP
FastAPI 後端 (本地 port 8000 / Azure 雲端)
    ↕
PyTorch GNN 模型 (ecochain_gnn.pt)   ← 需要先 train.py 產生
    ↕
Firebase Firestore (存檔 / 排行榜)   ← 需要金鑰，本地可跳過
```

**本地開發只需要「後端 + 前端HTML」就能玩遊戲**，Firebase 是選配（沒有金鑰時後端會用降級模式）。

---

## 本地跑起來（5分鐘版）

> 適合組員第一次 clone，只想先看到程式動起來。

### 第一步：clone & 進入資料夾

```bash
git clone https://github.com/你的帳號/ecochain.git
cd ecochain
```

### 第二步：確認 Python 版本

這個專案需要 **Python 3.11**。

```bash
python3 --version   # macOS / Linux
python --version    # Windows
```

如果版本不對，去 https://www.python.org/downloads/ 下載 3.11。

### 第三步：建虛擬環境 & 安裝套件

```bash
# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows（PowerShell）
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
```

成功的話終端機最左邊會出現 `(venv)` 字樣。

安裝套件：

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch Geometric 在某些系統需要額外步驟**（裝完 requirements.txt 後如果 import 失敗再做這步）：
> ```bash
> pip install torch-geometric
> ```

### 第三步：設定環境變數

```bash
cp .env.example .env
```

`.env` 裡已經有預設值，**本地測試直接用不需要改**。  
（如果你有自己的 Firebase 金鑰，把 `serviceAccountKey.json` 放進專案根目錄再改 `FIREBASE_CREDS_PATH`）

### 第四步：訓練模型（第一次必做，之後不用）

```bash
python train.py
```

這會產生 `models/ecochain_gnn.pt`，大約 3–5 分鐘。  
如果你已經從組員那邊拿到 `ecochain_gnn.pt`，可以直接跳過這步，把檔案放在 `models/` 資料夾下。

### 第五步：啟動後端

```bash
uvicorn main:app --reload --port 8000
```

看到這行就代表成功：
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 第六步：打開前端

用瀏覽器**直接開** `ecochain_v6_redesign.html`，  
或者在 VSCode 用 Live Server 外掛開（推薦）。

🎉 **這樣就跑起來了！**

---

## 完整環境設定

> 需要 Firebase 存檔功能的組員才需要做這段。

### Firebase 設定

1. 向專案管理員索取 `serviceAccountKey.json`
2. 把它放在專案根目錄（**不要 git add 這個檔案，.gitignore 已排除**）
3. 確認 `.env` 裡的 `FIREBASE_CREDS_PATH=serviceAccountKey.json`

### 環境變數說明

| 變數 | 說明 | 需要修改嗎 |
|------|------|-----------|
| `FIREBASE_CREDS_PATH` | Firebase 金鑰路徑 | 沒有金鑰就不用動 |
| `TOKEN_SECRET` | Session token 簽章用 | 本地開發不用改 |
| `RETRAIN_SECRET` | `/retrain` 端點密碼 | 本地開發不用改 |
| `ALLOW_ORIGINS` | CORS 允許來源 | 部署才需要設定 |

---

## Git 分支協作指南

這份指南讓你和組員可以**同時開發不同功能，不會互相衝突**。

### 分支策略（我們用這個）

```
main          ← 穩定版本，永遠可以跑，合併前要測試
  └── dev     ← 整合分支，組員的功能合進這裡
        ├── feat/gnn-改進     ← 你的功能分支
        ├── feat/前端UI       ← 組員A的分支
        └── fix/分數bug       ← 組員B的分支
```

**規則：不直接 commit 到 main，功能做好了開 PR 合進 dev，dev 測試沒問題再合進 main。**

---

### 第一次設定（管理員做）

```bash
# 在 GitHub 建立 repo 後
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/你的帳號/ecochain.git
git push -u origin main

# 建立 dev 分支並推上去
git checkout -b dev
git push -u origin dev
```

---

### 組員第一次 clone 後的設定

```bash
# clone 下來
git clone https://github.com/你的帳號/ecochain.git
cd ecochain

# 看目前有哪些分支
git branch -a

# 切到 dev 分支
git checkout dev

# 確認現在在 dev
git branch
# * dev
#   main
```

---

### 日常開發流程（每次開始新功能）

**第一步：確保 dev 是最新的**

```bash
git checkout dev
git pull origin dev
```

**第二步：從 dev 開一個新的功能分支**

```bash
# 命名規則：feat/功能名稱 或 fix/問題名稱
git checkout -b feat/你的功能名稱

# 例如：
git checkout -b feat/gnn-species2vec
git checkout -b fix/leaderboard-bug
git checkout -b feat/前端動畫
```

**第三步：做你的修改，然後存檔**

```bash
# 查看哪些檔案被改了
git status

# 把修改加進暫存區（全部）
git add .

# 或者只加特定檔案
git add gnn.py train.py

# 寫 commit 訊息
git commit -m "feat: 加入 species2vec 嵌入向量"
```

> **Commit 訊息格式建議：**
> - `feat: 新功能說明`
> - `fix: 修了什麼 bug`
> - `refactor: 重構了什麼`
> - `docs: 更新文件`

**第四步：推上 GitHub**

```bash
git push origin feat/你的功能名稱

# 第一次推新分支用 -u
git push -u origin feat/你的功能名稱
```

**第五步：開 Pull Request**

1. 去 GitHub → 你的 repo → 點「Compare & pull request」
2. 設定：`feat/你的功能名稱` → `dev`（不是 main！）
3. 寫說明，tag 組員 review
4. 組員 approve 後，按「Merge pull request」

---

### 同步最新進度（每天開始工作前）

```bash
# 切回自己的功能分支
git checkout feat/你的功能名稱

# 先拉 dev 的最新進度
git fetch origin dev

# 把 dev 的更新合進自己的分支（rebase 讓 log 比較乾淨）
git rebase origin/dev

# 如果有衝突：
# 1. 打開衝突的檔案，手動解決（看 <<<<<<< 和 >>>>>>> 的部分）
# 2. git add 解決好的檔案
# 3. git rebase --continue
```

---

### 處理衝突（有人改了同一個檔案）

衝突看起來像這樣，打開檔案你會看到：

```python
<<<<<<< HEAD（你的版本）
def predict(species_counts):
    # 你改的版本
=======
def predict(species_counts, threshold=0.5):
    # 組員改的版本
>>>>>>> origin/dev
```

解法：手動選擇要保留哪個版本（或兩個都要的話合併起來），然後：

```bash
git add gnn.py
git rebase --continue   # 如果是在 rebase 中
# 或
git commit              # 如果是在 merge 中
```

---

### 常用指令速查

```bash
# 查看目前在哪個分支
git branch

# 查看所有分支（含遠端）
git branch -a

# 切換分支
git checkout 分支名稱

# 建立新分支並切過去
git checkout -b 新分支名稱

# 拉最新進度
git pull origin 分支名稱

# 查看修改了什麼
git status
git diff

# 查看 commit 歷史
git log --oneline --graph

# 刪除本地分支（合併後清理用）
git branch -d feat/功能名稱

# 刪除遠端分支
git push origin --delete feat/功能名稱

# 反悔最後一次 commit（但保留修改）
git reset --soft HEAD~1

# 反悔所有未 commit 的修改（危險！改過的都會消失）
git restore .
```

---

### 緊急狀況處理

**誤 commit 進 main 了：**
```bash
# 找到上一個好的 commit hash
git log --oneline

# 回到那個點（--soft 保留檔案修改）
git reset --soft abc1234
git push origin main --force  # 要先跟組員說！
```

**想把別人某個 commit 搬過來：**
```bash
git cherry-pick commit的hash值
```

**想暫時存放目前的修改（還沒做完不想 commit）：**
```bash
git stash          # 暫存
git stash pop      # 取回
```

---

## 部署到 Azure（管理員用）

### GitHub Secrets 設定
GitHub repo → Settings → Secrets → Actions → New repository secret

| Secret | 值 |
|--------|-----|
| `AZURE_WEBAPP_PUBLISH_PROFILE` | Azure App Service → Get publish profile |

### Azure 環境變數
Azure App Service → Settings → Environment variables

| 變數 | 說明 |
|------|------|
| `FIREBASE_CREDS_JSON` | `serviceAccountKey.json` 的完整 JSON 字串（整個貼上去） |
| `TOKEN_SECRET` | 執行 `python -c "import secrets; print(secrets.token_hex(32))"` 產生 |
| `RETRAIN_SECRET` | 自訂密碼 |
| `ALLOW_ORIGINS` | 前端網址 |
| `ENV` | `production` |

### 推送觸發自動部署

```bash
git checkout develop
git push origin develop
# → GitHub Actions 自動部署到 Azure Web App
```

---

## 檔案說明

| 檔案 | 說明 | 要改嗎 |
|------|------|--------|
| `main.py` | FastAPI 後端主程式 | 加 API 端點時改 |
| `gnn.py` | GNN 模型定義 | 改模型架構時改 |
| `train.py` | GNN 訓練腳本 | 改訓練邏輯時改 |
| `firebase_db.py` | Firebase 資料庫操作 | 改資料庫結構時改 |
| `species2vec.py` | Species2Vec 預訓練腳本 | 不用動 |
| `fetch_mangal.py` | 下載真實食物網資料 | 不用動 |
| `holdout_test.py` | 模型評估腳本 | 不用動 |
| `ecochain_v6_redesign.html` | 前端遊戲 | 改 UI 時改 |
| `real_food_webs.json` | 真實食物網資料集 | 不用動 |
| `requirements.txt` | Python 套件清單 | 加套件時更新 |
| `_env` | `.env` 範本 | **複製成 `.env` 然後填入自己的值** |
| `.github/workflows/azure-webapp.yml` | Azure Web App 部署流程 | 部署設定變更時才需要動 |

> ⚠️ **不要 commit 的檔案**（.gitignore 已設定）：
> `.env`, `serviceAccountKey.json`, `models/ecochain_gnn.pt`, `species2vec.pt`

---

## 常見錯誤排除

**Q: `ModuleNotFoundError: No module named 'torch_geometric'`**

```bash
pip install torch-geometric
```

**Q: `FileNotFoundError: models/ecochain_gnn.pt`**

模型還沒訓練，跑一次：
```bash
python train.py
```
或向組員要 `ecochain_gnn.pt` 放進 `models/` 資料夾。

**Q: 後端啟動報 Firebase 錯誤**

沒有 Firebase 金鑰是正常的，後端會自動降級（存檔功能失效，其他功能正常）。  
想要存檔功能的話向管理員要 `serviceAccountKey.json`。

**Q: 前端顯示「🔴 後端離線（本機模式）」**

後端沒在跑。確認你有在終端機執行：
```bash
uvicorn main:app --reload --port 8000
```

**Q: `git push` 失敗，顯示 rejected**

有人比你先推，先拉再推：
```bash
git pull origin 你的分支名稱 --rebase
git push origin 你的分支名稱
```

**Q: 環境變數改了但後端沒更新**

重啟後端（Ctrl+C 然後再跑一次 uvicorn）。

---

## 模型效能指標（參考）

根據 hold-out 測試（n=100，Leave-one-out）：

| 方法 | Hit@5 |
|------|-------|
| GNN | 92.0% |
| 混合公式（GNN + Jaccard）| 100.0% |
| 純 Jaccard | 100.0% |
| 純分數公式 | 13.0% |

目前生產使用**混合公式**（30% GNN 分數 + 70% Jaccard 共現）。
