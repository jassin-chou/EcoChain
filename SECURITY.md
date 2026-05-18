# EcoChain — 資安強化說明文件

## 本次修改的檔案

| 檔案 | 變動類型 |
|------|---------|
| `main.py` | 全面重寫（向下相容） |
| `ecochain_v5_backend.html` | JS 區段 patch（auth、cloud sync、leaderboard） |
| `requirements.txt` | 無新依賴（不需要額外套件） |

---

## 修復的問題 & 解法

### ① 最核心問題：任何人知道 player_id 就能讀/寫他人存檔

**問題根源**

原本的 `/save` 和 `/load/{player_id}` 完全相信請求裡的 player_id：

```bash
# Postman 直接這樣打就能讀別人存檔
GET http://your-api.com/load/player_abc123

# 也能蓋掉別人存檔
POST http://your-api.com/save
{"player_id": "player_abc123", "save_data": {...}}
```

**解法：Session Token（HMAC-SHA256）**

```
前端建立新存檔時：
  POST /auth/token/new  { slot_id, password_hash }
  → 後端簽發 token = "slotId:timestamp:HMAC(secret, slotId:timestamp)"
  → 儲存在 localStorage['ecochain_tokens']

之後每次 /save 或 /load：
  Authorization: Bearer <token>
  → 後端從 token 取出 slot_id（驗 HMAC + 時效）
  → body 裡不需要也不信任任何 player_id 欄位
```

沒有 `TOKEN_SECRET`（env var）就算知道 slot_id 也無法偽造 token。

**token 格式**（純文字，不是 JWT，不依賴第三方）：
```
slot_id:issued_at_unix:hmac_sha256_hex
```

**時效**：30 天後失效，重新呼叫 `/auth/token` 換新的（DB 有記錄就自動換）。

---

### ② /load 改為 GET /load（不再接受 URL path 中的 player_id）

原本：`GET /load/{player_id}` — player_id 在 URL 裡，直接枚舉攻擊。

現在：`GET /load` — slot_id 完全從 Bearer token 取，URL 不帶任何 ID。

```bash
# 攻擊者就算知道 slot_id 也拿不到資料
GET /load
Authorization: Bearer <valid_token_for_that_slot>  ← 沒有這個就 401
```

---

### ③ Rate Limiting（防止暴力枚舉 & API 濫用）

```python
RATE_LIMITS = {
    "/gnn/analyze":        30,  # 每 IP 每分鐘
    "/gnn/analyze/detail": 30,
    "/save":               10,
    "/auth/token":         20,
    "/analytics":          30,
}
```

超過限制 → HTTP 429，header 帶 `Retry-After: 60`。

實作是 in-memory（dict），適合單一 worker。若 Render 多 worker 或需要持久化，
換成 Redis（`slowapi` 套件）即可，改動只在 middleware 那段。

---

### ④ Input Validation（防止注入與資料污染）

| 欄位 | 舊版 | 新版 |
|------|------|------|
| `cells[].id` | 任何字串 | 嚴格比對白名單（43 種 species） |
| `cells` 長度 | 無限制 | max 35 |
| `player_name` | 無限制 | max 30 字元 |
| `coins` | 無限制 | 0 ~ 10,000,000 |
| `gps_event_bonus` | 無限制（可送 9999.0 刷分） | 0.5 ~ 3.0 |
| `eco_score` | 無限制 | 0 ~ 100 |

未知的 species id 會直接回 `422 Unprocessable Entity`，不進 GNN。

---

### ⑤ /retrain Secret 不再放在 URL query string

原本：`POST /retrain?secret=eco-retrain-2024`（secret 會進 access log）

現在：`POST /retrain` + `X-Retrain-Secret: eco-retrain-2024`（只在 header）

用 `hmac.compare_digest` 做比較，防 timing attack。

---

### ⑥ Leaderboard 不再洩漏 player_id

原本 API 回傳每個玩家的 player_id，攻擊者可以：
1. 打開排行榜
2. 複製所有 player_id
3. 用 Postman 一個個打 `/load/{player_id}`

現在 leaderboard 回傳：`player_name, eco_score, coins, species_count`，不含 player_id。

---

### ⑦ CORS 收緊

原本：`allow_origins=["*"]`（任何網站都能打 API）

現在透過 env var 設定：

```bash
# .env 或 Render Environment Variables
ALLOW_ORIGINS=https://your-frontend.vercel.app,https://your-other-domain.com
```

不設 `ALLOW_ORIGINS` 時 fallback 為 `["*"]`（本地開發用）。

---

### ⑧ /health 不再洩漏內部狀態

原本：`{"status": "ok", "db_ready": true}`（洩漏 DB 連線狀態）

現在：`{"status": "ok"}`

---

### ⑨ Swagger UI 在 production 環境關閉

```python
# ENV=production 時自動關閉
docs_url=None if os.getenv("ENV") == "production" else "/docs",
```

在 Render 加入環境變數 `ENV=production` 即可。

---

## 部署 Checklist

```bash
# Render Environment Variables（必加）
TOKEN_SECRET=<至少 32 字元的隨機字串>   # python -c "import secrets; print(secrets.token_hex(32))"
ALLOW_ORIGINS=https://你的前端網址.vercel.app
ENV=production
RETRAIN_SECRET=<自訂密碼>
GOOGLE_SHEET_ID=<你的 Sheets ID>
GOOGLE_CREDS_JSON=<google_creds.json 全文>
```

---

## 仍然不在本方案覆蓋範圍

| 項目 | 說明 |
|------|------|
| 密碼強度 | 前端 djb2 hash 非密碼學安全，但密碼本身不上傳後端，只保存在 localStorage |
| 跨裝置登入 | 換裝置後需重新建立角色（或手動複製 localStorage token）|
| HTTPS | 由 Render 自動處理，本地開發用 localhost 即可 |
| DB injection | Google Sheets API 本身無 SQL，不存在 injection 問題 |
| 多 worker Rate Limit | 若 Render 開多個 instance，rate limit 不共享 → 換 Redis |
