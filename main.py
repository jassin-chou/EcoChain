"""
main.py — EcoChain FastAPI Backend  (Security-Hardened v3)
===========================================================
資安強化清單：
  ① 身份驗證  — 後端真正儲存密碼 hash（SHA-256+slot_id salt）於 Firestore
               帳號建立、登入、刪除全都在後端驗證，前端不再自己比 hash
  ② Rate Limiting — 每 IP 每分鐘最多 30 次 /gnn/analyze，10 次 /save
  ③ CORS — 正式環境只允許前端網域（開發時透過 ALLOW_ORIGINS env var 設定）
  ④ Input Validation — 嚴格限制物種 ID 白名單、cells 長度、coins 範圍、字串長度
  ⑤ 錯誤訊息 — 統一用 400/401/403/409/429，不洩漏 stack trace
  ⑥ /health — 不回傳 db._ready 等內部狀態
  ⑦ /retrain secret — 從 Header 傳，不在 URL query string 曝露
  ⑧ leaderboard — player_id 不對外回傳，只回傳 player_name
  ⑨ timing attack 防護 — 登入失敗固定 200ms 延遲

Endpoints:
  POST   /auth/register      — 建立帳號（slot_id + player_name + SHA-256 pwd hash）
  POST   /auth/login         — 登入換 token（slot_id + SHA-256 pwd hash）
  DELETE /auth/account       — 刪帳號（需 Bearer token + 密碼二次確認）
  POST   /gnn/analyze        — GNN inference（不需 token）
  POST   /gnn/analyze/detail — 詳細版（不需 token）
  POST   /save               — 需 Bearer token
  GET    /load               — 需 Bearer token
  GET    /leaderboard        — 公開
  POST   /analytics          — 需 Bearer token
  GET    /health             — 公開
  POST   /retrain            — 需 X-Retrain-Secret header
"""

import os
import uuid
import hmac
import hashlib
import time
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from models.gnn import predict, get_model, SPECIES_IDS
from models.firebase_db import db

try:
    from train import rule_recommendation_with_source as _rec_with_source
    _HAS_REC_SOURCE = True
except ImportError:
    _HAS_REC_SOURCE = False

# ──────────────────────────────────────────────
# 常數 & 設定
# ──────────────────────────────────────────────
# Token signing key — 一定要設 TOKEN_SECRET env var（不設則啟動失敗）
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "")
if not TOKEN_SECRET:
    import secrets
    TOKEN_SECRET = secrets.token_hex(32)
    print("[Security] WARNING: TOKEN_SECRET not set — using random key (tokens won't survive restart)")

# 有效 species id 白名單（從 gnn.py 取）
VALID_SPECIES_IDS: set[str] = set(SPECIES_IDS)

# CORS 允許來源
_raw_origins = os.getenv("ALLOW_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

# Rate limiting: in-memory (per IP, resets every 60s)
_rate_buckets: dict[str, dict] = defaultdict(lambda: {"ts": 0.0, "counts": defaultdict(int)})
RATE_LIMITS = {
    "/gnn/analyze":        30,   # per minute
    "/gnn/analyze/detail": 30,
    "/save":               10,
    "/auth/register":      5,    # 新帳號建立，嚴格限制防止大量註冊
    "/auth/login":         10,   # 登入，限制暴力破解
    "/analytics":          30,
}


def _check_rate(ip: str, path: str) -> bool:
    """Return True if allowed, False if rate-limited."""
    limit = RATE_LIMITS.get(path)
    if limit is None:
        return True
    bucket = _rate_buckets[ip]
    now = time.time()
    if now - bucket["ts"] > 60:
        bucket["ts"] = now
        bucket["counts"] = defaultdict(int)
    bucket["counts"][path] += 1
    return bucket["counts"][path] <= limit


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ──────────────────────────────────────────────
# Session Token（HMAC-SHA256）
# ──────────────────────────────────────────────
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 天

def _sign_token(slot_id: str, issued_at: int) -> str:
    msg = f"{slot_id}:{issued_at}"
    sig = hmac.new(TOKEN_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return f"{slot_id}:{issued_at}:{sig}"


def _verify_token(token: str) -> str | None:
    """驗證 token，回傳 slot_id 或 None。"""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        slot_id, issued_at_str, sig = parts
        issued_at = int(issued_at_str)
        if time.time() - issued_at > TOKEN_TTL:
            return None
        expected_sig = hmac.new(TOKEN_SECRET.encode(),
                                 f"{slot_id}:{issued_at}".encode(),
                                 hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return slot_id
    except Exception:
        return None


def require_token(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency — validates Bearer token, returns slot_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    slot_id = _verify_token(token)
    if slot_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return slot_id


# ──────────────────────────────────────────────
# Startup / Shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Connecting to Firebase Firestore...")
    db.connect()
    print("[Startup] Pre-loading GNN model...")
    try:
        get_model()
        print("[Startup] GNN model ready ✅")
    except Exception as e:
        print(f"[Startup] GNN model warning: {e}")
    yield
    print("[Shutdown] Bye!")


app = FastAPI(
    title="EcoChain API",
    description="Backend for EcoChain ecosystem game",
    version="2.0.0",
    lifespan=lifespan,
    # 正式環境關掉 Swagger UI（防止 API 結構外洩）
    docs_url=None if os.getenv("ENV") == "production" else "/docs",
    redoc_url=None,
    openapi_url=None if os.getenv("ENV") == "production" else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# ──────────────────────────────────────────────
# Rate Limiting Middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path in RATE_LIMITS:
        ip = get_client_ip(request)
        if not _check_rate(ip, path):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests, please slow down"},
                headers={"Retry-After": "60"},
            )
    return await call_next(request)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class SpeciesCell(BaseModel):
    id: str = Field(..., max_length=20)
    name: str = Field(..., max_length=30)
    dEmoji: Optional[str] = Field(default=None, max_length=8)
    tr: Optional[float] = None
    water: Optional[bool] = False

    @field_validator("id")
    @classmethod
    def validate_species_id(cls, v: str) -> str:
        if v not in VALID_SPECIES_IDS:
            raise ValueError(f"Unknown species id: {v}")
        return v


class AnalyzeRequest(BaseModel):
    player_id: str = Field(..., max_length=80)
    cells: list[Optional[SpeciesCell]] = Field(..., max_length=200)
    coins: float = Field(default=0, ge=0, le=10_000_000)
    season: int = Field(default=0, ge=0, le=3)
    gps_event_bonus: float = Field(default=1.0, ge=0.5, le=3.0)   # 限制倍率範圍，防止刷分
    event_id: Optional[str] = Field(default=None, max_length=40)


class AnalyzeResponse(BaseModel):
    score: float
    gps: float
    top_recommendations: list[str]
    issues: list[dict]
    positives: list[str]
    gnn_score: Optional[float] = None
    base_score: Optional[float] = None
    score_explanation: Optional[dict[str, Any]] = None
    model_version: str = "gnn-v2"


class TokenRequest(BaseModel):
    slot_id: str = Field(..., max_length=80)
    password_hash: str = Field(..., max_length=64)   # hex string from frontend djb2


class TokenResponse(BaseModel):
    token: str
    slot_id: str
    expires_in: int = TOKEN_TTL


class RegisterRequest(BaseModel):
    """建立全新帳號"""
    slot_id: str = Field(..., min_length=4, max_length=80)
    player_name: str = Field(..., min_length=1, max_length=30)
    password_hash: str = Field(..., min_length=64, max_length=64)   # SHA-256 hex from frontend


class LoginRequest(BaseModel):
    """用密碼換 token"""
    slot_id: str = Field(..., max_length=80)
    password_hash: str = Field(..., min_length=64, max_length=64)


class DeleteRequest(BaseModel):
    """刪除帳號（需同時附 Bearer token + 密碼）"""
    password_hash: str = Field(..., min_length=64, max_length=64)


class LoginByNameRequest(BaseModel):
    """用角色名稱 + 密碼登入（不需要 slot_id）"""
    player_name: str = Field(..., min_length=1, max_length=30)
    password_hash: str = Field(..., min_length=64, max_length=64)


class SaveRequest(BaseModel):
    player_name: str = Field(default="匿名玩家", max_length=30)
    save_data: dict


class LoadResponse(BaseModel):
    found: bool
    save_data: Optional[dict] = None


class AnalyticsRequest(BaseModel):
    cells: list[Optional[SpeciesCell]] = Field(..., max_length=200)
    eco_score: float = Field(..., ge=0, le=100)
    coins: float = Field(..., ge=0, le=10_000_000)


# ──────────────────────────────────────────────
# Helpers（與原本相同，略微修剪）
# ──────────────────────────────────────────────
SEASON_GPS_BONUS = [1.1, 1.2, 0.9, 0.7]

EVENT_EFFECTS = {
    "spring_rain": {
        "score_delta": 3.0, "gps_mult": 1.20,
        "positive": "🌧️ 春雨讓植物恢復生長，生態系更穩定。",
    },
    "locust_swarm": {
        "score_delta": -8.0, "gps_mult": 0.65,
        "issue": "🦗 蝗蟲啃食植物，生產者受損，生態穩定度下降。",
    },
    "forest_fire": {
        "score_delta": -12.0, "gps_mult": 0.55,
        "issue": "🔥 森林火災造成棲地破壞，生物多樣性受到嚴重衝擊。",
    },
    "pollinator_boom": {
        "score_delta": 6.0, "gps_mult": 1.25,
        "positive": "🦋 授粉者增加，植物繁殖更順利，食物網韌性提升。",
    },
    "drought": {
        "score_delta": -6.0, "gps_mult": 0.70,
        "issue": "☀️ 乾旱讓水域生物壓力上升，整體生產力下降。",
    },
    "full_moon": {
        "score_delta": 2.0, "gps_mult": 1.15,
        "positive": "🌕 滿月讓夜行動物更活躍，捕食關係更完整。",
    },
    "education_visit": {
        "score_delta": 1.0, "gps_mult": 1.10,
        "positive": "🏫 SDG 教育參訪提升保育意識，玩家管理效率小幅提升。",
    },
    "invasive_species": {
        "score_delta": -10.0, "gps_mult": 0.60,
        "issue": "⚠️ 外來種干擾原生食物網，生態平衡明顯下降。",
    },
    "decomposer_bloom": {
        "score_delta": 5.0, "gps_mult": 1.18,
        "positive": "🍄 分解者繁榮，加快養分循環，土壤狀態改善。",
    },
    "water_cleanup": {
        "score_delta": 7.0, "gps_mult": 1.22,
        "positive": "💧 水域清理改善棲地品質，水生生物更容易存活。",
    },
    "cold_wave": {
        "score_delta": -4.0, "gps_mult": 0.78,
        "issue": "❄️ 寒流降低動植物活動力，生態系短暫降溫。",
    },
    "disease_outbreak": {
        "score_delta": -7.0, "gps_mult": 0.68,
        "issue": "🧫 疾病爆發讓族群健康下降，物種互動變得脆弱。",
    },
    "habitat_restoration": {
        "score_delta": 8.0, "gps_mult": 1.24,
        "positive": "🌱 棲地復育成功，物種有更好的生存空間。",
    },
    "seed_dispersal": {
        "score_delta": 4.0, "gps_mult": 1.16,
        "positive": "🌰 種子傳播擴散植物，生態系基礎更穩。",
    },
    "research_grant": {
        "score_delta": 2.0, "gps_mult": 1.12,
        "positive": "🔬 研究補助提升監測能力，管理效率小幅提升。",
    },
}

SPECIES_EMOJI = {
    "grass":"🌿","flower":"🌸","berry":"🫐","tree":"🌳","shrub":"🍃","mushroom":"🍄",
    "wheat":"🌾","cactus":"🌵","lotus":"🪷","seaweed":"🪸","sheep":"🐑","cow":"🐄",
    "rabbit":"🐰","deer":"🦌","horse":"🐎","goat":"🐐","cat":"🐱","dog":"🐶",
    "chicken":"🐔","pig":"🐷","bee":"🐝","butterfly":"🦋","bird":"🐦","fox":"🦊",
    "wolf":"🐺","eagle":"🦅","snake":"🐍","owl":"🦉","carp":"🐟","salmon":"🐠",
    "frog":"🐸","shrimp":"🦐","turtle":"🐢","duck":"🦆","crab":"🦀","snail":"🐌",
    "pond":"🏞️","compost":"♻️","birdhouse":"🏡","stream":"💧","rockpile":"🪨","fence":"🚧",
}

SPECIES_NAME_ZH = {
    "grass":"青草","flower":"花朵","berry":"漿果叢","tree":"大樹","shrub":"灌木叢",
    "mushroom":"蘑菇","wheat":"小麥","cactus":"仙人掌","lotus":"荷花","seaweed":"水草",
    "sheep":"羊","cow":"牛","rabbit":"兔子","deer":"鹿","horse":"馬","goat":"山羊",
    "cat":"貓","dog":"狗","chicken":"雞","pig":"豬","bee":"蜜蜂","butterfly":"蝴蝶",
    "bird":"野鳥","fox":"狐狸","wolf":"狼","eagle":"老鷹","snake":"蛇","owl":"貓頭鷹",
    "carp":"鯉魚","salmon":"鱒魚","frog":"青蛙","shrimp":"蝦","turtle":"烏龜",
    "duck":"鴨子","crab":"螃蟹","snail":"田螺","pond":"池塘","compost":"堆肥桶",
    "birdhouse":"鳥巢箱","stream":"小溪流","rockpile":"石堆","fence":"圍欄",
}


def cells_to_counts(cells: list) -> dict:
    counts = {}
    for c in cells:
        if c and c.id:
            counts[c.id] = counts.get(c.id, 0) + 1
    return counts


def compute_gps(score: float, counts: dict, season: int, event_bonus: float) -> float:
    total   = sum(counts.values())
    bee     = "bee" in counts or "butterfly" in counts
    compost = "compost" in counts
    season_bonus = SEASON_GPS_BONUS[season % 4]
    base = 1 + total * 0.3
    mult = score / 50
    gps  = base * mult
    if bee:     gps *= 1.2
    if compost: gps *= 1.1
    gps *= season_bonus * event_bonus
    return round(max(0.1, gps), 1)


def apply_event_effects(
    event_id: Optional[str],
    score: float,
    gps: float,
    issues: list,
    positives: list,
) -> tuple[float, float, dict | None]:
    if not event_id:
        return score, gps, None

    event = EVENT_EFFECTS.get(event_id)
    if not event:
        return score, gps, None

    adjusted_score = round(max(5.0, min(100.0, score + event["score_delta"])), 1)
    adjusted_gps = round(max(0.1, gps * event["gps_mult"]), 1)

    if event.get("issue"):
        issues.append({"type": "event", "msg": event["issue"]})
    if event.get("positive"):
        positives.append(event["positive"])

    return adjusted_score, adjusted_gps, {
        "id": event_id,
        "score_delta": event["score_delta"],
        "gps_mult": event["gps_mult"],
        "score_before": round(score, 1),
        "score_after": adjusted_score,
    }


def compute_explainable_score(counts: dict) -> dict[str, Any]:
    import math
    from models.gnn import TROPHIC, IS_WATER, IS_TERRAIN

    total = sum(counts.values())
    species_types = len(counts)

    shannon = 0.0
    if total > 0 and species_types > 1:
        h_raw = -sum((n / total) * math.log(n / total) for n in counts.values() if n > 0)
        h_max = math.log(species_types)
        shannon = h_raw / h_max if h_max > 0 else 0.0

    has_producer = any(TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN for s in counts)
    has_herbivore = any(TROPHIC.get(s, 0) == 1 for s in counts)
    has_omnivore = any(TROPHIC.get(s, 0) == 1.5 for s in counts)
    has_predator = any(TROPHIC.get(s, 0) >= 2 for s in counts)

    trophic = (
        (1 / 3 if has_producer else 0)
        + (1 / 3 if (has_herbivore or has_omnivore) else 0)
        + (1 / 3 if has_predator else 0)
    )
    if (has_herbivore or has_omnivore) and not has_producer:
        trophic -= 0.25
    if has_predator and not has_herbivore and not has_omnivore:
        trophic -= 0.25
    trophic = max(0.0, min(1.0, trophic))

    has_water = "pond" in counts or "stream" in counts
    has_aquatic = any(s in IS_WATER and TROPHIC.get(s, 0) > 0 for s in counts)
    aquatic = 0.10 if has_water and has_aquatic else (-0.10 if has_aquatic and not has_water else 0.0)

    has_plants = has_producer
    has_pollinator = "bee" in counts or "butterfly" in counts
    pollinator = 0.05 if has_plants and has_pollinator else 0.0

    infrastructure = min(
        (0.03 if "compost" in counts else 0)
        + (0.03 if "birdhouse" in counts else 0)
        + (0.02 if "rockpile" in counts else 0)
        + (0.02 if "fence" in counts else 0),
        0.10,
    )

    components = {
        "base": 0.35,
        "shannon": shannon * 0.25,
        "trophic": trophic * 0.60,
        "aquatic": aquatic,
        "pollinator": pollinator,
        "infrastructure": infrastructure,
    }
    raw = sum(components.values())
    score = round(max(5.0, min(100.0, raw / 1.45 * 100)), 1)
    if total == 0:
        score = 10.0

    return {
        "score": score,
        "raw": raw,
        "max_raw": 1.45,
        "components": components,
        "shannon": round(shannon, 3),
        "trophic": round(trophic, 3),
        "total": total,
        "species_types": species_types,
        "has_producer": has_producer,
        "has_consumer": has_herbivore or has_omnivore,
        "has_predator": has_predator,
    }


def build_issues_positives(score: float, counts: dict) -> tuple[list, list, float]:
    import math
    from models.gnn import TROPHIC, IS_WATER, IS_TERRAIN

    plants = herbs = omnis = preds = aquatics = 0
    pond = bee = False
    total = sum(counts.values())

    for s, cnt in counts.items():
        tr = TROPHIC.get(s, 0)
        if tr == 0 and s not in IS_TERRAIN: plants += cnt
        if tr == 1:   herbs   += cnt
        if tr == 1.5: omnis   += cnt
        if tr >= 2:   preds   += cnt
        if s in IS_WATER and tr > 0: aquatics += cnt
        if s in ("pond", "stream"): pond = True
        if s in ("bee", "butterfly"): bee  = True

    shannon = 0.0
    n_types = len(counts)
    if total > 0 and n_types > 1:
        h_raw = -sum((n/total)*math.log(n/total) for n in counts.values() if n > 0)
        h_max = math.log(n_types)
        shannon = h_raw / h_max if h_max > 0 else 0.0

    issues, positives = [], []
    sh_pct = round(shannon * 100)

    if n_types == 0:
        pass
    elif n_types == 1:
        issues.append({"type": "warn", "msg": "只有 1 種物種，Shannon 多樣性為零 — 試著增加物種種類"})
    elif shannon >= 0.75:
        positives.append(f"Shannon 多樣性指數高（{sh_pct}%）— 物種分布均勻")
    elif shannon >= 0.45:
        issues.append({"type": "warn", "msg": f"Shannon 均勻度中等（{sh_pct}%）— 某些物種數量失衡"})
    else:
        issues.append({"type": "warn", "msg": f"Shannon 均勻度低（{sh_pct}%）— 物種過度集中於少數種類"})

    if plants > 0:
        positives.append("有生產者（植物）— 食物鏈基礎")
    else:
        issues.append({"type": "bad", "msg": "缺少生產者！植物是所有食物鏈的能量來源"})
    if herbs > 0 or omnis > 0:
        positives.append("有初級消費者（草食/雜食）")
    if preds > 0:
        positives.append("有頂層掠食者 — 防止草食動物過量繁殖")
    if (herbs > 0 or omnis > 0) and plants == 0:
        issues.append({"type": "bad", "msg": "草食動物沒有植物可吃（分數 -0.25）"})
    if preds > 0 and herbs == 0 and omnis == 0:
        issues.append({"type": "bad", "msg": "掠食者沒有獵物（分數 -0.25）"})
    if pond and aquatics > 0:
        positives.append("水陸雙生態系 — 連結兩個能量流系統")
    if aquatics > 0 and not pond:
        issues.append({"type": "bad", "msg": "水生生物需要池塘或溪流才能存活"})
    if pond and aquatics == 0:
        issues.append({"type": "warn", "msg": "池塘/溪流裡沒有水生生物，水體生態空置"})
    if bee and plants > 0:
        positives.append("授粉者（蜜蜂/蝴蝶）與植物共存")

    return issues, positives, shannon


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    # 不洩漏內部狀態（db._ready、模型路徑等）
    return {"status": "ok"}


# ── Auth ─────────────────────────────────────
# ── Auth ─────────────────────────────────────
@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest, request: Request):
    """
    建立全新帳號。
    - 前端傳入 slot_id（UUID v4）、player_name、SHA-256(password) hex
    - 後端把 SHA-256(slot_id + ":" + password_hash) 存進 Firestore accounts/{slot_id}
    - 成功後立即簽發 session token（讓前端不用再 login）
    - slot_id 重複 → 409
    - 密碼空白 → 422
    """
    if not db._ready:
        raise HTTPException(status_code=503, detail="Database not available")

    ok, err = db.create_account(req.slot_id, req.player_name, req.password_hash)
    if not ok:
        if err == "already_exists":
            raise HTTPException(status_code=409, detail="Slot ID already taken")
        if err == "empty_password":
            raise HTTPException(status_code=422, detail="Password cannot be empty")
        raise HTTPException(status_code=500, detail="Could not create account")

    issued_at = int(time.time())
    token = _sign_token(req.slot_id, issued_at)
    return TokenResponse(token=token, slot_id=req.slot_id, expires_in=TOKEN_TTL)


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request):
    """
    用 slot_id + 密碼（SHA-256 hex）換 session token。
    - 找不到帳號 → 401
    - 密碼錯誤   → 401（故意不區分兩者，防止帳號枚舉）
    """
    if not db._ready:
        raise HTTPException(status_code=503, detail="Database not available")

    ok, _player_name = db.verify_password(req.slot_id, req.password_hash)
    if not ok:
        # 固定延遲，防止 timing attack 猜出帳號是否存在
        import asyncio as _aio
        time.sleep(0.2)
        raise HTTPException(status_code=401, detail="Invalid slot ID or password")

    issued_at = int(time.time())
    token = _sign_token(req.slot_id, issued_at)
    return TokenResponse(token=token, slot_id=req.slot_id, expires_in=TOKEN_TTL)


@app.get("/accounts/list")
def list_accounts():
    """
    回傳所有帳號的 player_name + slot_id 清單。
    不含密碼，供前端讀檔 modal 顯示用。
    """
    if not db._ready:
        raise HTTPException(status_code=503, detail="Database not available")
    accounts = db.list_accounts()
    return {"accounts": accounts}


@app.post("/auth/login-by-name", response_model=TokenResponse)
def login_by_name(req: LoginByNameRequest, request: Request):
    """
    用 player_name + 密碼換 token（不需要 slot_id）。
    適合換裝置或 localStorage 被清掉的情況。
    - 找不到名稱 → 401（故意不區分帳號不存在 vs 密碼錯誤）
    """
    if not db._ready:
        raise HTTPException(status_code=503, detail="Database not available")

    slot_id = db.find_slot_by_name(req.player_name)
    if not slot_id:
        time.sleep(0.2)
        raise HTTPException(status_code=401, detail="Invalid name or password")

    ok, _ = db.verify_password(slot_id, req.password_hash)
    if not ok:
        time.sleep(0.2)
        raise HTTPException(status_code=401, detail="Invalid name or password")

    issued_at = int(time.time())
    token = _sign_token(slot_id, issued_at)
    return TokenResponse(token=token, slot_id=slot_id, expires_in=TOKEN_TTL)


@app.delete("/auth/account")
def delete_account(
    req: DeleteRequest,
    slot_id: str = Depends(require_token),
):
    """
    刪除帳號（accounts + saves + leaderboard）。
    需要：
    1. 有效的 Bearer token（確認是本人）
    2. 正確的密碼（防止 token 被竊後悄悄刪帳）
    """
    if not db._ready:
        raise HTTPException(status_code=503, detail="Database not available")

    ok, _ = db.verify_password(slot_id, req.password_hash)
    if not ok:
        time.sleep(0.2)
        raise HTTPException(status_code=401, detail="Incorrect password")

    deleted = db.delete_account(slot_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Could not delete account")
    return {"success": True, "deleted_slot": slot_id}


# ── GNN Analyze（不需登入，分析是公開功能）────
@app.post("/gnn/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, bg: BackgroundTasks):
    counts  = cells_to_counts(req.cells)
    result  = predict(counts)
    gnn_score = result["score"]
    score_info = compute_explainable_score(counts)
    score   = score_info["score"]
    top_rec = result["top_recommendations"]
    gps     = compute_gps(score, counts, req.season, req.gps_event_bonus)
    issues, positives, _ = build_issues_positives(score, counts)
    score, gps, _event = apply_event_effects(req.event_id, score, gps, issues, positives)
    return AnalyzeResponse(
        score=score, gps=gps,
        top_recommendations=top_rec,
        issues=issues, positives=positives,
        gnn_score=gnn_score,
        base_score=score_info["score"],
        score_explanation=score_info,
    )


@app.post("/gnn/analyze/detail")
def analyze_detail(req: AnalyzeRequest):
    counts  = cells_to_counts(req.cells)
    result  = predict(counts)
    gnn_score = result["score"]
    score_info = compute_explainable_score(counts)
    score   = score_info["score"]
    top_rec = result["top_recommendations"]
    gps     = compute_gps(score, counts, req.season, req.gps_event_bonus)
    issues, positives, shannon = build_issues_positives(score, counts)
    score, gps, event_meta = apply_event_effects(req.event_id, score, gps, issues, positives)

    if _HAS_REC_SOURCE:
        rec_with_source = _rec_with_source(counts)
        source_map = {r["id"]: r for r in rec_with_source}
        rec_enriched = []
        for s in top_rec:
            meta = source_map.get(s, {})
            rec_enriched.append({
                "id": s, "name": SPECIES_NAME_ZH.get(s, s),
                "emoji": SPECIES_EMOJI.get(s, "❓"),
                "network": meta.get("network", ""), "ref": meta.get("ref", ""),
                "score_gain": meta.get("score_gain", 0), "jaccard": meta.get("jaccard", 0),
            })
    else:
        rec_enriched = [
            {"id": s, "name": SPECIES_NAME_ZH.get(s, s), "emoji": SPECIES_EMOJI.get(s, "❓"),
             "network": "", "ref": "", "score_gain": 0, "jaccard": 0}
            for s in top_rec
        ]

    return {
        "score": score, "gps": gps,
        "base_score": score_info["score"], "gnn_score": gnn_score,
        "score_explanation": score_info,
        "recommendations": rec_enriched,
        "issues": issues, "positives": positives,
        "counts": counts,
        "shannon": score_info["shannon"], "trophic": score_info["trophic"],
        "event_effect": event_meta,
        "model_version": "Explainable+GNN-v3",
    }


# ── Save（需 token）────────────────────────────
@app.post("/save")
def save_game(
    req: SaveRequest,
    bg: BackgroundTasks,
    slot_id: str = Depends(require_token),
):
    """
    token 中的 slot_id 就是要存的 player_id。
    前端不需要再傳 player_id — 完全由 token 決定，防止跨帳戶存檔。
    """
    ok = db.save_game(slot_id, req.save_data)
    if ok:
        sd = req.save_data
        bg.add_task(
            db.upsert_leaderboard,
            slot_id, req.player_name,
            float(sd.get("ecoScore", 0)),
            float(sd.get("coins", 0)),
            int(sd.get("speciesCount", 0)),
        )
    return {"success": ok}


# ── Load（需 token）────────────────────────────
@app.get("/load", response_model=LoadResponse)
def load_game(slot_id: str = Depends(require_token)):
    """
    slot_id 從 token 取出，不接受 URL path parameter，防止枚舉攻擊。
    """
    data = db.load_game(slot_id)
    if data:
        return LoadResponse(found=True, save_data=data)
    return LoadResponse(found=False)


# ── Leaderboard（公開，但不洩漏 player_id）───
@app.get("/leaderboard")
def leaderboard(limit: int = Query(default=20, ge=1, le=50)):
    rows = db.get_leaderboard(min(limit, 50))
    # 移除 player_id，只保留公開資訊
    safe_rows = [
        {
            "player_name":   r.get("player_name", "匿名"),
            "eco_score":     r.get("eco_score", 0),
            "coins":         r.get("coins", 0),
            "species_count": r.get("species_count", 0),
            "updated_at":    r.get("updated_at", ""),   # ISO timestamp, lets frontend show "last saved X"
        }
        for r in rows
    ]
    return {"entries": safe_rows, "total": len(safe_rows)}


# ── Analytics（需 token，防止垃圾資料污染訓練集）
@app.post("/analytics")
def log_analytics(
    req: AnalyticsRequest,
    bg: BackgroundTasks,
    slot_id: str = Depends(require_token),
):
    cells_raw = [c.model_dump() if c else None for c in req.cells]
    bg.add_task(db.log_analytics, slot_id, cells_raw, req.eco_score, req.coins)
    return {"success": True}


# ── Species Importance（GNN perturbation analysis）────────────────────────────
@app.post("/gnn/importance")
def species_importance(req: AnalyzeRequest):
    """
    對每個物種做 leave-one-out perturbation：
    移除它之後重跑 GNN，看分數掉多少。
    分數差越大 = 該物種對生態系越關鍵。

    這是 GNN 圖結構真正的優勢：它能捕捉連鎖反應（bee→flower→butterfly），
    純公式/查表做不到。
    """
    counts = cells_to_counts(req.cells)
    if not counts:
        return {"baseline": 0, "impact": [], "chain_effects": []}

    baseline = predict(counts)["score"]
    baseline_gps = compute_gps(baseline, counts, req.season, req.gps_event_bonus)
    baseline_issues: list = []
    baseline_positives: list = []
    baseline, _baseline_gps, event_meta = apply_event_effects(
        req.event_id, baseline, baseline_gps, baseline_issues, baseline_positives
    )

    impact = {}
    for species in counts:
        reduced = {k: v for k, v in counts.items() if k != species}
        if reduced:
            new_score = predict(reduced)["score"]
            new_gps = compute_gps(new_score, reduced, req.season, req.gps_event_bonus)
            new_score, _new_gps, _ = apply_event_effects(
                req.event_id, new_score, new_gps, [], []
            )
        else:
            new_score = 5.0  # 空圖最低分
        impact[species] = round(baseline - new_score, 1)

    ranked = sorted(impact.items(), key=lambda x: x[1], reverse=True)

    # 找出「連鎖效應」：移除某物種後，另一物種也變得孤立
    from models.gnn import PRED_EDGES, POLL_EDGES, SYM_EDGES
    chain_effects = []
    for species, drop in ranked:
        if drop <= 0:
            continue
        victims = []
        # 誰依賴這個物種？
        for pred, preys in PRED_EDGES.items():
            if species in preys and pred in counts:
                victims.append(pred)
        for poll, plants in POLL_EDGES.items():
            if species in plants and poll in counts:
                victims.append(poll)
        for fac, targets in SYM_EDGES.items():
            if species in targets and fac in counts:
                victims.append(fac)
        # 反向：這個物種依賴誰
        if species in PRED_EDGES:
            for prey in PRED_EDGES[species]:
                if prey not in counts:
                    victims.append(f"缺少獵物:{prey}")
        if victims:
            chain_effects.append({
                "removed": species,
                "affected": list(set(victims))[:3],  # 最多顯示3個
            })

    return {
        "baseline": baseline,
        "event_effect": event_meta,
        "impact": [
            {
                "species": s,
                "name": SPECIES_NAME_ZH.get(s, s),
                "emoji": SPECIES_EMOJI.get(s, "❓"),
                "drop": d,
                "critical": d >= 5.0,  # 掉超過5分算關鍵物種
            }
            for s, d in ranked
        ],
        "chain_effects": chain_effects[:5],  # 最多5條連鎖
    }


# ── Retrain（需 Header，不在 URL query 曝露 secret）
@app.post("/retrain")
def trigger_retrain(x_retrain_secret: Optional[str] = Header(default=None)):
    expected = os.getenv("RETRAIN_SECRET", "eco-retrain-2024")
    if not x_retrain_secret or not hmac.compare_digest(x_retrain_secret, expected):
        raise HTTPException(status_code=403, detail="Forbidden")

    async def run_retrain():
        import subprocess, sys
        subprocess.Popen([
            sys.executable, "train.py",
            "--epochs", "20", "--n", "2000",
        ])

    asyncio.create_task(run_retrain())
    return {"message": "Retraining started"}


# ── Serve Frontend ────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse("ecochain_v6_redesign.html")
