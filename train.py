"""
train.py — Train the EcoChain GNN
==================================
Run once before starting the server:

    python train.py

    # 先跑 fetch_mangal.py 產生 real_food_webs.json（強烈建議）：
    python fetch_mangal.py --offline      # 不需要網路，用內建資料
    python fetch_mangal.py                # 從 Mangal API 下載最新資料

改動說明（v3）
--------------
1. eco_score 使用有學術依據的公式（同 v2）：
   - Shannon 多樣性指數 H（Pielou's J evenness）
   - 營養層完整性 T（生產者/草食/掠食三層完整 → 滿分）
   - 水域生態加成、授粉者加成、基礎設施加成

2. 訓練資料比例調整（v3 改動）：
   - 10% 純隨機合成（只保留少量作資料擴增用）
   - 60% 真實 Mangal 食物網子集（主要學習來源）
   - 30% 真實食物網帶擾動（增加多樣性）
   優先讀取 real_food_webs.json；若不存在，fallback 到手工整理的 6 個食物網。

3. 推薦系統權重調整（v3 改動）：
   - 以前：60% 分數增益 + 40% Jaccard
   - 現在：30% 分數增益 + 70% Jaccard
   讓 GNN 推薦的物種更貼近「真實食物網中共存的組合」，而非純粹最大化人造公式。

4. 推薦結果帶來源標注（v3 改動）：
   rule_recommendation_with_source() 回傳每個推薦物種的食物網來源，
   讓前端可顯示「基於 Ythan Estuary（Huxham et al. 1992）」。

5. 訓練標籤改為食物網共現推薦（v4 改動）：
   - 以前：訓練標籤來自 rule_recommendation()（公式計算分數增益 + Jaccard 混合）
   - 現在：訓練標籤來自 foodweb_recommendation()（純 Jaccard co-occurrence）
   GNN 推薦頭現在直接學「真實食物網裡什麼物種共存」，
   而不是學公式的分數增益——讓 GNN 能提供公式沒有的結構性推薦。
   rule_recommendation() 保留，仍用於前端即時推薦（不需要重新訓練）。
"""

import os
import sys
import random
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Species2Vec：GNN 訓練前先預訓練物種 embedding ──────────────────────────
# 若 species2vec.pt 已存在則跳過，強制重訓可傳 --retrain_s2v
try:
    from species2vec import train_species2vec, DEFAULT_OUTPUT as S2V_PATH
    _HAS_S2V = True
except ImportError:
    _HAS_S2V = False
    S2V_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "species2vec.pt")


def _ensure_species2vec(force: bool = False):
    """
    確保 species2vec.pt 存在。
    - 若已存在且 force=False → 跳過（避免每次訓練都重跑）
    - 若不存在或 force=True  → 執行預訓練

    在 train() 前面呼叫，讓 build_graph() 能讀到 embedding。
    """
    if not _HAS_S2V:
        print("[Train] species2vec.py not found — skipping Species2Vec pre-training")
        print("        Node features will use fallback 5-dim hand-crafted features.")
        return

    if os.path.exists(S2V_PATH) and not force:
        print(f"[Train] species2vec.pt already exists ({S2V_PATH}) — skipping pre-training")
        print("        Use --retrain_s2v to force re-training.")
        return

    print("[Train] ── Species2Vec Pre-training ─────────────────────────────────")
    print("[Train] Learning ecological niche embeddings from real food web data...")
    train_species2vec(embed_dim=16, epochs=300, lr=0.01)
    print("[Train] ─────────────────────────────────────────────────────────────")

# ══════════════════════════════════════════════════════
# 讀取 real_food_webs.json（由 fetch_mangal.py 產生）
# 若不存在，fallback 到下方手工整理的 6 個食物網
# ══════════════════════════════════════════════════════
_REAL_WEBS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_food_webs.json")

def _load_real_webs_json() -> list[dict]:
    """載入 fetch_mangal.py 產生的 real_food_webs.json"""
    if not os.path.exists(_REAL_WEBS_JSON_PATH):
        return []
    try:
        with open(_REAL_WEBS_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        webs = data.get("game_webs", [])
        print(f"[Train] Loaded {len(webs)} subsets from real_food_webs.json")
        return webs
    except Exception as e:
        print(f"[Train] Warning: could not load real_food_webs.json: {e}")
        return []

from models.gnn import (
    EcoGNN, build_graph, SPECIES_IDS, N_SPECIES, SPECIES_IDX,
    TROPHIC, IS_WATER, IS_TERRAIN, MODEL_PATH,
    PRED_EDGES, POLL_EDGES, SYM_EDGES,
)

# ══════════════════════════════════════════════════════
# 真實食物網資料（來源：Mangal 資料庫 / 生態學文獻）
# 每個 set 是一個真實生態系中「共存」的物種子集
# 只保留遊戲中有的物種 ID
# ══════════════════════════════════════════════════════

# Ythan Estuary (Scotland) — 代表性河口食物網
# Huxham et al. 1992 / Mangal network #98
YTHAN_ESTUARY = [
    {"grass", "shrimp", "crab", "duck"},
    {"seaweed", "snail", "crab", "duck"},
    {"grass", "shrimp", "frog", "snake", "bird"},
    {"seaweed", "shrimp", "crab", "salmon"},
    {"grass", "snail", "duck", "crab"},
    {"seaweed", "turtle", "crab", "shrimp"},
]

# Tuesday Lake (Michigan) — 湖泊食物網
# Post et al. 2000 / Mangal network #154
TUESDAY_LAKE = [
    {"grass", "carp", "duck", "frog"},
    {"seaweed", "carp", "shrimp", "frog", "snake"},
    {"lotus", "carp", "shrimp", "duck", "turtle"},
    {"seaweed", "shrimp", "frog", "snake", "bird"},
    {"grass", "rabbit", "frog", "snake"},
    {"lotus", "carp", "turtle", "shrimp"},
]

# Yellowstone grassland — 草原食物網
# Ripple & Beschta 2004 / wolf reintroduction studies
YELLOWSTONE = [
    {"grass", "deer", "wolf", "eagle", "bird"},
    {"grass", "rabbit", "fox", "eagle"},
    {"grass", "deer", "wolf", "bird"},
    {"shrub", "rabbit", "fox", "owl"},
    {"grass", "sheep", "wolf", "eagle"},
    {"grass", "deer", "rabbit", "fox", "wolf"},
    {"shrub", "deer", "wolf", "owl"},
    {"grass", "goat", "wolf", "eagle"},
    {"tree", "deer", "wolf", "owl", "bird"},
]

# Coastal marine — 沿岸食物網
# Based on Patagonian shelf / Mangal network #267
COASTAL_MARINE = [
    {"seaweed", "shrimp", "crab", "salmon", "eagle"},
    {"seaweed", "carp", "salmon", "bird"},
    {"lotus", "shrimp", "turtle", "crab"},
    {"seaweed", "snail", "crab", "salmon"},
    {"grass", "shrimp", "frog", "snake", "bird"},
    {"lotus", "carp", "shrimp", "duck"},
    {"seaweed", "shrimp", "crab", "salmon", "duck"},
]

# Farmland / agricultural — 農地生態系
# Benton et al. 2003 / UK farmland bird studies
FARMLAND = [
    {"wheat", "sheep", "chicken", "fox", "owl"},
    {"grass", "cow", "chicken", "fox", "bird"},
    {"wheat", "rabbit", "fox", "owl", "bird"},
    {"grass", "sheep", "cow", "bee", "butterfly"},
    {"flower", "bee", "butterfly", "bird", "fox"},
    {"wheat", "chicken", "pig", "dog", "fox"},
    {"grass", "rabbit", "sheep", "fox", "eagle"},
    {"berry", "rabbit", "fox", "owl"},
    {"mushroom", "rabbit", "fox", "bird"},
]

# Temperate forest — 溫帶森林
# Białowieża Forest / European Forest Biodiversity studies
TEMPERATE_FOREST = [
    {"tree", "deer", "wolf", "owl", "bird"},
    {"shrub", "rabbit", "fox", "owl"},
    {"tree", "mushroom", "snail", "bird", "fox"},
    {"berry", "deer", "fox", "wolf"},
    {"tree", "bird", "bee", "owl"},
    {"shrub", "deer", "rabbit", "fox", "wolf"},
    {"mushroom", "snail", "bird", "owl"},
    {"tree", "deer", "snake", "eagle"},
    {"berry", "rabbit", "fox", "eagle"},
]

ALL_REAL_WEBS: list[set] = (
    YTHAN_ESTUARY +
    TUESDAY_LAKE +
    YELLOWSTONE +
    COASTAL_MARINE +
    FARMLAND +
    TEMPERATE_FOREST
)

# 過濾掉遊戲裡沒有的物種，並去掉太小的子集
GAME_SPECIES = set(SPECIES_IDS)
_FILTERED_BUILTIN = [
    w for w in (
        {s for s in web if s in GAME_SPECIES}
        for web in ALL_REAL_WEBS
    )
    if len(w) >= 3
]

# ── 嘗試讀取 real_food_webs.json（fetch_mangal.py 的輸出）──────────────
_json_webs = _load_real_webs_json()
if _json_webs:
    # JSON 格式：{"species": [...], "network": "...", "ref": "..."}
    # 轉成 set 供 Jaccard 使用，同時保留 metadata
    REAL_WEBS_FILTERED: list[dict] = []
    for w in _json_webs:
        sp_set = {s for s in w.get("species", []) if s in GAME_SPECIES}
        if len(sp_set) >= 2:
            REAL_WEBS_FILTERED.append({
                "species": sp_set,
                "network": w.get("network", "unknown"),
                "ref":     w.get("ref", ""),
            })
    print(f"[Train] Using {len(REAL_WEBS_FILTERED)} real-web subsets from JSON")
else:
    # Fallback：把手工整理的 6 個食物網包成相同格式
    REAL_WEBS_FILTERED = [
        {"species": w, "network": "built-in", "ref": "see train.py"}
        for w in _FILTERED_BUILTIN
    ]
    print(f"[Train] Using {len(REAL_WEBS_FILTERED)} built-in real-web subsets (run fetch_mangal.py for better data)")


# ══════════════════════════════════════════════════════
# 科學化 eco_score
# ══════════════════════════════════════════════════════

def shannon_diversity(counts: dict) -> float:
    """
    Shannon-Wiener 多樣性指數（Pielou's J evenness）
    H' = -Σ(p_i × ln(p_i))，正規化到 0~1
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = -sum(
        (n / total) * math.log(n / total)
        for n in counts.values() if n > 0
    )
    h_max = math.log(max(len(counts), 1))
    return h / h_max if h_max > 0 else 0.0


def trophic_completeness(counts: dict) -> float:
    """
    營養層完整性（0~1）：
      +1/3 有生產者，+1/3 有初級消費者，+1/3 有高階消費者
    斷鏈懲罰：草食無植物 −0.25，掠食無獵物 −0.25
    """
    has_producer  = any(
        TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN
        for s in counts
    )
    has_herbivore = any(TROPHIC.get(s, 0) == 1   for s in counts)
    has_omnivore  = any(TROPHIC.get(s, 0) == 1.5 for s in counts)
    has_predator  = any(TROPHIC.get(s, 0) >= 2   for s in counts)

    score = (
        (1/3 if has_producer else 0) +
        (1/3 if (has_herbivore or has_omnivore) else 0) +
        (1/3 if has_predator else 0)
    )
    if (has_herbivore or has_omnivore) and not has_producer:
        score -= 0.25   # 草食動物沒有植物
    if has_predator and not has_herbivore and not has_omnivore:
        score -= 0.25   # 掠食者沒有獵物

    return max(0.0, min(1.0, score))


def aquatic_bonus(counts: dict) -> float:
    """水域生態：有水體 + 水生動物 → +0.1，有水生動物但無水體 → −0.1"""
    has_water   = "pond" in counts or "stream" in counts
    has_aquatic = any(
        s in IS_WATER and TROPHIC.get(s, 0) > 0
        for s in counts
    )
    if has_water and has_aquatic:
        return 0.1
    if has_aquatic and not has_water:
        return -0.1
    return 0.0


def pollinator_bonus(counts: dict) -> float:
    """授粉者（蜜蜂/蝴蝶）+ 有植物 → +0.05"""
    has_plants    = any(
        TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN
        for s in counts
    )
    has_pollinator = "bee" in counts or "butterfly" in counts
    return 0.05 if (has_plants and has_pollinator) else 0.0


def infrastructure_bonus(counts: dict) -> float:
    """人工設施加分（上限 0.10）"""
    bonus = (
        0.03 * ("compost"   in counts) +
        0.03 * ("birdhouse" in counts) +
        0.02 * ("rockpile"  in counts) +
        0.02 * ("fence"     in counts)
    )
    return min(bonus, 0.10)


def eco_score(species_counts: dict) -> float:
    """
    最終生態分數 0~100：
      25% Shannon 多樣性（Pielou's J evenness）
      60% 營養層完整性（最重要：有沒有完整食物鏈）
      15% 水域生態 + 授粉者 + 基礎設施

    設計依據：
    - 單一植物 → ~38，植物+草食 → ~68，完整三層 → ~81
    - 加水域、授粉者、設施後可達 ~90
    - 斷鏈（草食無植物、掠食無獵物）大幅扣分

    raw 理論最大值 = 0.25 + 0.60 + 0.35 + 0.10 + 0.05 + 0.10 = 1.45
    """
    if not species_counts:
        return 10.0

    H = shannon_diversity(species_counts)
    T = trophic_completeness(species_counts)
    A = aquatic_bonus(species_counts)
    P = pollinator_bonus(species_counts)
    I = infrastructure_bonus(species_counts)

    raw   = H * 0.25 + T * 0.60 + 0.35 + A + P + I
    score = raw / 1.45 * 100
    return round(max(5.0, min(100.0, score)), 1)


# 向後相容別名（main.py 不需改動）
def rule_score(species_counts: dict) -> float:
    return eco_score(species_counts)


# ══════════════════════════════════════════════════════
# 推薦系統（分數增益 × 60% + 真實食物網相符性 × 40%）
# ══════════════════════════════════════════════════════

def _real_web_affinity(species: str, present: set) -> tuple[float, str, str]:
    """
    計算「加入此物種」與真實食物網的最大 Jaccard 相似度。
    回傳 (score, network_name, reference)。
    分數越高 → 越符合真實存在的生態組合。
    """
    best_score = 0.0
    best_network = ""
    best_ref = ""
    for entry in REAL_WEBS_FILTERED:
        web  = entry["species"] if isinstance(entry, dict) else entry
        name = entry.get("network", "") if isinstance(entry, dict) else ""
        ref  = entry.get("ref", "")    if isinstance(entry, dict) else ""
        if species not in web:
            continue
        intersection = len(present & web)
        union        = len(present | web)
        score        = intersection / union if union > 0 else 0.0
        if score > best_score:
            best_score   = score
            best_network = name
            best_ref     = ref
    return best_score, best_network, best_ref


def rule_recommendation(species_counts: dict) -> list[int]:
    """
    綜合推薦，回傳 top-5 物種 index：
      30% 分數增益（eco_score 公式）   ← v3: 從 60% 降到 30%
      70% 真實食物網 Jaccard 相符性   ← v3: 從 40% 升到 70%

    降低公式權重，讓推薦更貼近「真實生態系中實際共存的組合」。
    """
    present = set(species_counts.keys())
    base    = eco_score(species_counts)

    gains = []
    for s in SPECIES_IDS:
        if s in present:
            continue
        trial = dict(species_counts)
        trial[s] = 1
        score_gain, _, _ = (eco_score(trial) - base), "", ""
        score_gain       = eco_score(trial) - base          # -30~+30
        web_affinity, _, _ = _real_web_affinity(s, present) # 0~1

        # v3: 30% 分數增益 + 70% Jaccard
        combined = score_gain / 30 * 0.30 + web_affinity * 0.70
        gains.append((combined, SPECIES_IDX[s]))

    gains.sort(reverse=True)
    return [idx for _, idx in gains[:5]]


def rule_recommendation_with_source(species_counts: dict) -> list[dict]:
    """
    同 rule_recommendation，但回傳完整 metadata 供前端顯示來源。
    回傳格式：
      [{"id": "fox", "idx": 23, "network": "Yellowstone", "ref": "Ripple & Beschta 2004",
        "score_gain": 8.2, "jaccard": 0.42, "combined": 0.37}, ...]
    """
    present = set(species_counts.keys())
    base    = eco_score(species_counts)

    gains = []
    for s in SPECIES_IDS:
        if s in present:
            continue
        trial = dict(species_counts)
        trial[s] = 1
        sg             = eco_score(trial) - base
        jacc, net, ref = _real_web_affinity(s, present)
        combined       = sg / 30 * 0.30 + jacc * 0.70
        gains.append({
            "id":         s,
            "idx":        SPECIES_IDX[s],
            "score_gain": round(sg, 2),
            "jaccard":    round(jacc, 3),
            "combined":   combined,
            "network":    net,
            "ref":        ref,
        })

    gains.sort(key=lambda x: x["combined"], reverse=True)
    return gains[:5]


# ══════════════════════════════════════════════════════
# 食物網共現推薦（v4 新增）
# 推薦標籤完全來自真實食物網的 co-occurrence，不碰公式
# ══════════════════════════════════════════════════════

def jaccard_only_recommendation(species_counts: dict, top_k: int = 5) -> list[str]:
    """
    純食物網 Jaccard 共現推薦，回傳物種名稱字串列表（非 index）。
    與 foodweb_recommendation() 邏輯相同，但回傳格式為 list[str]
    以便 holdout_test.py 直接使用。

    這是 GNN 推薦頭的訓練目標，理論上是 GNN 的 ceiling。
    """
    present = set(species_counts.keys())

    has_plant = any(TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN for s in present)
    has_herb  = any(TROPHIC.get(s, 0) in (1, 1.5) for s in present)
    has_pred  = any(TROPHIC.get(s, 0) >= 2 for s in present)

    scores = []
    for s in SPECIES_IDS:
        if s in present:
            continue
        best_jacc = 0.0
        for entry in REAL_WEBS_FILTERED:
            web = entry["species"] if isinstance(entry, dict) else entry
            if s not in web:
                continue
            inter = len(present & web)
            union = len(present | web)
            jacc  = inter / union if union > 0 else 0.0
            if jacc > best_jacc:
                best_jacc = jacc

        tr = TROPHIC.get(s, -1)
        tiebreak = 0.0
        if not has_plant and tr == 0 and s not in IS_TERRAIN:
            tiebreak = 0.01
        elif not has_herb and tr in (1, 1.5):
            tiebreak = 0.008
        elif not has_pred and tr >= 2:
            tiebreak = 0.005

        scores.append((best_jacc + tiebreak, s))

    scores.sort(reverse=True)
    return [s for _, s in scores[:top_k]]


def score_gain_only_recommendation(species_counts: dict, top_k: int = 5) -> list[str]:
    """
    純分數增益推薦（eco_score delta），回傳物種名稱字串列表。
    完全不使用食物網結構，代表「只靠公式能猜到多少」。
    GNN 若能超過此線，代表圖結構學習有正向貢獻。
    holdout_test.py 用來作為 Score-only baseline。
    """
    present = set(species_counts.keys())
    base    = eco_score(species_counts)
    scores  = []
    for s in SPECIES_IDS:
        if s in present:
            continue
        trial = dict(species_counts)
        trial[s] = 1
        gain = eco_score(trial) - base
        scores.append((gain, s))
    scores.sort(reverse=True)
    return [s for _, s in scores[:top_k]]


def foodweb_recommendation(species_counts: dict) -> list[int]:
    """
    基於真實食物網共現關係的推薦，回傳 top-5 物種 index。

    邏輯：
      對每個候選物種 s，在所有真實食物網裡尋找「含有 s 的子集」，
      計算該子集與當前物種組合的 Jaccard 相似度（intersection/union），
      取最高分（best-match）作為 s 的共現分數。

      完全不使用 eco_score 公式——純粹學「什麼物種在自然界共存」。

    額外 tiebreak：若 Jaccard 分數相同，優先推薦能補齊營養層缺口的物種
    （有植物沒草食 → 草食優先；有草食沒掠食 → 掠食優先），
    讓推薦在食物網結構合理的前提下，仍有生態學意義。
    """
    present = set(species_counts.keys())

    # 計算當前營養層缺口（用於 tiebreak，不影響主排序）
    has_plant = any(TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN for s in present)
    has_herb  = any(TROPHIC.get(s, 0) in (1, 1.5) for s in present)
    has_pred  = any(TROPHIC.get(s, 0) >= 2 for s in present)

    scores = []
    for s in SPECIES_IDS:
        if s in present:
            continue

        # 主分：在真實食物網中與當前物種的最大 Jaccard
        best_jacc = 0.0
        for entry in REAL_WEBS_FILTERED:
            web = entry["species"] if isinstance(entry, dict) else entry
            if s not in web:
                continue
            inter = len(present & web)
            union = len(present | web)
            jacc  = inter / union if union > 0 else 0.0
            if jacc > best_jacc:
                best_jacc = jacc

        # Tiebreak：補齊食物鏈缺口的物種加微小 bonus（0.01，不壓過主分）
        tr = TROPHIC.get(s, -1)
        tiebreak = 0.0
        if not has_plant and tr == 0 and s not in IS_TERRAIN:
            tiebreak = 0.01   # 缺植物 → 植物加分
        elif not has_herb and tr in (1, 1.5):
            tiebreak = 0.008  # 缺草食 → 草食加分
        elif not has_pred and tr >= 2:
            tiebreak = 0.005  # 缺掠食 → 掠食加分

        scores.append((best_jacc + tiebreak, SPECIES_IDX[s]))

    scores.sort(reverse=True)
    return [idx for _, idx in scores[:5]]


# ══════════════════════════════════════════════════════
# 合成樣本生成
# ══════════════════════════════════════════════════════

plant_ids   = [s for s in SPECIES_IDS if TROPHIC.get(s, 0) == 0 and s not in IS_TERRAIN]
herb_ids    = [s for s in SPECIES_IDS if TROPHIC.get(s, 0) == 1]
pred_ids    = [s for s in SPECIES_IDS if TROPHIC.get(s, 0) >= 2]
terrain_ids = list(IS_TERRAIN)
water_ids   = [s for s in SPECIES_IDS if s in IS_WATER]


def generate_sample(n_cells: int = 35) -> dict:
    """純隨機生成（至少一種植物）"""
    n_placed = random.randint(1, min(n_cells, 20))
    chosen = {}
    p = random.choice(plant_ids)
    chosen[p] = chosen.get(p, 0) + random.randint(1, 4)
    n_placed -= 1
    pool = plant_ids * 3 + herb_ids * 2 + pred_ids + terrain_ids + water_ids
    for _ in range(n_placed):
        s = random.choice(pool)
        chosen[s] = chosen.get(s, 0) + 1
    return chosen


def generate_from_real_web() -> dict:
    """從真實食物網採樣子集（帶隨機擾動）"""
    entry   = random.choice(REAL_WEBS_FILTERED)
    web     = entry["species"] if isinstance(entry, dict) else entry
    n       = random.randint(2, len(web))
    chosen  = {s: random.randint(1, 4) for s in random.sample(sorted(web), n)}
    if random.random() < 0.3:                   # 30% 機率加一個額外物種
        extra = random.choice(SPECIES_IDS)
        chosen[extra] = chosen.get(extra, 0) + 1
    return chosen


def generate_from_real_web_perturbed() -> dict:
    """
    真實食物網子集 + 較大幅度擾動（v3 新增）：
    - 保留核心物種（佔原 web 的 50~80%）
    - 額外加入 1~3 個隨機物種（模擬玩家實驗性配置）
    - 有時移除 1 個物種（模擬生態破碎）
    讓 GNN 學習「接近真實但不完美」的生態組合。
    """
    entry  = random.choice(REAL_WEBS_FILTERED)
    web    = entry["species"] if isinstance(entry, dict) else entry
    web_l  = sorted(web)

    # 保留 50~80% 的核心物種
    keep_n = max(2, random.randint(int(len(web_l) * 0.5), int(len(web_l) * 0.8 + 1)))
    keep_n = min(keep_n, len(web_l))
    chosen = {s: random.randint(1, 3) for s in random.sample(web_l, keep_n)}

    # 加入 1~3 個隨機物種
    n_extra = random.randint(1, 3)
    for _ in range(n_extra):
        s = random.choice(SPECIES_IDS)
        chosen[s] = chosen.get(s, 0) + 1

    # 10% 機率移除一個物種（生態破碎）
    if len(chosen) > 2 and random.random() < 0.10:
        remove = random.choice(list(chosen.keys()))
        del chosen[remove]

    return chosen


# ══════════════════════════════════════════════════════
# Dataset / DataLoader
# ══════════════════════════════════════════════════════

class EcoDataset(Dataset):
    def __init__(self, samples, scores, recs):
        self.samples = samples
        self.scores  = scores
        self.recs    = recs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        data  = build_graph(self.samples[idx])
        score = torch.tensor([self.scores[idx] / 100.0], dtype=torch.float)
        return data, score, self.recs[idx]


def collate_fn(batch):
    from torch_geometric.data import Batch as PygBatch
    graphs, scores, recs = zip(*batch)
    batched = PygBatch.from_data_list(list(graphs))
    scores  = torch.cat(scores, dim=0)
    return batched, scores, list(recs)


# ══════════════════════════════════════════════════════
# Google Sheets 真實玩家資料
# ══════════════════════════════════════════════════════

def load_sheets_data(sheet_id: str, creds_path: str) -> list[dict]:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(sheet_id).sheet1
        rows = ws.get_all_records()
        results = []
        for row in rows:
            try:
                cells_json = json.loads(row["cells_json"])
                counts = {}
                for c in cells_json:
                    if c and c.get("id"):
                        counts[c["id"]] = counts.get(c["id"], 0) + 1
                # 用新公式重算（不沿用舊分數）
                results.append({"counts": counts, "score": eco_score(counts)})
            except Exception:
                continue
        print(f"[Sheets] Loaded {len(results)} records (rescored with v2 formula)")
        return results
    except Exception as e:
        print(f"[Sheets] Skipping: {e}")
        return []


# ══════════════════════════════════════════════════════
# 主訓練迴圈
# ══════════════════════════════════════════════════════

def train(
    n_synthetic: int = 5000,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 3e-4,
    sheet_id: str | None = None,
    creds_path: str = "google_creds.json",
    retrain_s2v: bool = False,
):
    # ── Step 0：確保 Species2Vec 已訓練 ───────────────
    # 必須在 build_graph() 被呼叫前完成（EcoDataset.__getitem__ 會呼叫它）
    _ensure_species2vec(force=retrain_s2v)

    # 重置 gnn.py 裡的 embedding cache（如果剛重訓了 s2v 的話）
    import models.gnn as _gnn_mod
    _gnn_mod._S2V_EMBEDDINGS = None
    in_dim = _gnn_mod.get_node_feature_dim()
    print(f"[Train] Node feature dim: {in_dim}  "
          f"({'Species2Vec ' + str(in_dim-1) + 'd + count' if in_dim > 5 else 'fallback 5d hand-crafted'})")

    # ── 建立樣本 ──────────────────────────────────────
    # v3: 10% 純隨機 + 60% 真實食物網 + 30% 真實食物網帶擾動
    n_real_base    = int(n_synthetic * 0.60)   # ← 從 30% → 60%
    n_real_perturb = int(n_synthetic * 0.30)   # ← 新增：帶擾動版
    n_random       = n_synthetic - n_real_base - n_real_perturb  # 剩下 10%
    print(f"[Train] {n_random} random + {n_real_base} real-web + {n_real_perturb} real-web-perturbed")

    samples, scores, recs = [], [], []

    for _ in range(n_random):
        sc = generate_sample()
        samples.append(sc); scores.append(eco_score(sc)); recs.append(foodweb_recommendation(sc))

    for _ in range(n_real_base):
        sc = generate_from_real_web()
        samples.append(sc); scores.append(eco_score(sc)); recs.append(foodweb_recommendation(sc))

    for _ in range(n_real_perturb):
        sc = generate_from_real_web_perturbed()
        samples.append(sc); scores.append(eco_score(sc)); recs.append(foodweb_recommendation(sc))

    if sheet_id:
        for item in load_sheets_data(sheet_id, creds_path):
            samples.append(item["counts"])
            scores.append(item["score"])
            recs.append(foodweb_recommendation(item["counts"]))
        print(f"[Train] Total: {len(samples)}")

    print(f"[Train] Score distribution: "
          f"min={min(scores):.1f}  mean={sum(scores)/len(scores):.1f}  max={max(scores):.1f}")

    # ── Split 90/10 ───────────────────────────────────
    n   = len(samples)
    idx = list(range(n)); random.shuffle(idx)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    def make_ds(ii):
        return EcoDataset([samples[i] for i in ii],
                          [scores[i]  for i in ii],
                          [recs[i]    for i in ii])

    tr_dl = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    va_dl = DataLoader(make_ds(va_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ── 模型 ──────────────────────────────────────────
    model     = EcoGNN(in_dim=in_dim)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss  = nn.MSELoss()

    best_val = float("inf")
    print("[Train] Starting...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for graphs, score_labels, rec_labels in tr_dl:
            optimizer.zero_grad()
            score_pred, rec_pred = model(graphs)

            loss_score = mse_loss(score_pred, score_labels)

            rec_target = torch.zeros(len(rec_labels), N_SPECIES)
            for i, top5 in enumerate(rec_labels):
                for j in top5:
                    rec_target[i, j] = 1.0
            loss_rec = F.binary_cross_entropy_with_logits(rec_pred, rec_target)

            loss = loss_score + 0.3 * loss_rec
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss, score_errs = 0.0, []
        with torch.no_grad():
            for graphs, score_labels, _ in va_dl:
                sp, _ = model(graphs)
                val_loss += mse_loss(sp, score_labels).item()
                score_errs.extend((sp * 100 - score_labels * 100).abs().tolist())

        mae = np.mean(score_errs)
        print(f"  Epoch {epoch:3d} | train={total_loss/len(tr_dl):.4f} "
              f"| val={val_loss/len(va_dl):.4f} | score_MAE={mae:.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Saved best model")

    print(f"\n[Train] Done! Best model → {MODEL_PATH}")
    print(f"[Train] Final score MAE: {mae:.2f} / 100")
    _print_score_examples()


def _print_score_examples():
    """訓練後印出分數範例供驗證"""
    examples = [
        ("空生態系",          {}),
        ("只有草",            {"grass": 5}),
        ("草 + 兔",           {"grass": 4, "rabbit": 2}),
        ("草 + 兔 + 狐",      {"grass": 4, "rabbit": 3, "fox": 1}),
        ("草 + 鹿 + 狼",      {"grass": 5, "deer": 3, "wolf": 1}),
        ("完整草原",          {"grass": 4, "shrub": 2, "deer": 3,
                               "rabbit": 2, "wolf": 1, "eagle": 1, "bee": 1}),
        ("水陸雙生態",        {"grass": 3, "carp": 2, "frog": 2,
                               "pond": 1, "duck": 1, "snake": 1, "eagle": 1}),
        ("Yellowstone 子集",  {"grass": 4, "deer": 3, "wolf": 2,
                               "eagle": 1, "bird": 2}),
        ("水生動物無水體",    {"shrimp": 3, "crab": 2, "salmon": 1}),
        ("完整水域",          {"seaweed": 3, "shrimp": 2, "carp": 2,
                               "pond": 1, "duck": 1, "frog": 1}),
    ]
    print("\n── Score Examples (v2 formula) ──")
    for name, counts in examples:
        print(f"  {name:<22} → {eco_score(counts):5.1f}")


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EcoChain GNN")
    parser.add_argument("--n",          type=int,   default=5000,
                        help="合成樣本數（30%% 會來自真實食物網）")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--sheet_id",   type=str,   default=None)
    parser.add_argument("--creds",      type=str,   default="google_creds.json")
    parser.add_argument("--score_test", action="store_true",
                        help="只印分數範例，不訓練（不需要 models/gnn.py）")
    parser.add_argument("--retrain_s2v", action="store_true",
                        help="強制重新訓練 Species2Vec（即使 species2vec.pt 已存在）")
    args = parser.parse_args()

    if args.score_test:
        # 快速驗證公式，不需要 torch / gnn
        _print_score_examples()
    else:
        train(
            n_synthetic=args.n,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            sheet_id=args.sheet_id,
            creds_path=args.creds,
            retrain_s2v=args.retrain_s2v,
        )
