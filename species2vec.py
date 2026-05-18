"""
species2vec.py — Species Co-occurrence Embedding Pre-trainer
=============================================================
從真實食物網資料學習每個物種的 16 維生態位向量（ecological niche embedding）。

原理（類似 Word2Vec skip-gram）：
  - 在同一個食物網子集裡出現的物種 → 正樣本（label=1）
  - 從未在同一子集出現的物種 → 負樣本（label=0）
  - 訓練一個小型神經網路預測「兩物種會不會共存」
  - 學完後，每個物種的 embedding 向量代表它的「生態位」

輸出：
  species2vec.pt — { "embeddings": Tensor(N_SPECIES, 16), "species_ids": [...] }

使用方式：
  python species2vec.py                    # 訓練並儲存
  python species2vec.py --epochs 500 --dim 16   # 自訂參數

在 gnn.py 的 build_graph() 裡，這個 embedding 取代原本手動設計的 5 維節點特徵：
  [trophic/3, is_water, is_terrain, count, degree]  ← 舊（人工規則）
  [embed_16dim..., log_count]                       ← 新（資料驅動，17 維）
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations

# ──────────────────────────────────────────────
# 物種清單（與 gnn.py 完全相同）
# ──────────────────────────────────────────────
SPECIES_IDS = [
    "grass","flower","berry","tree","shrub","mushroom","wheat","cactus","lotus","seaweed",
    "sheep","cow","rabbit","deer","horse","goat",
    "cat","dog","chicken","pig","bee","butterfly","bird",
    "fox","wolf","eagle","snake","owl",
    "carp","salmon","frog","shrimp","turtle","duck","crab","snail",
    "pond","compost","birdhouse","stream","rockpile","fence",
]
N_SPECIES = len(SPECIES_IDS)
SPECIES_IDX = {s: i for i, s in enumerate(SPECIES_IDS)}

# 預設輸出路徑（與 gnn.py 同目錄）
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "species2vec.pt")
REAL_WEBS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_food_webs.json")


# ──────────────────────────────────────────────
# Step 1：載入食物網資料
# ──────────────────────────────────────────────
def load_food_webs() -> list[list[str]]:
    """
    載入所有真實食物網子集，回傳 list of species lists。
    優先讀 real_food_webs.json，fallback 到內建資料。
    """
    # 嘗試讀取 real_food_webs.json
    if os.path.exists(REAL_WEBS_JSON):
        try:
            with open(REAL_WEBS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            webs = []
            for entry in data.get("game_webs", []):
                sp = [s for s in entry.get("species", []) if s in SPECIES_IDX]
                if len(sp) >= 2:
                    webs.append(sp)
            if webs:
                print(f"[Species2Vec] Loaded {len(webs)} food web subsets from real_food_webs.json")
                return webs
        except Exception as e:
            print(f"[Species2Vec] Warning: could not load real_food_webs.json: {e}")

    # Fallback：內建真實食物網（來自 train.py 的手工整理）
    print("[Species2Vec] Using built-in food web data (run fetch_mangal.py for more data)")
    return [
        # Ythan Estuary
        ["grass","shrimp","crab","duck"], ["seaweed","snail","crab","duck"],
        ["grass","shrimp","frog","snake","bird"], ["seaweed","shrimp","crab","salmon"],
        ["seaweed","turtle","crab","shrimp"],
        # Tuesday Lake
        ["grass","carp","duck","frog"], ["seaweed","carp","shrimp","frog","snake"],
        ["lotus","carp","shrimp","duck","turtle"], ["seaweed","shrimp","frog","snake","bird"],
        ["lotus","carp","turtle","shrimp"],
        # Yellowstone
        ["grass","deer","wolf","eagle","bird"], ["grass","rabbit","fox","eagle"],
        ["shrub","rabbit","fox","owl"], ["grass","sheep","wolf","eagle"],
        ["grass","deer","rabbit","fox","wolf"], ["tree","deer","wolf","owl","bird"],
        # Patagonian shelf
        ["seaweed","shrimp","crab","salmon","eagle"], ["seaweed","carp","salmon","bird"],
        ["lotus","shrimp","turtle","crab"], ["seaweed","snail","crab","salmon"],
        ["lotus","carp","shrimp","duck"],
        # UK Farmland
        ["wheat","sheep","chicken","fox","owl"], ["grass","cow","chicken","fox","bird"],
        ["wheat","rabbit","fox","owl","bird"], ["grass","sheep","cow","bee","butterfly"],
        ["flower","bee","butterfly","bird","fox"], ["berry","rabbit","fox","owl"],
        ["mushroom","rabbit","fox","bird"],
        # Białowieża Forest
        ["tree","deer","wolf","owl","bird"], ["shrub","rabbit","fox","owl"],
        ["tree","mushroom","snail","bird","fox"], ["berry","deer","fox","wolf"],
        ["tree","bird","bee","owl"], ["shrub","deer","rabbit","fox","wolf"],
        ["tree","deer","snake","eagle"], ["berry","rabbit","fox","eagle"],
        # Broadstone Stream / Skipwith Pond
        ["seaweed","shrimp","frog","snake"], ["lotus","shrimp","frog","duck"],
        ["seaweed","snail","frog","bird"], ["lotus","carp","shrimp","frog","snake"],
        ["seaweed","shrimp","carp","frog"], ["lotus","shrimp","duck","frog","snake"],
        ["seaweed","snail","crab","duck"],
        # Silwood Park
        ["grass","bee","butterfly","bird"], ["flower","bee","butterfly","bird"],
        ["shrub","bee","bird","fox"],
    ]


# ──────────────────────────────────────────────
# Step 2：建立共現矩陣
# ──────────────────────────────────────────────
def build_cooccurrence(food_webs: list[list[str]]) -> np.ndarray:
    """
    統計哪些物種常常一起出現在同一個食物網子集裡。
    回傳 (N_SPECIES × N_SPECIES) 的共現次數矩陣（對稱）。
    """
    cooccur = np.zeros((N_SPECIES, N_SPECIES), dtype=np.float32)
    for web in food_webs:
        valid = [s for s in web if s in SPECIES_IDX]
        for a, b in combinations(valid, 2):
            i, j = SPECIES_IDX[a], SPECIES_IDX[b]
            cooccur[i][j] += 1
            cooccur[j][i] += 1  # 對稱

    n_pairs = int((cooccur > 0).sum() / 2)
    print(f"[Species2Vec] Co-occurrence matrix: {N_SPECIES}×{N_SPECIES}, "
          f"{n_pairs} positive pairs from {len(food_webs)} webs")
    return cooccur


# ──────────────────────────────────────────────
# Step 3：訓練資料（正負樣本）
# ──────────────────────────────────────────────
def make_training_pairs(cooccur: np.ndarray, neg_ratio: float = 3.0):
    """
    從共現矩陣生成訓練樣本。
    正樣本：共現次數 > 0 的物種對 (label=1.0)
    負樣本：從未共現的物種對 (label=0.0)，數量 = 正樣本 × neg_ratio

    neg_ratio > 1 是因為負樣本遠多於正樣本（多數物種不共存），
    需要過採樣正樣本或欠採樣負樣本以平衡。
    """
    pos_pairs, neg_pairs = [], []
    for i in range(N_SPECIES):
        for j in range(i + 1, N_SPECIES):
            if cooccur[i][j] > 0:
                pos_pairs.append((i, j, 1.0))
            else:
                neg_pairs.append((i, j, 0.0))

    # 限制負樣本數量（否則嚴重不平衡）
    n_neg = min(len(neg_pairs), int(len(pos_pairs) * neg_ratio))
    import random
    random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:n_neg]

    all_pairs = pos_pairs + neg_pairs
    print(f"[Species2Vec] Training pairs: {len(pos_pairs)} positive + {len(neg_pairs)} negative")
    return all_pairs


# ──────────────────────────────────────────────
# Step 4：Species2Vec 模型
# ──────────────────────────────────────────────
class Species2Vec(nn.Module):
    """
    每個物種學一個 embed_dim 維的生態位向量。
    訓練目標：兩物種向量的點積 → 預測共存機率。

    物種共存 (label=1)  → 點積正、向量相近
    物種不共存 (label=0) → 點積負、向量相遠

    訓練後：
    - wolf 和 deer 的向量相近（常共存）
    - wolf 和 shrimp 的向量相遠（從不共存）
    - 同一個食物網裡的物種形成語義叢集
    """
    def __init__(self, n_species: int = N_SPECIES, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(n_species, embed_dim)
        # 用小的初始化值，避免梯度爆炸
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, species_a: torch.Tensor, species_b: torch.Tensor) -> torch.Tensor:
        """
        回傳兩個物種的共現 logit（未過 sigmoid）。
        越大 = 越可能共存。
        """
        ea = self.embedding(species_a)   # (batch, embed_dim)
        eb = self.embedding(species_b)   # (batch, embed_dim)
        return (ea * eb).sum(dim=-1)     # 點積 → (batch,)

    def get_embeddings(self) -> torch.Tensor:
        """回傳所有物種的 embedding，shape: (N_SPECIES, embed_dim)"""
        return self.embedding.weight.detach()


# ──────────────────────────────────────────────
# Step 5：訓練
# ──────────────────────────────────────────────
def train_species2vec(
    embed_dim: int = 16,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 0.01,
    output_path: str = DEFAULT_OUTPUT,
) -> torch.Tensor:
    """
    訓練 Species2Vec，回傳並儲存 embedding 矩陣。
    """
    food_webs = load_food_webs()
    cooccur   = build_cooccurrence(food_webs)
    pairs     = make_training_pairs(cooccur)

    model     = Species2Vec(n_species=N_SPECIES, embed_dim=embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    # 轉成 tensor（一次性）
    a_all = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    b_all = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    y_all = torch.tensor([p[2] for p in pairs], dtype=torch.float)

    n = len(pairs)
    print(f"[Species2Vec] Training {N_SPECIES} species × {embed_dim}d embedding "
          f"for {epochs} epochs...")

    model.train()
    for epoch in range(1, epochs + 1):
        # Mini-batch SGD
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx   = perm[start:start + batch_size]
            pred  = model(a_all[idx], b_all[idx])
            loss  = loss_fn(pred, y_all[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch % 50 == 0 or epoch == epochs:
            avg_loss = total_loss / n
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")

    # ── 取出 embedding 並儲存 ──────────────────────
    embeddings = model.get_embeddings()   # (N_SPECIES, embed_dim)

    # 正規化（讓每個向量的 L2 norm ≈ 1，讓 GNN 輸入更穩定）
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-6)
    embeddings_normed = embeddings / norms

    save_data = {
        "embeddings": embeddings_normed,
        "species_ids": SPECIES_IDS,
        "embed_dim": embed_dim,
        "n_webs": len(food_webs),
    }
    torch.save(save_data, output_path)
    print(f"\n[Species2Vec] Saved to {output_path}")
    print(f"  Shape: {embeddings_normed.shape}")

    # ── 驗證：印出幾個物種的最近鄰 ──────────────────
    _print_nearest_neighbors(embeddings_normed, k=5)

    return embeddings_normed


def _print_nearest_neighbors(embeddings: torch.Tensor, k: int = 5):
    """印出幾個代表性物種的最近鄰，用於驗證 embedding 品質"""
    test_species = ["wolf", "grass", "shrimp", "bee", "salmon"]
    print("\n[Species2Vec] Nearest neighbors (sanity check):")
    for s in test_species:
        if s not in SPECIES_IDX:
            continue
        i = SPECIES_IDX[s]
        v = embeddings[i]
        sims = (embeddings @ v).tolist()  # cosine sim（已正規化）
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        neighbors = [SPECIES_IDS[j] for j, _ in ranked[1:k+1]]  # 跳過自己
        print(f"  {s:<12} → {', '.join(neighbors)}")


# ──────────────────────────────────────────────
# 載入（供 gnn.py 使用）
# ──────────────────────────────────────────────
_CACHED_EMBEDDINGS: torch.Tensor | None = None

def load_species2vec(path: str = DEFAULT_OUTPUT) -> torch.Tensor | None:
    """
    載入預訓練 embedding。若檔案不存在回傳 None。
    結果會被 cache，多次呼叫只讀一次磁碟。
    """
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS

    if not os.path.exists(path):
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        emb  = data["embeddings"]   # (N_SPECIES, embed_dim)
        _CACHED_EMBEDDINGS = emb
        print(f"[Species2Vec] Loaded embeddings {emb.shape} from {path}")
        return emb
    except Exception as e:
        print(f"[Species2Vec] Warning: could not load {path}: {e}")
        return None


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Species2Vec ecological niche embeddings")
    parser.add_argument("--dim",    type=int,   default=16,  help="Embedding dimension (default: 16)")
    parser.add_argument("--epochs", type=int,   default=300, help="Training epochs (default: 300)")
    parser.add_argument("--lr",     type=float, default=0.01,help="Learning rate (default: 0.01)")
    parser.add_argument("--batch",  type=int,   default=64,  help="Batch size (default: 64)")
    parser.add_argument("--output", type=str,   default=DEFAULT_OUTPUT, help="Output .pt path")
    args = parser.parse_args()

    train_species2vec(
        embed_dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        output_path=args.output,
    )
