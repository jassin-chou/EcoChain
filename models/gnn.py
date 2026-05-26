"""
EcoChain GNN Model
==================
Graph Neural Network for ecosystem health scoring and species recommendation.

Architecture:
  - Node features (17d): Species2Vec embedding (16d) + log_count (1d)
    ↑ 如果 species2vec.pt 不存在，fallback 到原本手工設計的 5 維特徵
  - Edge features: [relation_type_onehot(3)]  — predation / pollination / symbiosis
  - 3-layer GAT with skip connections → global mean pool + max pool → MLP head
  - Output 1: eco_score  (0–100, regression)
  - Output 2: next_species logits (len=N_SPECIES, multi-class recommendation)

節點特徵說明（v2：Species2Vec）：
  原本 (5維)  : [trophic/4, is_water, is_terrain, log_count/5, degree]  ← 人工規則
  現在 (17維) : [embed_0..embed_15, log_count]                          ← 資料驅動
  
  Species2Vec embedding 由 species2vec.py 預訓練：
  - 輸入：real_food_webs.json 中的真實食物網共現關係
  - 目標：同一食物網子集中的物種 → 向量相近
  - 結果：wolf 和 deer 向量相近，wolf 和 shrimp 向量相遠
  
  GNN 架構本身不變，只有輸入維度 in_dim: 5 → 17
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ──────────────────────────────────────────────
# Species catalog (mirrors frontend CAT)
# ──────────────────────────────────────────────
SPECIES_IDS = [
    # plants
    "grass","flower","berry","tree","shrub","mushroom","wheat","cactus","lotus","seaweed",
    # herbivore
    "sheep","cow","rabbit","deer","horse","goat",
    # omnivore
    "cat","dog","chicken","pig","bee","butterfly","bird",
    # predator
    "fox","wolf","eagle","snake","owl",
    # aquatic
    "carp","salmon","frog","shrimp","turtle","duck","crab","snail",
    # terrain
    "pond","compost","birdhouse","stream","rockpile","fence",
]
N_SPECIES = len(SPECIES_IDS)
SPECIES_IDX = {s: i for i, s in enumerate(SPECIES_IDS)}

TROPHIC = {
    "grass":0,"flower":0,"berry":0,"tree":0,"shrub":0,"mushroom":0,"wheat":0,"cactus":0,"lotus":0,"seaweed":0,
    "sheep":1,"cow":1,"rabbit":1,"deer":1,"horse":1,"goat":1,
    "cat":1.5,"dog":1.5,"chicken":1.5,"pig":1.5,"bee":0.5,"butterfly":0.5,"bird":1.5,
    "fox":2,"wolf":2,"eagle":3,"snake":2,"owl":2.5,
    "carp":1,"salmon":1.5,"frog":1.5,"shrimp":1,"turtle":1.5,"duck":1.5,"crab":1,"snail":0.5,
    "pond":-1,"compost":-1,"birdhouse":-1,"stream":-1,"rockpile":-1,"fence":-1,
}
IS_WATER = {"lotus","seaweed","carp","salmon","frog","shrimp","turtle","duck","crab","snail"}
IS_TERRAIN = {"pond","compost","birdhouse","stream","rockpile","fence"}

# Predation edges (predator → prey)
PRED_EDGES = {
    "fox":["rabbit","chicken"],"wolf":["sheep","deer","rabbit"],"eagle":["rabbit","chicken","bird","frog"],
    "snake":["rabbit","frog","shrimp"],"owl":["rabbit","chicken","bird"],"salmon":["carp","shrimp","frog"],
    "frog":["bee","butterfly"],"bird":["bee","butterfly","berry"],
}
POLL_EDGES = {"bee":["flower","berry","lotus","wheat"],"butterfly":["flower","berry"]}
SYM_EDGES  = {
    "compost":["grass","flower","wheat","shrub","berry","tree"],
    "birdhouse":["bird","owl","eagle"],
    "stream":["frog","duck","carp","salmon"],
    "rockpile":["snake"],
    "fence":["sheep","cow","rabbit","deer"],
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ecochain_gnn.pt")

# species2vec.pt 可能在：
#   (A) 與 gnn.py 同一層（專案根目錄直接放 gnn.py 時）
#   (B) gnn.py 在 models/ 子資料夾，pt 在上一層（最常見情況）
def _find_s2v_path() -> str:
    """往上最多找兩層，找到就回傳，找不到回傳預設路徑（讓後續自然 fallback）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate_dir in [here, os.path.dirname(here)]:
        p = os.path.join(candidate_dir, "species2vec.pt")
        if os.path.exists(p):
            return p
    return os.path.join(here, "species2vec.pt")   # 不存在時的佔位路徑

S2V_PATH = _find_s2v_path()

# ──────────────────────────────────────────────
# Species2Vec embedding loader（lazy，只讀一次）
# ──────────────────────────────────────────────
_S2V_EMBEDDINGS: torch.Tensor | None = None   # (N_SPECIES, 16)
_S2V_DIM: int = 16

def _load_s2v() -> tuple[torch.Tensor | None, int]:
    """
    嘗試載入 species2vec.pt。
    回傳 (embeddings, embed_dim)；若不存在回傳 (None, 0)。
    """
    global _S2V_EMBEDDINGS, _S2V_DIM
    if _S2V_EMBEDDINGS is not None:
        return _S2V_EMBEDDINGS, _S2V_DIM

    if not os.path.exists(S2V_PATH):
        return None, 0

    try:
        data = torch.load(S2V_PATH, map_location="cpu", weights_only=True)
        emb  = data["embeddings"]   # (N_SPECIES, embed_dim)
        _S2V_EMBEDDINGS = emb
        _S2V_DIM        = emb.shape[1]
        print(f"[GNN] Species2Vec loaded: {emb.shape} from {S2V_PATH}")
        return emb, _S2V_DIM
    except Exception as e:
        print(f"[GNN] Warning: could not load species2vec.pt: {e}")
        return None, 0


def get_node_feature_dim() -> int:
    """
    動態偵測節點特徵維度：
    - 有 species2vec.pt → embed_dim + 1（count feature）
    - 沒有 → 5（原本的手工特徵，向後相容）
    """
    if _FORCE_HANDCRAFTED_FEATURES:
        return 5
    _, dim = _load_s2v()
    return dim + 1 if dim > 0 else 5


# ──────────────────────────────────────────────
# Build a PyG Data object from a species count dict
# ──────────────────────────────────────────────
def build_graph(species_counts: dict) -> "Data":
    """
    species_counts: { "grass": 3, "rabbit": 2, ... }
    Returns a PyG Data object.

    節點特徵（v2，有 species2vec.pt）：
      [embed_0..embed_15, log_count]  → 17 維，資料驅動
    節點特徵（fallback，無 species2vec.pt）：
      [trophic/4, is_water, is_terrain, log_count/5, degree]  → 5 維，人工規則
    """
    emb_matrix, emb_dim = _load_s2v()
    use_s2v = (emb_matrix is not None) and not _FORCE_HANDCRAFTED_FEATURES

    # 空圖的 dummy 維度
    feat_dim = (emb_dim + 1) if use_s2v else 5

    present = [s for s in species_counts if species_counts[s] > 0]
    if not present:
        x = torch.zeros((1, feat_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 3), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    batch=torch.zeros(1, dtype=torch.long))

    idx_map = {s: i for i, s in enumerate(present)}

    # ── 節點特徵 ──────────────────────────────────
    x_rows = []
    for s in present:
        cnt = np.log1p(species_counts.get(s, 1)) / 5.0

        if use_s2v:
            # v2（資料驅動）：Species2Vec embedding + count
            sp_idx = SPECIES_IDX.get(s)
            if sp_idx is not None:
                embed = emb_matrix[sp_idx]          # (embed_dim,)
            else:
                embed = torch.zeros(emb_dim)        # 未知物種用零向量
            count_feat = torch.tensor([cnt], dtype=torch.float)
            x_rows.append(torch.cat([embed, count_feat]))   # (embed_dim+1,)
        else:
            # fallback（人工規則）：原本的 5 維
            tr  = (TROPHIC.get(s, 0) + 1) / 4.0   # normalise –1..3 → 0..1
            iw  = float(s in IS_WATER)
            it  = float(s in IS_TERRAIN)
            x_rows.append(torch.tensor([tr, iw, it, cnt, 0.0], dtype=torch.float))

    x = torch.stack(x_rows, dim=0)   # (n_present, feat_dim)

    # ── 邊 ──────────────────────────────────────
    src, dst, eattr = [], [], []
    def add_edge(a, b, etype):
        if a in idx_map and b in idx_map:
            src.append(idx_map[a]); dst.append(idx_map[b])
            oh = [0.0, 0.0, 0.0]; oh[etype] = 1.0
            eattr.append(oh)
    for pred, preys in PRED_EDGES.items():
        for prey in preys: add_edge(pred, prey, 0)
    for poll, plants in POLL_EDGES.items():
        for p in plants: add_edge(poll, p, 1)
    for fac, targets in SYM_EDGES.items():
        for t in targets: add_edge(fac, t, 2)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float)
        # degree 只在 fallback 模式下填入 x[:,4]
        if not use_s2v:
            deg = torch.zeros(len(present))
            for s in src: deg[s] += 1
            x[:, 4] = deg / (deg.max() + 1e-6)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 3), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=torch.zeros(len(present), dtype=torch.long))


# ──────────────────────────────────────────────
# GNN Architecture
# ──────────────────────────────────────────────
class EcoGNN(nn.Module):
    """
    3-layer GAT + global pool → dual-head MLP
    Head 1: eco_score  (scalar 0–1, multiply ×100 for display)
    Head 2: species recommendation logits (N_SPECIES)

    in_dim 由外部傳入（由 get_node_feature_dim() 動態決定）：
    - 有 species2vec.pt → 17（16d embedding + 1d count）
    - 沒有 → 5（原本手工特徵，向後相容）
    """
    def __init__(self, in_dim: int | None = None, hidden: int = 64, n_species: int = N_SPECIES):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric not installed")
        if in_dim is None:
            in_dim = get_node_feature_dim()
        self.in_dim = in_dim
        self.conv1 = GATConv(in_dim,   hidden, heads=4, concat=True,  dropout=0.1, edge_dim=3)
        self.conv2 = GATConv(hidden*4, hidden, heads=4, concat=True,  dropout=0.1)
        self.conv3 = GATConv(hidden*4, hidden, heads=1, concat=False, dropout=0.1)

        self.bn1 = nn.BatchNorm1d(hidden*4)
        self.bn2 = nn.BatchNorm1d(hidden*4)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        # Recommendation head
        self.rec_head = nn.Sequential(
            nn.Linear(hidden*2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_species)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ea = data.edge_attr if data.edge_attr.shape[0] > 0 else None

        # Layer 1 (with edge_attr only if edges exist)
        if ea is not None and edge_index.shape[1] > 0:
            x1 = F.elu(self.bn1(self.conv1(x, edge_index, ea)))
        else:
            x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        x3 = F.elu(self.bn3(self.conv3(x2, edge_index)))

        # Global readout: concat mean + max
        g_mean = global_mean_pool(x3, batch)
        g_max  = global_max_pool(x3, batch)
        g = torch.cat([g_mean, g_max], dim=-1)

        score  = self.score_head(g).squeeze(-1)   # (batch,)
        rec    = self.rec_head(g)                  # (batch, N_SPECIES)
        return score, rec


# ──────────────────────────────────────────────
# Inference helper (singleton)
# ──────────────────────────────────────────────
_model: EcoGNN | None = None
_FORCE_HANDCRAFTED_FEATURES = False

def get_model() -> EcoGNN:
    global _model, _FORCE_HANDCRAFTED_FEATURES
    if _model is None:
        in_dim = get_node_feature_dim()
        model = EcoGNN(in_dim=in_dim)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            # 若 checkpoint 的 in_dim 與當前不符（舊模型 5d vs 新模型 17d），重建模型
            first_key = "conv1.lin_src.weight"
            if first_key not in state and "conv1.lin.weight" in state:
                first_key = "conv1.lin.weight"
            if first_key in state:
                ckpt_in_dim = state[first_key].shape[1]
                if ckpt_in_dim != in_dim:
                    print(f"[GNN] Checkpoint in_dim={ckpt_in_dim} ≠ current in_dim={in_dim}")
                    print(f"[GNN] Rebuilding model with in_dim={ckpt_in_dim} (re-run train.py to retrain)")
                    _FORCE_HANDCRAFTED_FEATURES = (ckpt_in_dim == 5)
                    model = EcoGNN(in_dim=ckpt_in_dim)
            model.load_state_dict(state)
            model.eval()
            _model = model
            print(f"[GNN] Loaded trained weights (in_dim={_model.in_dim}) from {MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"GNN weights not found at {MODEL_PATH}. Run `python train.py` first."
            )
    return _model


def predict(species_counts: dict) -> dict:
    """
    Input : { "grass": 3, "rabbit": 2 }
    Output: { "score": 72.4, "top_recommendations": ["fox","bee","pond"] }
    """
    model = get_model()
    data  = build_graph(species_counts)

    with torch.no_grad():
        score_t, rec_t = model(data)

    score = float(score_t[0]) * 100.0
    score = round(max(5.0, min(100.0, score)), 1)

    # Mask out already-present species and get top-5 recommendations
    present_set = set(species_counts.keys())
    logits = rec_t[0].numpy()
    masked = logits.copy()
    for s in present_set:
        if s in SPECIES_IDX:
            masked[SPECIES_IDX[s]] -= 1000.0  # suppress present species

    top_idx = np.argsort(masked)[::-1][:5]
    top_rec = [SPECIES_IDS[i] for i in top_idx]

    return {
        "score": score,
        "top_recommendations": top_rec,
    }
