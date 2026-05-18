#!/usr/bin/env python3
"""
holdout_test.py — GNN vs. 公式 真實食物網 Hold-out 測試
========================================================
測試邏輯：
  1. 從 real_food_webs.json 取出每個真實食物網子集
  2. 隨機藏掉其中一個「非植物」物種（藏起來當正確答案）
  3. 讓 GNN 和公式各自推薦 Top-5
  4. 看誰猜到了被藏起來的物種（Hit@5）
  5. 比較命中率

為什麼這樣設計：
  - 真實食物網提供了「這些物種應該共存」的生態學依據
  - 藏掉一個物種再猜，有真正的正確答案可以對照
  - 這是推薦系統標準評估方法（類似 Netflix 藏掉評分再預測）

產生：
  holdout_results.png  — 命中率長條圖 + 詳細案例
  holdout_report.txt   — 純文字摘要
"""

import os, sys, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gnn import predict, TROPHIC, IS_TERRAIN, SPECIES_IDS
from train import (
    eco_score,
    rule_recommendation_with_source,
    jaccard_only_recommendation,
    score_gain_only_recommendation,
    REAL_WEBS_FILTERED,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

EMOJI = {
    "grass":"🌿","flower":"🌸","berry":"🫐","tree":"🌳","shrub":"🍃","mushroom":"🍄",
    "wheat":"🌾","cactus":"🌵","lotus":"🪷","seaweed":"🪸","sheep":"🐑","cow":"🐄",
    "rabbit":"🐰","deer":"🦌","horse":"🐎","goat":"🐐","cat":"🐱","dog":"🐶",
    "chicken":"🐔","pig":"🐷","bee":"🐝","butterfly":"🦋","bird":"🐦","fox":"🦊",
    "wolf":"🐺","eagle":"🦅","snake":"🐍","owl":"🦉","carp":"🐟","salmon":"🐠",
    "frog":"🐸","shrimp":"🦐","turtle":"🐢","duck":"🦆","crab":"🦀","snail":"🐌",
    "pond":"🏞️","compost":"♻️","birdhouse":"🏡","stream":"💧","rockpile":"🪨","fence":"🚧",
}

# ════════════════════════════════════════════════════════════
# 核心測試函式
# ════════════════════════════════════════════════════════════

def run_holdout(n_trials=100, top_k=5, seed=42):
    """
    從真實食物網做 hold-out 測試。
    回傳每個 trial 的結果 list。
    """
    random.seed(seed)
    results = []
    used = 0

    # 把 REAL_WEBS_FILTERED 展開成可用的 (species_set, network, ref) 列表
    webs = []
    for entry in REAL_WEBS_FILTERED:
        sp = entry["species"] if isinstance(entry, dict) else entry
        nm = entry.get("network", "unknown") if isinstance(entry, dict) else "unknown"
        rf = entry.get("ref", "") if isinstance(entry, dict) else ""
        if len(sp) >= 3:
            webs.append((set(sp), nm, rf))

    if not webs:
        print("[Error] 沒有可用的真實食物網資料")
        return []

    print(f"[Holdout] 可用真實食物網子集: {len(webs)} 個")
    print(f"[Holdout] 執行 {n_trials} 次 hold-out 測試（Top-{top_k}）\n")

    attempts = 0
    while used < n_trials and attempts < n_trials * 10:
        attempts += 1
        sp_set, network, ref = random.choice(webs)
        sp_list = list(sp_set)

        # 找可以「藏掉」的候選物種：非植物、非地形、至少剩 2 個物種
        candidates = [
            s for s in sp_list
            if TROPHIC.get(s, 0) > 0 and s not in IS_TERRAIN
        ]
        if not candidates:
            continue

        hidden = random.choice(candidates)
        remaining = {s: random.randint(1, 3) for s in sp_list if s != hidden}
        if len(remaining) < 2:
            continue

        # GNN 推薦
        gnn_result = predict(remaining)
        gnn_recs   = gnn_result["top_recommendations"][:top_k]
        gnn_hit    = hidden in gnn_recs

        # 混合公式推薦（30% 分數增益 + 70% Jaccard）
        formula_recs_raw = rule_recommendation_with_source(remaining)
        formula_recs     = [r["id"] for r in formula_recs_raw[:top_k]]
        formula_hit      = hidden in formula_recs

        # 純 Jaccard baseline
        jacc_recs = jaccard_only_recommendation(remaining)
        jacc_hit  = hidden in jacc_recs

        # 純分數增益 baseline
        score_recs = score_gain_only_recommendation(remaining)
        score_hit  = hidden in score_recs

        results.append({
            "network":      network,
            "ref":          ref,
            "hidden":       hidden,
            "remaining":    list(remaining.keys()),
            "gnn_recs":     gnn_recs,
            "formula_recs": formula_recs,
            "jacc_recs":    jacc_recs,
            "score_recs":   score_recs,
            "gnn_hit":      gnn_hit,
            "formula_hit":  formula_hit,
            "jacc_hit":     jacc_hit,
            "score_hit":    score_hit,
            # 分析 GNN vs 各 baseline 的獨立貢獻
            "gnn_only":          gnn_hit and not formula_hit,
            "formula_only":      formula_hit and not gnn_hit,
            "gnn_beats_jacc":    gnn_hit and not jacc_hit,
            "gnn_beats_score":   gnn_hit and not score_hit,
            "neither":           not gnn_hit and not formula_hit,
        })
        used += 1

        if used % 20 == 0:
            g  = sum(r["gnn_hit"]     for r in results)
            f  = sum(r["formula_hit"] for r in results)
            j  = sum(r["jacc_hit"]    for r in results)
            s  = sum(r["score_hit"]   for r in results)
            print(f"  {used}/{n_trials} — "
                  f"GNN: {g/used*100:.1f}%  "
                  f"混合: {f/used*100:.1f}%  "
                  f"純Jaccard: {j/used*100:.1f}%  "
                  f"純分數: {s/used*100:.1f}%")

    return results


# ════════════════════════════════════════════════════════════
# 視覺化
# ════════════════════════════════════════════════════════════

def plot_results(results, top_k, save_path):
    n = len(results)
    gnn_hits   = sum(r["gnn_hit"]   for r in results)
    mix_hits   = sum(r["formula_hit"] for r in results)
    jacc_hits  = sum(r["jacc_hit"]  for r in results)
    score_hits = sum(r["score_hit"] for r in results)

    gnn_rate   = gnn_hits   / n * 100
    mix_rate   = mix_hits   / n * 100
    jacc_rate  = jacc_hits  / n * 100
    score_rate = score_hits / n * 100

    gnn_only     = sum(r["gnn_only"]        for r in results)
    formula_only = sum(r["formula_only"]    for r in results)
    gnn_beats_j  = sum(r["gnn_beats_jacc"]  for r in results)
    gnn_beats_s  = sum(r["gnn_beats_score"] for r in results)
    neither      = sum(r["neither"]         for r in results)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0F172A")

    # ── 左上：四方法命中率主圖 ────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1E293B")
    labels = [f"GNN\nHit@{top_k}", f"混合公式\n(30%+70%)", f"純Jaccard\n(baseline)", f"純分數增益\n(baseline)"]
    rates  = [gnn_rate, mix_rate, jacc_rate, score_rate]
    colors = ["#16A34A", "#9333EA", "#22D3EE", "#F59E0B"]
    bars = ax1.bar(labels, rates, color=colors, width=0.55, edgecolor="#334155")
    for bar, val in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.0,
                 f"{val:.1f}%", ha="center", color="#F1F5F9",
                 fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("命中率 (%)", color="#CBD5E1")
    ax1.set_title(f"四方法 Hit@{top_k} 比較（n={n}）", color="#F1F5F9", fontsize=11)
    ax1.tick_params(colors="#94A3B8")
    for spine in ax1.spines.values(): spine.set_edgecolor("#334155")
    # GNN vs 各 baseline 差距標注
    for i, (r, c) in enumerate(zip([mix_rate, jacc_rate, score_rate],
                                   ["#9333EA", "#22D3EE", "#F59E0B"])):
        diff = gnn_rate - r
        sign = "+" if diff >= 0 else ""
        ax1.annotate(f"vs {['混合','Jacc','分數'][i]}: {sign}{diff:.1f}%",
                     xy=(0, gnn_rate), xytext=(0.02 + i*0.33, 0.88),
                     xycoords=("axes fraction", "data"),
                     textcoords="axes fraction",
                     color=c, fontsize=7.5, fontweight="bold")

    # ── 右上：GNN 獨立貢獻分析（橫條圖）─────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1E293B")
    cats   = ["GNN猜到\n混合沒猜到", "混合猜到\nGNN沒猜到",
              "GNN猜到\n純Jacc沒猜到", "GNN猜到\n純分數沒猜到", "兩者都沒猜到"]
    vals   = [gnn_only, formula_only, gnn_beats_j, gnn_beats_s, neither]
    colors2 = ["#16A34A", "#A855F7", "#22D3EE", "#F59E0B", "#EF4444"]
    bars2 = ax2.barh(cats, vals, color=colors2, edgecolor="#334155", height=0.55)
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val} ({val/n*100:.1f}%)", va="center",
                 color="#F1F5F9", fontsize=9)
    ax2.set_xlim(0, n * 0.6)
    ax2.set_xlabel("試驗數", color="#CBD5E1")
    ax2.set_title("GNN 獨立貢獻分析", color="#F1F5F9", fontsize=11)
    ax2.tick_params(colors="#94A3B8")
    for spine in ax2.spines.values(): spine.set_edgecolor("#334155")

    # ── 左下：各食物網四方法命中率 ───────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#1E293B")
    network_stats = {}
    for r in results:
        nm = r["network"] or "unknown"
        if nm not in network_stats:
            network_stats[nm] = {"gnn":0, "mix":0, "jacc":0, "score":0, "total":0}
        network_stats[nm]["total"] += 1
        if r["gnn_hit"]:     network_stats[nm]["gnn"]   += 1
        if r["formula_hit"]: network_stats[nm]["mix"]   += 1
        if r["jacc_hit"]:    network_stats[nm]["jacc"]  += 1
        if r["score_hit"]:   network_stats[nm]["score"] += 1

    nms = sorted([k for k, v in network_stats.items() if v["total"] >= 3],
                 key=lambda k: network_stats[k]["total"], reverse=True)[:5]
    x = np.arange(len(nms))
    w = 0.20
    for i, (key, col) in enumerate([("gnn","#16A34A"),("mix","#9333EA"),
                                     ("jacc","#22D3EE"),("score","#F59E0B")]):
        rates_nm = [network_stats[nm][key] / network_stats[nm]["total"] * 100 for nm in nms]
        ax3.bar(x + (i-1.5)*w, rates_nm, w, label=["GNN","混合","純Jacc","純分數"][i],
                color=col, edgecolor="#334155")
    ax3.set_xticks(x)
    ax3.set_xticklabels([nm[:10] for nm in nms], rotation=20, ha="right",
                        color="#94A3B8", fontsize=7)
    ax3.set_ylabel("命中率 (%)", color="#CBD5E1")
    ax3.set_title("各食物網命中率", color="#F1F5F9", fontsize=11)
    ax3.set_ylim(0, 110)
    ax3.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9",
               fontsize=7, ncol=2)
    ax3.tick_params(colors="#94A3B8")
    for spine in ax3.spines.values(): spine.set_edgecolor("#334155")

    # ── 右下：GNN 超越純 Jaccard 的案例 ─────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#1E293B")
    ax4.axis("off")

    gnn_wins  = [r for r in results if r["gnn_beats_jacc"]][:4]
    jacc_wins = [r for r in results if r["jacc_hit"] and not r["gnn_hit"]][:4]

    lines = [f"GNN猜到、純Jaccard沒猜到（共{gnn_beats_j}例）\n"]
    for r in gnn_wins:
        em = EMOJI.get(r["hidden"], "")
        ctx = ", ".join(r["remaining"][:3])
        lines.append(f"  ✅ 藏: {em}{r['hidden']}  情境: {ctx}...")
    jacc_only_cnt = sum(1 for r in results if r["jacc_hit"] and not r["gnn_hit"])
    lines.append(f"\n純Jaccard猜到、GNN沒猜到（共{jacc_only_cnt}例）\n")
    for r in jacc_wins:
        em = EMOJI.get(r["hidden"], "")
        ctx = ", ".join(r["remaining"][:3])
        lines.append(f"  ✅ 藏: {em}{r['hidden']}  情境: {ctx}...")

    ax4.text(0.02, 0.95, "\n".join(lines),
             transform=ax4.transAxes, va="top", ha="left",
             color="#E2E8F0", fontsize=7.5, fontfamily="monospace")
    ax4.set_title("GNN vs 純Jaccard 差異案例", color="#F1F5F9", fontsize=11)

    plt.suptitle(
        "EcoChain — GNN vs. 混合公式 vs. 純Jaccard vs. 純分數增益  Hold-out 評估",
        color="#F1F5F9", fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"\n[Plot] 已儲存 → {save_path}")
    return gnn_rate, mix_rate, jacc_rate, score_rate


# ════════════════════════════════════════════════════════════
# 文字報告
# ════════════════════════════════════════════════════════════

def write_report(results, top_k, gnn_rate, mix_rate, jacc_rate, score_rate, save_path):
    n = len(results)
    gnn_hits   = sum(r["gnn_hit"]     for r in results)
    mix_hits   = sum(r["formula_hit"] for r in results)
    jacc_hits  = sum(r["jacc_hit"]    for r in results)
    score_hits = sum(r["score_hit"]   for r in results)
    gnn_only   = sum(r["gnn_only"]    for r in results)
    f_only     = sum(r["formula_only"]for r in results)
    beats_jacc = sum(r["gnn_beats_jacc"]  for r in results)
    beats_score= sum(r["gnn_beats_score"] for r in results)
    neither    = sum(r["neither"]     for r in results)

    diff_mix   = gnn_rate - mix_rate
    diff_jacc  = gnn_rate - jacc_rate
    diff_score = gnn_rate - score_rate

    lines = [
        "EcoChain — 真實食物網 Hold-out 評估報告（v4 四方法比較）",
        "=" * 60,
        "",
        "【測試設計】",
        "  方法   : Leave-one-out（從真實食物網藏掉一個物種，看推薦系統能否猜到）",
        "  資料來源: real_food_webs.json（Mangal 資料庫 + 手工整理食物網）",
        f"  試驗次數: {n} 次",
        f"  評估指標: Hit@{top_k}（Top-{top_k}推薦中有沒有猜到被藏掉的物種）",
        "",
        "【四方法核心結果】",
        f"  GNN          Hit@{top_k} : {gnn_rate:.1f}%  （{gnn_hits}/{n}）",
        f"  混合公式     Hit@{top_k} : {mix_rate:.1f}%  （{mix_hits}/{n}）  ← 30%分數增益 + 70%Jaccard",
        f"  純Jaccard    Hit@{top_k} : {jacc_rate:.1f}%  （{jacc_hits}/{n}）  ← 只看食物網共現",
        f"  純分數增益   Hit@{top_k} : {score_rate:.1f}%  （{score_hits}/{n}）  ← 只看eco_score公式",
        "",
        "【GNN vs 各 baseline 差距】",
        f"  GNN vs 混合公式   : {diff_mix:+.1f}%",
        f"  GNN vs 純Jaccard  : {diff_jacc:+.1f}%",
        f"  GNN vs 純分數增益 : {diff_score:+.1f}%",
        "",
        "【獨立貢獻分析】",
        f"  GNN猜到、混合公式沒猜到 : {gnn_only} 例  ({gnn_only/n*100:.1f}%)",
        f"  混合公式猜到、GNN沒猜到 : {f_only} 例  ({f_only/n*100:.1f}%)",
        f"  GNN猜到、純Jaccard沒猜到: {beats_jacc} 例  ({beats_jacc/n*100:.1f}%)",
        f"  GNN猜到、純分數增益沒猜到: {beats_score} 例  ({beats_score/n*100:.1f}%)",
        f"  全部都沒猜到            : {neither} 例  ({neither/n*100:.1f}%)",
        "",
        "【解讀】",
    ]

    # GNN vs 純Jaccard 是最關鍵的比較
    if diff_jacc > 5:
        lines += [
            f"  ★ 關鍵發現：GNN 命中率比純Jaccard高 {diff_jacc:.1f}%。",
            "  這表示 GNN 的圖結構訊息傳遞學到了 Jaccard 單純共現之外的資訊——",
            "  物種間的捕食/授粉/共生邊關係對推薦有實質貢獻。",
        ]
    elif diff_jacc >= 0:
        lines += [
            f"  GNN 命中率與純Jaccard接近（差距 {diff_jacc:+.1f}%）。",
            "  GNN 的圖結構優勢尚未超越純共現統計，",
            "  原因是 46 個食物網子集對 GAT 模型而言訓練量仍然有限。",
        ]
    else:
        lines += [
            f"  GNN 命中率低於純Jaccard（{diff_jacc:.1f}%）。",
            "  這說明在此資料規模下，直接查食物網共現比 GNN 更可靠。",
            "  GNN 的推薦頭需要更多食物網樣本才能超越 lookup-table 方法。",
        ]

    # 補充：混合公式 vs 純分數增益 的拆解
    lines += [
        "",
        f"  混合公式 vs 純分數增益差距：{mix_rate - score_rate:+.1f}%",
        "  這個差距來自 Jaccard 成分（70%）的貢獻。",
        "  混合公式命中率高，主要由 Jaccard 成分驅動，而非公式本身。",
        "",
        "【限制說明】",
        "  1. 訓練資料的 score head ground truth 來自 eco_score() 公式，",
        "     GNN 的分數頭學的是公式，推薦頭從 v4 起改用食物網 co-occurrence。",
        "  2. 測試食物網與訓練食物網來自同一個 real_food_webs.json（n=46），",
        "     存在資料重疊的風險，是小型資料集的固有限制。",
        "  3. 純Jaccard 命中率 = GNN 推薦頭在當前資料規模下的能力上限參考點。",
        "     GNN 超越它才代表圖結構帶來了真正的資訊增益。",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] 已儲存 → {save_path}")
    print("\n" + "\n".join(lines))


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=100, help="試驗次數")
    parser.add_argument("--top_k", type=int, default=5,   help="Top-K 推薦數")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    results = run_holdout(n_trials=args.n, top_k=args.top_k, seed=args.seed)

    if not results:
        print("沒有有效結果，請確認 real_food_webs.json 存在且格式正確")
        sys.exit(1)

    img_path    = os.path.join(OUTPUT_DIR, "holdout_results.png")
    report_path = os.path.join(OUTPUT_DIR, "holdout_report.txt")

    gnn_rate, mix_rate, jacc_rate, score_rate = plot_results(results, args.top_k, img_path)
    write_report(results, args.top_k, gnn_rate, mix_rate, jacc_rate, score_rate, report_path)

    print(f"\n{'='*50}")
    print(f"  完成！產生的檔案：")
    print(f"    ✅ holdout_results.png")
    print(f"    ✅ holdout_report.txt")
    print(f"{'='*50}")
