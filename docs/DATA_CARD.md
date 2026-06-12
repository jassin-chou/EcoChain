# EcoChain Data Card

> Purpose: document the datasets used to train, evaluate, and explain the EcoChain ecosystem recommendation system.
> This card follows the spirit of Google PAIR's Data Cards Playbook: describe what the data represents, where it comes from, how it is prepared, how it should be used, and what its limitations are.

## 1. Dataset Overview

| Field | Description |
|---|---|
| Dataset name | EcoChain ecological graph training data |
| Project | EcoChain WebApp |
| Primary files | `real_food_webs.json`, generated samples from `train.py`, optional external sheet data |
| Related model files | `models/ecochain_gnn.pt`, `species2vec.pt` |
| Main task | Species recommendation and ecosystem graph analysis |
| Data type | Graph-structured ecological game states |
| Human personal data | No direct human personal data is included in the ML training data |
| Current documentation date | 2026-06-12 |

## 2. Motivation

EcoChain represents a player-built ecosystem as a graph. Species and facilities become nodes, while ecological interactions such as predation, pollination, habitat support, and facility support become edges.

The dataset exists to support two goals:

1. Train a GNN to recommend species that are structurally compatible with the current ecosystem.
2. Provide ecosystem examples grounded in real food-web patterns instead of only random game states.

The dataset is not intended to represent a complete biological database. It is a game-oriented, simplified ecological dataset designed for educational ecosystem simulation.

## 3. Data Sources

### 3.1 Real Food-Web Subsets

`real_food_webs.json` contains curated game-sized subsets derived from food-web references and generated through `fetch_mangal.py`.

Current local snapshot:

| Property | Value |
|---|---:|
| Number of game food-web subsets | 48 |
| Species covered in these subsets | 34 |
| Subset size range | 2 to 6 species |
| Average subset size | 4.44 species |

Observed source networks in the current file include:

| Network | Subsets |
|---|---:|
| Yellowstone | 6 |
| Ythan Estuary | 5 |
| Tuesday Lake | 5 |
| UK Farmland | 5 |
| Bialowieza Forest | 5 |
| Broadstone Stream | 4 |
| Patagonian shelf | 3 |
| Silwood Park | 3 |
| Coachella Valley | 3 |
| Skipwith Pond | 2 |
| St. Martin Island | 2 |
| Mediterranean Coast | 2 |
| Yellowstone NP | 1 |
| UK Countryside | 1 |
| East Africa | 1 |

The file includes references such as Huxham et al. 1992, Post et al. 2000, Ripple & Beschta 2004, and GBIF occurrence-based entries. The project should preserve these references when presenting recommendation sources.

### 3.2 Generated Game States

`train.py` generates additional training examples from the game species catalog. The default synthetic training mix is:

| Source type | Approximate share |
|---|---:|
| Random game states | 10% |
| Real food-web based states | 60% |
| Perturbed real food-web states | 30% |

Generated states are scored with the project's rule-based ecological score function and paired with recommendation labels derived from real food-web co-occurrence logic.

### 3.3 Species2Vec Data

`species2vec.py` trains species embeddings from real food-web co-occurrence patterns. The generated artifact is `species2vec.pt`.

Current local artifact:

| File | Size | Last modified |
|---|---:|---|
| `species2vec.pt` | 4,869 bytes | 2026-06-02 |

Species2Vec embeddings are used as node features when available. If the file is unavailable, the GNN code falls back to hand-crafted node features.

## 4. Data Schema

### 4.1 Food-Web Subset Entry

Each `real_food_webs.json` game subset has the following practical structure:

```json
{
  "species": ["grass", "shrimp", "crab", "duck"],
  "network": "Ythan Estuary",
  "ref": "Huxham et al. 1992"
}
```

### 4.2 Training Sample Representation

Training samples are species-count dictionaries:

```json
{
  "grass": 3,
  "rabbit": 2,
  "fox": 1
}
```

These are converted into PyTorch Geometric graph objects by `models/gnn.py`.

### 4.3 Graph Construction

| Component | Representation |
|---|---|
| Node | Present species or facility |
| Node feature | Species2Vec embedding + log count, or fallback hand-crafted features |
| Edge | Ecological relation between present nodes |
| Edge attributes | One-hot relation type: predation, pollination, symbiosis/facility support |
| Graph label | Rule-based eco score |
| Recommendation label | Food-web co-occurrence recommendation target |

## 5. Labeling and Targets

### 5.1 Eco Score Label

The score label used during training comes from `eco_score()` in `train.py`. It combines:

- Shannon / Pielou evenness
- Trophic completeness
- Aquatic habitat bonus or penalty
- Pollinator bonus
- Infrastructure bonus

The user-facing runtime score is now computed by the explainable backend function `compute_explainable_score()` in `main.py`, rather than directly using the GNN prediction.

### 5.2 Recommendation Label

The recommendation target is based on real food-web co-occurrence. The current training logic uses `foodweb_recommendation()` so that the recommendation head learns which species tend to co-occur in real or curated ecological subsets.

## 6. Preparation and Processing

1. Load or generate real food-web subsets through `real_food_webs.json`.
2. Generate random, real-web, and perturbed real-web samples in `train.py`.
3. Compute score labels with the ecological scoring function.
4. Compute recommendation labels using food-web co-occurrence logic.
5. Convert species-count dictionaries into graph objects with node features and ecological edges.
6. Split data into training and validation sets with a 90/10 split.

## 7. Intended Uses

Appropriate uses:

- Training EcoChain's GNN recommendation model.
- Evaluating species recommendation behavior inside the EcoChain game.
- Explaining how graph-based ecosystem states are represented.
- Demonstrating ML documentation and MLsecOps practices in an educational project.

Out-of-scope uses:

- Real-world ecological policy decisions.
- Conservation planning.
- Scientific ecological forecasting.
- Human-impact or biodiversity risk assessment outside the game context.

## 8. Known Limitations

- The dataset is small and game-oriented.
- The species list is simplified and does not cover full ecological taxonomies.
- Food-web subsets are mapped into game species, which may lose ecological detail.
- Generated samples may overrepresent rules from the game's scoring function.
- Recommendation labels rely heavily on Jaccard-style co-occurrence logic and may favor known subsets.
- Some real ecological relationships are simplified into three edge types.
- The dataset does not include player behavior, long-term user telemetry, or real deployment traffic.

## 9. Privacy and Security Considerations

- The ML dataset does not require user account data, passwords, tokens, or Firebase credentials.
- Files such as `.env`, `serviceAccountKey.json`, and Azure publish profiles must not be included in dataset releases.
- If future training uses player-generated boards, those boards should be documented as a separate dataset version and reviewed for privacy, consent, retention, and deletion policy.

## 10. Evaluation Data

The repository includes a hold-out report generated by `holdout_test.py`.

Current local hold-out summary from `holdout_report.txt`:

| Method | Hit@5 |
|---|---:|
| GNN | 92.0% |
| Rule / hybrid recommendation | 100.0% |
| Pure Jaccard baseline | 100.0% |
| Score-only baseline | 13.0% |

Important note: this is a small local evaluation with 100 trials. It should be reported as project evidence, not as a broad scientific benchmark.

## 11. Maintenance

Recommended maintenance process:

1. Update this Data Card when `real_food_webs.json`, `train.py`, or data generation logic changes.
2. Record dataset version, generation command, date, and random seed when retraining.
3. Keep hold-out reports with the matching model artifact.
4. Avoid overwriting evaluation artifacts without noting the model and dataset version.

## 12. References

- Google PAIR, Data Cards Playbook: https://sites.research.google/datacardsplaybook/
- Pushkarna, Zaldivar, and Kjartansson, "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI": https://arxiv.org/abs/2204.01075
- Mitchell et al., "Model Cards for Model Reporting": https://arxiv.org/abs/1810.03993
