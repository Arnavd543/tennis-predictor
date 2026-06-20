# Tennis Predictor: Roadmap to 80%+ Accuracy

## Current State

- **Model**: XGBoost + LightGBM weighted ensemble + Platt calibration
- **Features**: 43 engineered features (Elo, H2H, form, market intelligence, tournament context)
- **Data**: Kaggle dataset — rankings, odds, score; no serve/point-level stats
- **Accuracy**: ~67.5% ATP / ~67.4% WTA (2024-2025 test set)
- **Ceiling with current data**: ~68–70% — the Kaggle dataset is the bottleneck

---

## Why 80%+ Is Achievable (and What It Requires)

The fundamental limit of any model using only **rankings + odds + basic match info** is approximately **68–70%**. Professional betting markets, which incorporate all public information efficiently, sit at 65–68%. To beat them meaningfully requires **signals they don't already price in**.

The research literature shows (verified 2024–2025 papers):
- **Random Forest with serve-focused features**: 80%+ (Gao & Kowalczyk 2019, arxiv:1910.03203)
- **Linear model with Elo + points + age**: 79.5% (Statistical Enhanced Learning 2025, arxiv:2502.01613)
- **XGBoost + NSGA-II genetic optimizer**: 93% accuracy (but on limited dataset)
- **Soft voting ensemble with mid-match momentum**: 97.5% — but this uses in-match data (set scores, current momentum), not pre-match
- **Baseline XGBoost/LightGBM with rankings only**: ~65–68%

**The single most important finding**: The jump from 67% to 80%+ is achieved by adding **Serve Points Won % (SPW%)** and **Return Points Won % (RPW%)**. These two features, combined with break point conversion rates, account for most of the gap. Algorithm choice (XGBoost vs TabNet vs neural net) matters far less than having these features.

> **TabNet and deep learning are NOT recommended for this use case.** Comparative studies show XGBoost outperforms TabNet on 8 of 11 sports datasets. The bottleneck is features, not model architecture.

**The primary lever is data quality, not algorithm complexity.** Random Forest alone won't get you to 80% on the same features. Switching data sources will.

---

## Phase 1 — Switch Data Source (Highest Impact, +3–5%)

### The Problem with the Current Kaggle Dataset

The current dataset (`atp_tennis.csv` / `wta.csv`) contains only:
```
Tournament, Date, Series, Court, Surface, Round, Best of,
Player_1, Player_2, Winner, Rank_1, Rank_2, Pts_1, Pts_2,
Odd_1, Odd_2, Score
```

No serve statistics. No point-level data. This caps accuracy.

### Jeff Sackmann's `tennis_atp` / `tennis_wta` GitHub Repositories

The gold standard free data source. Available at:
- `github.com/JeffSackmann/tennis_atp` (ATP, 1968–present)
- `github.com/JeffSackmann/tennis_wta` (WTA, 1920–present)

**Key additional columns** (available from ~2000 for ATP, ~2007 for WTA):
```
w_ace      l_ace       # Aces per match (winner / loser)
w_df       l_df        # Double faults
w_svpt     l_svpt      # Total service points played
w_1stIn    l_1stIn     # First serves in
w_1stWon   l_1stWon    # First serve points won
w_2ndWon   l_2ndWon    # Second serve points won
w_SvGms    l_SvGms     # Service games played
w_bpSaved  l_bpSaved   # Break points saved
w_bpFaced  l_bpFaced   # Break points faced

# Derived on the fly:
first_serve_pct     = w_1stIn / w_svpt
first_serve_won_pct = w_1stWon / w_1stIn
second_serve_won_pct= w_2ndWon / (w_svpt - w_1stIn)
service_hold_pct    = (w_SvGms - bp_broken) / w_SvGms
break_point_conv    = bp_won / w_bpFaced
```

Also adds: `winner_rank`, `loser_rank`, `winner_age`, `loser_age`, `winner_ht`, `loser_ht`, `winner_hand`, `loser_hand`, `tourney_level` (Grand Slam / Masters / etc.)

**License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 — free to use for non-commercial projects.

### Tennis Abstract Match Charting Project

A second free dataset (tennisabstract.com) with 17,700+ matches charted point-by-point. Provides significantly richer serving metrics:

| Metric | Meaning |
|---|---|
| `Unret%` | Unreturned serve % (aces + service winners + return errors) |
| `RiP W%` | Return-in-play winning % (server wins when return comes back) |
| `SvImpact` | Serve impact score — serve influence even on successful returns |
| `BP Wide%` | Break-point specific serving direction patterns |
| `BreakBack%` | Frequency of breaking back immediately after losing serve |

These situational/pressure-situation stats are more predictive than raw ace counts because they capture consistency under pressure.

### Migration Plan

1. Download Sackmann CSV files (free, no API key):
   ```bash
   git clone https://github.com/JeffSackmann/tennis_atp.git /tmp/tennis_atp
   git clone https://github.com/JeffSackmann/tennis_wta.git /tmp/tennis_wta
   ```

2. Concatenate annual files (`atp_matches_2000.csv` ... `atp_matches_2025.csv`)

3. Update `download_and_preprocess.py` management command to pull from Sackmann instead of / in addition to Kaggle

4. Add rolling-window player averages for each serve stat (last 10 matches, last 30 days, surface-specific)

**Expected gain: +2–4% accuracy** (serve % and break point conversion are among the strongest pre-match signals)

---

## Phase 2 — New Serve-Based Features (+2–3%)

Once Sackmann data is integrated, these rolling features become available for each player:

### Serve Power Features
```python
ace_rate_diff          # Player 1 ace rate minus Player 2 ace rate (last 20 matches)
df_rate_diff           # Double fault rate difference
first_serve_pct_diff   # First serve percentage difference
first_serve_won_diff   # First serve points won % difference
second_serve_won_diff  # Second serve won % difference
```

### Pressure / Clutch Features
```python
bp_conversion_diff     # Break point conversion rate difference
bp_save_rate_diff      # Break point save rate difference (very strong signal)
service_hold_pct_diff  # Service hold % difference (captures serve dominance)
return_pts_won_diff    # Return points won % difference
```

### Physical / Biographical Features
```python
age_diff               # Age at match date (direct from Sackmann)
height_diff            # Height in cm (taller = serve advantage, especially on grass)
age_x_surface          # Interaction: older players often outperform on clay (experience)
```

### Surface-Specific Serve Rolling Averages
Compute rolling ace%, service hold%, and break point save rate separately per surface:
```python
clay_service_hold_diff
grass_service_hold_diff
hard_service_hold_diff
```

---

## Phase 3 — Better Models (+0.5–1.5%)

> **Key finding from research**: Algorithm choice is secondary to feature quality. On sports prediction tasks, XGBoost outperformed TabNet and neural nets in 8 of 11 comparative studies. Invest in data first, models second.

### 3.1 Random Forest in Ensemble

Random Forest is already partially wired in (`create_ensemble_model`). It should be added as a 4th component in `tune_hyperparameters()`. RF is deliberately different from gradient boosting (random feature subsets, bagging) so it adds genuine diversity to the ensemble:

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

rf = RandomForestClassifier(
    n_estimators=400, max_depth=12, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
```

Expect **+0.3–0.5%** from adding RF + ExtraTrees to the ensemble.

### 3.2 ~~TabNet~~  — Not Recommended

Research shows TabNet underperforms XGBoost/LightGBM on sports tabular data. Skip it and focus on serve-stat features instead.

### 3.3 Stacking (Meta-Learner)

Replace the fixed-weight ensemble with a meta-learner that learns which model to trust for which input patterns:

```
Level 0: XGBoost, LightGBM, CatBoost, Random Forest, TabNet
Level 1: Logistic Regression trained on OOF predictions from Level 0
```

Use temporal out-of-fold (not random splits) to avoid leakage:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
oof_probs = np.zeros((len(X_train), n_models))

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    for j, base_model in enumerate(base_models):
        base_model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_probs[val_idx, j] = base_model.predict_proba(X_train[val_idx])[:, 1]

meta = LogisticRegression(C=1.0, max_iter=200)
meta.fit(oof_probs, y_train)
```

**Expected gain: +0.3–0.7%** over fixed-weight ensemble.

### 3.4 Neural Network with Player Embeddings

Train a small neural network where each player gets a learned embedding vector (like word embeddings, but for players). This captures stylistic tendencies that statistics don't fully represent:

```python
# Architecture:
# - Player 1 ID → Embedding(n_players, 32)
# - Player 2 ID → Embedding(n_players, 32)
# - Match features → Dense(128) → Dense(64)
# - Concatenate all → Dense(64) → Dense(1, sigmoid)

import torch.nn as nn

class TennisNet(nn.Module):
    def __init__(self, n_players, n_features, emb_dim=32):
        super().__init__()
        self.player_emb = nn.Embedding(n_players, emb_dim)
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 2 * emb_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, p1_id, p2_id, features):
        e1 = self.player_emb(p1_id)
        e2 = self.player_emb(p2_id)
        f  = self.feature_net(features)
        return self.head(torch.cat([f, e1, e2], dim=1))
```

This requires ~5,000+ matches per player pool for stable embeddings. With 107K+ historical matches it's feasible.

**Expected gain: +0.5–1.5%** (especially for matchups involving well-known players with many historical matches).

---

## Phase 4 — Feature Engineering Upgrades (+0.5–1%)

### 4.1 Ranking Trajectory Curve
Instead of single 90-day trend, fit a linear regression to the last 6 months of ranking points and use the slope and intercept as features:
```python
from numpy.polynomial import polynomial as P
coeff = np.polyfit(range(len(recent_ranks)), recent_ranks, 1)
rank_slope = coeff[0]   # Rising vs falling trajectory
```

### 4.2 Fatigue Index
Cumulative games played in the last 14 days (not just last match):
```python
fatigue_14d = sum(total_games for matches in last_14_days)
```

### 4.3 Surface Specialist Score
How much better (or worse) a player performs on the current surface vs their overall win rate:
```python
surface_specialist = surface_win_pct - overall_win_pct
# Positive = specialist advantage, negative = weakened on this surface
```

### 4.4 Clutch Performance Index
Win rate in matches that went to a deciding set, across the player's career:
```python
clutch_index = deciding_set_wins / total_deciding_set_matches
```
(Already partially captured by `deciding_set_diff`, but career-level rather than just recent.)

### 4.5 Opponent Quality-Adjusted Win Rate
Weight recent wins by opponent ranking:
```python
quality_wins = sum(1/opponent_rank for wins in recent_20_matches)
```
Beating the world #1 counts 100x more than beating #100.

---

## Phase 5 — Advanced Validation (+confidence calibration)

### 5.1 Proper Temporal Cross-Validation

Instead of single 3-way split, use 5-fold time-series walk-forward:
```
Fold 1: Train 2010-2018 | Val 2019
Fold 2: Train 2010-2019 | Val 2020
Fold 3: Train 2010-2020 | Val 2021
Fold 4: Train 2010-2021 | Val 2022
Fold 5: Train 2010-2022 | Val 2023
Final:  Train 2010-2023 | Test 2024-2025
```

This gives stable performance estimates and reduces the risk of overfitting to a single validation year.

### 5.2 Isotonic Regression Calibration

Platt scaling (sigmoid) is already in the pipeline. For better probability calibration, also try **isotonic regression**:
```python
from sklearn.calibration import CalibratedClassifierCV
model_iso = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
```

### 5.3 Match Uncertainty Flags

Instead of just prediction + confidence, flag matches where the model is uncertain:
- Both implied probabilities within 5% of each other → "Coin flip" flag
- Model confidence < 55% → "Low confidence" flag
- Market and model disagree → "Contrarian" flag (potentially highest value)

---

## Realistic Accuracy Targets

| Phase | What Changes | Expected ATP Accuracy |
|---|---|---|
| Current | 43-feature XGBoost+LGB+CAT ensemble | ~67.5% |
| + Round importance fix | Already done (this session) | ~67.5–68% |
| + Random Forest + ExtraTrees | Add to existing ensemble | ~68–68.5% |
| + Sackmann data (SPW%, RPW%) | **The main unlock — new data pipeline** | ~72–75% |
| + Serve stat rolling features | Extended from Sackmann | ~75–77% |
| + Tennis Abstract situational stats | Break-point serving patterns | ~77–79% |
| + Stacking meta-learner | Better model combination | ~79–80% |
| + Player embeddings neural net | Learned style representation | ~80–81% |
| **Combined (realistic, pre-match)** | | **~78–80%** |

**Research-backed honest ceiling for pre-match prediction**:
- Gao & Kowalczyk (2019) achieved **80%+ with Random Forest using serve-focused features**
- Statistical Enhanced Learning (2025) achieved **79.5% with Elo + ranking points + age alone**
- These results confirm **80% is achievable** with the right features — it's not a ceiling, it's a milestone

**What you cannot predict no matter the model**:
- Injuries that occur during a match or are undisclosed pre-match
- Mental health / personal issues (Naomi Osaka retirement type situations)
- Extreme weather / court-condition surprises

**The 97.5% papers** use **in-match data** (current set scores, game momentum, physical tracking). That's a different, harder engineering problem.

---

## Implementation Priority Order

```
1. [Day 1–2] Add Random Forest + ExtraTrees to ensemble
   → ml_utils.py: tune_hyperparameters() — add Phase 2.75 with RF/ET
   → Easiest win, already have the infrastructure

2. [Week 1] Migrate to Sackmann data — THE MAIN UNLOCK
   → download_and_preprocess.py: git clone tennis_atp, merge CSV files
   → ml_utils.py: add SPW%, RPW%, break point conversion rolling features
   → This alone should push from ~68% to ~75%

3. [Week 2] Add Tennis Abstract situational serve stats
   → Scraper for tennisabstract.com (or use existing charting project CSVs)
   → Add Unret%, RiP W%, BP Wide% as rolling player averages

4. [Week 2–3] Stacking meta-learner
   → ml_utils.py: replace WeightedEnsemble with TimeSeriesSplit OOF stacking
   → Meta-learner: LogisticRegression on OOF probability outputs

5. [Week 3–4] Walk-forward cross-validation
   → ml_utils.py: 5-fold temporal CV in train_atp_model() and train_wta_model()
   → Better model selection + uncertainty estimates

6. [Week 4–5] Player embeddings neural net (optional — depends on results from 1–4)
   → New file: predictor/nn_model.py
   → Only worthwhile if still below 78% after steps 1–4
```

## Data Sources

| Source | URL | Key Data | Cost |
|---|---|---|---|
| Jeff Sackmann ATP | github.com/JeffSackmann/tennis_atp | Rankings, results, serve stats (aces, 1st serve %, SvGms, bpFaced) from 1991+ | Free (CC BY-NC-SA) |
| Jeff Sackmann WTA | github.com/JeffSackmann/tennis_wta | Same for WTA | Free (CC BY-NC-SA) |
| Tennis Abstract Charting | tennisabstract.com | Point-by-point: Unret%, RiP W%, situational serve patterns, 17,700+ matches | Free |
| Current Kaggle dataset | Kaggle | Rankings, odds, scores — no serve stats | Free |

---

## Key Files to Modify

| File | Change |
|---|---|
| `predictor/management/commands/download_and_preprocess.py` | Switch to Sackmann repo, merge serve stats |
| `predictor/ml_utils.py` | New serve features, RF in ensemble, TabNet, stacking |
| `predictor/predictor_utils.py` | Handle new feature dimensions, multi-model loading |
| `predictor/views.py` | Expose serve stat inputs (optional — can use career averages) |
| `requirements.txt` | Add `pytorch-tabnet`, `torch` |
| New: `predictor/nn_model.py` | Player embedding neural network |

---

## Quick Wins vs Big Bets

### Quick wins (can implement today, no new data):
- Add `RandomForestClassifier` + `ExtraTreesClassifier` to the ensemble in `tune_hyperparameters()` — existing infrastructure supports it, ~30 min of code
- Use walk-forward CV instead of single 3-way split — better evaluation, same model

### Medium effort (1–2 weeks):
- Sackmann data migration — needs new ETL pipeline, biggest accuracy gain
- Serve stat feature engineering — ~200 lines in `ml_utils.py`

### Big bets (2–4 weeks):
- TabNet integration — new dependency, hyperparameter tuning needed
- Player embeddings — new architecture, training loop, inference changes
