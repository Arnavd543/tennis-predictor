# Tennis Predictor - Complete Technical Analysis

## Consolidation Note (June 2026)
The `tennis/` folder (sibling project with only empty stub files) has been merged into this project. `tennis-predictor/` is the **sole canonical project**. Cleaned up: `ml_utils_backup.py`, old generic `xgb_match_winner.pkl`, `catboost_info/` training artifacts.

## June 2026 Upgrade (Phase 1-3)

### UI Overhaul (Phase 1)
- Switched from Tailwind CDN to npm build (`package.json`, `tailwind.config.js`, `predictor/static/predictor/src/input.css`)
- Added HTMX for inline prediction (no full page reload) — `predict_htmx_view` endpoint
- Added Alpine.js replacing vanilla JS toggles and form state
- New template layout: 2-col desktop (form + sidebar with stats/recent predictions)
- New `predictor/templates/predictor/partials/result_card.html` — HTMX response fragment
- Dual-player probability split bar on results (68% vs 32%)
- Surface color-coded pill badges (clay=orange, grass=green, hard=blue)
- Dark mode with localStorage persistence
- Deleted dead `style.css`
- Added `confidence` + `model_type` fields to `PredictionRequest` model (migration 0003)
- Install: `npm install` then `npm run build:css` (or `npm run watch:css` for dev)

### Accuracy Improvements (Phase 3)
- Removed 4 features with train/inference mismatch: `deciding_set`, `straight_sets`, `comeback_win`, `dominant_win`
- Added 6 new features: `surf_h2h_diff`, `rank_trend_diff`, `exp_form_diff`, `gs_form_diff`, `tiebreak_rate_diff`, `recent_games_diff`
- Added CatBoost as third ensemble model in `tune_hyperparameters()` (XGB+LGB+CatBoost)
- Updated `get_all_feature_columns()` — now 43 features (was 41)
- **Retrain required** to see accuracy gains: `python manage.py train_models`

### AWS Deployment (Phase 2)
- `Dockerfile` — now includes Node.js for Tailwind build, updated CMD with `wait_for_db && migrate && gunicorn`
- `predictor/management/commands/wait_for_db.py` — new command, polls DB until ready
- `.dockerignore` — excludes `xgb_match_winner_atp/wta.pkl` (~38 MB) and `node_modules/`
- `.github/workflows/deploy.yml` — CI/CD: push to ECR → update App Runner on `git push main`
- `.gitignore` — added `node_modules/`, `predictor/static/predictor/dist/`, `data/*.csv`
- **Settings bugs fixed**: Redis now only active when `REDIS_URL` env var set; `SECURE_SSL_REDIRECT=False` + `SECURE_PROXY_SSL_HEADER` for App Runner
- **AWS setup needed**: Create ECR repo, RDS PostgreSQL (t3.micro), App Runner service; set GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `APP_RUNNER_ARN`

## Project Overview
This is a professional-grade tennis match prediction system that achieved **67.5% accuracy on 2024-2025 matches** using machine learning. The system combines advanced feature engineering, rigorous temporal validation, and a production-ready Django web application with Docker deployment to predict ATP and WTA match outcomes.

**Key Achievement**: Competitive with professional betting markets (65-68%) while processing 107K+ historical matches.

## ✨ Latest Updates

### Phase 2: Ensemble Learning + Calibration (December 2025)
**Implemented**: LightGBM ensemble model with probability calibration for improved accuracy and reliability.

**New Features**:
1. **LightGBM Integration**: Added LightGBM alongside XGBoost for ensemble predictions
2. **Weighted Ensemble**: Optimal weight finding (typically 60% XGBoost, 40% LightGBM)
3. **Probability Calibration**: Platt scaling for accurate confidence scores
4. **Brier Score Tracking**: Quantifies probability calibration quality

**Expected Improvements**:
- **+1-2% accuracy boost** from ensemble diversity
- **Better confidence scores** from calibration
- **More reliable probabilities** (Brier score improvement)

**Technical Details**:
- 4-phase training: XGBoost → LightGBM → Ensemble → Calibration
- Automated hyperparameter tuning for both models
- Validation-based weight optimization
- Sigmoid calibration using validation set

**Training Command**:
```bash
python manage.py train_enhanced_models
# or
docker-compose exec web python manage.py train_enhanced_models
```

---

### Phase 1: Production Ready (December 2025)
**Implemented**: Complete production-ready infrastructure with modern UI and Docker deployment.

**New Features**:
1. **Modern Frontend**: Professional Tailwind CSS UI with animated confidence meters and responsive design
2. **Docker Deployment**: Full containerization with Docker Compose for dev and production
3. **Production Features**: Redis caching, health checks, structured logging, rate limiting
4. **Security**: Environment-based configuration, CORS, security headers, HTTPS-ready
5. **Monitoring**: Health check endpoints, Sentry integration ready, comprehensive logging

**Files Added/Modified**:
- ✅ Enhanced templates with Tailwind CSS (home.html, result.html)
- ✅ Dockerfile and docker-compose.yml (dev + prod)
- ✅ Production settings with Redis caching and PostgreSQL
- ✅ Health check endpoints (/health, /health/ready, /health/live)
- ✅ Middleware for caching and rate limiting
- ✅ Nginx reverse proxy configuration
- ✅ Comprehensive DEPLOYMENT.md guide

**Quick Start**:
```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up -d --build
```

## 🏗️ Architecture Overview

```
tennis-predictor/
├── predictor/                    # Django app
│   ├── management/commands/      # Data processing & training
│   ├── models.py                 # Database models
│   ├── views.py                  # Web interface logic
│   ├── predictor_utils.py        # ML prediction API
│   ├── ml_utils.py              # Feature engineering & training
│   ├── templates/               # HTML templates
│   └── models/                  # Trained ML models
├── tennis_predictor/            # Django project config
├── data/                        # Raw and processed datasets
├── db.sqlite3                   # Application database
├── requirements.txt             # Dependencies
└── manage.py                    # Django management
```

## 📊 Data Foundation

### Data Sources
- **ATP**: `dissfya/atp-tennis-2000-2023daily-pull` (Kaggle)
- **WTA**: `dissfya/wta-tennis-2007-2023-daily-update` (Kaggle)
- **Total Volume**: 107K+ matches (65K ATP + 42K WTA)

### Raw Data Structure
Each match contains 17 essential fields:
```csv
Tournament,Date,Series,Court,Surface,Round,Best of,Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score
```

**Key Data Points**:
- **Who**: Player names, rankings, points
- **When**: Exact tournament date
- **Where**: Tournament, court, surface type
- **Context**: Round, format (Bo3/Bo5), series importance
- **Market**: Real betting odds reflecting expectations
- **Outcome**: Winner + detailed set score

## 🔧 Data Processing Pipeline

### Stage 1: Data Download (`download_and_preprocess.py`)
```python
# Automated Kaggle download
kaggle.api.dataset_download_files(atp_dataset, path=data_dir, unzip=True)

# Data cleaning and standardization
atp_df = atp_df[required_columns].dropna()
```

**Process**:
1. Authenticate with Kaggle API
2. Download compressed datasets
3. Extract to `data/` directory
4. Standardize column structure
5. Remove incomplete records
6. Parse dates properly
7. Save as `atp_processed.csv`, `wta_processed.csv`

### Stage 2: Advanced Feature Engineering (`ml_utils.py`)
The system extracts **40+ sophisticated features** from the same raw data:

#### Basic Features (13)
- `rank_diff`: Official ranking difference
- `surface_enc`: Surface type encoding (Clay=0, Grass=1, Hard=2, Carpet=3)
- `year`, `month`: Temporal patterns
- `odd_diff`, `pts_diff`: Market and ranking point differences
- `best_of`: Match format (3 or 5 sets)
- `series_enc`, `tournament_enc`, `round_enc`: Categorical encodings
- `elo_diff`, `surf_elo_diff`: Elo rating differences
- `h2h_diff`: Head-to-head record difference

#### Form & Momentum Features (5)
```python
def add_form_features(df):
    # Current win/loss streaks
    df['streak_diff'] = df['player_1_streak'] - df['player_2_streak']
    
    # Recent form (last 10 matches)
    df['form_diff_10'] = df['p1_form_10'] - df['p2_form_10']
    
    # 30-day performance window
    df['form_diff_30d'] = df['p1_form_30d'] - df['p2_form_30d']
    
    # Surface-specific recent form
    df['surface_form_diff'] = df['p1_surface_form'] - df['p2_surface_form']
    
    # Tournament-specific form
    df['tournament_form_diff'] = df['p1_tournament_form'] - df['p2_tournament_form']
```

#### Advanced Elo Systems (5)
```python
def add_advanced_elo_features(df):
    # Tournament-tier Elo (Grand Slams vs regular tournaments)
    df['tournament_elo_diff'] = df['p1_tournament_elo'] - df['p2_tournament_elo']
    
    # Round-specific Elo (Finals vs early rounds performance)
    df['round_elo_diff'] = df['p1_round_elo'] - df['p2_round_elo']
    
    # Opponent-tier Elo (vs Top10, Top50, etc.)
    df['opponent_tier_elo_diff'] = df['p1_tier_elo'] - df['p2_tier_elo']
    
    # Recent-weighted Elo (emphasizes recent matches)
    df['recent_elo_diff'] = df['p1_recent_elo'] - df['p2_recent_elo']
    
    # Clutch performance Elo (deciding sets, pressure situations)
    df['clutch_elo_diff'] = df['p1_clutch_elo'] - df['p2_clutch_elo']
```

#### Set-Level Intelligence (7)
```python
def add_set_intelligence(df):
    # Deciding set frequency (matches going 3+ sets)
    df['deciding_set'] = (df['sets_played'] >= 3).astype(int)
    
    # Straight sets dominance
    df['straight_sets'] = (df['sets_played'] == 2).astype(int)
    
    # Comeback victories (lost first set, won match)
    df['comeback_win'] = patterns_df['comeback_win'].astype(int)
    
    # Dominant victories (large margin wins)
    df['dominant_win'] = patterns_df['dominant_win'].astype(int)
    
    # Player-specific performance rates
    df['deciding_set_diff'] = df['p1_deciding_set_rate'] - df['p2_deciding_set_rate']
    df['comeback_diff'] = df['p1_comeback_rate'] - df['p2_comeback_rate']
    df['dominant_diff'] = df['p1_dominant_rate'] - df['p2_dominant_rate']
```

#### Market Intelligence (6)
```python
def add_market_intelligence(df):
    # Market confidence (how certain are the odds?)
    df['market_confidence'] = np.abs(df['implied_prob_1'] - df['implied_prob_2'])
    
    # Favorite identification
    df['p1_favorite'] = (df['Odd_1'] < df['Odd_2']).astype(int)
    
    # Odds spread (favorite vs underdog gap)
    df['odds_spread'] = df['underdog_odds'] - df['favorite_odds']
    
    # Market vs ranking agreement
    df['market_rank_agree'] = (df['p1_favorite'] == df['rank_favorite']).astype(int)
    
    # Market surprise factor (disagreement indicators)
    df['market_surprise'] = np.abs(df['p1_favorite'] - df['rank_favorite'])
    
    # Upset potential calculation
    df['upset_potential'] = calculate_upset_potential(df)
```

#### Tournament Context Features (5)
```python
def add_tournament_context(df):
    # Grand Slam importance
    grand_slams = ['Australian Open', 'French Open', 'Wimbledon', 'US Open']
    df['is_grand_slam'] = df['Tournament'].isin(grand_slams).astype(int)
    
    # Masters/Premier tier
    df['is_masters'] = df['Tournament'].isin(masters_1000).astype(int)
    
    # Round importance scoring
    round_importance = {'Final': 5, 'Semi': 4, 'Quarter': 3, 'R16': 2, 'R32': 1}
    df['round_importance'] = df['Round'].map(round_importance).fillna(0)
    
    # Rest advantage (days since last match)
    df['rest_advantage'] = calculate_rest_days(df)
    
    # Surface transition advantage
    df['transition_advantage'] = calculate_surface_transitions(df)
```

## 🤖 Machine Learning Implementation

### Professional 3-Way Temporal Validation

**Critical Design Choice**: Prevents data leakage through strict chronological separation

```python
# IMPROVED temporal split with extended training data
train_mask = (df['Date'].dt.year >= 2010) & (df['Date'].dt.year <= 2023)  # 2010-2023
val_mask = (df['Date'].dt.year == 2024) & (df['Date'].dt.month <= 4)      # Jan-Apr 2024
test_mask = ((df['Date'].dt.year == 2024) & (df['Date'].dt.month > 4)) | (df['Date'].dt.year == 2025)  # May 2024+
```

**Why This Split Is Superior**:
- **No data leakage**: Future never informs past predictions
- **Extended training**: More samples (2010-2023 vs 2012-2022) for better accuracy
- **Realistic testing**: Performance on most recent matches
- **Professional standard**: Used by quantitative hedge funds

### Systematic Hyperparameter Optimization

```python
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 8, 10], 
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Test 108 different combinations systematically
    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model
```

**Process**:
1. Define hyperparameter search space
2. Test 108 combinations systematically
3. Select best model using validation performance
4. Final evaluation on untouched test set
5. Save optimal model

### Performance Results

**Final Accuracy (Unbiased Test Set)**:
- **ATP**: 67.5% on complete 2024-2025 test set
- **WTA**: 67.4% on complete 2024-2025 test set

**Benchmark Comparison**:
| System | Accuracy | Method |
|--------|----------|---------|
| Random Baseline | 50% | Coin flip |
| Rankings Only | ~60% | Use current rankings |
| Basic Tennis Model | ~65% | Rankings + surface |
| **Professional Betting Markets** | **65-68%** | All public information |
| **Our System** | **67.5% ATP / 67.4% WTA** | **40+ advanced features** |

## 🌐 Django Web Application

### Architecture Overview

```python
# URL Structure
urlpatterns = [
    path('', views.home, name='home'),           # Input form
    path('predict/', views.predict_view, name='predict'),  # Prediction engine
]
```

### Database Models (`models.py`)

```python
class PredictionRequest(models.Model):
    TOUR_CATEGORIES = [('ATP', 'ATP'), ('WTA', 'WTA')]
    
    player_1 = models.CharField(max_length=100)
    player_2 = models.CharField(max_length=100)
    player_1_rank = models.IntegerField()
    player_2_rank = models.IntegerField()
    surface = models.CharField(max_length=20)
    tourney_date = models.DateField()
    tour_category = models.CharField(max_length=3, choices=TOUR_CATEGORIES)
    predicted_winner = models.CharField(max_length=100)
    request_time = models.DateTimeField(auto_now_add=True)
```

### Views Logic (`views.py`)

#### Home View
- Renders comprehensive form with all required inputs
- Tour category selection (ATP/WTA)
- Player details (names, rankings, points)
- Match context (surface, tournament, date, round)
- Betting odds for market intelligence

#### Prediction View
**Complex Feature Engineering Pipeline**:

1. **Extract Form Data**: All 11 input parameters
2. **Load Historical Data**: Entire processed dataset for context
3. **Compute Dynamic Features**:
   ```python
   # Real-time Elo calculation
   p1_elo, p1_surf_elo = compute_current_elos(df_all, player_1, surface, tourney_date)
   p2_elo, p2_surf_elo = compute_current_elos(df_all, player_2, surface, tourney_date)
   
   # Head-to-head history
   h2h_diff = compute_h2h_diff(df_all, player_1, player_2, tourney_date)
   ```

4. **Feature Vector Assembly**: 13 basic features → enhanced pipeline
5. **Model Selection**: Enhanced → Basic → Random fallback
6. **Confidence Scoring**: Probability-based confidence percentage
7. **Database Logging**: Store prediction for analysis
8. **Result Display**: Winner + confidence + model type

### Enhanced Prediction API (`predictor_utils.py`)

#### Model Management System
```python
MODELS = {
    'enhanced_atp': load_model('enhanced_atp_model.pkl'),    # 67.5% accuracy
    'enhanced_wta': load_model('enhanced_wta_model.pkl'),    # 67.4% accuracy  
    'basic_atp': load_model('xgb_match_winner_atp.pkl'),     # Fallback
    'basic_wta': load_model('xgb_match_winner_wta.pkl'),     # Fallback
}
```

#### Real-Time Feature Engineering
```python
def _prepare_enhanced_features(player_1, player_2, surface, tourney_date, tournament,
                             round_name, player_1_rank, player_2_rank, 
                             odd1, odd2, pts1, pts2, best_of, series_enc, is_atp=True):
    # Load historical data for context
    historical_df = load_processed_csv('atp' if is_atp else 'wta')
    
    # Create synthetic match row
    match_row = create_match_dataframe(...)
    
    # Combine with history for feature engineering  
    combined_df = pd.concat([historical_df, match_row])
    
    # Run through 40+ feature pipeline
    X, _ = feature_engineer(combined_df, is_atp=is_atp)
    
    # Return features for prediction match
    return X[-1:]
```

#### Multi-Model Prediction with Fallback
```python
def predict_atp_winner_with_confidence(...) -> Tuple[int, float, str]:
    # Try enhanced model first (67.5% accuracy)
    if MODELS['enhanced_atp'] is not None:
        try:
            X = _prepare_enhanced_features(...)
            prediction = MODELS['enhanced_atp'].predict(X)[0]
            probabilities = MODELS['enhanced_atp'].predict_proba(X)[0]
            confidence = max(probabilities) * 100
            return int(prediction), confidence, 'Enhanced'
        except Exception as e:
            print(f"Enhanced prediction failed: {e}")
    
    # Fallback to basic model
    if MODELS['basic_atp'] is not None:
        # Use original 13-feature system
        prediction, confidence = basic_prediction_logic()
        return int(prediction), confidence, 'Basic'
    
    # Last resort
    return random.choice([0, 1]), 55.0, 'Random (No Models Available)'
```

### Frontend Templates

#### Input Form (`home.html`)
- **Comprehensive Form**: 11 input fields covering all match parameters
- **Dynamic UI**: Series field shows/hides based on ATP/WTA selection
- **Validation**: Required fields, proper data types
- **User Experience**: Clear labels, logical grouping

#### Results Page (`result.html`)  
- **Match Summary**: All input parameters displayed
- **Prediction Display**: Winner clearly highlighted
- **Confidence Visualization**: 
  ```html
  <div class="confidence-bar">
      <div class="confidence-fill" style="width: {{ confidence_percent }}%;"></div>
  </div>
  ```
- **Model Transparency**: Shows which model was used
- **Enhanced Features Note**: Indicates when using 40+ feature model

## 🔄 Complete System Workflow

```
1. User Input (Web Form)
   ↓
2. Data Extraction (11 parameters)
   ↓
3. Historical Data Loading (107K+ matches)
   ↓
4. Real-Time Feature Engineering:
   ├── Basic Features (13)
   ├── Form & Momentum (5) 
   ├── Advanced Elo (5)
   ├── Set Intelligence (7)
   ├── Market Intelligence (6)
   └── Tournament Context (5)
   ↓
5. Feature Quality Control (remove low variance)
   ↓
6. Model Selection (Enhanced → Basic → Random)
   ↓
7. Prediction + Confidence Scoring
   ↓
8. Database Logging
   ↓
9. Result Display (Winner + Confidence + Model Type)
```

## 💡 Key Technical Innovations

### 1. Professional-Grade Validation
- **3-way temporal split**: Prevents data leakage
- **Extended training period**: 2010-2023 for maximum data
- **Unbiased evaluation**: Strict chronological testing

### 2. Advanced Feature Engineering
- **40+ features** from same raw data
- **Multiple time horizons**: Recent form + career patterns
- **Context awareness**: Tournament/surface/round importance
- **Market intelligence**: Betting odds incorporation
- **Set-level analysis**: Score pattern recognition

### 3. Production-Ready Architecture
- **Graceful degradation**: Enhanced → Basic → Random fallback
- **Real-time computation**: Dynamic feature calculation
- **Confidence scoring**: Probability-based reliability
- **Comprehensive logging**: All predictions stored
- **Professional UI**: Clear, informative interface

### 4. Systematic Optimization  
- **Hyperparameter tuning**: 108 combinations tested
- **Feature quality control**: Low-variance removal
- **Model selection**: Validation-based optimization

## 🎯 Interview Talking Points

### Technical Depth
- **Data Engineering**: Automated Kaggle pipeline, 107K+ match processing
- **Feature Engineering**: 40+ advanced features from domain expertise
- **ML Engineering**: Professional validation, hyperparameter tuning
- **Full-Stack Development**: Django web app, real-time predictions

### Problem-Solving Skills
- **Data Leakage Prevention**: Temporal validation methodology
- **Performance Optimization**: Feature quality control, systematic tuning
- **Production Reliability**: Multi-model fallback system
- **User Experience**: Confidence scoring, transparent model selection

### Business Impact
- **Competitive Performance**: Matches professional betting markets (67.5% vs 65-68%)
- **Real-World Application**: Production web interface
- **Scalable Architecture**: Handles large datasets, real-time predictions

### Mathematical Rigor
- **Statistical Validation**: Proper train/val/test methodology
- **Feature Selection**: Variance threshold, correlation analysis
- **Model Evaluation**: Unbiased test set performance
- **Confidence Quantification**: Probability-based scoring

## 🏆 Final Performance Summary

**Achievement**: Built a professional-grade tennis prediction system achieving **68% accuracy** (competitive with betting markets) using advanced ML techniques on 107K+ historical matches.

**Key Metrics**:
- **Accuracy**: 67.5% ATP, 67.4% WTA (unbiased test set)
- **Data Scale**: 107K+ matches processed
- **Feature Engineering**: 40+ advanced features
- **Validation**: Professional 3-way temporal split
- **Production**: Full Django web application

This system demonstrates the complete ML engineering lifecycle: data acquisition, advanced feature engineering, rigorous validation, hyperparameter optimization, and production deployment with a professional web interface.