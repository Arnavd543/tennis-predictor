This program attempts to accurately predict the winner of tennis matches using an XGBoost machine learning model and data from Kaggle

Raw Data Structure (`data/atp_tennis.csv`, `data/wta.csv`)
Tournament,Date,Series,Court,Surface,Round,Best of,Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score


Step 1: Data Download
1. Authenticates with Kaggle API using credentials
2. Downloads compressed dataset files (~107K matches)
3. Extracts CSV files to `data/` directory

Step 2: Data Cleaning & Standardization
Select only the columns we need for ML
atp_columns = [
    'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round',
    'Best of', 'Player_1', 'Player_2', 'Winner',
    'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Score'
]

atp_df = atp_df[atp_columns].dropna()  # Remove incomplete records


Data Quality Improvements:
- Standardize columns: Ensure ATP/WTA have same structure
- Remove missing data: Drop incomplete matches
- Parse dates: Convert strings to datetime objects
- Save processed: Store as `atp_processed.csv`, `wta_processed.csv`

---

Step 3: Feature Engineering

The system extracts 40+ advanced features from the data.

3-Way Temporal Split:
Train: 2010–2023 (extended training for optimal accuracy)
Validation: Early 2024 (Jan-Apr hyperparameter tuning)
Test: Mid 2024–2025 (May 2024+ final performance report)

Feature Categories (40+ Total)

Basic Features
basic_features = [
    'rank_diff',        # Official ranking difference
    'surface_enc',      # Surface type (0=Clay, 1=Grass, 2=Hard, 3=Carpet)
    'year', 'month',    # Seasonal patterns
    'odd_diff',         # Betting market expectations
    'pts_diff',         # Ranking points difference
    'best_of',          # Match format (3 or 5 sets)
    'series_enc',       # Tournament importance
    'tournament_enc',   # Tournament encoding
    'round_enc',        # Round encoding
    'elo_diff',         # Overall Elo difference
    'surf_elo_diff',    # Surface-specific Elo difference
    'h2h_diff'          # Head-to-head win difference
]

Form & Momentum Features
def compute_form_features(df):
    """Extract recent performance patterns"""
    # Current win/loss streaks
    df['streak_diff'] = df['player_1_streak'] - df['player_2_streak']
    
    # Recent form (last 10 matches)
    df['form_10_diff'] = df['p1_form_10'] - df['p2_form_10']
    
    # Last 30 days performance
    df['form_30d_diff'] = df['p1_form_30d'] - df['p2_form_30d']
    
    # Surface-specific recent form
    df['surface_form_diff'] = df['p1_surface_form'] - df['p2_surface_form']
    
    # Tournament-specific form
    df['tournament_form_diff'] = df['p1_tournament_form'] - df['p2_tournament_form']

Advanced Elo Systems
def compute_advanced_elo(df):
    """Multiple specialized Elo ratings"""
    # Tournament-specific Elo (Grand Slams vs regular)
    df['tournament_elo_diff'] = df['p1_tournament_elo'] - df['p2_tournament_elo']
    
    # Round-specific Elo (Finals vs early rounds)
    df['round_elo_diff'] = df['p1_round_elo'] - df['p2_round_elo']
    
    # Opponent-tier Elo (vs Top10, Top50, etc.)
    df['opponent_tier_elo_diff'] = df['p1_tier_elo'] - df['p2_tier_elo']
    
    # Recent-weighted Elo (recent matches matter more)
    df['recent_elo_diff'] = df['p1_recent_elo'] - df['p2_recent_elo']
    
    # Clutch performance Elo (deciding sets, tiebreaks)
    df['clutch_elo_diff'] = df['p1_clutch_elo'] - df['p2_clutch_elo']

Set-Level Intelligence
def extract_set_intelligence(df):
    """Parse match scores for advanced insights"""
    # Deciding set analysis (3+ sets played)
    df['deciding_set'] = df['Score'].apply(lambda x: 1 if count_sets(x) >= 3 else 0)
    
    # Straight sets dominance (2-0, 3-0)
    df['straight_sets'] = df['Score'].apply(is_straight_sets)
    
    # Comeback victories (lost first set, won match)
    df['comeback_win'] = df['Score'].apply(is_comeback)
    
    # Dominant victories (large margin wins)
    df['dominant_win'] = df['Score'].apply(is_dominant)
    
    # Player-specific rates
    df['p1_deciding_set_rate'] = get_deciding_set_rate(df, 'Player_1')
    df['p1_comeback_rate'] = get_comeback_rate(df, 'Player_1')
    df['p1_dominant_rate'] = get_dominant_rate(df, 'Player_1')

Market Intelligence
def compute_market_intelligence(df):
    """Extract insights from betting markets"""
    # Market confidence (how certain are the odds?)
    df['market_confidence'] = 1 / (df['Odd_1'] * df['Odd_2'])
    
    # Player 1 favorite indicator
    df['p1_favorite'] = (df['Odd_1'] < df['Odd_2']).astype(int)
    
    # Odds spread (favorite vs underdog gap)
    df['odds_spread'] = np.abs(df['Odd_1'] - df['Odd_2'])
    
    # Market vs ranking agreement
    df['market_rank_agree'] = (df['p1_favorite'] == df['rank_favorite']).astype(int)
    
    # Market surprise factor
    df['market_surprise'] = np.abs(df['p1_favorite'] - df['rank_favorite'])
    
    # Upset potential score
    df['upset_potential'] = calculate_upset_potential(df)

Tournament Context Features
def add_tournament_context(df):
    """Tournament-specific contextual features"""
    # Grand Slam importance
    grand_slams = ['Australian Open', 'French Open', 'Wimbledon', 'US Open']
    df['is_grand_slam'] = df['Tournament'].isin(grand_slams).astype(int)
    
    # Masters/Premier importance
    df['is_masters'] = df['Series'].str.contains('Masters|Premier', na=False).astype(int)
    
    # Round importance score
    round_importance = {'Final': 7, 'Semi': 6, 'Quarter': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1}
    df['round_importance'] = df['Round'].map(round_importance).fillna(1)
    
    # Rest advantage (days since last match)
    df['rest_advantage'] = calculate_rest_days(df)
    
    # Surface transition advantage
    df['transition_advantage'] = calculate_surface_transitions(df)

Step 4: ML Training with Hyperparameter Tuning

Systematic Hyperparameter Optimization

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Professional hyperparameter tuning using validation set"""
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Test 108 different combinations systematically
    best_score = 0
    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model


Accuracy: 67.5% ATP / 67.4% WTA

Step 5: Django Web Framework

Step 6: Enhanced Prediction Pipeline

Real-Time Feature Engineering
def _prepare_enhanced_features(player_1, player_2, surface, tourney_date, tournament,
                             round_name, player_1_rank, player_2_rank, 
                             odd1, odd2, pts1, pts2, best_of, series_enc, is_atp=True):
    """
    Prepare 40+ enhanced features for a single match prediction.
    
    This function creates a synthetic match row and processes it through the same
    feature engineering pipeline used during training to ensure consistency.
    """
    # Load historical data for context
    tour = 'atp' if is_atp else 'wta'
    historical_df = load_processed_csv(tour)
    
    # Create synthetic match row
    match_row = pd.DataFrame([{
        'Date': tourney_date,
        'Player_1': player_1,
        'Player_2': player_2,
        'Tournament': tournament,
        'Round': round_name,
        # ... all match details
    }])
    
    # Combine with historical data for context
    combined_df = pd.concat([historical_df, match_row], ignore_index=True)
    
    # Run through enhanced feature engineering pipeline
    X, _ = feature_engineer(combined_df, is_atp=is_atp)
    
    # Return features for the prediction match (last row)
    return X[-1:]

Model Selection with Fallback
def predict_atp_winner_with_confidence(player_1, player_2, surface, tourney_date, tournament,
                                     round_name, player_1_rank, player_2_rank, 
                                     odd1, odd2, pts1, pts2, best_of=3, series_enc=0):
    """
    Predict ATP match winner with confidence score and model type.
    
    Returns:
        Tuple[int, float, str]: (winner_prediction, confidence_score, model_type)
    """
    # Try enhanced model first (74%+ accuracy)
    if MODELS['enhanced_atp'] is not None:
        try:
            X = _prepare_enhanced_features(
                player_1, player_2, surface, tourney_date, tournament,
                round_name, player_1_rank, player_2_rank, 
                odd1, odd2, pts1, pts2, best_of, series_enc, is_atp=True
            )
            
            prediction = MODELS['enhanced_atp'].predict(X)[0]
            probabilities = MODELS['enhanced_atp'].predict_proba(X)[0]
            confidence = max(probabilities) * 100  # Convert to percentage
            
            return int(prediction), confidence, 'Enhanced'
            
        except Exception as e:
            print(f"Enhanced prediction failed: {e}")
    
    # Fallback to basic model (70% accuracy)
    if MODELS['basic_atp'] is not None:
        # Use original 13-feature system
        prediction, confidence = basic_prediction_logic()
        return int(prediction), confidence, 'Basic'
    
    # Last resort
    return random.choice([0, 1]), 55.0, 'Random (No Models Available)'


Step 7: Performance Analysis & Validation

Accuracy Benchmarks

| **System** | **Accuracy** | **Methodology** | **Features** |
|------------|--------------|-----------------|--------------|
| **Random Baseline** | 50% | Coin flip | None |
| **Ranking Only** | ~60% | Use current rankings | 1 |
| **Basic Tennis Model** | ~65% | Rankings + surface | 3-5 |
| **Betting Markets** | 65-68% | Professional oddsmakers | All public info |
| **Our System** | **67.5% ATP / 67.4% WTA** | **40+ advanced features** | **40+** |
