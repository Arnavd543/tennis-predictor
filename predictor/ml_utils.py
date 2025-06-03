import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from scipy.stats import randint, uniform

# ────────────────────────────────────────────────────────────────────────────────
# Paths & Directories
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.getcwd()
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')                # expects atp_processed.csv & wta_processed.csv
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'predictor', 'models')  # will save models here
os.makedirs(MODEL_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: Load the preprocessed CSV for ATP or WTA
# ────────────────────────────────────────────────────────────────────────────────
def load_processed_csv(tour):
    """
    tour: 'atp' or 'wta'
    Loads and returns a pandas DataFrame from data/atp_processed.csv or data/wta_processed.csv.
    Raises FileNotFoundError if the expected file is missing.
    """
    filename = f"{tour}_processed.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Make sure you ran download_and_preprocess."
        )
    # low_memory=False avoids mixed‐type warnings when reading large CSVs
    df = pd.read_csv(path, low_memory=False)
    return df

# ────────────────────────────────────────────────────────────────────────────────
# Compute Elo Ratings for Every Match
# ────────────────────────────────────────────────────────────────────────────────
def compute_elo(df, K=20, initial_elo=1500):
    """
    Given a DataFrame sorted chronologically by 'Date' with columns:
       ['Player_1', 'Player_2', 'Winner', 'Date', 'Surface'], etc.
    this function computes:
      - 'elo_1': Player_1's Elo **before** the match
      - 'elo_2': Player_2's Elo **before** the match

    It returns a copy of the DataFrame with those two new columns appended.

    Steps:
      1) Sort df by Date (ascending).
      2) Initialize every player’s Elo to initial_elo.
      3) For each match row:
          a) Record pre‐match Elo of p1 and p2 (elo_1, elo_2).
          b) Compute expected scores exp1, exp2 via standard Elo formula.
          c) Update Elo for both players based on match outcome (Winner).
    """
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Date'])
    df_copy = df_copy.sort_values('Date').reset_index(drop=True)

    # Create a set of all unique players
    all_players = pd.unique(pd.concat([df_copy['Player_1'], df_copy['Player_2']]))

    # Initialize Elo for each player
    elo_dict = {player: initial_elo for player in all_players}

    pre_elo_1 = []
    pre_elo_2 = []

    for _, row in df_copy.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        w  = row['Winner']

        e1 = elo_dict.get(p1, initial_elo)
        e2 = elo_dict.get(p2, initial_elo)

        # Record pre‐match Elo
        pre_elo_1.append(e1)
        pre_elo_2.append(e2)

        # Expected scores
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        exp2 = 1 / (1 + 10 ** ((e1 - e2) / 400))

        # Actual results
        res1 = 1 if w == p1 else 0
        res2 = 1 - res1

        # Update Elo
        elo_dict[p1] = e1 + K * (res1 - exp1)
        elo_dict[p2] = e2 + K * (res2 - exp2)

    df_copy['elo_1'] = pre_elo_1
    df_copy['elo_2'] = pre_elo_2

    return df_copy

# ────────────────────────────────────────────────────────────────────────────────
# Main Feature Engineering (adds Elo, Surface‐Elo, Head‐to‐Head)
# ────────────────────────────────────────────────────────────────────────────────
def feature_engineer(df, is_atp=True):
    """
    Given a DataFrame df with columns:
        ['Date', 'Player_1', 'Player_2', 'Winner',
         'Rank_1', 'Rank_2', 'Surface',
         'Odd_1', 'Odd_2', 'Pts_1', 'Pts_2',
         'Best of', 'Court', 'Round'] 
        (and 'Series' if is_atp=True),
    this function computes a rich feature set and returns:
      X: numpy array of shape (n_samples, 13) containing:
         [rank_diff, surface_enc, year, month,
          odd_diff, pts_diff, best_of,
          series_enc, court_enc, round_enc,
          elo_diff, surf_elo_diff, h2h_diff]
      y: numpy array of shape (n_samples,) with binary target (1 if Player_1 won).

    Steps:
      1) Convert 'Date' → datetime; drop invalid rows.
      2) Sort df by date.
      3) Initialize:
         • elo_dict for overall Elo
         • surf_elo_dict for surface‐specific Elo
         • h2h_dict for head‐to‐head counts
      4) Iterate in chronological order:
         a) Record pre‐match:
            - elo_1, elo_2
            - surf_elo_1, surf_elo_2 (on match’s surface)
            - h2h_diff = (# wins p1 vs p2) − (# wins p2 vs p1)
         b) Update:
            - overall Elo of p1, p2
            - surface Elo of p1, p2 on that surface
            - head‐to‐head counts
      5) Add the arrays as columns to df.
      6) Build remaining features:
         • rank_diff = Rank_1 − Rank_2
         • surface_enc: map “Clay/Grass/Hard/Carpet” → {0,1,2,3}; else 4
         • odd_diff = Odd_1 − Odd_2     (after coercing to numeric)
         • pts_diff = Pts_1 − Pts_2     (after coercing to numeric)
         • best_of = numeric(“Best of”), default to 3 if missing
         • series_enc = LabelEncode(Series) if ATP; else 0
         • court_enc = LabelEncode(Court)
         • round_enc = LabelEncode(Round)
         • elo_diff = elo_1 − elo_2
         • surf_elo_diff = surf_elo_1 − surf_elo_2
         • h2h_diff (from step 4a)
         • target = 1 if Winner == Player_1 else 0
      7) Drop any rows containing NaN in those feature or target columns.
      8) Return X (all 13 features) and y (target).
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Build list of all unique players
    all_players = pd.unique(pd.concat([df['Player_1'], df['Player_2']]))

    # 3a) Initialize overall Elo
    initial_elo = 1500
    elo_dict = {p: initial_elo for p in all_players}

    # 3b) Initialize surface Elo for each surface × each player
    surfaces = ['Clay', 'Grass', 'Hard', 'Carpet']
    surf_elo_dict = {}
    for p in all_players:
        for s in surfaces:
            surf_elo_dict[(p, s)] = initial_elo

    # 3c) Initialize head‐to‐head dictionary
    # h2h_dict[(winner, loser)] = count of how many times 'winner' beat 'loser'
    h2h_dict = {}

    # Prepare arrays to store pre‐match Elo/Surface‐Elo/H2H
    elo_1_list        = []
    elo_2_list        = []
    surf_elo_1_list   = []
    surf_elo_2_list   = []
    h2h_diff_list     = []

    K = 20  # Elo k‐factor

    # 4) Iterate chronologically
    for _, row in df.iterrows():
        p1    = row['Player_1']
        p2    = row['Player_2']
        w     = row['Winner']
        surf  = row['Surface']

        # 4a) Pre‐match overall Elo
        e1 = elo_dict.get(p1, initial_elo)
        e2 = elo_dict.get(p2, initial_elo)
        elo_1_list.append(e1)
        elo_2_list.append(e2)

        # 4b) Pre‐match surface Elo
        se1 = surf_elo_dict.get((p1, surf), initial_elo)
        se2 = surf_elo_dict.get((p2, surf), initial_elo)
        surf_elo_1_list.append(se1)
        surf_elo_2_list.append(se2)

        # 4c) Head‐to‐head difference
        h1 = h2h_dict.get((p1, p2), 0)
        h2 = h2h_dict.get((p2, p1), 0)
        h2h_diff_list.append(h1 - h2)

        # 4d) Compute expected scores (overall Elo)
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        exp2 = 1 / (1 + 10 ** ((e1 - e2) / 400))

        res1 = 1 if w == p1 else 0
        res2 = 1 - res1

        # Update overall Elo
        elo_dict[p1] = e1 + K * (res1 - exp1)
        elo_dict[p2] = e2 + K * (res2 - exp2)

        # 4e) Compute expected scores (surface Elo)
        exp_se1 = 1 / (1 + 10 ** ((se2 - se1) / 400))
        exp_se2 = 1 / (1 + 10 ** ((se1 - se2) / 400))

        surf_elo_dict[(p1, surf)] = se1 + K * (res1 - exp_se1)
        surf_elo_dict[(p2, surf)] = se2 + K * (res2 - exp_se2)

        # 4f) Update head‐to‐head counts
        if w == p1:
            h2h_dict[(p1, p2)] = h2h_dict.get((p1, p2), 0) + 1
        else:
            h2h_dict[(p2, p1)] = h2h_dict.get((p2, p1), 0) + 1

    # Assign new Elo/H2H columns
    df['elo_1']       = elo_1_list
    df['elo_2']       = elo_2_list
    df['surf_elo_1']  = surf_elo_1_list
    df['surf_elo_2']  = surf_elo_2_list
    df['h2h_diff']    = h2h_diff_list

    # 5) Build remaining features

    # (a) Year, Month
    df['year']  = df['Date'].dt.year
    df['month'] = df['Date'].dt.month

    # (b) Rank difference
    df['rank_diff'] = df['Rank_1'] - df['Rank_2']

    # (c) Surface encoding
    surface_map = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3}
    df['surface_enc'] = df['Surface'].map(lambda x: surface_map.get(x, 4))

    # (d) Odds difference
    df['Odd_1'] = pd.to_numeric(df['Odd_1'], errors='coerce')
    df['Odd_2'] = pd.to_numeric(df['Odd_2'], errors='coerce')
    df['odd_diff'] = df['Odd_1'] - df['Odd_2']

    # (e) Points difference
    df['Pts_1'] = pd.to_numeric(df['Pts_1'], errors='coerce')
    df['Pts_2'] = pd.to_numeric(df['Pts_2'], errors='coerce')
    df['pts_diff'] = df['Pts_1'] - df['Pts_2']

    # (f) Best‐of (3 or 5)
    df['best_of'] = pd.to_numeric(df['Best of'], errors='coerce').fillna(3).astype(int)

    # (g) Series encoding (ATP only)
    if is_atp:
        le_series = LabelEncoder()
        df['series_enc'] = le_series.fit_transform(df['Series'].astype(str))
    else:
        df['series_enc'] = 0

    # (h) Court encoding
    le_court = LabelEncoder()
    df['court_enc'] = le_court.fit_transform(df['Court'].astype(str))

    # (i) Round encoding
    le_round = LabelEncoder()
    df['round_enc'] = le_round.fit_transform(df['Round'].astype(str))

    # (j) Elo difference
    df['elo_diff'] = df['elo_1'] - df['elo_2']

    # (k) Surface Elo difference
    df['surf_elo_diff'] = df['surf_elo_1'] - df['surf_elo_2']

    # (l) Target variable
    df['player_1_win'] = (df['Winner'] == df['Player_1']).astype(int)

    # 6) Drop rows with any NaN in feature or target columns
    feature_cols = [
        'rank_diff', 'surface_enc', 'year', 'month',
        'odd_diff', 'pts_diff', 'best_of',
        'series_enc', 'court_enc', 'round_enc',
        'elo_diff', 'surf_elo_diff', 'h2h_diff',
        'player_1_win'
    ]
    df = df.dropna(subset=feature_cols)

    # 7) Assemble X and y
    X = df[
        ['rank_diff', 'surface_enc', 'year', 'month',
         'odd_diff', 'pts_diff', 'best_of',
         'series_enc', 'court_enc', 'round_enc',
         'elo_diff', 'surf_elo_diff', 'h2h_diff']
    ].values
    y = df['player_1_win'].values

    return X, y

# ────────────────────────────────────────────────────────────────────────────────
# Train the ATP Model (with hyperparameter tuning)
# ────────────────────────────────────────────────────────────────────────────────
def train_atp_model():
    """
    1) Loads data from data/atp_processed.csv
    2) Feature‐engineers → X, y
    3) Splits train/test (stratified)
    4) Uses RandomizedSearchCV to find best XGBoost hyperparameters
    5) Retrains final model on full train set with best params
    6) Evaluates on test set and prints accuracy
    7) Saves model to predictor/models/xgb_match_winner_atp.pkl
    """
    df = load_processed_csv('atp')
    X, y = feature_engineer(df, is_atp=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Hyperparameter distributions
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5)
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Running RandomizedSearchCV for ATP model (this may take a few minutes)...")
    rand_search.fit(X_train, y_train)
    best_params = rand_search.best_params_
    print(f"Best hyperparameters for ATP: {best_params}")

    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        **best_params
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ATP XGBoost] Validation Accuracy: {acc:.4f}")

    atp_model_path = os.path.join(MODEL_DIR, 'xgb_match_winner_atp.pkl')
    joblib.dump(final_model, atp_model_path)
    print(f"Saved ATP model to {atp_model_path}")

# ────────────────────────────────────────────────────────────────────────────────
# Train the WTA Model (with hyperparameter tuning)
# ────────────────────────────────────────────────────────────────────────────────
def train_wta_model():
    """
    1) Loads data from data/wta_processed.csv
    2) Feature‐engineers → X, y
    3) Splits train/test (stratified)
    4) Uses RandomizedSearchCV to find best XGBoost hyperparameters
    5) Retrains final model on full train set with best params
    6) Evaluates on test set and prints accuracy
    7) Saves model to predictor/models/xgb_match_winner_wta.pkl
    """
    df = load_processed_csv('wta')
    X, y = feature_engineer(df, is_atp=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5)
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Running RandomizedSearchCV for WTA model (this may take a few minutes)...")
    rand_search.fit(X_train, y_train)
    best_params = rand_search.best_params_
    print(f"Best hyperparameters for WTA: {best_params}")

    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        **best_params
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[WTA XGBoost] Validation Accuracy: {acc:.4f}")

    wta_model_path = os.path.join(MODEL_DIR, 'xgb_match_winner_wta.pkl')
    joblib.dump(final_model, wta_model_path)
    print(f"Saved WTA model to {wta_model_path}")

# ────────────────────────────────────────────────────────────────────────────────
# Combined Entrypoint: Train Both Models
# ────────────────────────────────────────────────────────────────────────────────
def train_all_models():
    print("=== Training ATP Model ===")
    train_atp_model()
    print("\n=== Training WTA Model ===")
    train_wta_model()

if __name__ == '__main__':
    train_all_models()
