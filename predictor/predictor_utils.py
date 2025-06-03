# predictor/predictor_utils.py

import os
import joblib
import numpy as np

# Paths to saved models
PROJECT_ROOT = os.getcwd()
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'predictor', 'models')

ATP_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_match_winner_atp.pkl')
WTA_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_match_winner_wta.pkl')

try:
    atp_model = joblib.load(ATP_MODEL_PATH)
except Exception as e:
    atp_model = None
    print(f"ERROR: Could not load ATP model from {ATP_MODEL_PATH}: {e}")

try:
    wta_model = joblib.load(WTA_MODEL_PATH)
except Exception as e:
    wta_model = None
    print(f"ERROR: Could not load WTA model from {WTA_MODEL_PATH}: {e}")


def _prepare_features(player_1_rank, player_2_rank, surface, tourney_date,
                      odd1, odd2, pts1, pts2, best_of, series_enc, court_enc, round_enc):
    """
    Replicate exactly the feature engineering from training:
      • rank_diff   = player_1_rank - player_2_rank
      • surface_enc = map surface string to int
      • year, month from tourney_date
      • odd_diff    = odd1 - odd2
      • pts_diff    = pts1 - pts2
      • best_of     = already numeric (3 or 5)
      • series_enc  = numeric encoding of Series (0 for WTA if not used)
      • court_enc   = numeric encoding of Court
      • round_enc   = numeric encoding of Round

    Returns a numpy array of shape (1, 10) in exactly this column order:
      [rank_diff, surface_enc, year, month, odd_diff, pts_diff,
       best_of, series_enc, court_enc, round_enc]
    """
    # 1) rank difference
    rank_diff = player_1_rank - player_2_rank

    # 2) surface encoding
    surface_map = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3}
    surface_enc = surface_map.get(surface, 4)

    # 3) year, month
    year  = tourney_date.year
    month = tourney_date.month

    # 4) odds difference
    odd_diff = odd1 - odd2

    # 5) points difference
    pts_diff = pts1 - pts2

    # 6) best_of (already numeric; e.g. 3 or 5)
    #    assume best_of is passed in as an int

    # 7) series_enc (if ATP, a precomputed integer; if WTA, pass in 0)

    # 8) court_enc  (precomputed integer)
    # 9) round_enc  (precomputed integer)

    return np.array([[rank_diff, surface_enc, year, month,
                      odd_diff, pts_diff, best_of, series_enc, court_enc, round_enc]])


def predict_atp_winner(player_1_rank, player_2_rank, surface, tourney_date,
                       odd1, odd2, pts1, pts2, best_of, series_enc, court_enc, round_enc):
    """
    Returns 1 if ATP model predicts Player_1 wins, else 0.
    Raises RuntimeError if ATP model is not loaded.
    """
    if atp_model is None:
        raise RuntimeError(f"ATP model not loaded. Expected at {ATP_MODEL_PATH}")

    X = _prepare_features(player_1_rank, player_2_rank, surface, tourney_date,
                          odd1, odd2, pts1, pts2, best_of, series_enc, court_enc, round_enc)
    return int(atp_model.predict(X)[0])


def predict_wta_winner(player_1_rank, player_2_rank, surface, tourney_date,
                       odd1, odd2, pts1, pts2, best_of, series_enc, court_enc, round_enc):
    """
    Returns 1 if WTA model predicts Player_1 wins, else 0.
    Raises RuntimeError if WTA model is not loaded.
    """
    if wta_model is None:
        raise RuntimeError(f"WTA model not loaded. Expected at {WTA_MODEL_PATH}")

    X = _prepare_features(player_1_rank, player_2_rank, surface, tourney_date,
                          odd1, odd2, pts1, pts2, best_of, series_enc, court_enc, round_enc)
    return int(wta_model.predict(X)[0])
