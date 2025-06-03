# predictor/views.py

from django.shortcuts import render
from django.http import HttpResponseRedirect
from datetime import datetime
import pandas as pd
import numpy as np

from .predictor_utils import predict_atp_winner, predict_wta_winner
from .models import PredictionRequest
from .ml_utils import load_processed_csv

# ────────────────────────────────────────────────────────────────────────────────
# Mappings (must match exactly what you used in ml_utils.feature_engineer)
# ────────────────────────────────────────────────────────────────────────────────

SERIES_CATEGORIES_ATP = [
    'Grand Slam',
    'Masters 1000',
    'ATP 500',
    'ATP 250',
    'ATP Finals'
]

# Map series string → integer label
SERIES_MAPPING = {name: idx for idx, name in enumerate(SERIES_CATEGORIES_ATP)}

COURT_CATEGORIES = [
    'Centre Court',
    'Outside Court 1',
    'Outside Court 2',
    'Court Philippe‐Chatrier',
    'Court Suzanne‐Lenglen',
    'Main Arena',
    # (add any other court names exactly as in your CSV)
]

# Map court string → integer label
COURT_MAPPING = {name: idx for idx, name in enumerate(COURT_CATEGORIES)}

ROUND_CATEGORIES = [
    'R128',
    'R64',
    'R32',
    'R16',
    'Quarter',
    'Semi',
    'Final',
    # (add any other round names exactly as in your CSV)
]

# Map round string → integer label
ROUND_MAPPING = {name: idx for idx, name in enumerate(ROUND_CATEGORIES)}


# ────────────────────────────────────────────────────────────────────────────────
# Helpers for prediction‐time Elo and H2H
# ────────────────────────────────────────────────────────────────────────────────

def compute_current_elos(df, player, surface, reference_date, K=20, initial_elo=1500):
    """
    Computes BOTH:
      • overall Elo of `player` just *before* reference_date
      • surface‐specific Elo of `player` on `surface` just *before* reference_date

    df: the entire processed dataframe (ATP or WTA) with columns:
       ['Date','Player_1','Player_2','Winner','Surface']
    player: string name of the player whose Elo we want
    surface: string name of the surface ('Clay','Grass','Hard','Carpet', etc.)
    reference_date: a Python date (not datetime) representing match date
    """
    # 1) Filter to all matches strictly before reference_date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    past = df.loc[df['Date'] < pd.to_datetime(reference_date)]

    # 2) Sort by date ascending
    past = past.sort_values('Date').reset_index(drop=True)

    # 3) Initialize Elo dictionaries
    #    overall Elo per player
    all_players = pd.unique(pd.concat([past['Player_1'], past['Player_2']]))
    elo_dict = {p: initial_elo for p in all_players}

    #    surface Elo per player‐surface pair
    surfaces = ['Clay','Grass','Hard','Carpet']
    surf_elo = {(p, s): initial_elo for p in all_players for s in surfaces}

    # 4) Walk through each past match and update Elo
    for _, row in past.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        w  = row['Winner']
        s  = row['Surface']

        e1 = elo_dict.get(p1, initial_elo)
        e2 = elo_dict.get(p2, initial_elo)

        # expected scores (overall Elo)
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        exp2 = 1 / (1 + 10 ** ((e1 - e2) / 400))
        res1 = 1 if w == p1 else 0
        res2 = 1 - res1

        elo_dict[p1] = e1 + K * (res1 - exp1)
        elo_dict[p2] = e2 + K * (res2 - exp2)

        # update surface Elo as well for whichever surface this match was on
        se1 = surf_elo.get((p1, s), initial_elo)
        se2 = surf_elo.get((p2, s), initial_elo)
        exp_se1 = 1 / (1 + 10 ** ((se2 - se1) / 400))
        exp_se2 = 1 / (1 + 10 ** ((se1 - se2) / 400))

        surf_elo[(p1, s)] = se1 + K * (res1 - exp_se1)
        surf_elo[(p2, s)] = se2 + K * (res2 - exp_se2)

    # 5) Return pre‐match Elo for requested player
    overall_elo = elo_dict.get(player, initial_elo)
    surf_elo_before = surf_elo.get((player, surface), initial_elo)
    return overall_elo, surf_elo_before


def compute_h2h_diff(df, player_1, player_2, reference_date):
    """
    Returns (# times player_1 beat player_2) − (# times player_2 beat player_1),
    considering only matches that occurred strictly before reference_date.

    df: entire processed DataFrame with columns ['Date','Player_1','Player_2','Winner']
    player_1, player_2: strings (the two players)
    reference_date: Python date of the new match
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    past = df.loc[df['Date'] < pd.to_datetime(reference_date)]

    # Count how many times each won
    p1_wins = past.loc[
        (past['Player_1'] == player_1) & (past['Player_2'] == player_2) & (past['Winner'] == player_1)
    ].shape[0] + past.loc[
        (past['Player_1'] == player_2) & (past['Player_2'] == player_1) & (past['Winner'] == player_1)
    ].shape[0]

    p2_wins = past.loc[
        (past['Player_1'] == player_2) & (past['Player_2'] == player_1) & (past['Winner'] == player_2)
    ].shape[0] + past.loc[
        (past['Player_1'] == player_1) & (past['Player_2'] == player_2) & (past['Winner'] == player_2)
    ].shape[0]

    return p1_wins - p2_wins


# ────────────────────────────────────────────────────────────────────────────────
# Main Views
# ────────────────────────────────────────────────────────────────────────────────

def home(request):
    """
    Renders the homepage with a form to choose ATP/WTA, two players, ranks, surface, date, etc.
    """
    return render(request, 'predictor/home.html')


def predict_view(request):
    """
    Handles the POST from home.html, rebuilds the full 13‐feature vector,
    calls the correct model, saves a PredictionRequest, and renders result.html.
    """
    if request.method != 'POST':
        return HttpResponseRedirect('/')

    # ─── 1) Extract basic form fields ────────────────────────────────────────────
    tour_category   = request.POST.get('tour_category')        # 'ATP' or 'WTA'
    player_1        = request.POST.get('player_1')
    player_2        = request.POST.get('player_2')
    player_1_rank   = int(request.POST.get('player_1_rank'))
    player_2_rank   = int(request.POST.get('player_2_rank'))
    surface         = request.POST.get('surface')
    tourney_date_s  = request.POST.get('tourney_date')        # 'YYYY-MM-DD'
    tourney_date    = datetime.strptime(tourney_date_s, '%Y-%m-%d').date()

    # ─── 2) Load processed CSV (ATP or WTA) for computing Elo/H2H ───────────────
    df_all = load_processed_csv('atp') if tour_category == 'ATP' else load_processed_csv('wta')
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')

    # ─── 3) Compute Elo & surface Elo for each player ───────────────────────────
    p1_elo, p1_surf_elo = compute_current_elos(df_all, player_1, surface, tourney_date)
    p2_elo, p2_surf_elo = compute_current_elos(df_all, player_2, surface, tourney_date)

    elo_diff      = p1_elo - p2_elo
    surf_elo_diff = p1_surf_elo - p2_surf_elo

    # ─── 4) Compute head‐to‐head difference ────────────────────────────────────
    h2h_diff = compute_h2h_diff(df_all, player_1, player_2, tourney_date)

    # ─── 5) Extract Odds and Points ─────────────────────────────────────────────
    odd1   = float(request.POST.get('odd_1'))
    odd2   = float(request.POST.get('odd_2'))
    pts1   = float(request.POST.get('pts_1'))
    pts2   = float(request.POST.get('pts_2'))
    best_of = int(request.POST.get('best_of'))

    odd_diff = odd1 - odd2
    pts_diff = pts1 - pts2

    # ─── 6) Series encoding (ATP only) ────────────────────────────────────────
    if tour_category == 'ATP':
        series_str = request.POST.get('series')
        series_enc = SERIES_MAPPING.get(series_str, len(SERIES_CATEGORIES_ATP))
    else:
        series_enc = 0

    # ─── 7) Court encoding ────────────────────────────────────────────────────
    court_str = request.POST.get('court')
    court_enc = COURT_MAPPING.get(court_str, len(COURT_CATEGORIES))

    # ─── 8) Round encoding ────────────────────────────────────────────────────
    round_str = request.POST.get('round')
    round_enc = ROUND_MAPPING.get(round_str, len(ROUND_CATEGORIES))

    # ─── 9) Year & month features ─────────────────────────────────────────────
    year  = tourney_date.year
    month = tourney_date.month

    # ─── 10) Surface encoding (matches ml_utils.surface_map) ───────────────────
    surface_map = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3}
    surface_enc = surface_map.get(surface, 4)

    # ─── 11) Rank difference ──────────────────────────────────────────────────
    rank_diff = player_1_rank - player_2_rank

    # ─── 12) Assemble feature vector in the same order ml_utils used ──────────
    features = np.array([[
        rank_diff,
        surface_enc,
        year,
        month,
        odd_diff,
        pts_diff,
        best_of,
        series_enc,
        court_enc,
        round_enc,
        elo_diff,
        surf_elo_diff,
        h2h_diff
    ]])

    # ─── 13) Call the correct model ────────────────────────────────────────────
    if tour_category == 'ATP':
        flag = predict_atp_winner(
            player_1_rank=player_1_rank,
            player_2_rank=player_2_rank,
            surface=surface,
            tourney_date=tourney_date,
            odd1=odd1,
            odd2=odd2,
            pts1=pts1,
            pts2=pts2,
            best_of=best_of,
            series_enc=series_enc,
            court_enc=court_enc,
            round_enc=round_enc,
            elo_diff=elo_diff,
            surf_elo_diff=surf_elo_diff,
            h2h_diff=h2h_diff
        )
    else:  # WTA
        flag = predict_wta_winner(
            player_1_rank=player_1_rank,
            player_2_rank=player_2_rank,
            surface=surface,
            tourney_date=tourney_date,
            odd1=odd1,
            odd2=odd2,
            pts1=pts1,
            pts2=pts2,
            best_of=best_of,
            series_enc=0,
            court_enc=court_enc,
            round_enc=round_enc,
            elo_diff=elo_diff,
            surf_elo_diff=surf_elo_diff,
            h2h_diff=h2h_diff
        )

    predicted_winner = player_1 if flag == 1 else player_2

    # ─── 14) Log to database ──────────────────────────────────────────────────
    pr = PredictionRequest(
        player_1        = player_1,
        player_2        = player_2,
        player_1_rank   = player_1_rank,
        player_2_rank   = player_2_rank,
        surface         = surface,
        tourney_date    = tourney_date,
        tour_category   = tour_category,
        predicted_winner= predicted_winner
    )
    pr.save()

    # ─── 15) Render result.html ───────────────────────────────────────────────
    context = {
        'tour_category':   tour_category,
        'player_1':        player_1,
        'player_2':        player_2,
        'player_1_rank':   player_1_rank,
        'player_2_rank':   player_2_rank,
        'surface':         surface,
        'tourney_date':    tourney_date,
        'odd_1':           odd1,
        'odd_2':           odd2,
        'pts_1':           pts1,
        'pts_2':           pts2,
        'best_of':         best_of,
        'series':          series_str if tour_category=='ATP' else '',
        'court':           court_str,
        'round':           round_str,
        'predicted_winner':predicted_winner
    }
    return render(request, 'predictor/result.html', context)
