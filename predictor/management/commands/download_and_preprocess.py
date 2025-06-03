# predictor/management/commands/download_and_preprocess.py

import os
import pandas as pd
from django.core.management.base import BaseCommand

# We attempt to import Kaggle CLI first; if unavailable, fall back to kagglehub.
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False


class Command(BaseCommand):
    help = 'Download and preprocess ATP & WTA tennis match data from Kaggle'

    def handle(self, *args, **options):
        # ─────────────────────────────────────────────────────────────────
        # 1) Kaggle dataset identifiers (replace with your own if different):
        #    - ATP dataset:      dissfya/atp-tennis-2000-2023daily-pull
        #    - WTA dataset:      dissfya/wta-tennis-2007-2023-daily-update
        #
        #    After unzipping, we expect:
        #      data/atp_tennis.csv    (ATP matches, columns as in your screenshot)
        #      data/wta.csv           (WTA matches, columns as in your screenshot)
        #
        atp_dataset = 'dissfya/atp-tennis-2000-2023daily-pull'
        wta_dataset = 'dissfya/wta-tennis-2007-2023-daily-update'
        # ─────────────────────────────────────────────────────────────────

        # 2) Create a local 'data/' folder if it doesn't exist
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        # 3) Download (and unzip) using Kaggle or KaggleHub
        if KAGGLE_AVAILABLE:
            self.stdout.write('Downloading ATP dataset via Kaggle CLI...')
            kaggle.api.dataset_download_files(atp_dataset, path=data_dir, unzip=True)
            self.stdout.write('ATP download complete.')

            self.stdout.write('Downloading WTA dataset via Kaggle CLI...')
            kaggle.api.dataset_download_files(wta_dataset, path=data_dir, unzip=True)
            self.stdout.write('WTA download complete.')

        elif KAGGLEHUB_AVAILABLE:
            self.stdout.write('Downloading ATP dataset via kagglehub...')
            kagglehub.dataset_download(atp_dataset, output_path=data_dir)
            self.stdout.write('ATP download complete.')

            self.stdout.write('Downloading WTA dataset via kagglehub...')
            kagglehub.dataset_download(wta_dataset, output_path=data_dir)
            self.stdout.write('WTA download complete.')

        else:
            self.stdout.write(self.style.ERROR(
                'ERROR: Neither kaggle nor kagglehub is installed. Please install one of them.'
            ))
            return

        # 4) Locate the unzipped CSV files. Adjust names if your dataset uses different filenames:
        atp_csv = os.path.join(data_dir, 'atp_tennis.csv')
        wta_csv = os.path.join(data_dir, 'wta.csv')

        if not os.path.exists(atp_csv):
            self.stdout.write(self.style.ERROR(
                f'Could not find expected file: {atp_csv}\n'
                'Make sure your Kaggle dataset actually produced "atp_tennis.csv" in the data folder.'
            ))
            return

        if not os.path.exists(wta_csv):
            self.stdout.write(self.style.ERROR(
                f'Could not find expected file: {wta_csv}\n'
                'Make sure your Kaggle dataset actually produced "wta.csv" in the data folder.'
            ))
            return

        # ─────────────────────────────────────────────────────────────────
        # 5) Read & preprocess each CSV. We know the “Date” column exists (not “tourney_date”).
        #    The headers—exactly as in your screenshots—are:
        #      Tournament, Date, Series (ATP only), Court, Surface, Round,
        #      Best of, Player_1, Player_2, Winner, Rank_1, Rank_2,
        #      Pts_1, Pts_2, Odd_1, Odd_2, Score
        #
        #    We will:
        #      • parse the “Date” column
        #      • select only those exact columns
        #      • drop any rows with missing values
        #      • save as “atp_processed.csv” and “wta_processed.csv” under data/
        # ─────────────────────────────────────────────────────────────────

        self.stdout.write('Loading ATP CSV and preprocessing...')
        atp_df = pd.read_csv(
            atp_csv,
            parse_dates=['Date']  # only parse the “Date” column
        )

        # Confirm the exact column names
        # print(atp_df.columns)  # Uncomment if you want to double-check

        # Select only the columns you showed in your screenshot:
        #   “Tournament”, “Date”, “Series”, “Court”, “Surface”, “Round”,
        #   “Best of”, “Player_1”, “Player_2”, “Winner”,
        #   “Rank_1”, “Rank_2”, “Pts_1”, “Pts_2”, “Odd_1”, “Odd_2”, “Score”
        atp_columns = [
            'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round',
            'Best of', 'Player_1', 'Player_2', 'Winner',
            'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Score'
        ]

        # If your CSV has slightly different names (e.g. 'Best of ' vs 'Best of'),
        # adjust this list accordingly. Otherwise, this will raise a KeyError.
        try:
            atp_df = atp_df[atp_columns].dropna()
        except KeyError as e:
            missing = list(set(atp_columns) - set(atp_df.columns))
            self.stdout.write(self.style.ERROR(
                'ATP preprocessing failed. Check for missing columns:\n'
                f'  Missing: {missing}'
            ))
            return

        # Save the cleaned ATP data
        atp_outpath = os.path.join(data_dir, 'atp_processed.csv')
        atp_df.to_csv(atp_outpath, index=False)
        self.stdout.write(f'ATP preprocessing complete. Saved to {atp_outpath}')

        # ─────────────────────────────────────────────────────────────────

        self.stdout.write('Loading WTA CSV and preprocessing...')
        wta_df = pd.read_csv(
            wta_csv,
            parse_dates=['Date']  # WTA also uses “Date”
        )

        wta_columns = [
            'Tournament', 'Date', 'Court', 'Surface', 'Round',
            'Best of', 'Player_1', 'Player_2', 'Winner',
            'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Score'
        ]

        try:
            wta_df = wta_df[wta_columns].dropna()
        except KeyError as e:
            missing = list(set(wta_columns) - set(wta_df.columns))
            self.stdout.write(self.style.ERROR(
                'WTA preprocessing failed. Check for missing columns:\n'
                f'  Missing: {missing}'
            ))
            return

        # Save the cleaned WTA data
        wta_outpath = os.path.join(data_dir, 'wta_processed.csv')
        wta_df.to_csv(wta_outpath, index=False)
        self.stdout.write(f'WTA preprocessing complete. Saved to {wta_outpath}')

        self.stdout.write(self.style.SUCCESS('Data download & preprocessing finished.'))
