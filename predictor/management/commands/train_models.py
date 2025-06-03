# predictor/management/commands/train_models.py

from django.core.management.base import BaseCommand
from predictor.ml_utils import train_all_models

class Command(BaseCommand):
    help = 'Train both ATP and WTA match‐winner XGBoost models'

    def handle(self, *args, **options):
        self.stdout.write("Starting to train all models...")
        train_all_models()
        self.stdout.write(self.style.SUCCESS("Finished training both ATP and WTA models."))
