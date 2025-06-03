# predictor/models.py

from django.db import models

class PredictionRequest(models.Model):
    TOUR_CATEGORIES = [
        ('ATP', 'ATP'),
        ('WTA', 'WTA'),
    ]

    player_1         = models.CharField(max_length=100)
    player_2         = models.CharField(max_length=100)
    player_1_rank    = models.IntegerField()
    player_2_rank    = models.IntegerField()
    surface          = models.CharField(max_length=20)
    tourney_date     = models.DateField()
    tour_category    = models.CharField(max_length=3, choices=TOUR_CATEGORIES)
    predicted_winner = models.CharField(max_length=100)
    request_time     = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.tour_category}] {self.player_1} vs {self.player_2} on {self.tourney_date}"
