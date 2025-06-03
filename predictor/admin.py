from django.contrib import admin
from .models import PredictionRequest

@admin.register(PredictionRequest)
class PredictionRequestAdmin(admin.ModelAdmin):
    list_display = (
        'player_1', 'player_2', 'tourney_date',
        'surface', 'tour_category', 'predicted_winner', 'request_time'
    )
    list_filter = ('tour_category', 'surface')
    ordering = ('-request_time',)
