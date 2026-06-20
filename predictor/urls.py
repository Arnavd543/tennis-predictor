from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('predict/htmx/', views.predict_htmx_view, name='predict_htmx'),
]