from django.urls import path
from . import views

urlpatterns = [
    path('', views.movie_list, name='movie_list'),
    path('movie/<int:pk>/', views.movie_detail, name='movie_detail'),
    path('recommendations/', views.recommendations_view, name='recommendations_view'),
    path('explain/<int:movie_id>/', views.explain_view, name='explain_view'),  # Add this line
]
