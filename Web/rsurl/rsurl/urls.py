# rsurl/urls.py
from django.urls import path
from . import views
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('movies/', include('movie.urls')),
    path('accounts/', include('accounts.urls')),
    path('recommendations/<int:user_id>/', views.recommendations_view, name='recommendations'),
]
