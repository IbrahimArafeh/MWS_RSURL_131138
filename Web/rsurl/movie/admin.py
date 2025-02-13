# movie/admin.py
from django.contrib import admin
from .models import Movie, Genre

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'release_date', 'imdb_url')
    search_fields = ('title',)

@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)
