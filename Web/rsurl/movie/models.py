# movie/models.py
from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Movie(models.Model):
    movie_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    release_date = models.DateField(null=True, blank=True)
    imdb_url = models.URLField(max_length=200, null=True, blank=True)
    genres = models.ManyToManyField(Genre)

    def __str__(self):
        return self.title
