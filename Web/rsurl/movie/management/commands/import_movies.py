import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from movie.models import Movie, Genre
import os

class Command(BaseCommand):
    help = 'Import movies from u.item file'

    def handle(self, *args, **options):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'u.item')

        with open(file_path, 'r', encoding='latin-1') as file:
            reader = csv.reader(file, delimiter='|')
            genres = list(Genre.objects.all())
            for row in reader:
                movie_id, title, release_date_str, video_release_date, imdb_url = row[:5]
                genre_flags = row[5:]

                if release_date_str:
                    release_date = datetime.strptime(release_date_str, '%d-%b-%Y').date()
                else:
                    release_date = None

                # Check if the movie already exists
                movie, created = Movie.objects.update_or_create(
                    movie_id=movie_id,
                    defaults={
                        'title': title,
                        'release_date': release_date,
                        'imdb_url': imdb_url,
                    }
                )

                # Update genres only if the movie is newly created
                if created:
                    for i, flag in enumerate(genre_flags):
                        if flag == '1':
                            movie.genres.add(genres[i])

                self.stdout.write(self.style.SUCCESS(f'Imported movie: {title}'))
