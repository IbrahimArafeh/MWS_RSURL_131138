from django.db import migrations

def create_genres(apps, schema_editor):
    Genre = apps.get_model('movie', 'Genre')
    genres = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's",
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    for genre in genres:
        Genre.objects.create(name=genre)

class Migration(migrations.Migration):

    dependencies = [
        ('movie', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(create_genres),
    ]
