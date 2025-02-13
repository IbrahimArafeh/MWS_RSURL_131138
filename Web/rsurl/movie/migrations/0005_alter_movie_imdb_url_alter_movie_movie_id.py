# Generated by Django 5.0.3 on 2024-07-04 15:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie', '0004_movie_movie_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie',
            name='imdb_url',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='movie',
            name='movie_id',
            field=models.IntegerField(default=0, unique=True),
            preserve_default=False,
        ),
    ]
