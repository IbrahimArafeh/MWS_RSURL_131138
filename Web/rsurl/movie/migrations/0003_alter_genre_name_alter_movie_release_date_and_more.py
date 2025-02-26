# Generated by Django 5.0.3 on 2024-06-25 15:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie', '0002_create_genres'),
    ]

    operations = [
        migrations.AlterField(
            model_name='genre',
            name='name',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='movie',
            name='release_date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='movie',
            name='title',
            field=models.CharField(max_length=255),
        ),
    ]
