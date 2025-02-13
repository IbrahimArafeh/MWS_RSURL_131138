# accounts/models.py
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    external_user_id = models.IntegerField(unique=True)  # Renamed from user_id
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=[('0', 'Male'), ('1', 'Female')])
    salary = models.DecimalField(max_digits=10, decimal_places=2)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)

    def __str__(self):
        return self.user.username
