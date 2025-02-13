# accounts/admin.py
from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('external_user_id', 'user', 'age', 'gender', 'salary', 'latitude', 'longitude')
    search_fields = ('user__username', 'user__email', 'external_user_id', 'age', 'gender', 'salary', 'latitude', 'longitude')
