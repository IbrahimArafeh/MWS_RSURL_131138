# accounts/management/commands/clear_users.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import UserProfile

class Command(BaseCommand):
    help = 'Clear all users and user profiles from the database'

    def handle(self, *args, **kwargs):
        UserProfile.objects.all().delete()
        User.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully deleted all users and user profiles.'))
