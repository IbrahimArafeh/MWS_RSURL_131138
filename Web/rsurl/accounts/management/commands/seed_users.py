import csv
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from faker import Faker
from accounts.models import UserProfile

class Command(BaseCommand):
    help = 'Seed the database with users from a CSV file'

    def handle(self, *args, **kwargs):
        faker = Faker()

        try:
            with open('users.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Generate realistic username and email using Faker
                    username = faker.unique.user_name()
                    email = faker.unique.email()

                    # Create the user
                    user = User.objects.create_user(
                        username=username,
                        email=email,
                        password='defaultpassword'  # set a default password
                    )

                    # Create the user profile
                    UserProfile.objects.create(
                        user=user,
                        external_user_id=int(row['UserID']),  # Use UserID from the CSV
                        age=int(row['Age']),
                        gender=row['Gender'],
                        salary=float(row['Salary']),
                        latitude=float(row['Lattitude']),
                        longitude=float(row['Longtitude'])
                    )

            self.stdout.write(self.style.SUCCESS('Successfully seeded the database with users.'))
        
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR('CSV file not found. Ensure the file is in the correct directory.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'An error occurred: {e}'))
