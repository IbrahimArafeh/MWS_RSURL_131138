import os
import pandas as pd
import requests
import random  # Import random module
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from .models import Movie, Genre
from accounts.models import UserProfile  # Import UserProfile model
from django.contrib.auth.decorators import login_required

# Define absolute paths for CSV files
ratings_file_path = os.path.join(settings.BASE_DIR, "ratings.csv")
movies_type_path = os.path.join(settings.BASE_DIR, "movies_type.csv")

# Load data
df = pd.read_csv(ratings_file_path)

def filter_movies_by_user_and_rating(movies_type, data_path, user_id, min_rating):
    df = pd.read_csv(data_path)
    df1 = pd.read_csv(movies_type)
    filtered_df = df[(df['UserID'] == user_id) & (df['Rating'] > min_rating)]
    combined_df = filtered_df.merge(df1, on='MovieID', how='left')
    return combined_df

def find_movie_type(movie_id, movies_type):
    type_df = pd.read_csv(movies_type)
    movie_row = type_df[type_df['MovieID'] == movie_id]
    if not movie_row.empty:
        return movie_row['type'].values[0]
    else:
        return None

def count_differences(str1, str2):
    min_length = min(len(str1), len(str2))
    differences = sum(1 for i in range(min_length) if str1[i] != str2[i])
    differences += abs(len(str1) - len(str2))
    return differences

def find_most_similar_3_movie_type(movie_id, movies_type, filtered_df):
    target_type = find_movie_type(movie_id, movies_type)
    if target_type is None:
        return None

    differences_list = []
    for _, row in filtered_df.iterrows():
        current_type = row['type']
        current_movie_id = row['MovieID']
        difference = count_differences(target_type, current_type)
        differences_list.append((current_movie_id, current_type, difference))
    
    differences_list.sort(key=lambda x: x[2])
    most_similar_movies = differences_list[:3]
    return most_similar_movies

def explain(user_id, movie_id):
    min_rating = 3
    filtered_df = filter_movies_by_user_and_rating(movies_type_path, ratings_file_path, user_id, min_rating)
    most_similar_movies = find_most_similar_3_movie_type(movie_id, movies_type_path, filtered_df)
    if most_similar_movies is None:
        return f"No similar movies found for movie ID: {movie_id}"
    else:
        return [movie[0] for movie in most_similar_movies]

def movie_list(request):
    movies = Movie.objects.all()
    genres = Genre.objects.all()
    
    # Add image index to each movie
    for index, movie in enumerate(movies):
        movie.image_index = (index % 18) + 1  # Cycles through image numbers 1 to 18
    
    return render(request, 'movie/movie_list.html', {'movies': movies, 'genres': genres})

def movie_detail(request, pk):
    movie = get_object_or_404(Movie, pk=pk)
    return render(request, 'movie/movie_detail.html', {'movie': movie})

@login_required
def recommendations_view(request):
    user = request.user
    user_profile = get_object_or_404(UserProfile, user=user)
    external_user_id = user_profile.external_user_id  # Get External user ID

    print(f"Fetching recommendations for external_user_id: {external_user_id}")  # Debugging line

    url = "http://127.0.0.1:7000/recommendations"  # FastAPI URL
    payload = {"user_id": external_user_id}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"Received data: {data}")  # Debugging line

        recommended_movie_ids = data.get("recommendations", [])
        print(f"Recommended movie IDs: {recommended_movie_ids}")  # Debugging line
        
        # Fetch movie details from the database
        recommended_movies = Movie.objects.filter(movie_id__in=recommended_movie_ids)
        
        # Add image index to each movie
        for index, movie in enumerate(recommended_movies):
            movie.image_index = (index % 18) + 1  # Cycles through image numbers 1 to 18
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recommendations: {e}")
        recommended_movies = Movie.objects.none()

    genres = Genre.objects.all()
    
    return render(request, 'movie/recommendations.html', {'movies': recommended_movies, 'genres': genres})

@login_required
def explain_view(request, movie_id):
    user = request.user
    user_profile = get_object_or_404(UserProfile, user=user)
    external_user_id = user_profile.external_user_id  # Get External user ID

    similar_movie_ids = explain(external_user_id, movie_id)
    
    similar_movies = Movie.objects.filter(movie_id__in=similar_movie_ids)
    
    # Add random image index to each movie
    for movie in similar_movies:
        movie.image_index = random.randint(1, 18)
    
    genres = Genre.objects.all()
    
    return render(request, 'movie/explain.html', {'movie_id': movie_id, 'similar_movies': similar_movies, 'genres': genres})
