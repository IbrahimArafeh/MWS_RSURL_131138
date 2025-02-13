# recommendations/recommendations.py

from .GenerateRecommendations import get_states_movies_average_ratings, get_start_state, generate_recommendations
import pandas as pd
import pickle
import random


data_path = "..\\..\\Data\\ml-100k"
biclustering_results_path = "..\\..\\Data\\biclustering results"
excel_file_path = f"{data_path}\\ratings.csv"
clusteres_rows = f"{biclustering_results_path}\\clusteres_rows.txt"
clusteres_columns = f"{biclustering_results_path}\\clusters_columns.txt"
top_100_users_indices_path = f"{biclustering_results_path}\\top_100_users_indices.txt"
top_100_movies_indices_path = f"{biclustering_results_path}\\top_100_movies_indices.txt"
qlearning_results_path = "..\\..\\Data\\Qlearning results"
state_space_path = f"{qlearning_results_path}\\state_space.pkl"
qtable_path = f"{qlearning_results_path}\\qtable_path.pkl"
users_info_path = f"{data_path}\\users.csv"
excel_file_path = f"{data_path}\\ratings.csv"
qlearning_results_path = "..\\..\\Data\\Qlearning results"
state_space_path = f"{qlearning_results_path}\\state_space.pkl"
qtable_path = f"{qlearning_results_path}\\qtable_path.pkl"
users = pd.read_csv(users_info_path)
grid_size = 8

class state:
    def __init__(self):
        self.users = [] # Users Ids
        self.movies = [] # Movies Ids
        
# Load state_space from the saved file
with open(state_space_path, 'rb') as f:
    state_space = pickle.load(f)
    
# Load qltable from the saved file
with open(qtable_path, 'rb') as f:
    Qtable = pickle.load(f)
    
df = pd.read_csv(excel_file_path)
ratings_matrix = df.pivot(index='UserID', columns='MovieID', values='Rating')

def get_favourite_movies_for_user(userId, ratings_matrix):
    user_ratings = ratings_matrix.loc[userId]  # Get ratings for the specified user
    movies_with_ratings_above_3 = user_ratings[user_ratings >= 3]
    movie_ids = movies_with_ratings_above_3.index.tolist()  # Get the movie IDs
    return movie_ids

 

def get_recommendations(_id):
    states_movies_average_ratings = GenerateRecommendations.get_states_movies_average_ratings(state_space, ratings_matrix)
    start_state, start_state_row_index, start_state_col_index = GenerateRecommendations.get_start_state(state_space, states_movies_average_ratings, _id, ratings_matrix, users, cold_start=True)
    recommended_movies = GenerateRecommendations.generate_recommendations(state_space, start_state, start_state_row_index, start_state_col_index, Qtable, grid_size)
    return recommended_movies