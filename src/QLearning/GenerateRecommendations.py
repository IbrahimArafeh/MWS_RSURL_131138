import pickle
import pandas as pd
import numpy as np

data_path = "..\\..\\Data\\ml-100k"
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


def get_states_movies_average_ratings(state_space, ratings_matrix):
    states_movies_average_ratings = {}
    for i, row in enumerate(state_space):
        for j, col in enumerate(row):
            state_movies_average_ratings = []
            for movieId in col.movies:
                movie_average_rating = 0
                for userId in col.users:
                    movie_average_rating += ratings_matrix.loc[userId + 1, movieId + 1]
                movie_average_rating/= len(col.users)
                state_movies_average_ratings.append(movie_average_rating)
            states_movies_average_ratings[(i, j)] = state_movies_average_ratings
    return states_movies_average_ratings

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_start_state(state_space, states_movies_average_ratings, userId, ratings_matrix, users, cold_start = False):
    if cold_start:
        max_similarity = -1
        user = users.iloc[userId - 1]
        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                similarities = 0
                for user_id in col.users:
                    similarities += cosine_similarity(user, users.iloc[user_id])
                avg_similarity = similarities/len(col.users)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    start_state = col
                    start_state_row_inedx= i
                    start_state_col_inedx= j
    else:
        max_similarity = -1
        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                user_movies_ratings = []
                for movieId in col.movies:
                    user_movies_ratings.append(ratings_matrix.loc[userId, movieId + 1])
                similarity = cosine_similarity(np.array(states_movies_average_ratings[(i, j)]), np.array(user_movies_ratings))
                if similarity > max_similarity:
                    max_similarity = similarity
                    start_state = col
                    start_state_row_inedx= i
                    start_state_col_inedx= j
    return start_state, start_state_row_inedx, start_state_col_inedx

def generate_recommendations(state_space, start_state,start_state_row_index, start_state_col_index, QTable, grid_size):
    recommended_movies = []
    current_row = start_state_row_index
    current_col = start_state_col_index
    current_state = start_state
    while(1):
        state_movies = current_state.movies
        new_movies_found = False
        for movie_id in state_movies:
            if (movie_id + 1) not in recommended_movies:
                recommended_movies.append(movie_id + 1)
                new_movies_found = True
        if not new_movies_found:
            break
        next_step_index = np.argmax(np.array(QTable[current_row][current_col]))
        # 1:top, 2:right, 3:bottom, 4: left
        if next_step_index == 0:
            current_row -= 1
        elif next_step_index == 1:
            current_col += 1
        elif next_step_index == 2:
            current_row += 1
        else:
            current_col -= 1
        if (current_row < 0 or current_col < 0 or current_row >= grid_size or current_col >= grid_size):
            break
        current_state= state_space[current_row][current_col]
    return recommended_movies      

userId = 89
states_movies_average_ratings = get_states_movies_average_ratings(state_space, ratings_matrix)     
start_state, start_state_row_index, start_state_col_index = get_start_state(state_space, states_movies_average_ratings, userId, ratings_matrix, users, cold_start= True)   
recommended_movies =  generate_recommendations(start_state,start_state_row_index, start_state_col_index, Qtable, grid_size)    
print(recommended_movies)
        
        
        
        