from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

# Define the `state` class as it was defined when the `state_space` object was pickled
class state:
    def __init__(self):
        self.users = []  # Users IDs
        self.movies = []  # Movies IDs

def get_states_movies_average_ratings(state_space, ratings_matrix):
    states_movies_average_ratings = {}
    for i, row in enumerate(state_space):
        for j, col in enumerate(row):
            state_movies_average_ratings = []
            for movieId in col.movies:
                movie_average_rating = sum(ratings_matrix.loc[userId + 1, movieId + 1] for userId in col.users)
                movie_average_rating /= len(col.users)
                state_movies_average_ratings[(i, j)] = state_movies_average_ratings
    return states_movies_average_ratings

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def get_start_state(state_space, states_movies_average_ratings, userId, ratings_matrix, users, cold_start=False):
    max_similarity = -1
    start_state = None
    start_state_row_index = None
    start_state_col_index = None

    user = users.iloc[userId - 1]
    
    for i, row in enumerate(state_space):
        for j, col in enumerate(row):
            if cold_start:
                similarities = sum(cosine_similarity(user, users.iloc[user_id]) for user_id in col.users)
                avg_similarity = similarities / len(col.users)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    start_state = col
                    start_state_row_index = i
                    start_state_col_index = j
            else:
                user_movies_ratings = [ratings_matrix.loc[userId, movieId + 1] for movieId in col.movies]
                similarity = cosine_similarity(np.array(states_movies_average_ratings[(i, j)]), np.array(user_movies_ratings))
                if similarity > max_similarity:
                    max_similarity = similarity
                    start_state = col
                    start_state_row_index = i
                    start_state_col_index = j

    return start_state, start_state_row_index, start_state_col_index

def generate_recommendations(state_space, start_state, start_state_row_index, start_state_col_index, QTable, grid_size):
    recommended_movies = []
    current_row = start_state_row_index
    current_col = start_state_col_index
    current_state = start_state
    
    while True:
        state_movies = current_state.movies
        new_movies_found = False
        for movie_id in state_movies:
            if (movie_id + 1) not in recommended_movies:
                recommended_movies.append(movie_id + 1)
                new_movies_found = True
        if not new_movies_found:
            break
        
        next_step_index = np.argmax(QTable[current_row][current_col])
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
        
        current_state = state_space[current_row][current_col]
    
    return recommended_movies

ratings_file_path = "ratings.csv"
state_space_path = "state_space.pkl"
qtable_path = "qtable_path.pkl"
users_info_path = "users.csv"
movies_type_path = "movies_type.csv"

# Load data
users = pd.read_csv(users_info_path)
df = pd.read_csv(ratings_file_path)
ratings_matrix = df.pivot(index='UserID', columns='MovieID', values='Rating')

with open(state_space_path, 'rb') as f:
    state_space = pickle.load(f)

with open(qtable_path, 'rb') as f:
    QTable = pickle.load(f)

grid_size = 8

def get_recommendations(userId):
    states_movies_average_ratings = get_states_movies_average_ratings(state_space, ratings_matrix)
    start_state, start_state_row_index, start_state_col_index = get_start_state(state_space, states_movies_average_ratings, userId, ratings_matrix, users, cold_start=True)
    return generate_recommendations(state_space, start_state, start_state_row_index, start_state_col_index, QTable, grid_size)

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

class CombinedRequest(BaseModel):
    user_id: int
    movie_id: int

@app.post("/combined_recommendations")
def combined_recommendations(request: CombinedRequest):
    user_id = request.user_id
    movie_id = request.movie_id

    if user_id < 1 or user_id > len(users):
        raise HTTPException(status_code=400, detail="Invalid user ID")

    try:
        recommendations = get_recommendations(user_id)
        
        filtered_df = filter_movies_by_user_and_rating(movies_type_path, ratings_file_path, user_id, min_rating=3)
        most_similar_movies = find_most_similar_3_movie_type(movie_id, movies_type_path, filtered_df)

        if most_similar_movies is None:
            raise HTTPException(status_code=404, detail="Movie ID not found or no similar movies found")

        similar_movies_ids = [movie[0] for movie in most_similar_movies]

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "movie_id": movie_id,
        "similar_movies": similar_movies_ids
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastApiApp:app", host="0.0.0.0", port=7000, reload=True)
