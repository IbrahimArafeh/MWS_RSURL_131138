import GenerateRecommendations
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
qtable_path = f"{qlearning_results_path}\\dice_coefficient_qtable_path.pkl"
users_info_path = f"{data_path}\\users.csv"
excel_file_path = f"{data_path}\\ratings.csv"
qlearning_results_path = "..\\..\\Data\\Qlearning results"
state_space_path = f"{qlearning_results_path}\\state_space.pkl"
qtable_path = f"{qlearning_results_path}\\dice_coefficient_qtable_path.pkl"
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

all_users_ids = ratings_matrix.index.tolist()
users_train_set = []
with open(top_100_users_indices_path, 'r') as file:
    for line in file:
        users_train_set.append(int(line.strip()) + 1)     
movies_train_set = []
with open(top_100_movies_indices_path, 'r') as file:
    for line in file:
        movies_train_set.append(int(line.strip()) + 1)   

random_user_ids = random.sample([user_id for user_id in all_users_ids if user_id not in users_train_set], 100)

TP_list = []
TN_list = []
FP_list = []
FN_list = []
Precision_list = []
Recall_list = []
F1_score_list = []
Accuracy_list = []

for _id in random_user_ids:
    states_movies_average_ratings = GenerateRecommendations.get_states_movies_average_ratings(state_space, ratings_matrix)
    start_state, start_state_row_index, start_state_col_index = GenerateRecommendations.get_start_state(state_space, states_movies_average_ratings, _id, ratings_matrix, users, cold_start=True)
    recommended_movies = GenerateRecommendations.generate_recommendations(state_space, start_state, start_state_row_index, start_state_col_index, Qtable, grid_size)
    favourite_movies = get_favourite_movies_for_user(_id, ratings_matrix)
    actual_movies = [m_id for m_id in favourite_movies if m_id in movies_train_set]
    # Calculate TP, TN, FP, FN manually
    TP = sum((m in recommended_movies) and (m in actual_movies) for m in recommended_movies)
    TN = sum((m not in recommended_movies) and (m not in actual_movies) for m in movies_train_set)
    FP = sum((m in recommended_movies) and (m not in actual_movies) for m in recommended_movies)
    FN = sum((m not in recommended_movies) and (m in actual_movies) for m in actual_movies)
    
    # Calculate Precision, Recall, F1-score, and Accuracy manually
    Precision = TP / (TP + FP) if TP + FP != 0 else 0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Append metrics to respective lists
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    Precision_list.append(Precision)
    Recall_list.append(Recall)
    F1_score_list.append(F1_score)
    Accuracy_list.append(Accuracy)

avg_TP = sum(TP_list) / len(TP_list)
avg_FP = sum(FP_list) / len(FP_list)
avg_TN = sum(TN_list) / len(TN_list)
avg_FN = sum(FN_list) / len(FN_list)
avg_prec = sum(Precision_list) / len(Precision_list)
avg_rec = sum(Recall_list) / len(Recall_list)
avg_f1 = sum(F1_score_list) / len(F1_score_list)
avg_acc = sum(Accuracy_list) / len(Accuracy_list)
