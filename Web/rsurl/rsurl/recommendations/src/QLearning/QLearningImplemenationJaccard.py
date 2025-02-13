import pandas as pd
import numpy as np
from FitneessFunction import bicluster_fitness_value
import random
import pickle
import matplotlib.pyplot as plt


data_path = "..\\..\\Data\\ml-100k"
biclustering_results_path = "..\\..\\Data\\biclustering results"
excel_file_path = f"{data_path}\\ratings.csv"
clusteres_rows = f"{biclustering_results_path}\\clusteres_rows.txt"
clusteres_columns = f"{biclustering_results_path}\\clusters_columns.txt"
top_100_users_indices_path = f"{biclustering_results_path}\\top_100_users_indices.txt"
top_100_movies_indices_path = f"{biclustering_results_path}\\top_100_movies_indices.txt"
qlearning_results_path = "..\\..\\Data\\Qlearning results"
state_space_path = f"{qlearning_results_path}\\state_space.pkl"
qtable_path = f"{qlearning_results_path}\\jaccard_qtable_path.pkl"


def select_values_by_indices(lst, indices):
    selected_values = [lst[i] for i in indices]
    return selected_values

def get_grid_biclusters():
    df = pd.read_csv(excel_file_path)
    pivot_matrix = df.pivot(index='UserID', columns='MovieID', values='Rating')
    ratings_matrix = ((pivot_matrix.to_numpy() >= 3).astype(int))
    pivot_matrix = pivot_matrix.to_numpy().astype(int)
    ratings_per_user = (ratings_matrix != 0).sum(axis=1)
    users_rated_50_or_above_indices = np.where(ratings_per_user >= 50)[0]
    mask = np.isin(np.arange(ratings_matrix.shape[0]), users_rated_50_or_above_indices)
    original_users_indices = []
    for e, element in enumerate(mask):
        if element:
            original_users_indices.append(e)
    filtered_ratings_matrix = ratings_matrix[mask] #Binary matrix
    filtered_pivot_matrix = pivot_matrix[mask] #Original matrix
    total_ratings_per_user = filtered_ratings_matrix.sum(axis=1)
    sorted_users_indices = np.argsort(total_ratings_per_user)[::-1]
    top_100_users_indices = sorted_users_indices[:100]
    top_100_users = filtered_ratings_matrix[top_100_users_indices, :] #Binary Matrix
    top_100_pivot_users = filtered_pivot_matrix[top_100_users_indices, :] #Original matrix
    indices = []
    for i in range(top_100_users.shape[1]):
      if(top_100_users[:, i].sum(axis=0) >= 50):
          if len(indices) == 100:
              break
          indices.append(i)
    final_pivot_matrix = top_100_pivot_users[:, indices] #Original matrix
    with open(clusteres_rows, 'r') as file:
        lines = file.readlines()
    rows = []
    for line in lines:
        values = line.strip().split()  # Split by whitespace and remove trailing newline
        bool_values = [value == 'True' for value in values]  # Convert 'True'/'False' strings to boolean
        rows.append(bool_values)
    rows = np.array(rows) 
    with open(clusteres_columns, 'r') as file:
        lines = file.readlines()
    columns = []
    for line in lines:
        values = line.strip().split()  # Split by whitespace and remove trailing newline
        bool_values = [value == 'True' for value in values]  # Convert 'True'/'False' strings to boolean
        columns.append(bool_values)
    columns = np.array(columns)
    fitness_values = bicluster_fitness_value(rows, columns, final_pivot_matrix)
    indexed_list = list(enumerate(fitness_values))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    sorted_values = [item[1] for item in sorted_indexed_list]
    sorted_indices = [item[0] for item in sorted_indexed_list]
    final_rows = rows[sorted_indices[:64]]
    final_columns = columns[sorted_indices[:64]]
    return select_values_by_indices(original_users_indices, top_100_users_indices), indices, final_rows, final_columns

top_100_users_indices, top_100_movies_indices, final_rows, final_columns = get_grid_biclusters()

with open(top_100_users_indices_path, 'w') as file:
    for item in top_100_users_indices:
        file.write(str(item) + '\n')

with open(top_100_movies_indices_path, 'w') as file:
    for item in top_100_movies_indices:
        file.write(str(item) + '\n')
############################################################################

class state:
    def __init__(self):
        self.users = [] # Users Ids
        self.movies = [] # Movies Ids
        
def generate_zigzag_indices(size):
    indices = []
    row, col = 0, 0
    indices.append((row, col))
    direction = "up"
    steps = 1
    half_grid_passed = False
    while(steps != 0):
        if direction == "up":
            if half_grid_passed:
                row += 1
            else:
                col += 1
            indices.append((row, col))
            step = 1
            while (step <= steps):
                row += 1
                col -= 1
                indices.append((row, col))
                step += 1
            direction = "down"
        elif direction == "down":
            if half_grid_passed:
                col += 1
            else:
                row += 1
            indices.append((row, col))
            step = 1
            while (step <= steps):
                row -= 1
                col += 1
                indices.append((row, col))
                step += 1
            direction = "up"    
        if steps == size - 1:
            half_grid_passed = True
        if half_grid_passed:
            steps -= 1
        else:
            steps += 1
    indices.append(((size -1), (size - 1)))
    return indices

def define_state_space(top_100_users_indices, top_100_movies_indices, final_rows, final_columns):
    state_space = [[state() for _ in range(8)] for _ in range(8)]
    grid_size = 8
    zigzag_indices = generate_zigzag_indices(grid_size)
    for i, idx in enumerate(zigzag_indices):
        usersIds = []
        moviesIds = []
        for r_i, r in enumerate(final_rows[i]):
            if r:
                usersIds.append(top_100_users_indices[r_i])

        for c_i, c in enumerate(final_columns[i]):
            if c:
                moviesIds.append(top_100_movies_indices[c_i])  
        state_space[idx[0]][idx[1]].movies = moviesIds
        state_space[idx[0]][idx[1]].users = usersIds
    return state_space    
        
state_space = define_state_space(top_100_users_indices, top_100_movies_indices, final_rows, final_columns)

# Save state_space to a file
with open(state_space_path, 'wb') as f:
    pickle.dump(state_space, f)
#########################################################################################

def Jaccard_index(s1, s2):
    intersection = set(s1).intersection((set(s2)))
    print(s1)
    print(s1)
    print(s2)
    return len(intersection)/len(set(s1).union(set(s2)))

def epsilon_greedy(epsilon = 0.3, first_option = "random", second_option = "reward"):
    rand_num = random.uniform(0, 1)
    if rand_num < epsilon:
        return first_option
    else:
        return second_option

def evaluate(state_space, start_state,start_state_row_index, start_state_col_index, QTable, grid_size):
    recommended_movies = []
    current_row = start_state_row_index
    current_col = start_state_col_index
    current_state = start_state
    cumulative_rewards = 0
    while(1):
        state_movies = current_state.movies
        new_movies_found = False
        for movie_id in state_movies:
            if movie_id not in recommended_movies:
                recommended_movies.append(movie_id)
                new_movies_found = True
        if not new_movies_found:
            break
        next_step_index = np.argmax(np.array(QTable[current_row][current_col]))
        next_step_value = np.max(np.array(QTable[current_row][current_col]))
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
        cumulative_rewards += next_step_value
        current_state= state_space[current_row][current_col]
    return cumulative_rewards


def smooth_rewards(episodes_rewards, window_size=10):
    # Create a new array to hold smoothed values
    smoothed_rewards = np.zeros_like(episodes_rewards)
    
    # Apply moving average smoothing
    for i in range(len(episodes_rewards)):
        # Determine window boundaries
        start = max(0, i - window_size + 1)
        end = i + 1
        
        # Calculate the average within the window
        smoothed_rewards[i] = np.mean(episodes_rewards[start:end])
    
    return smoothed_rewards

def QLearning(state_space, trials = 150, max_steps_per_episode = 10, learning_rate = 0.05, discount_factor = 1, grid_size = 8):
    QTable = [[[0] * 4 for _ in range(grid_size)] for _ in range(grid_size)]
    for i,_ in enumerate(QTable):
        for j,_ in enumerate(QTable[i]):
            for k,_ in enumerate((QTable[i][j])):
                QTable[i][j][k]= random.uniform(-1, 1)
    episodes_lengths = []
    episodes_rewards = []
    for trial in range(trials):
        random_row = random.randint(0, grid_size - 1)
        random_col = random.randint(0, grid_size - 1)
        current_state = (random_row, random_col)
        game_over = False
        step = 1
        tracked_movies = []
        off_the_grid = False
        new_movies_found = True
        final_reward = 0
        while (step <= max_steps_per_episode):
            rewards = np.zeros(5)
            max_reward = -1
            max_reward_state = 1 # 1:top, 2:right, 3:bottom, 4: left
            if (current_state[0] - 1 >= 0):
                reward_left = Jaccard_index(state_space[current_state[0]][current_state[1]].users, state_space[current_state[0] - 1][current_state[1]].users)
                rewards[4] = reward_left
                if reward_left > max_reward:
                    max_reward = reward_left
                    max_reward_state = 4
            if (current_state[0] + 1 <= grid_size - 1):
                reward_right = Jaccard_index(state_space[current_state[0]][current_state[1]].users, state_space[current_state[0] + 1][current_state[1]].users)
                rewards[2] = reward_right
                if reward_right > max_reward:
                    max_reward = reward_right
                    max_reward_state = 2
            if (current_state[1] + 1 <= grid_size - 1):
                reward_bottom = Jaccard_index(state_space[current_state[0]][current_state[1]].users, state_space[current_state[0]][current_state[1] + 1].users)
                rewards[3] = reward_bottom
                if reward_bottom > max_reward:
                    max_reward = reward_bottom
                    max_reward_state = 3
            if (current_state[1] - 1 >= 0):
                reward_top = Jaccard_index(state_space[current_state[0]][current_state[1]].users, state_space[current_state[0]][current_state[1] - 1].users)
                rewards[1] = reward_top
                if reward_top > max_reward:
                    max_reward = reward_top
                    max_reward_state = 1        
            next_state_status = epsilon_greedy(0.3)
            if (next_state_status == "reward"):
                action = max_reward_state
            elif (next_state_status == "random"):
                possible_actions = [act for act in range(1,5) if act != max_reward_state]
                action = random.choice(possible_actions)
            if action == 1:
                next_state_row = current_state[0]
                next_state_col = current_state[1] - 1
            if action == 2:
                next_state_row = current_state[0] + 1
                next_state_col = current_state[1]
            if action == 3:
                next_state_row = current_state[0]
                next_state_col = current_state[1] + 1
            if action == 4:
                next_state_row = current_state[0] - 1
                next_state_col = current_state[1]
            reward = rewards[action]
            if ((next_state_row < 0 or next_state_row >= grid_size) or (next_state_col < 0 or next_state_col >= grid_size)):
                off_the_grid = True
                game_over = True
                break
            optimal_future_value = max(QTable[next_state_row][next_state_col][0], QTable[next_state_row][next_state_col][1], QTable[next_state_row][next_state_col][2], QTable[next_state_row][next_state_col][3])
            QTable[current_state[0]][current_state[1]][action-1] += learning_rate * (reward + discount_factor * optimal_future_value - QTable[current_state[0]][current_state[1]][action-1]) 
            new_movies = state_space[current_state[0]][current_state[1]].movies
            new_movies_found = False
            for n_movie in new_movies:
                if n_movie not in tracked_movies:
                    new_movies_found = True
                    tracked_movies.append(n_movie)
            if new_movies_found == False:
                game_over = True
                break
            else:
                current_state = (next_state_row, next_state_col)
                step += 1
        for i, row in enumerate(state_space):
            for j, col in enumerate(row):
                final_reward += evaluate(state_space, state_space[i][j], i, j, QTable, grid_size)
        episodes_lengths.append(step)
        episodes_rewards.append(final_reward)
    return QTable, episodes_lengths, episodes_rewards

QTable, episodes_lengths, episodes_rewards = QLearning(state_space)
smoothed_rewards = smooth_rewards(episodes_rewards, window_size=10)

with open(qtable_path, 'wb') as f:
    pickle.dump(QTable, f) 

plt.figure(figsize=(10, 6))  # Adjust the figure size for better readability

# Plot the original rewards
plt.plot(episodes_rewards, label='Original Rewards')

# Plot the smoothed rewards with a different line style and color for better distinction
plt.plot(smoothed_rewards, label='Smoothed Rewards', linestyle='--', color='orange')

# Set labels and title for the plot
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode during Q-Learning in Jaccard')

# Add legend to differentiate between the lines
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate average rewards
average_rewards = np.mean(episodes_rewards, axis=0)



# Flatten the Q-table for easier visualization
all_q_values = np.array([q for row in QTable for q in row]).flatten()

# Plot the distribution of Q-values (consider using plt.hist or sns.kdeplot)
plt.figure(figsize=(8, 5))
plt.hist(all_q_values)  # Replace with sns.kdeplot for a smoother density plot
plt.xlabel('Q-Value')
plt.ylabel('Frequency')
plt.title('Distribution of Q-Values in the Q-Table  in Jaccard')
plt.grid(True)
plt.tight_layout()
plt.show()

