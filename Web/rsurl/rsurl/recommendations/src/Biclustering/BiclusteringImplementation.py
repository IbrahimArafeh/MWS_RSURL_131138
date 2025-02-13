import pandas as pd
import numpy as np
from Bimax import BiMax
from FitneessFunction import bicluster_fitness_value
import matplotlib.pyplot as plt

### Prepare the Data
data_path = "..\\..\\Data\\ml-100k"
biclustering_results_path = "..\\..\\Data\\biclustering results"
excel_file_path = f"{data_path}\\ratings.csv"
clusteres_rows = f"{biclustering_results_path}\\clusteres_rows.txt"
clusteres_columns = f"{biclustering_results_path}\\clusters_columns.txt"
clusters_fitness_values = f"{biclustering_results_path}\\clusters_fitness_values.txt"
df = pd.read_csv(excel_file_path)
# Pivot the DataFrame to create a matrix with UserIds as rows, MovieIds as columns, and Ratings as values
pivot_matrix = df.pivot(index='UserID', columns='MovieID', values='Rating')
# Convert the matrix to binary matrix in order to make Bimax able to work on it
ratings_matrix = ((pivot_matrix.to_numpy() >= 3).astype(int))
pivot_matrix = pivot_matrix.to_numpy().astype(int)
# Calculate the number of non-zero ratings for each user (row)
ratings_per_user = (ratings_matrix != 0).sum(axis=1)
# Get the indices of users who have rated at least 50 movies
users_rated_50_or_above_indices = np.where(ratings_per_user >= 50)[0]
# Create a mask for selecting rows (users) that meet the condition
mask = np.isin(np.arange(ratings_matrix.shape[0]), users_rated_50_or_above_indices)
# Apply the mask to select the rows (users) that have rated at least 50 movies
filtered_ratings_matrix = ratings_matrix[mask] #Binary matrix
filtered_pivot_matrix = pivot_matrix[mask] #Original matrix
# Calculate the total number of ratings for each user
total_ratings_per_user = filtered_ratings_matrix.sum(axis=1)
# Sort the users based on the total number of ratings (descending order)
sorted_users_indices = np.argsort(total_ratings_per_user)[::-1]
# Get the indices of the top 100 users (based on the total number of ratings)
top_100_users_indices = sorted_users_indices[:100]
# Get the ratings matrix that include only the top 100 users (based on the total number of ratings)
top_100_users = filtered_ratings_matrix[top_100_users_indices, :] #Binary Matrix
top_100_pivot_users = filtered_pivot_matrix[top_100_users_indices, :] #Original matrix
# Get the indices of the movies that have been rated 50 time at least
indices = []
for i in range(top_100_users.shape[1]):
  if(top_100_users[:, i].sum(axis=0) >= 50):
      if len(indices) == 100:
          break
      indices.append(i)
# Get the ratings matrix that include only the movies that have been rated 50 time at least
final_matrix = top_100_users[:, indices] #Binary Matrix
final_pivot_matrix = top_100_pivot_users[:, indices] #Original matrix

############################################################################################################

### Execute the Algorithm
model = BiMax()
model.fit(final_matrix)

############################################################################################################

### Biclusters Selection
fitness_values = bicluster_fitness_value(model.rows_, model.columns_, final_pivot_matrix)
indexed_list = list(enumerate(fitness_values))
sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
sorted_values = [item[1] for item in sorted_indexed_list]
sorted_indices = [item[0] for item in sorted_indexed_list]
final_rows = model.rows_[sorted_indices[:64]]
final_columns = model.columns_[sorted_indices[:64]]
plt.plot(range(64), sorted_values[:64])
plt.plot(range(len(sorted_values)), sorted_values)

# get largest bicluster
idx = np.argmax(list(model.rows_[i].sum() * model.columns_[i].sum()
                     for i in range(len(model.rows_))))
bc = np.outer(model.rows_[idx], model.columns_[idx])

# plot data and overlay largest bicluster
plt.pcolor(final_matrix, cmap=plt.cm.Greys, shading='faceted')
plt.pcolor(bc, cmap=plt.cm.Greys, alpha=0.7, shading='faceted')
plt.axis('scaled')
plt.xticks([])
plt.yticks([])

# Save the generated biclusters
np.savetxt(clusteres_rows, model.rows_, fmt='%s') 
np.savetxt(clusteres_columns, model.columns_, fmt='%s') 

# Save the biclusters fitness values
with open (clusters_fitness_values, 'w') as file:
    for item in fitness_values:
        file.write(str(item) + '\n')