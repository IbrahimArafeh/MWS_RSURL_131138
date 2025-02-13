import pandas as pd

# Read the data
data_path = "..\\..\\Data\\ml-100k"
ratings_info_path = f"{data_path}\\u.data"
movies_info_path = f"{data_path}\\u.item"
users_info_path = f"{data_path}\\users.csv"
users = pd.read_csv(users_info_path)
ratings_columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
ratings = pd.read_csv(ratings_info_path, sep='\t', header=None,
                    names=ratings_columns)
ratings = ratings.drop('TimeStamp', axis=1)
print(ratings.head())
with open(movies_info_path, 'r') as file:
    lines = file.readlines()
movies_ids = list([line.split('|')[0] for line in lines])

###########################################

# Encode Gender Info
users['Gender'] = users['Gender'].map({'M': 0, 'F': 1})
print(users.head())
users.to_csv(f"{data_path}\\users.csv", index=False)

###########################################

# Preapre the ratings data to be entrerd into Biclustering Algorithm
users_who_rated = ratings['UserID'].unique().astype(int)
users_ratings_with_all_movies = pd.DataFrame([(user, movie) for user in users_who_rated for movie in movies_ids],
                               columns=['UserID', 'MovieID']).astype(int)
rating_dict = {(row['UserID'], row['MovieID']): row['Rating'] for _, row in ratings.iterrows()}

def get_rating(user_id, movie_id):
    return rating_dict.get((user_id, movie_id), 0)

users_ratings_with_all_movies['Rating'] = users_ratings_with_all_movies.apply(
    lambda x: get_rating(x['UserID'], x['MovieID']), axis=1
)
users_ratings_with_all_movies.to_csv(f"{data_path}\\ratings.csv", index=False)