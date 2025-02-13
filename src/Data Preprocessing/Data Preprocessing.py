import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Statistics about the data generated manual by passing the value to Google maps and get the longtiude and altitiude...
data_path = "..\\..\\Data\\ml-100k"
users_info_path = f"{data_path}\\users.csv"
geo_mapping_file_path = f"{data_path}\\Geographic Mapping of Null Values.xlsx"
users = pd.read_csv(users_info_path)
pd.set_option('display.max_columns', None)
print(users.drop("Gender", axis = 1).describe())

################################################

# Displaying Null values
null_counts = users.isnull().sum()
print(null_counts)
plt.figure(figsize=(10, 6))
null_counts.plot(kind='bar', color='skyblue')
plt.title('Null Values in Users Columns')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

###############################################

# Handling Null values
average_salary = int(users['Salary'].mean())
users['Salary'].fillna(average_salary, inplace=True)
users_with_null_coordinates = users[users['Lattitude'].isnull() | users['Longtitude'].isnull()]['UserID']
geo_mapping_for_users_with_null_coordinates = pd.read_excel(geo_mapping_file_path, names = ['UserID','Lattitude', 'Longtitude'])
for index, row in users.iterrows():
    if row['UserID'] in list(users_with_null_coordinates):
        geo_row = geo_mapping_for_users_with_null_coordinates[geo_mapping_for_users_with_null_coordinates['UserID'] == row['UserID']]
        if not geo_row.empty:
            users.at[index, 'Longtitude'] = geo_row['Longtitude'].values[0]
            users.at[index, 'Lattitude'] = geo_row['Lattitude'].values[0]

###############################################

# Saving the cleaned users.csv file
null_counts = users.isnull().sum()
print(null_counts)
users.to_csv(f"{data_path}\\users.csv", index=False)