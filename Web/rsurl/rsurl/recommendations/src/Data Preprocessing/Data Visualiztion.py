import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data_path = "..\\..\\Data\\ml-100k"
users_info_path = f"{data_path}\\users.csv"
users = pd.read_csv(users_info_path)
print(users.head())

###########################################

# Visualize charts about the data distributions
plt.figure(figsize=(8, 5))
bin_ranges = range(0,200001,25000)
plt.hist(users['Salary'], bins=bin_ranges, color='skyblue', edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Distribution of Salaries')
plt.grid(True)
plt.show()

gender_counts = users['Gender'].value_counts()
plt.figure(figsize=(8, 5))
gender_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(users['Longtitude'], users['Lattitude'], color='skyblue', s=50, alpha=0.7)
plt.xlabel('Longtitude')
plt.ylabel('Lattitude')
plt.title('Locations')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
bin_ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
plt.hist(users['Age'], bins=bin_ranges, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.grid(True)
plt.show()