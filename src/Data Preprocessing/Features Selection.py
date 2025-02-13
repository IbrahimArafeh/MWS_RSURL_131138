import pandas as pd
import sys
sys.path.append("../Packages/")
from modified_geocodio import geocodio

# Getting Users Info
columns = ['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode']
data_path = "..\\..\\Data\\ml-100k"
users_info_path = f"{data_path}\\u.user"
users = pd.read_csv(users_info_path, sep='|', header=None,
                    names=columns)
print(users.head())

#########################################################################

# Replace ZipCode feature with Longtitude and Lattitude features
Geocodio_API_Key = "15fd00606b1d51db557f11b0116a1275a1fa571"
client = geocodio.GeocodioClient(Geocodio_API_Key)

def geocode_location(zipcode):
    try:
        location = client.geocode(zipcode)
        return location.coords[0], location.coords[1]
    except:
        return None, None

users['Lattitude'] = None
users['Longtitude'] = None
for index, row in users.iterrows():
    client.geocode(row['ZipCode'])
    latitude, longitude = geocode_location(row['ZipCode'])
    users.at[index, 'Lattitude'] = latitude
    users.at[index, 'Longtitude'] = longitude

users.drop(columns=['ZipCode'], inplace=True)    
users.to_csv(f"{data_path}\\users.csv", index=False)
print(users.head())

#########################################################################

# Replace Occupation feature with Salary feature
jobs_avg_annual_salaries = pd.read_excel(f"{data_path}\\Job Average Salaries.xlsx")
for index, row in users.iterrows():
    users.at[index, 'Occupation'] = int(jobs_avg_annual_salaries[jobs_avg_annual_salaries["Job"] == users.at[index, 'Occupation']]["Aveage Annual Salary in usd"])
users.rename(columns={'Occupation': 'Salary'}, inplace=True)
users.to_csv(f"{data_path}\\users.csv", index=False)
print(users.head())