import pandas as pd
import re

# Define data paths (consider using a configuration file for reusability)
data_path = "..\\..\\..\\Data\\ml-100k"
movies_info_path = f"{data_path}\\u.item"
temp_movie = f"{data_path}\\u_item_no_pipes.txt"
movies_type_path = f"{data_path}\\movies_type.csv"


def extract_movie_ids(input_file, id_array):
  """
  Extracts movie IDs (integers) from a file using regular expressions and stores them in an array.

  Args:
      input_file (str): Path to the input file.
      id_array (list): List to store extracted movie IDs.
  """

  with open(input_file, "r") as in_file:
    for line in in_file:
      # Use regular expression to match digits before the first pipe
      match = re.search(r"^\d+", line)  # Matches digits at the beginning of the line

      if match:
        movie_id = int(match.group())  # Extract the matched digits and convert to integer
        id_array.append(movie_id)
      else:
        # Handle lines without a matching pattern (optional: log errors, skip the line, etc.)
        print(f"Warning: Line doesn't match ID pattern: {line}")


def extract_last_38_chars(input_file, alist):
  """
  Extracts the last 38 characters from each line in a file, handling lines shorter than 38 characters.

  Args:
      input_file (str): Path to the input file.
      alist (list): List to store the last 38 characters of each data element.
  """

  with open(input_file, "r") as in_file:
    for line in in_file:
      # Remove trailing newline (if present)
      line = line.rstrip()

      if len(line) < 38:
        alist.append(line)  # Append the entire string if it's less than 38 characters
      else:
        alist.append(line[-38:])  # Extract and append the last 38 characters



def combine_and_save_to_csv(data_list, id_array, output_filename):
  """
  Combines two lists (data_list and id_array) into a DataFrame and saves it as a CSV file.

  Args:
      data_list (list): List containing data for each movie (e.g., last 38 characters).
      id_array (list): List containing movie IDs.
      output_filename (str): Name of the CSV file to save the DataFrame.
  """

  # Create a dictionary with lists as values (assuming data_list and id_array have the same length)
  data = {"id": id_array, "type": data_list}

  # Create a DataFrame from the dictionary
  df = pd.DataFrame(data)

  # Save the DataFrame as a CSV file
  df.to_csv(output_filename, index=False)  # Set index=False to avoid an extra index column
  print(f"Dataframe saved as '{output_filename}'.")


if __name__ == "__main__":
  # Extract movie IDs
  id_array = []
  extract_movie_ids(movies_info_path, id_array)

  # Extract last 38 characters from each line (handling short lines)
  type_list = []
  extract_last_38_chars(movies_info_path, type_list)

  # Combine and save to CSV
combine_and_save_to_csv(type_list, id_array, movies_type_path)
