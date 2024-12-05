import numpy as np

def view_first_100_lines(npz_file):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Print out the keys in the .npz file
    print("Keys in the .npz file:", data.files)
    
    # Iterate over each key and print the first 100 lines of its data
    for key in data.files:
        print(f"\nData for key: {key}")
        # Get the array corresponding to the key
        array_data = data[key]
        
        # Determine the number of lines to print
        num_lines = min(100, array_data.size)
        
        # Print the first 100 lines
        if array_data.ndim == 1:
            print(array_data[:num_lines])
        elif array_data.ndim == 2:
            print(array_data[:num_lines, :])
        else:
            print(array_data[:num_lines])
        
        print("\n" + "-"*40 + "\n")

# Path to your .npz file
npz_file_path = "datasets/raw_data/PEMS03/PEMS03.npz"

# Call the function
view_first_100_lines(npz_file_path)

import pandas as pd
import pickle

def inspect_pkl_file(pkl_file):
    try:
        # Load the .pkl file
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
        
        # Print type and some content to understand the structure
        print(f"Type of data: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", list(data.keys())[:10])  # Print first 10 keys
            for key in data.keys():
                print(f"Type of data[{key}]: {type(data[key])}")
                print(f"Data[{key}]:", data[key][:5])  # Print first 5 elements of each key
                break  # Inspect first key
        elif isinstance(data, list):
            print("First element type:", type(data[0]))
            print("First element:", data[0])
        else:
            print(data)
        
        return data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_to_dataframe(data):
    if isinstance(data, dict):
        # Attempt to convert to DataFrame
        try:
            df = pd.DataFrame(data)
            return df
        except ValueError:
            # If conversion fails, try alternative ways
            pass
        
        # If the values are lists, try to construct a DataFrame manually
        if all(isinstance(v, list) for v in data.values()):
            max_len = max(len(v) for v in data.values())
            for k, v in data.items():
                if len(v) < max_len:
                    data[k].extend([None] * (max_len - len(v)))
            df = pd.DataFrame(data)
            return df
    
    elif isinstance(data, list):
        # Convert list of dictionaries to DataFrame
        if all(isinstance(i, dict) for i in data):
            df = pd.DataFrame(data)
            return df
        # Convert list of lists to DataFrame
        elif all(isinstance(i, list) for i in data):
            df = pd.DataFrame(data)
            return df
    
    # For other data types, attempt to create a DataFrame
    df = pd.DataFrame([data])
    return df

def main():
    pkl_file_path = "datasets/raw_data/PEMS03/adj_PEMS03.pkl"


    data = inspect_pkl_file(pkl_file_path)
    
    if data is not None:
        df = convert_to_dataframe(data)
        if df is not None:
            print("Tabular view of the first 10 rows:")
            print(df.head(10))
        else:
            print("Unable to convert data to a tabular format.")

if __name__ == "__main__":
    main()

