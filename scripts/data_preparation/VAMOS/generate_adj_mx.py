import os
import csv
import pickle

import numpy as np
import pandas as pd
from geopy.distance import geodesic

def extract_lat_long(geom):
    if pd.isna(geom):
        return None, None
    geom = geom.strip("POINT ()")
    lat, long = map(float, geom.split())
    return lat, long

def get_adjacency_matrix_vamos(distance_df_filename: str):
   
    data = pd.read_csv(distance_df_filename)

    # Ignore sensors where "geom" is empty
    data = data.dropna(subset=["geom"])

    data[['longitude', 'latitude']] = data['geom'].apply(lambda x: pd.Series(extract_lat_long(x)))

    num_of_vertices = len(data)
    adjacency_matrix_connectivity = np.ones((num_of_vertices, num_of_vertices), dtype=np.float32)
    adjacency_matrix_distance = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    # Calculate distance-based adjacency matrix
    for i in range(num_of_vertices):
        for j in range(i + 1, num_of_vertices):
            distance = geodesic((data.iloc[i]['latitude'], data.iloc[i]['longitude']),
                                (data.iloc[j]['latitude'], data.iloc[j]['longitude'])).meters
            adjacency_matrix_distance[i, j] = distance
            adjacency_matrix_distance[j, i] = distance

    return adjacency_matrix_connectivity, adjacency_matrix_distance

def generate_adj_vamos():
    distance_df_filename = "datasets/raw_data/VAMOS/pcs_meta.csv"

    adj_mx, distance_mx = get_adjacency_matrix_vamos(distance_df_filename)
        
    with open("datasets/raw_data/VAMOS/adj_VAMOS.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    with open("datasets/raw_data/VAMOS/adj_VAMOS_distance.pkl", "wb") as f:
        pickle.dump(distance_mx, f)

if __name__ == "__main__":
    generate_adj_vamos()
