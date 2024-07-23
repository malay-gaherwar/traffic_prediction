import os
import csv
import pickle

import numpy as np
import pandas as pd
from geopy.distance import geodesic

def get_adjacency_matrix_vamos(distance_df_filename: str, num_of_vertices: int) -> tuple:
    """Generate adjacency matrix for VAMOS data.

    Args:
        distance_df_filename (str): path of the csv file containing sensor information
        num_of_vertices (int): number of vertices

    Returns:
        tuple: two adjacency matrices
            np.array: fully connected adjacency matrix (all entries are 1)
            np.array: distance-based adjacency matrix (geodesic distances)
    """
    adjacency_matrix_connectivity = np.ones((num_of_vertices, num_of_vertices), dtype=np.float32)
    adjacency_matrix_distance = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    data = pd.read_csv(distance_df_filename)
    for i in range(num_of_vertices):
        for j in range(i + 1, num_of_vertices):
            distance = geodesic((data.iloc[i]['latitude'], data.iloc[i]['longitude']),
                                (data.iloc[j]['latitude'], data.iloc[j]['longitude'])).meters
            adjacency_matrix_distance[i, j] = distance
            adjacency_matrix_distance[j, i] = distance

    return adjacency_matrix_connectivity, adjacency_matrix_distance

def generate_adj_vamos():
    distance_df_filename = "datasets/raw_data/VAMOS/sample_vamos_data_2022_23.csv"
    num_of_vertices = len(pd.read_csv(distance_df_filename))

    adj_mx, distance_mx = get_adjacency_matrix_vamos(distance_df_filename, num_of_vertices)
    
    # Note: For VAMOS, adding a self-loop might not be necessary, if required uncomment below.
    add_self_loop = False
    if add_self_loop:
        print("adding self loop to adjacency matrices.")
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        print("kindly note that there is no self loop in adjacency matrices.")
        
    with open("datasets/raw_data/VAMOS/adj_VAMOS.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    with open("datasets/raw_data/VAMOS/adj_VAMOS_distance.pkl", "wb") as f:
        pickle.dump(distance_mx, f)

if __name__ == "__main__":
    generate_adj_vamos()
