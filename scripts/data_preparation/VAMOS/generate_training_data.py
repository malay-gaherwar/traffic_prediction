import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

from geopy.distance import geodesic

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform

"""
def generate_adj_matrix(data, threshold_distance=500):

    num_sensors = len(data)
    adjacency_matrix = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            distance = geodesic((data.iloc[i]['latitude'], data.iloc[i]['longitude']),
                                (data.iloc[j]['latitude'], data.iloc[j]['longitude'])).meters
            if distance <= threshold_distance:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    return adjacency_matrix

"""


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets for Vamos data."""

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel

    # read data
    df = pd.read_csv(data_file_path)

    # Convert 'time' column to datetime, with the correct format
    df['Datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    # Set 'Datetime' as the index
    df.set_index('Datetime', inplace=True)

    # Keep only the s_idx and trafficVolumeLight columns
    df = df[['s_idx', 'trafficVolumeLight']]

    df_pivoted = df.pivot(columns='s_idx', values='trafficVolumeLight')

    df_pivoted.columns.name = None
    df_pivoted.index.name = None

    df_resampled = df_pivoted.resample('5T').sum()
    data = df_resampled.head(5000)

    print("raw time series shape: ", data.shape)

    # Reshape the data to add a third dimension for features
    data = data.values[:, :, np.newaxis]  # Add a third dimension for features

    # Now unpack the shape
    l, n, f = data.shape

    # split data
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len,
                       norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        tod = (df_resampled.index.values - df_resampled.index.values.astype("datetime64[D]")) / np.timedelta64(1,
                                                                                                                      "D")
        tod_tiled = np.expand_dims(tod, axis=-1)
        tod_tiled = np.repeat(tod_tiled, n, axis=1)
        feature_list.append(np.expand_dims(tod_tiled, axis=-1))  # Ensure 3D shape

    if add_day_of_week:
        dow = df_resampled.index.dayofweek / 7
        dow_tiled = np.expand_dims(dow, axis=-1)
        dow_tiled = np.repeat(dow_tiled, n, axis=1)
        feature_list.append(np.expand_dims(dow_tiled, axis=-1))  # Ensure 3D shape

    if add_day_of_month:
        dom = (df_resampled.index.day - 1) / 31  # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.expand_dims(dom, axis=-1)
        dom_tiled = np.repeat(dom_tiled, n, axis=1)
        feature_list.append(np.expand_dims(dom_tiled, axis=-1))  # Ensure 3D shape

    if add_day_of_year:
        doy = (df_resampled.index.dayofyear - 1) / 366  # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.expand_dims(doy, axis=-1)
        doy_tiled = np.repeat(doy_tiled, n, axis=1)
        feature_list.append(np.expand_dims(doy_tiled, axis=-1))  # Ensure 3D shape

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale),
              "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale),
              "wb") as f:
        pickle.dump(data, f)


"""
    # generate and save adjacency matrix
    adj_matrix = generate_adj_matrix(df)
    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(adj_matrix, f)
"""

if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0, 1, 2, 3, 4, 5]  # target channels
    STEPS_PER_DAY = 288

    DATASET_NAME = "VAMOS"
    TOD = True  # if add time_of_day feature
    DOW = True  # if add day_of_week feature
    DOM = True  # if add day_of_month feature
    DOY = True  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "/home/ijaradar/Work/dl-trafic-prediction-gsm/datasets/raw_data/VAMOS/vamos_394-97_sample.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Normalize each channel separately.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
