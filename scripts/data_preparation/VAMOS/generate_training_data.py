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


def generate_adj_matrix(data, threshold_distance=500):
    """Generate adjacency matrix based on geographical distances."""
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


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets for Vamos data."""

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel

    # read data
    df = pd.read_csv(data_file_path)
    df['Datetime'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M')
    df.set_index('Datetime', inplace=True)

    data = df[['speed', 'netTimeGap', 'occupiedTime', 'vehicleLength', 'trafficVolumeHeavy', 'trafficVolumeLight']].values
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, f, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, f, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)

    # generate and save adjacency matrix
    adj_matrix = generate_adj_matrix(df)
    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(adj_matrix, f)


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0, 1, 2, 3, 4, 5]    # target channels
    STEPS_PER_DAY = 288

    DATASET_NAME = "VAMOS"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/VAMOS/sample_vamos_data_2022_23.csv"

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