import os
import sys
import shutil
import pickle
import argparse
import logging

import numpy as np
import pandas as pd

from geopy.distance import geodesic
from tqdm import tqdm

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def setup_logger(output_dir):
    """Sets up the logger to log info to both console and file."""
    logger = logging.getLogger('DataProcessingLogger')
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(output_dir, 'data_processing.log'))

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def log_statistical_analysis(data, logger, label="Data"):
    """Log statistical analysis of the data."""
    flattened_data = data.flatten()
    logger.info(f"--- {label} Statistical Analysis ---")
    logger.info(f"Min: {np.min(flattened_data):.4f}")
    logger.info(f"Max: {np.max(flattened_data):.4f}")
    logger.info(f"Mean: {np.mean(flattened_data):.4f}")
    logger.info(f"Standard Deviation: {np.std(flattened_data):.4f}")
    logger.info(f"25th Percentile: {np.percentile(flattened_data, 25):.4f}")
    logger.info(f"50th Percentile (Median): {np.percentile(flattened_data, 50):.4f}")
    logger.info(f"75th Percentile: {np.percentile(flattened_data, 75):.4f}")
    logger.info("-------------------------------")


def generate_data(args: argparse.Namespace, logger):
    """Preprocess and generate train/valid/test datasets for Vamos data."""

    logger.info("Starting data generation process...")

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

    logger.info(f"Reading data from {data_file_path}...")
    df = pd.read_csv(data_file_path)

    logger.info("Processing datetime column...")
    df['Datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['s_idx', 'trafficVolumeLight']]

    logger.info("Pivoting the data...")
    df_pivoted = df.pivot(columns='s_idx', values='trafficVolumeLight')
    df_pivoted.columns.name = None
    df_pivoted.index.name = None

    logger.info("Resampling the data...")
    df_resampled = df_pivoted.resample('5T').sum()

    logger.info(f"Raw time series shape: {df_resampled.shape}")
    data = df_resampled.values[:, :, np.newaxis]  # Add a third dimension for features
    l, n, f = data.shape

    logger.info("Performing initial statistical analysis...")
    log_statistical_analysis(data, logger, label="Raw Data")

    logger.info("Splitting data into train/validation/test sets...")
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    logger.info(f"Training samples: {train_num}, Validation samples: {valid_num}, Test samples: {test_num}")

    index_list = []
    for t in tqdm(range(history_seq_len, num_samples + history_seq_len), desc="Generating index list"):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num + valid_num: train_num + valid_num + test_num]

    logger.info("Normalizing data...")
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len,
                       norm_each_channel=norm_each_channel)

    logger.info("Performing statistical analysis on normalized data...")
    log_statistical_analysis(data_norm, logger, label="Normalized Data")

    logger.info("Adding temporal features...")
    feature_list = [data_norm]
    if add_time_of_day:
        tod = (df_resampled.index.values - df_resampled.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod_tiled = np.expand_dims(tod, axis=-1)
        tod_tiled = np.repeat(tod_tiled, n, axis=1)
        feature_list.append(np.expand_dims(tod_tiled, axis=-1))

    if add_day_of_week:
        dow = df_resampled.index.dayofweek / 7
        dow_tiled = np.expand_dims(dow, axis=-1)
        dow_tiled = np.repeat(dow_tiled, n, axis=1)
        feature_list.append(np.expand_dims(dow_tiled, axis=-1))

    if add_day_of_month:
        dom = (df_resampled.index.day - 1) / 31
        dom_tiled = np.expand_dims(dom, axis=-1)
        dom_tiled = np.repeat(dom_tiled, n, axis=1)
        feature_list.append(np.expand_dims(dom_tiled, axis=-1))

    if add_day_of_year:
        doy = (df_resampled.index.dayofyear - 1) / 366
        doy_tiled = np.expand_dims(doy, axis=-1)
        doy_tiled = np.repeat(doy_tiled, n, axis=1)
        feature_list.append(np.expand_dims(doy_tiled, axis=-1))

    processed_data = np.concatenate(feature_list, axis=-1)

    logger.info("Performing statistical analysis on processed data with temporal features...")
    log_statistical_analysis(processed_data, logger, label="Processed Data with Temporal Features")

    logger.info("Saving processed data and index files...")
    index = {
        "train": train_index,
        "valid": valid_index,
        "test": test_index
    }
    with open(os.path.join(output_dir, f"index_in_{history_seq_len}_out_{future_seq_len}_rescale_{if_rescale}.pkl"),
              "wb") as f:
        pickle.dump(index, f)

    data = {"processed_data": processed_data}
    with open(os.path.join(output_dir, f"data_in_{history_seq_len}_out_{future_seq_len}_rescale_{if_rescale}.pkl"),
              "wb") as f:
        pickle.dump(data, f)

    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
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
    DATA_FILE_PATH = "/home/ijaradar/Work/dl-trafic-prediction-gsm/datasets/raw_data/VAMOS/filtered_dataset_23.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str, default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD, help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW, help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--dom", type=bool, default=DOM, help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY, help="Add feature day_of_week.")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Normalize each channel separately.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = setup_logger(args.output_dir)

    args.norm_each_channel = True
    generate_data(args, logger)
    args.norm_each_channel = False
    generate_data(args, logger)
