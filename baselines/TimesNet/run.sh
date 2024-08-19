#!/bin/bash
python experiments/train.py -c baselines/TimesNet/ETTh1.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/ETTh2.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/ETTm1.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/ETTm2.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/Electricity.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/ExchangeRate.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/Weather.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/TimesNet/PEMS08.py --gpus '0'
