"""
This script processes raw dalia dataset to into npz file.

One processed experiment is npz file with 7 subfiles: ['PPG', 'timestamps', 'ACC', 'time_HR', 'HR', 'user_info'].
array['PPG'].shape = (number of timestamps, 1) - 1 PPG channel.
array['ACC].shape = (number of timestamps, 3) - acc_x, acc_y, acc_z
array['subject'] is a number of experiment
array['user_info'] consist of information about user and experiment.
array['time_sensors'].shape = (number of timestamps) - timestamps of ACC data
array['time_hr'].shape = (number of timestamps) - timestamps of PPG data
array['time_beats'].shape - timestamps of heart beats
array['time_activity'].shape = (number of timestamps) - timestamps of activity type
array['HR'] consists of person's heart rate
---------------------------------------------------------------------
Usage:
    polyn_process.py /path/to/raw/data /path/to/out/folder

"""

import numpy as np
import pickle
import pandas as pd
import time
import re
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to the folder with experiments')
parser.add_argument('out', help='output directory')
args = parser.parse_args()

data_path = args.data
save_dir = args.out
subjects = [exp_folder for exp_folder in os.listdir(data_path) if re.match('S[1-9]+$', exp_folder) is not None]


acc_freq = 32
ppg_freq = 64
activity_freq = 4
hr_freq = 0.5

max_freq = max([acc_freq, ppg_freq, activity_freq, hr_freq])

for s in tqdm(subjects):
    
    acc_start_time = int(re.findall(r"(\d+).", pd.read_csv(data_path+s+'/ACC.csv').columns[0])[0])
    bvp_start_time = int(re.findall(r"(\d+).", pd.read_csv(data_path+s+'/BVP.csv').columns[0])[0])
    hr_start_time = int(re.findall(r"(\d+).", pd.read_csv(data_path+s+'/HR.csv').columns[0])[0])
    all_data = pd.read_pickle(data_path+s+f'/{s}.pkl')
    
    final_data = {}
    data = {}

    # идентификатор эксперимента
    final_data['subject'] = s 

    # PPG на запястье 64 Hz
    final_data['PPG'] = all_data['signal']['wrist']['BVP'] 

    # ACC на запястье 32 Hz в масштабе 1/64*g, g - ускорение своодного падения
    final_data['ACC'] = np.repeat(all_data['signal']['wrist']['ACC'], max_freq//acc_freq, axis=0) 

    # типы активностей с частотой замера 4 Hz
    final_data['activity'] = all_data['activity']

    # среднее кол-во сердечных ударов за 8-сек окно со смещением в 2 секунды - нужно так же считать 
    # предсказание для регрессии (усреднять значения PPG и ACC)
    final_data['HR'] = all_data['label'] 
    
    # сведения об испытуемом в формате словаря
    final_data['user_info'] = all_data['questionnaire']
    final_data['user_info']['Name'] = int(s[1:])
    # моменты ударов
    beats = all_data['rpeaks']
    final_data['time_beats'] = np.array([1000 * (acc_start_time + beat.astype(int)/1000) for beat in beats]).astype('datetime64[ms]') 
    
    final_data['time_hr'] = np.array([1000 * (hr_start_time + i/hr_freq) for i in range(len(final_data['HR']))]).astype('datetime64[ms]') 
    final_data['time_sensors'] = np.array([1000 * (bvp_start_time + i/ppg_freq) for i in range(len(final_data['PPG']))]).astype('datetime64[ms]')
    final_data['time_activity'] = np.array([1000 * (acc_start_time + i/activity_freq) for i in range(len(final_data['activity']))]).astype('datetime64[ms]')
    np.savez(save_dir+f'{s}.npz', **final_data)