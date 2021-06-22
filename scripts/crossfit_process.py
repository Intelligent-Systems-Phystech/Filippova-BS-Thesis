import sys
sys.path.append('..')
from glob import glob
import numpy as np
import os
import datetime
import argparse
import pickle
import pandas as pd


def extract_data(path, act_label):
    d = np.load(path, allow_pickle=True)
    acc = d[:3, :].T
    activities = act_label * np.ones(acc.shape[0])
    start_time = pd.to_datetime(datetime.datetime.today())
    freq = 100
    time = np.cumsum(np.concatenate([np.zeros(1), np.ones(activities.shape[0] - 1)]) * 1000 / freq).astype(
        'timedelta64[ms]')
    time = (start_time + time).astype('datetime64[ms]')
    data = {}
    data['user_info'] = {}
    name = os.path.basename(path).split('_')[-1].split('.')[0]
    data['user_info'].update({'Name': name})
    data['ACC'] = acc
    data['PPG'] = np.zeros((data['ACC'].shape[0], 1))
    data['HR'] = np.zeros(data['ACC'].shape[0])
    data['time_sensors'] = time
    data['time_hr'] = time
    data['time_activity'] = time
    data['activity'] = activities

    return data

def main(args):

    save_dir = args.out_path
    path = args.data_path
    paths = f'{path}HAR_Crossfit_Sensors_Data/data/constrained_workout/preprocessed_numpy_data/np_exercise_data/'
    paths_activity = glob(f'{paths}/*')
    paths_activity = [path for path in paths_activity if os.path.basename(path)!= 'Null']
    with open('crossfit_data_split.pkl', 'rb') as f:
        split = pickle.load(f)
        split['train']= [f'{save_dir}{i}' for i in split['train']]
        split['valid'] = [f'{save_dir}{i}' for i in split['valid']]
    with open('result_crossfit_data_split.pkl', 'wb') as f:
        pickle.dump(split, f)
    activity = ['KB Squat press','Squats', 'Box jumps', 'KB Press','Push ups',
                'Pull ups', 'Burpees', 'Dead lifts',
                'Wall balls', 'Crunches']
    activity_dict = dict(zip(activity, np.arange(len(activity))))
    for activity in paths_activity:
        act = os.path.basename(activity)
        act_label = activity_dict[act]
        paths = glob(f'{activity}/*')
        for path in paths:
            data = extract_data(path, act_label)
            activities = data['activity']
            if activities.shape[0] > 500:
                out_file_name = f'{save_dir}/{os.path.basename(path)[:-3]}npz'
                np.savez(out_file_name, **data)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', default='/data/crossfit_sensors_har/')
    parser.add_argument('-out_path', '--out_path', default='/data/crossfit_release/', help='path to split_try.pkl')

    args = parser.parse_args()
    main(args)

