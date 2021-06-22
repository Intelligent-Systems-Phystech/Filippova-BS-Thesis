import datetime
from glob import glob

import numpy as np
import pandas as pd

from polyn_lib.preparation.utils import *

split_ppg_acc_params = {
    'skiprows': 2,
    'header': None,
    'names': range(10),
    'sep': ';'
}
table_names = ['User Info:', 'Sensor Info:', 'Hydration Info:', 'Acc_AFE-meter-log']

def process_sensors(path, params):

    tables = split_csv_file(path, **split_ppg_acc_params)
    sensors_log_name = params['sensors_log_name']
    user_info_name = params['user_info_name']
    columns_names_sensors = params['columns_names_sensors']
    table_sensors = tables[sensors_log_name]
    table_user_info = tables[user_info_name]
    sensors_dict = extract_sensors_data(table_sensors, **columns_names_sensors)
    meta_data_dict = extract_meta_data(table_user_info)

    return sensors_dict, meta_data_dict


def process_hr(path):

    hr_table = pd.read_csv(path)
    hr_dict = extract_heart_rate(hr_table)

    return hr_dict


def process_data(path_to_folder, params, out_folder):

    data = {}
    pathes = glob(f'{path_to_folder}/*')
    for path in pathes:
        if path.find('Polar') != -1:
            hr_path = path
        if path.find('AFE') != -1:
            sensors_path = path
        if path.find('andromeda-protocols') != -1:
            activity_path = path
    params_sensors = params['params_sensors']
    params_act = params['params_act']
    sensors_dict, meta_data_dict = process_sensors(sensors_path, params_sensors)
    data.update(sensors_dict)
    data.update(meta_data_dict)
    hr_dict = process_hr(hr_path)
    data.update(hr_dict)
    activity_dict = process_act(activity_path,  **params_act)
    data.update(activity_dict)
    file_name = os.path.basename(path_to_folder)
    out_file_name = f'{out_folder}/{file_name}'
    np.savez(out_file_name, **data)

def process_act(activity_path, activities_dict, frequency):
    act_dict = {}
    act_info = pd.read_csv(activity_path, sep = ';')
    activity_table = label_activities(act_info, activities_dict)
    time_seconds = np.repeat(activity_table.index.values, frequency)
    time_seconds = pd.to_datetime(time_seconds)
    delta = np.concatenate(
        (np.cumsum(np.repeat(datetime.timedelta(milliseconds=40), frequency - 1)),
         np.array([datetime.timedelta(seconds=1)]))
    )
    size = activity_table.index.values.shape[0]
    time_ms = np.concatenate(size*[delta])
    time_array = time_ms + time_seconds
    activity_array = np.repeat(activity_table['activities'].values, frequency)
    act_dict['time_activity'] = time_array
    act_dict['activity'] = activity_array
    return act_dict

def label_activities(act_info, activities_dict):

    starts = np.array([pd.to_datetime(i[:-9]) for i in act_info['start'].values], dtype = 'datetime64[s]')
    ends = act_info['end'].astype('datetime64[s]').values
    points = list(zip(starts, ends))
    activities = act_info['activity_type'].values
    time_array = np.arange(starts[0], ends[-1], step = datetime.timedelta(seconds = 1), dtype = "datetime64[s]")
    activities_array = np.zeros(time_array.shape[0])
    table = pd.DataFrame(activities_array, columns = ['activities'], index = time_array)
    for index, point in enumerate(points):
        key = activities_dict[activities[index]]
        size = (table.loc[point[0]:point[1]].shape[0], 1)
        act_ids = np.ones(size)*key
        table[point[0]:point[1]] = act_ids

    return table



def extract_sensors_data(dataframe, ms_column='ms_ticker',
                         time_column = 'Tstamp',
                         ppg_columns=['led_1', 'led_2'],
                         acc_columns=['acc_x', 'acc_y', 'acc_z']):
    """
    Extracts sensors info (PPG signal (2 channels), ACC signal (3 channels))
    ------------------------------------------------------------
    """

    sensors_dict = {}
    sensors = dataframe.loc[1:, 1:]
    sensors_columns = dataframe.head(1).values[0]
    sensors_columns = [i.replace(" ", "") for i in sensors_columns if i.find('Index') == -1]
    sensors.columns = sensors_columns
    check_columns_exist(ppg_columns, sensors_columns)
    check_columns_exist(acc_columns, sensors_columns)
    check_columns_exist(ms_column, sensors_columns)
    check_columns_exist(time_column, sensors_columns)
    ppg = np.array(sensors[ppg_columns].values[1:, :], dtype=int)
    ms = np.array(sensors[ms_column].values[1:, ])
    ms_ints = np.array([int(str(i)[-3:]) for i in ms], dtype=float)
    ms_delta = [datetime.timedelta(milliseconds=i) for i in ms_ints]

    time = dataframe.loc[:,1].values[1:]
    time = np.array([pd.to_datetime(i) for i in time])
    time_with_ms = np.array(ms_delta) + time

    sensors_dict['PPG'] = ppg
    sensors_dict['time_sensors'] = time_with_ms.astype('datetime64[us]')
    sensors_dict['ms_ticker_sensors'] = ms
    acc = np.array(sensors[acc_columns].values[1:, :], dtype=float)
    sensors_dict['ACC'] = acc

    return sensors_dict


def extract_meta_data(dataframe):

    meta_data_dict = {}
    columns_name = [i.replace(':', '') for i in dataframe.loc[:, 0].values]
    dataframe = dataframe.T.loc[1:2, :]
    dataframe.columns = columns_name
    meta_data = dict(dataframe)
    meta_data_dict['meta_data'] = meta_data

    return meta_data_dict

def extract_heart_rate(dataframe):

    hr_dict = {}
    user_info = dict(dataframe.loc[0, :])
    start_time = user_info['Start time']
    start_date = user_info['Date']
    start = f'{start_time} {start_date}'
    hr_dict['HR'] = np.array(dataframe.iloc[2:, 2].values, dtype=int)
    start = pd.to_datetime(start)
    duration = datetime.timedelta(seconds=hr_dict['HR'].shape[0])
    end = start + duration
    step = datetime.timedelta(seconds = 1)
    time_array = np.arange(start, end, step)
    hr_dict['time_hr'] = time_array
    hr_dict['user_info'] = user_info

    return hr_dict


def split_csv_file(path, **split_ppg_acc_params):

    df = pd.read_csv(path, **split_ppg_acc_params)
    groups = df[0].isin(table_names).cumsum()
    tables = {g.iloc[0, 0]: g.iloc[1:] for k, g in df.groupby(groups)}

    return tables


def create_act_dict(folder_path):
    act_list = []
    pathes = glob(f'{folder_path}/*/export-andromeda-protocols*')
    for path in pathes:
        act_inf = pd.read_csv(path, sep = ';')
        activities = act_inf['activity_type'].values
        act_list.extend(activities)
    act_set = set(act_list)
    act_dict = dict(zip(act_set, np.arange(1, len(act_set)+1)))
    act_dict['unknown'] = 0
    return act_dict