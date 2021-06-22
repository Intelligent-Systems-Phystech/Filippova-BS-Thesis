import datetime
import numpy as np
import pandas as pd
import torch

from lib.utils import get_windows, interpolate


class SensorsExperiment:
    """
    Attributes
    ----------
    path: np.ndarray
        numpy array with info about experiment: sensors, hr, time, user info, meta data
    params: dict
        params for preprocessing
    """

    def __init__(self, path, params, ignore_zero=False):

        self.sampling_rate = params['sampling_rate']
        self.averaging_interval = params['slice_for_hr_label']
        self.ignore_zero = ignore_zero
        self.path = path
        self.load()
        self._parse_raw()
        self._process()

    def load(self):
        self.raw_experiment = np.load(self.path, allow_pickle=True)
        self.exist_attrs = [f.lower() for f in self.raw_experiment.files]

    def _parse_raw(self):
        attr_list = ['ppg', 'acc', 'hr', 'time_sensors', 'time_hr',
                     'user_info', 'meta_data', 'activity', 'time_activity']
        for attr in attr_list:
            if attr in self.exist_attrs:
                try:
                    setattr(self, attr, self.raw_experiment[attr])
                except KeyError:
                    setattr(self, attr, self.raw_experiment[attr.upper()])
            else:
                setattr(self, attr, None)

    def _generate_timestamps(self, length, start_time):
        deltas = np.cumsum(np.repeat(datetime.timedelta(milliseconds=1000 / self.sampling_rate), length - 1))
        zero_second = np.zeros(1, dtype='timedelta64[ms]')
        deltas = np.concatenate([zero_second, deltas]).astype('timedelta64[ms]')
        generated_time = start_time + deltas
        return generated_time

    def _correct_timestamps_sensors(self):
        try:
            length = self.ppg.shape[0]
            start_time = self.time_sensors[0]
            corrected_timestamps = self._generate_timestamps(length, start_time)
            self.time_sensors = corrected_timestamps
        except:
            raise RuntimeError("PPG signal isn't defined")

    def _get_global_timestamps(self):
        starts = []
        ends = []
        time_list = [self.time_sensors, self.time_hr, self.time_activity]
        for time in time_list:
            if time is not None:
                starts.append(time[0])
                ends.append(time[-1])
        step = datetime.timedelta(milliseconds=1000 / self.sampling_rate)
        start = starts[np.argmax(starts)]
        stop = ends[np.argmin(ends)]
        stop = pd.to_datetime(stop) + step
        time_for_interp = np.arange(start, stop, step=step).astype('datetime64[ms]')
        setattr(self, 'time', time_for_interp)
        return time_for_interp

    def _align_time(self):
        attr_list = ['ppg', 'acc', 'hr', 'activity']
        self._get_global_timestamps().astype('datetime64[ms]').astype(int)
        for attr_name in attr_list:
            attr = getattr(self, attr_name)
            try:
                time_attr = getattr(self, f'time_{attr_name}').astype('datetime64[ms]').astype(int)
            except AttributeError:
                time_attr = getattr(self, 'time_sensors').astype('datetime64[ms]').astype(int)
            if attr is not None:
                time_for_interp = self.time.astype(int)
                attr_interp = interpolate(time_for_interp, time_attr, attr)
                setattr(self, attr_name, attr_interp)

    def _get_hr_label(self, hr_window, window_size):
        averaging_interval = np.array(self.averaging_interval)
        averaging_interval_index = tuple((averaging_interval * window_size).astype(int))
        averaging_slice = slice(*averaging_interval_index)
        hr_label = hr_window[averaging_slice].mean()

        return hr_label

    def _process(self):
        # self._correct_timestamps_sensors()
        self._align_time()

    def compile_items(self, window_size, step_size):
        segments = zip(
            get_windows(torch.Tensor(self.acc), window_size, step_size),
            get_windows(torch.Tensor(self.ppg), window_size, step_size),
            get_windows(torch.Tensor(self.hr), window_size, step_size),
            get_windows(torch.Tensor(self.activity), window_size, step_size),
        )
        items = []
        for n_in_experiment, (acc, ppg, hr, activity) in enumerate(segments):
            activity = activity.squeeze()
            activity_label = torch.mode(activity, 0)[0].long()
            item = {
                    'acc': acc,
                    'ppg': ppg,
                    'hr': hr,
                    'hr_label': self._get_hr_label(hr, window_size).view(1),
                    'activity': activity,
                    'activity_label': activity_label,
                    'n_in_experiment': torch.tensor(n_in_experiment), 
                    'experiment_id': torch.tensor(int(self.user_info.item(0)['Name'])),
                    'user_id': torch.tensor(int(self.user_info.item(0)['Name']))
                    }
            if not self.ignore_zero:
                items.append(item)
            elif item['activity_label'] != 0:
                item['activity_label'] -= 1
                items.append(item)
                
        return items
