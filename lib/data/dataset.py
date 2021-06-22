import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List

from .experiment import SensorsExperiment
import copy
from collections import defaultdict

class SensorsDataset(Dataset):
    """
    Attributes
    ----------
    task: str
        type of task for dataloader: "hr", "har", "descriptor"
    ignore_train_ids: list
        indexes, that need to ignore in train, but it is requared in valid
    experiments_paths: List[i] = str
        paths to experiments
    processing_params: dict
        params for preprocessing
    segment_params: dict = {'window_size': int, 'step_size': int }
        params for get_windows function
    limit: int
        number of experiments to process

    Methods
    -------
    process_experiments()
        returns list of windows from all experiments in self.experiments_pathes
    __getitem__(item)
        returns data window
    """

    def __init__(self,
                 experiments_paths: List,
                 task_type: str='hr',
                 ignore_ids: list=[],
                 processing_params: dict={'sampling_rate': 25, 'slice_for_hr_label': (0.5, 1)},
                 segment_params: dict={'window_size': 500, 'step_size': 50},
                 limit: int=None,
                 ignore_zero: bool=False):

        assert task_type in ['hr', 'har', 'descriptor'], 'task not supported. use "hr", "har", or "descriptor"'

        self.task_type = task_type
        self.ignore_ids = ignore_ids
        self.experiments = []
        self.experiments_paths = experiments_paths
        self.processing_params = processing_params
        self.segment_params = segment_params
        self.limit = limit
        self.ignore_zero = ignore_zero
        self._segments = []
        self._activity_labels = []
        self._user_ids = []


    def process_experiments(self):
        self._experiment_indices = [0]

        indices = range(len(self.experiments_paths))[:self.limit]
        indices_iterator = tqdm(indices, desc='Processing experiments')
        for index in indices_iterator:
            experiment = SensorsExperiment(self.experiments_paths[index],
                                           self.processing_params,
                                           self.ignore_zero)
            self.experiments.append(experiment)
            items = self.experiments[index].compile_items(**self.segment_params)

            items = [item for item in items if item['activity_label'] not in self.ignore_ids]
            self._segments.extend(items)
            self._experiment_indices.append(len(self._segments))

        self._segments = np.array(self._segments)
        
        if self.task_type == 'descriptor':
            self._class_has_segments = self.generate_segments_for_classes()

    def generate_segments_for_classes(self):
        # each segment must know its index in its own activity class for creating triplets
        for segment in self._segments:
            self._activity_labels.append(segment['activity_label'])
            self._user_ids.append(segment['user_id'])

            # add ind_in_class for tracking equal of segments in self.generate_triplet()
            segment_class = int(segment['activity_label'])

        if (len(self._activity_labels) == 0) and (self.task_type == 'descriptor'):
            raise ValueError(f"dataset has only ignored activity_label={self.ignore_ids} or empty")

        # for each class collect all segments from it (pos-samples) and all segments not from it (neg-samples)
        class_has_segments = {}
        for label in tqdm(np.unique(self._activity_labels), desc='Prepare classes with experiments'):
            mask_by_label = self._activity_labels == label
            class_has_segments[label] = {}
            for user_id in np.unique(self._user_ids):
                pos_mask = mask_by_label * (self._user_ids == user_id)
                neg_mask = ~pos_mask
                class_has_segments[label][user_id] = {'pos': self._segments[pos_mask], 'neg': self._segments[neg_mask]}
        return class_has_segments

    def generate_triplet(self, segment_ind):
        # get anch sample as segment from all segments
        anch = self._segments[segment_ind]
        activity_ind = int(anch['activity_label'])
        user_id = int(anch['user_id'])

        # get pos and neg samples for this anch sample by activity_ind
        pos_segments = self._class_has_segments[activity_ind][user_id]['pos']
        neg_segments = self._class_has_segments[activity_ind][user_id]['neg']

        # get pos sample and check that pos sample is not anch sample
        pos_id = np.random.randint(0, high=len(pos_segments))
        if pos_id == anch['n_in_experiment']:
            pos = pos_segments[abs(pos_id-1)]
        else:
            pos = pos_segments[pos_id]
            
        #get neg sample
        neg = np.random.choice(neg_segments)

        # create triplet
        triplet = {}
        for key in anch.keys():
            # pass key then was needed only for prepare triplets
            if key == 'ind_in_class':
                continue
            if type(anch[key]) != dict:
                triplet[key] = torch.stack([anch[key], pos[key], neg[key]])
            else:
                triplet[key] = (anch[key], pos[key], neg[key])

        return triplet

    def get_processed_experiment(self, index):
        if self._segments is None:
            raise RuntimeError('experiments are not processed yet')
        if self.task_type in ['hr', 'har']:
            items_slice = slice(self._experiment_indices[index], self._experiment_indices[index + 1])
            return self._segments[items_slice]
        if self.task_type == 'descriptor':
            return None

    def __getitem__(self, item):

        if len(self._segments) == 0:
            raise ValueError("Segments haven't been compiled yet. Use class method self.process_experiments().")
        else:
            if self.task_type == 'descriptor':
                return self.generate_triplet(item)
            elif self.task_type in ['hr', 'har']:
                return self._segments[item]

    def __len__(self):

        return len(self._segments)