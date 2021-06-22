import pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import SensorsDataset

class SensorsDataModule(LightningDataModule):
    """
    LightningDataModule for training and evaluation pipelines

    Attributes
    ----------
    split_path: str
        path to train-valid split
    task: str
        type of task for dataloader: "hr", "har", "descriptor"
    ignore_train_ids: list
        indexes, that need to ignore in train, but it is requared in valid
    processing_params: dict
        params for preprocessing
    segment_params: dict = {'window_size': int, 'step_size': int }
        params for get_windows function
    limit: int
        number of experiments to process
    ignore_zero: int
        if it need to ignore 0-class in dataset
    batch_size: int
    num_workers: int

    Methods
    -------
    split_experiments()
        loads split and initializes train pathes and valid pathes
    train_dataloader()
        creates a dataloader for training
    val_dataloader()
        creates a dataloader for validation
    """
    name = 'SensorsHRDataModule'

    def __init__(self,
                 split_path: str,
                 task: str='hr',
                 ignore_train_ids: list=[],
                 processing_params: dict={'sampling_rate': 25, 'slice_for_hr_label': (0.5, 1)},
                 segment_params: dict={'window_size': 500, 'step_size': 50},
                 limit: int=None,
                 batch_size: int=64,
                 num_workers: int=1,
                 ignore_zero: bool=False,
                 *args,
                 **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert task in ['hr', 'har', 'descriptor'], 'task not supported. use "hr", "har", or "descriptor"'

        self.task_type = task
        self.ignore_train_ids = ignore_train_ids
        self.processing_params = processing_params
        self.segment_params = segment_params
        self.limit = limit
        self.split_path = split_path
        self.split_experiments()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.ignore_zero = ignore_zero
        
        self.train_dataset = SensorsDataset(task_type=self.task_type,
                                            ignore_ids=ignore_train_ids,
                                            experiments_paths=self.train_paths,
                                            processing_params=self.processing_params,
                                            segment_params=self.segment_params,
                                            limit=self.limit,
                                            ignore_zero=self.ignore_zero)
        self.train_dataset.process_experiments()
        self.valid_dataset = SensorsDataset(task_type=self.task_type,
                                            ignore_ids=[],
                                            experiments_paths=self.valid_paths,
                                            processing_params=self.processing_params,
                                            segment_params=self.segment_params,
                                            limit=self.limit,
                                            ignore_zero=self.ignore_zero)
        self.valid_dataset.process_experiments()

    def split_experiments(self):
        # To do: must change this part because of more data (use info about user)

        split = pickle.load(open(self.split_path, 'rb'))
        self.train_paths = split['train']
        self.valid_paths = split['valid']

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader
