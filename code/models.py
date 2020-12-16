import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from acclib.geometry import apply_quats
from acclib.nets.utils import get_traj_from_v_t, process_data
from acclib.nets.training import get_loaders, train
from acclib.postprocess import fix_trajectories
from acclib.show.prediction import show_trajectories
from acclib.save_load_utils import create_folder
from acclib.nets.models import LSTM, ResNet, ResNetLSTM

HYPER_RL = [6, 64, 64, 128, 256, 2, 100, 0.5]
HYPER_R = [6, 64, 64, 128, 256, 512, 0.5]
HYPER_L = [6, 100, 3, 0.5]


class TrajectoryEstimator:
    """
    Provides easy and convenient interface for training and evaluating neural networks in
    IMU navigation.

    """

    def __init__(self, name, model, window_size=200, step=50, trend=True, predict_correction=True,
                 train_correction=False, frequency=200, device='cpu', path="../experiments",
                 **kwargs):
        """Interface for neural network usage in IMU navigation.

        Parameters
        ----------
        name : string
            Name of the folder where all the results will be stored.
        model : torch.nn.Module
            A neural network that predicts velocities.
        window_size : int
            Window size for data preparation.
        step : int
            Step size for data preparation.
        trend : bool
            Whether to use gyroscope trend in world c.s. as an input channel.
        frequency : int, optional (default=200)
            Hz.
        device : string
            Device for a model.
        path : str, optional (default="../experiments")
            Path to the experiments' folder.

        """
        self.name = name
        self.window_size = window_size
        self.step = step
        self.device = torch.device(device)
        self.model = model
        self.model = self.model.to(device)
        self.predict_correction = predict_correction
        self.train_correction = train_correction
        self.trend = trend
        self.frequency = frequency
        self.train_params = None
        self.path = path

    def train_model(self, train_ind, valid_ind, storage, n_epochs=20, lr=1e-3, batch_size=128,
                    verbose=True, save_loss=False):
        """Trains a model of estimator.

        Parameters
        ----------
        train_ind : list of strings
            Names of the experiments in train set.
        valid_ind : list of strings
            Names of the experiments in validation set.
        storage : object of class Storage
            Storage which contains the experiments.
        n_epochs : int, optional (default=20)
            Number of epochs.
        lr : float, optional (default=0.001)
            Learning rate.
        batch_size : int, optional (default=128)
            Batch size.
        verbose : bool
            Whether to print loss on different epochs.
        save_loss : bool
            Whether to save loss in figure and csv.

        Returns
        -------
        pd.DataFrame
            DataFrame with a x_avg, y_avg, t columns which is trajectory of the object.

        """
        train_params = {'train_ind': train_ind,
                        'n_epochs': n_epochs,
                        'lr': lr,
                        'batch_size': batch_size}
        self.train_params = train_params
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        train_loader, valid_loader = get_loaders(train_ind, valid_ind, self.trend, self.window_size,
                                                 self.step,
                                                 self.train_correction, self.predict_correction,
                                                 storage, self.frequency, batch_size)

        train_loss, valid_loss = train(self.model, optimizer, train_loader, valid_loader,
                                       criterion=criterion, device=self.device,
                                       n_epochs=n_epochs, writer=None, verbose=verbose,
                                       scheduler=scheduler)
        if save_loss:
            create_folder(f"{self.path}/{self.name}")
            fig, ax = plt.subplots(1, 1)
            ax.plot(train_loss)
            ax.plot(valid_loss)
            plt.close()
            fig.savefig(f"{self.path}/{self.name}/loss.png")
            pd.DataFrame([train_loss, valid_loss]).to_csv(
                f"{self.path}/{self.name}/loss.csv")

    def predict(self, acc, gyro):
        """Predicts trajectory of the object woth model and acc, gyro data.

        Parameters
        ----------
        acc : pd.DataFrame
            DataFrame of accelerometer readings.
        gyro : pd.DataFrame
            DataFrame of gyroscope readings.

        Returns
        -------
        pd.DataFrame
            DataFrame with a x_avg, y_avg, t columns which is trajectory of the object.

        """
        if self.model is None:
            raise Exception('There is no model to use.')
        x, t, q = process_data(acc, gyro, window_size=self.window_size,
                               step=self.step, trend=self.trend,
                               q_correction=self.predict_correction, frequency=self.frequency)
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            v_pred = self.model(x).cpu().data.numpy()

        v_pred = np.hstack([v_pred, np.zeros((v_pred.shape[0], 1))])
        v_pred = apply_quats(v_pred, q) / (self.window_size / self.frequency)
        trajectory_pred = get_traj_from_v_t(v_pred, t)

        return trajectory_pred

    def evaluate(self, names, storage, print_metrics=False, save=True,
                 save_folder=None, save_path=None):
        """Evaluate model, save predictions, figures and metrics

        Parameters
        ----------
        names : list od strings
            Names of the experiments in validation set.
        storage : object of class Storage
            Storage which contains the experiments.
        print_metrics : bool
            Whether to print mean scores of the model on each object.
        save : bool
            Whether to save figures and metrics.
        save_folder : str, optional (default=None)
            Name of the folder in model's folder to save evaluation results.
        save_path : str, optional (default=None)
            Full path to folder to save evaluation results.

        Returns
        -------

        """
        metrics = []
        if save_path is None:
            if save_folder is None:
                save_path = f"{self.path}/{self.name}/evaluate"
            else:
                save_path = f"{self.path}/{self.name}/{save_folder}"
        path_fig = save_path + '/figures'
        path_preds = save_path + '/trajectories'
        if save:
            create_folder(save_path)
            create_folder(path_fig)
            create_folder(path_preds)

        for name in tqdm(names):
            acc, gyro = storage[name, 'acc'], storage[name, 'gyro']
            trajectory = storage[name, 'trajectory']
            prediction = self.predict(acc, gyro)
            traj_fix, pred_fix = fix_trajectories(trajectory, prediction,
                                                  coordinate_columns=('x_avg', 'y_avg'))
            fig, met = show_trajectories(traj_fix, pred_fix, coordinate_columns=('x_avg', 'y_avg'),
                                         show=False)
            if save:
                fig.savefig(path_fig + '/' + name + '.png')
                prediction.to_csv(path_preds + '/' + name + '.csv')
            metrics.append(met)
        metrics = pd.DataFrame(metrics, index=names)
        if save:
            metrics.to_csv(save_path + '/' + 'metrics.csv')
        if print_metrics:
            print(metrics.mean())

    def save(self):
        """Save params_dict and model to 'self.path/self.name'

        Returns
        -------

        """
        path = f"{self.path}/{self.name}"
        create_folder(path)
        parameters = {key: value for key, value in self.__dict__.items() if
                      not key.startswith('__') and not callable(key) and key != "model"}
        model_params = self.model.get_params()
        parameters["model_params"] = model_params
        with open(path + '/parameters_dict.pkl', 'wb') as f:
            pickle.dump(parameters, f)
        torch.save(self.model.to(torch.device("cpu")).state_dict(), path + '/model')
        self.model.to(self.device)

    def set_device(self, device="cpu"):
        """Sets device.

        Parameters
        ----------
        device : string, optional (default="cpu")
            Device.

        Returns
        -------

        """
        self.device = torch.device(device)
        self.model.to(self.device)

    @staticmethod
    def load(path, device="cpu"):
        """Load TrajectoryEstimator from a directory path

        Parameters
        ----------
        path : string
            Path to a folder where to create a new folder with parameters of estimator.
        device : string, optional (default="cpu")
            Device.

        Returns
        -------
        object of class TrajectoryEstimator
            Estimator.

        """
        with open(path + '/parameters_dict.pkl', 'rb') as f:
            params = pickle.load(f)
        params["device"] = device
        if params["model_params"]["class_name"] == "LSTM":
            model = LSTM(params["model_params"]["init_params"])
        elif params["model_params"]["class_name"] == "ResNet":
            model = ResNet(params["model_params"]["init_params"])
        elif params["model_params"]["class_name"] == "ResNetLSTM":
            model = ResNetLSTM(params["model_params"]["init_params"])
        else:
            raise ValueError("No class for such model.")
        model.load_state_dict(torch.load(path + '/model', map_location=params["device"]))
        estimator = TrajectoryEstimator(model=model, **params)

        return estimator
