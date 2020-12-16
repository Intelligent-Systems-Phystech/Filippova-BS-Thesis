from acclib.postprocess import remove_bias
from acclib.nets import TrajectoryEstimator


def get_trajectory(acc, gyro, path_to_model, device='cpu', postprocess=True):
    """Predicts trajectory.

    Parameters
    ----------
    acc : pd.DataFrame
        Accelerometer data.
    gyro : pd.DataFrame
        Gyroscope data (with the same timestamps).
    path_to_model : str
        Path to model's folder.
    device : str, optional (default='cpu')
        This parameter defines device.
    postprocess : bool, optional (default=True)
        If true then trajectory becomes closed. Otherwise not.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns `tmsp`, `x`, and `y` that contains predicted trajectory.

    """
    model = TrajectoryEstimator.load(path=path_to_model, device=device)

    trajectory_pred = model.predict(acc, gyro)
    if postprocess:
        trajectory_pred = remove_bias(trajectory_pred, coordinate_columns=("x_avg", "y_avg"))

    return trajectory_pred
