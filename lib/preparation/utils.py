import os


def check_columns_exist(columns, columns_list):
    if type(columns) == list:
        for column in columns:
            if column not in columns_list:
                raise KeyError("Names of columns aren't correct.")
    elif type(columns) == str:
        if columns not in columns_list:
            raise KeyError("Names of column aren't correct.")
    else:
        raise ValueError('Column name must have str type.')


def create_folder(path_to_folder):
    """Creates all folders contained in the path.

    Parameters
    ----------
    path_to_folder : str
        Path to the folder that should be created

    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
