# saving and loading sww meta data

import numpy as np


def save_sensing_plan_nparray(sensing_plan_name, sensing_plan_nparray):

    np.save(f"/home/banana/Desktop/B210_SWW/data_sensing_plans/{sensing_plan_name}.npy", sensing_plan_nparray)


def load_sensing_plan_nparray(sensing_plan_name):
    return np.load(f"/home/banana/Desktop/B210_SWW/data_sensing_plans/{sensing_plan_name}.npy", allow_pickle=True)


def save_to_matlab(sensing_plan_name):
    pass
