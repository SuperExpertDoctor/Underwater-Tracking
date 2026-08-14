# src/underwater_tracking/tracking/angles.py
import numpy as np


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi
