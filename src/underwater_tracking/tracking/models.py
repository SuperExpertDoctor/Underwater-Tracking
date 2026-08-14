# src/underwater_tracking/tracking/models.py
import numpy as np


def constant_turn(state: np.ndarray, dt: float, commanded_turn: float) -> np.ndarray:
    x, y, vx, vy, _ = state
    angle = commanded_turn * dt
    c, s = np.cos(angle), np.sin(angle)
    nvx, nvy = c * vx - s * vy, s * vx + c * vy
    return np.array([x + nvx * dt, y + nvy * dt, nvx, nvy, commanded_turn])


def bearing_measurement(state: np.ndarray, observer_xy: np.ndarray) -> float:
    return float(np.arctan2(state[1] - observer_xy[1], state[0] - observer_xy[0]))
