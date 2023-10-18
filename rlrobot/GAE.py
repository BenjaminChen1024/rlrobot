# GAE.py
# Generalized Advantage Estimation

from typing import Any
import numpy as np

class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.n_workers = n_workers
        self.worker_steps = worker_steps

    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        advantages = np.zeros_like((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        last_value = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        return advantages