"""Minimal space definitions (no gymnasium dependency)."""
import numpy as np


class Box:
    """Continuous box space."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x, dtype=self.dtype)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def clip(self, x):
        return np.clip(x, self.low, self.high).astype(self.dtype)

    def __repr__(self):
        return f"Box({self.low}, {self.high}, shape={self.shape}, dtype={self.dtype})"
