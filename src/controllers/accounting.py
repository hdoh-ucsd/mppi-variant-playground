"""Count calls to the existing plant; no dynamics or integration is reimplemented."""

import numpy as np


class CountedModel:
    def __init__(self, model):
        self.wrapped = model
        self.transitions = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def step(self, state, control, **kwargs):
        shape = np.broadcast_shapes(np.shape(state)[:-1], np.shape(control)[:-1])
        self.transitions += int(np.prod(shape)) if shape else 1
        return self.wrapped.step(state, control, **kwargs)
