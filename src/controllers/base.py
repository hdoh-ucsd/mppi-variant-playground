from abc import ABC, abstractmethod


class Controller(ABC):
    """Observations [px,py,heading,speed]; commands [acceleration,steering]."""

    @abstractmethod
    def reset(self, initial_state, seed): ...

    @abstractmethod
    def command(self, observed_state): ...

    @abstractmethod
    def get_diagnostics(self): ...
