from abc import ABC, abstractmethod


class Controller(ABC):
    """Common reset/command/diagnostics interface; state and inputs follow the supplied model."""

    @abstractmethod
    def reset(self, initial_state, seed): ...

    @abstractmethod
    def command(self, observed_state): ...

    @abstractmethod
    def get_diagnostics(self): ...
