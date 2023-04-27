"""Abstract class for systems of coordinates."""

__all__ = ['Coordinates']


from abc import ABC, abstractmethod


class Coordinates(ABC):
    """System of coordinates."""

    domain_dimension: int
    codomain_dimension: int

    @abstractmethod
    def __call__(self, X):
        ...
