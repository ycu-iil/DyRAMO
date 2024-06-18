"""
This script was retrieved from https://github.com/molecule-generator-collection/ChemTSv2/blob/4980a850bc2411fcdebe2adaab87609c2d75972e/reward/reward.py
"""

from abc import ABC, abstractmethod
from typing import List, Callable

from rdkit.Chem import Mol


class Reward(ABC):
    @staticmethod
    @abstractmethod
    def get_objective_functions(conf: dict) -> List[Callable[[Mol], float]]:
        raise NotImplementedError('Please check your reward file')
    
    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError('Please check your reward file')

class BatchReward(ABC):
    @staticmethod
    @abstractmethod
    def get_batch_objective_functions() -> List[Callable[[List[Mol], List[dict]], float]]:
        raise NotImplementedError('Please check your reward file')
    
    @staticmethod
    @abstractmethod
    def calc_reward_from_objective_values(values: List[float], conf: dict) -> float:
        raise NotImplementedError('Please check your reward file')
