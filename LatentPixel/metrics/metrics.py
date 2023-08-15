from typing import Any
import re

import torch

import editdistance
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    accuracy_score
)
from sklearn.feature_selection import r_regression
from scipy.stats import spearmanr

class Metric:

    def __init__(self) -> None:
        self.golden = []
        self.compare = []

    def accumulate(self, golden: Any, compare: Any) -> None:
        if isinstance(golden, list):
            self.golden += golden
            assert isinstance(compare, list)
            self.compare += compare
        else:
            self.golden.append(golden)
            self.compare.append(compare)
    
    def result(self) -> float:
        raise NotImplementedError('All metrics should implement this method')
    
    def metric_name(self) -> str:
        raise NotImplementedError('All metrics should implement this method')
    
    def __str__(self) -> str:
        return f'METRIC[{self.metric_name()}]: {self.result()}'


class EditDistance(Metric):

    def __init__(self) -> None:
        self.num_char = 0
        self.sum_dist = 0
        self.pattern = re.compile('\s+')

    def accumulate(self, golden: str | list[str], compare: str | list[str]) -> None:
        num_c, dist = self.distance(golden, compare)
        self.num_char += num_c
        self.sum_dist += dist

    def average_dist(self) -> float:
        return self.sum_dist / self.num_char

    def _distance(self, golden: str, compare: str) -> tuple[int, int]:
        golden = self.pattern.sub('', golden)
        compare = self.pattern.sub('', compare)
        return len(golden), editdistance.eval(golden, compare)
    
    def distance(self, golden: str | list[str], compare: str | list[str]) -> tuple[int, int]:
        if isinstance(golden, str):
            return self._distance(golden, compare)
        
        results = [self._distance(g, c) for g, c in zip(golden, compare)]
        results = list(zip(*results))

        return sum(results[0]), sum(results[1])
    

class Accuracy(Metric):

    def result(self) -> float:
        return accuracy_score(self.golden, self.compare)
    
    def metric_name(self) -> str:
        return 'Accuracy'
    

class F1(Metric):

    def result(self) -> float:
        return f1_score(self.golden, self.compare)
    
    def metric_name(self) -> str:
        return 'F1'


class MC(Metric):

    def result(self) -> float:
        return float(matthews_corrcoef(self.golden, self.compare))
    
    def metric_name(self) -> str:
        return 'Matthews_correlation'


class PC(Metric):

    def result(self) -> float:
        x = torch.tensor(self.compare).unsqueeze(1)
        y = torch.tensor(self.golden)
        return r_regression(x, y).tolist()[0]
    
    def metric_name(self) -> str:
        return 'Pearson_correlation'
    
class SC(Metric):

    def result(self) -> float:
        return float(spearmanr(self.golden, self.compare, nan_policy='omit').statistic)

    def metric_name(self) -> str:
        return 'Spearman_correlation'
