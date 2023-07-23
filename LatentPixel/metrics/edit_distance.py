import editdistance
import re


class EditDistance:

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
