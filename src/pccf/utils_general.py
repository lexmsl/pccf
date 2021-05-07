import os
from typing import List


def path_ext_low_with_dot(file_path):
    return os.path.splitext(os.path.split(file_path)[1])[1].lower()


def diff1(changes_list: List[int]) -> List[int]:
    distances = []
    for e in zip(changes_list[:-1], changes_list[1:]):
        distances.append(e[1] - e[0])
    return distances