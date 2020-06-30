import timeit
import unittest

import torch

_arr = torch.randn([512, 256], dtype=torch.float)


def _without_cpu_call() -> None:
    sum_ = 0.0
    size = _arr.size()
    for i in range(size[0]):
        for j in range(size[1]):
            sum_ += _arr[i, j].item()


def _with_cpu_call() -> None:
    in_list = _arr.tolist()
    sum_ = 0.0
    for row in in_list:
        for num in row:
            sum_ += num


if __name__ == "__main__":
    unittest.main()
