from typing import List

import torch
from torch import Size
from torch import Tensor


def block_diag(*args: Tensor) -> Tensor:
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """

    dims: List[int] = [t.dim() for t in args]
    assert set(dims) == {2}
    sizes: List[Size] = [t.size() for t in args]
    assert all([size[0] == size[1] for size in sizes])

    dtypes: List[torch.dtype] = [t.dtype for t in args]
    assert len(set(dtypes)) == 1

    new_t_size: int = sum([size[0] for size in sizes])
    new_t: Tensor = torch.zeros(new_t_size, new_t_size, dtype=dtypes.pop())

    offset: int = 0
    for i, t in enumerate(args):
        cur_dim: int = sizes[i][0]
        new_t[offset : offset + cur_dim, offset : offset + cur_dim] = t
        offset += cur_dim
    return new_t
