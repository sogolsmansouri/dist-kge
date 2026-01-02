import numpy as np
import torch
from torch import Tensor
from typing import List, Union


def get_sp_po_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], num_entities: int, sp_index: dict, po_index: dict, targets: set
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    sp_coords = sp_index.get_all(batch[:, [0, 1]], targets=targets)
    po_coords = po_index.get_all(batch[:, [1, 2]], targets=targets)
    po_coords[:, 1] += num_entities
    coords = torch.cat(
        (
            sp_coords,
            po_coords
        )
    )

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if coords.numel() == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float, device=device)
        return torch.sparse_coo_tensor(indices, values, (nrows, ncols), device=device)

    indices = coords.long().t().contiguous()
    if device != "cpu":
        indices = indices.to(device)
    values = torch.full(
        (indices.shape[1],),
        float(value),
        dtype=torch.float,
        device=device,
    )
    return torch.sparse_coo_tensor(indices, values, (nrows, ncols), device=device).coalesce()
