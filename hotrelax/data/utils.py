import abc
import copy
import json
import logging
from typing import Iterator, List, Optional, Union

import networkx as nx
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from ..utils import EnvPara


log = logging.getLogger(__name__)

__all__ = ["AtomsDataset", "atoms_collate_fn", "MaxNodeSampler", "MaxEdgeSampler"]

dataset_mapping = {}


def register_dataset(name: str):
    """
    Register a dataset class.

    Args:
        name: Dataset type name used in config.

    Returns:
        Decorator that stores the dataset class in the registry.
    """
    def decorator(cls):
        dataset_mapping[name] = cls
        return cls

    return decorator


# TODO: offset and scaling for different condition
class AtomsDataset(Dataset, abc.ABC):

    @staticmethod
    def atoms_to_data(
        atoms: Atoms,
        cutoff: float = 6.0,
        properties: Optional[List[str]] = None,
        spin: bool = False,
        max_neigh: Optional[int] = None,
    ):
        """
        Convert one ASE atoms object into model input tensors.

        Args:
            atoms: Input atomic structure.
            cutoff: Neighbor cutoff radius.
            properties: Kept for API compatibility with existing dataset callers.
            spin: Kept for API compatibility with existing dataset callers.
            max_neigh: Maximum number of neighbors per center atom.
            add_feat: Whether to attach graph-level handcrafted features.
            feat_json: Feature selection list or JSON file path.
            use_cycle: Whether to attach cycle basis and cycle offset tensors.

        Returns:
            A dictionary containing tensors required by the model and loss.
        """
        idx_i, idx_j, distance, offsets = neighbor_list("ijdS", atoms, cutoff, self_interaction=False)
        offsets = np.array(offsets)

        if max_neigh is not None:
            filter_index = []
            for i in range(len(atoms)):
                c_index = (idx_i == i).nonzero()[0]
                c_sorted = np.argsort(distance[c_index])[: max_neigh]
                filter_index.append(c_index[c_sorted])
            filter_index = np.concatenate(filter_index)
            idx_i = idx_i[filter_index]
            idx_j = idx_j[filter_index]
            offsets = offsets[filter_index]

        pos_u = torch.tensor(atoms.positions, dtype=EnvPara.FLOAT_PRECISION)
        cell_u = torch.tensor(
            np.array(atoms.get_cell()), dtype=EnvPara.FLOAT_PRECISION
        ).view(1, 3, 3)
        direct_pos_t = torch.tensor(atoms.info["direct_pos"], dtype=EnvPara.FLOAT_PRECISION)
        direct_cell_t = torch.tensor(atoms.info["direct_cell"], dtype=EnvPara.FLOAT_PRECISION)

        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long),
            "idx_i": torch.tensor(idx_i, dtype=torch.long),
            "idx_j": torch.tensor(idx_j, dtype=torch.long),
            "pos_u": pos_u,
            "n_atoms": torch.tensor([len(atoms)], dtype=torch.long),
            "n_edges": torch.tensor([len(idx_i)], dtype=torch.long),
            "offset": torch.tensor(offsets, dtype=EnvPara.FLOAT_PRECISION),
            "cell_u": cell_u,
            "ats_index": torch.tensor([atoms.info["index"]], dtype=torch.long),
            "direct_pos_t": direct_pos_t,
            "direct_cell_t": direct_cell_t.view(1, 3, 3),
        }

        return data

    def __init__(
        self,
        indices: Optional[List[int]] = None,
        cutoff: float = 6.0,
    ) -> None:
        self.indices = indices
        self.cutoff = cutoff

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass

    def subset(self, indices: List[int]):
        """
        Build a shallow subset dataset.

        Args:
            indices: Selected sample indices.

        Returns:
            A copied dataset view with updated indices.
        """
        ds = copy.copy(self)
        if ds.indices:
            ds.indices = [ds.indices[i] for i in indices]
        else:
            ds.indices = indices
        return ds


def atoms_collate_fn(batch):
    """
    Collate atomistic samples into one mini-batch.

    Args:
        batch: List of single-structure data dictionaries.

    Returns:
        A batched tensor dictionary for model forward and loss computation.
    """
    elem = batch[0]
    coll_batch = {}

    skip_keys = {"idx_i", "idx_j"}
    for key in elem:
        if key not in skip_keys:
            coll_batch[key] = torch.cat([d[key] for d in batch], dim=0)

    # idx_i and idx_j should to be converted like
    # [0, 0, 1, 1] + [0, 0, 1, 2] -> [0, 0, 1, 1, 2, 2, 3, 4]
    for key in ["idx_i", "idx_j"]:
        coll_batch[key] = torch.cat(
            [
                batch[i][key] + torch.sum(coll_batch["n_atoms"][:i])
                for i in range(len(batch))
            ],
            dim=0,
        )

    coll_batch["batch"] = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch["n_atoms"].to(torch.long), dim=0
    )
    return coll_batch


class MaxNodeSampler(Sampler):

    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:

        self.sampler = sampler
        self.batch_size = batch_size
        self.node_num = [0] * len(self.sampler)
        for idx in self.sampler:
            self.node_num[idx] = self.sampler.data_source[idx]["n_atoms"]
        if max(self.node_num) > self.batch_size:
            raise Exception(
                f"Max atoms in one structure {max(self.node_num)} > batch_size {self.batch_size}"
            )

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        node_in_batch = 0
        for idx in self.sampler:
            if node_in_batch + self.node_num[idx] > self.batch_size:
                yield batch
                node_in_batch = 0
                batch = []
            node_in_batch += self.node_num[idx]
            batch.append(idx)
        if len(batch) > 0:
            yield batch


class MaxEdgeSampler(Sampler):

    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:

        self.sampler = sampler
        self.batch_size = batch_size
        self.edge_num = [0] * len(self.sampler)
        for idx in self.sampler:
            self.edge_num[idx] = self.sampler.data_source[idx]["n_edges"]
        if max(self.edge_num) > self.batch_size:
            raise Exception(
                f"Max edge in one structure {max(self.edge_num)} > batch_size {self.batch_size}"
            )

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        edge_in_batch = 0
        for idx in self.sampler:
            if edge_in_batch + self.edge_num[idx] > self.batch_size:
                yield batch
                edge_in_batch = 0
                batch = []
            edge_in_batch += self.edge_num[idx]
            batch.append(idx)

        if len(batch) > 0:
            yield batch
