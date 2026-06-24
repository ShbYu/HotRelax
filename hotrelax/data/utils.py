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
from .crystgraph import HotRelax
from .graph_feat import compute_feature


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

def select_graph_features(graph: nx.MultiDiGraph, select_key: List[str]) -> np.ndarray:
    """
    Build the final graph feature vector.

    Args:
        select_key: Selected feature names.

    Returns:
        A flattened graph-level feature vector.
    """
    select_feat = []
    simple_graph = nx.Graph(graph)
    for key in select_key:
        select_feat.append(np.asarray(compute_feature(key, graph, simple_graph)).reshape(-1))
    return np.concatenate(select_feat, axis=0)

def build_graph(
    n_atoms: int,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    offsets: np.ndarray,
) -> nx.MultiGraph:
    """
    Build the training graph used for graph features and cycle data.

    Args:
        n_atoms: Number of atoms in the structure.
        idx_i: Edge source indices.
        idx_j: Edge destination indices.
        offsets: Periodic image offsets for each edge.

    Returns:
        A networkx MultiGraph with vector and direction edge attributes.
    """
    graph = nx.MultiGraph()
    for i in range(n_atoms):
        graph.add_node(i)
    for i, j, offset in zip(idx_i, idx_j, offsets):
        graph.add_edge(i, j, vector=np.asarray(offset), direction=(i, j))
    return graph

def sparse_cycle_basis(cycle_basis: np.ndarray) -> dict:
    """
    Convert a dense cycle basis matrix into sparse coordinate tensors.

    Args:
        cycle_basis: Dense cycle basis array with shape [n_cycle, n_edge].

    Returns:
        Dictionary containing sparse row, column, value, and shape tensors.
    """
    cycle_basis = np.asarray(cycle_basis)
    row, col = np.nonzero(cycle_basis)
    value = cycle_basis[row, col].astype(np.int8)
    return {
        "cycle_basis_row": torch.tensor(row, dtype=torch.long),
        "cycle_basis_col": torch.tensor(col, dtype=torch.long),
        "cycle_basis_value": torch.tensor(value, dtype=torch.int8),
        "cycle_basis_shape": torch.tensor([[cycle_basis.shape[0], cycle_basis.shape[1]]], dtype=torch.long),
    }


def collate_cycle_tensors(batch: List[dict]) -> dict:
    """
    Pad cycle tensors in a mini-batch and build masks.

    Args:
        batch: List of single-structure data dictionaries.

    Returns:
        Dictionary containing padded cycle tensors and valid-position masks.
    """
    batch_size = len(batch)
    offset_dtype = batch[0]["cycle_offset"].dtype
    offset_device = batch[0]["cycle_offset"].device

    offset_list = [item["cycle_offset"] for item in batch]
    cycle_offset = pad_sequence(offset_list, batch_first=True)

    n_cycles = torch.tensor(
        [offset.shape[0] for offset in offset_list],
        dtype=torch.long,
        device=offset_device,
    )
    n_edges = torch.tensor(
        [int(item["n_edges"].item()) for item in batch],
        dtype=torch.long,
        device=offset_device,
    )

    max_cycle = int(n_cycles.max().item())
    max_edge = int(n_edges.max().item())

    cycle_mask = (torch.arange(max_cycle, device=offset_device)[None, :] < n_cycles[:, None])
    edge_mask = (torch.arange(max_edge, device=offset_device)[None, :] < n_edges[:, None])

    cycle_basis = torch.zeros(
        (batch_size, max_cycle, max_edge),
        dtype=offset_dtype,
        device=offset_device,
    )

    if "cycle_basis_row" in batch[0]:
        nnz = torch.tensor(
            [item["cycle_basis_value"].numel() for item in batch],
            dtype=torch.long,
            device=offset_device,
        )
        total_nnz = int(nnz.sum().item())

        if total_nnz > 0:
            batch_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=offset_device),
                nnz,
            )
            rows = torch.cat(
                [item["cycle_basis_row"].to(offset_device, dtype=torch.long) for item in batch],
                dim=0,
            )
            cols = torch.cat(
                [item["cycle_basis_col"].to(offset_device, dtype=torch.long) for item in batch],
                dim=0,
            )
            values = torch.cat(
                [item["cycle_basis_value"].to(offset_device, dtype=offset_dtype) for item in batch],
                dim=0,
            )
            cycle_basis[batch_ids, rows, cols] = values
    else:
        for batch_idx, item in enumerate(batch):
            current_basis = item["cycle_basis"].to(offset_dtype)
            n_cycle = int(item["cycle_offset"].shape[0])
            n_edge = int(item["n_edges"].item())
            cycle_basis[batch_idx, :n_cycle, :n_edge] = current_basis[:, :n_edge]

    return {
        "cycle_basis": cycle_basis,
        "cycle_offset": cycle_offset,
        "cycle_mask": cycle_mask,
        "edge_mask": edge_mask,
    }


# TODO: offset and scaling for different condition
class AtomsDataset(Dataset, abc.ABC):

    @staticmethod
    def atoms_to_data(
        atoms: Atoms,
        cutoff: float = 6.0,
        properties: Optional[List[str]] = None,
        spin: bool = False,
        max_neigh: Optional[int] = None,
        add_feat: bool = False,
        feat_json: Optional[Union[str, List[str]]] = None,
        use_cycle: bool = False,
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
        offsets = np.asarray(offsets)

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

        if add_feat or use_cycle:
            graph = build_graph(len(atoms), idx_i, idx_j, offsets)

        if add_feat:
            with open(feat_json) as file_obj:
                select_key = json.load(file_obj)
            select_feat = select_graph_features(graph, select_key)
            data["graph_feat"] = torch.tensor(select_feat, dtype=EnvPara.FLOAT_PRECISION).view(1, -1)

        if use_cycle:
            hotR = HotRelax(atoms, atoms)
            hotR.read_graph(atoms, ori_G=graph)
            hotR.get_cycle()
            data.update(sparse_cycle_basis(hotR.cycle_basis))
            data["cycle_offset"] = torch.tensor(hotR.cycle_offset, dtype=EnvPara.FLOAT_PRECISION)

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

    skip_keys = {"idx_i", "idx_j", "cycle_basis", "cycle_offset", "cycle_basis_row", "cycle_basis_col", "cycle_basis_value", "cycle_basis_shape"}
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

    if "cycle_basis_row" in elem or "cycle_basis" in elem:
        cycle_tensors = collate_cycle_tensors(batch)
        coll_batch.update(cycle_tensors)

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
