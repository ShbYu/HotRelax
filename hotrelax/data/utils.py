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
from torch.utils.data import Dataset, Sampler

from ..utils import EnvPara
from .crystgraph import HotRelax
from .graph_feat import feat_dict


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


def _load_graph_feature_keys(
    feat_json: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
    """
    Load selected graph feature keys.

    Args:
        feat_json: Feature key list or a JSON file path storing the key list.

    Returns:
        Selected graph feature names, or None when all features should be used.
    """
    if feat_json is None:
        return None
    if isinstance(feat_json, list):
        return feat_json
    with open(feat_json) as file_obj:
        return json.load(file_obj)



def _flatten_graph_feature(value: np.ndarray) -> np.ndarray:
    """
    Flatten one graph feature into a 1D numpy array.

    Args:
        value: Scalar or array-like graph feature value.

    Returns:
        A flattened float array.
    """
    return np.asarray(value, dtype=np.float64).reshape(-1)



def _select_graph_features(
    all_feat: dict,
    select_key: Optional[List[str]],
) -> np.ndarray:
    """
    Build the final graph feature vector.

    Args:
        all_feat: Dictionary returned by feat_dict.
        select_key: Selected feature names. If None, use all features.

    Returns:
        A flattened graph-level feature vector.
    """
    if select_key is None:
        select_key = list(all_feat.keys())

    select_feat = []
    for key in select_key:
        select_feat.append(_flatten_graph_feature(all_feat[key]))
    if len(select_feat) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(select_feat, axis=0)



def _build_graph(
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
        graph.add_edge(int(i), int(j), vector=np.asarray(offset), direction=(int(i), int(j)))
    return graph



def _collate_cycle_basis(batch: List[dict]) -> torch.Tensor:
    """
    Build a block-diagonal cycle basis for a mini-batch.

    Args:
        batch: List of single-structure data dictionaries.

    Returns:
        A dense block-diagonal cycle basis tensor with shape [sum_cycle, sum_edge].
    """
    total_cycle = int(sum(item["cycle_basis"].shape[0] for item in batch))
    total_edge = int(sum(int(item["n_edges"].item()) for item in batch))
    dtype = batch[0]["cycle_basis"].dtype
    device = batch[0]["cycle_basis"].device
    cycle_basis = torch.zeros((total_cycle, total_edge), dtype=dtype, device=device)

    cycle_start = 0
    edge_start = 0
    for item in batch:
        current_basis = item["cycle_basis"]
        n_cycle, n_edge = current_basis.shape
        if n_cycle > 0 and n_edge > 0:
            cycle_basis[cycle_start: cycle_start + n_cycle, edge_start: edge_start + n_edge] = current_basis
        cycle_start += n_cycle
        edge_start += n_edge
    return cycle_basis


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
        del properties, spin
        idx_i, idx_j, distance, offsets = neighbor_list(
            "ijdS", atoms, cutoff, self_interaction=False
        )
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
        direct_pos_t = torch.tensor(
            atoms.info["direct_pos"], dtype=EnvPara.FLOAT_PRECISION
        )
        direct_cell_t = torch.tensor(
            atoms.info["direct_cell"], dtype=EnvPara.FLOAT_PRECISION
        )

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

        graph = None
        if add_feat or use_cycle:
            graph = _build_graph(len(atoms), idx_i, idx_j, offsets)

        if add_feat:
            all_feat = feat_dict(graph)
            select_key = _load_graph_feature_keys(feat_json)
            select_feat = _select_graph_features(all_feat, select_key)
            data["graph_feat"] = torch.tensor(
                select_feat, dtype=EnvPara.FLOAT_PRECISION
            ).view(1, -1)

        if use_cycle:
            hotR = HotRelax(atoms, atoms)
            hotR.read_graph(atoms, ori_G=graph)
            hotR.get_cycle()
            data["cycle_basis"] = torch.tensor(
                hotR.cycle_basis, dtype=EnvPara.FLOAT_PRECISION
            )
            data["cycle_offset"] = torch.tensor(
                hotR.cycle_offset, dtype=EnvPara.FLOAT_PRECISION
            )

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

    skip_keys = {"idx_i", "idx_j", "cycle_basis"}
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

    if "cycle_basis" in elem:
        coll_batch["cycle_basis"] = _collate_cycle_basis(batch)

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
