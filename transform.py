import copy
import os
import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from hotrelax.data import ASEData
from hotrelax.data.utils import sparse_cycle_basis


# def _collate_fn(batch):
#     """
#     Collate one mini-batch during offline tensor transformation.

#     Args:
#         batch: List of single-structure data dictionaries.

#     Returns:
#         Collated tensor dictionary and per-key row counts.
#     """
#     batch = [_convert_sparse_cycle_in_sample(copy.deepcopy(data)) for data in batch]
#     coll_batch = {}
#     number = {}
#     for key in batch[0]:
#         coll_batch[key] = torch.cat([d[key] for d in batch], dim=0)
#         number[key] = [len(d[key]) for d in batch]
#     return coll_batch, number

def _collate_fn(batch):
    coll_batch = {}
    number = {}
    for key in batch[0]:
        coll_batch[key] = torch.cat([d[key] for d in batch], dim=0)
        number[key] = [len(d[key]) for d in batch]
    return coll_batch, number

def _convert_sparse_cycle_in_sample(data: dict) -> dict:
    """
    Replace dense cycle basis with sparse coordinate fields in one sample.

    Args:
        data: One sample dictionary.

    Returns:
        Updated sample dictionary using sparse cycle basis fields.
    """
    if "cycle_basis" not in data:
        return data
    sparse_basis = sparse_cycle_basis(data["cycle_basis"].cpu().numpy())
    data.pop("cycle_basis")
    data.update(sparse_basis)
    return data


def main(
    *args,
    cutoff,
    indices=None,
    datapath_u=None,
    datapath=None,
    properties=None,
    spin=False,
    max_neigh=20,
    add_graph_feat=False,
    feat_json=None,
    add_atom_feat=False,
    use_cycle=False,
    batchsize=32,
    num_workers=4,
    **kwargs,
):
    """
    Transform ASE trajectory data into collated tensor files.
    """
    if indices is not None:
        indices = np.loadtxt(indices, dtype=int)

    if isinstance(datapath_u, list) and len(datapath_u) == 1:
        datapath_u = datapath_u[0]
    dataset = ASEData(
        frames=datapath_u,
        cutoff=cutoff,
        properties=properties,
        spin=spin,
        indices=indices,
        max_neigh=max_neigh,
        add_graph_feat=add_graph_feat,
        feat_json=feat_json,
        add_atom_feat=add_atom_feat,
        use_cycle=use_cycle,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    # sample_template = _convert_sparse_cycle_in_sample(copy.deepcopy(dataset[0]))
    # data_list = {k: [] for k in sample_template}
    # number_list = {k: [0] for k in sample_template}

    data_list = {k: [] for k in dataset[0]}
    number_list = {k: [0] for k in dataset[0]}
    for batch_data, number in tqdm.tqdm(data_loader):
        for key in data_list:
            data_list[key].append(copy.deepcopy(batch_data[key]))
            number_list[key].extend(copy.deepcopy(number[key]))

    collated_data = {key: torch.cat(data_list[key], dim=0) for key in data_list}
    collated_number = {
        key: torch.cumsum(torch.tensor(number_list[key], dtype=torch.long), dim=0)
        for key in number_list
    }
    meta_data = {
        "cutoff": cutoff,
        "n_data": len(dataset),
        "max_neigh": max_neigh,
    }
    if add_graph_feat:
        graph_feat_mean = collated_data["graph_feat"].mean(dim=0)
        graph_feat_std = collated_data["graph_feat"].std(dim=0, unbiased=False)
        meta_data["graph_feat_mean"] = graph_feat_mean
        meta_data["graph_feat_std"] = graph_feat_std
    if add_atom_feat:
        atom_feat_mean = collated_data["atom_feat"].mean(dim=0)
        atom_feat_std = collated_data["atom_feat"].std(dim=0, unbiased=False)
        meta_data["atom_feat_mean"] = atom_feat_mean
        meta_data["atom_feat_std"] = atom_feat_std
    torch.save((collated_data, collated_number, meta_data), os.path.join(datapath, "data.pt"))


if __name__ == "__main__":
    data_root = sys.argv[1]
    num_workers = int(sys.argv[2])
    for mode in ["train", "val", "test"]:
        datapath_u = os.path.join(data_root, mode, "data.traj")
        datapath = os.path.join(data_root, mode)
        feat_path = os.path.join(data_root, "feat.json")
        main(
            cutoff=6.0,
            datapath_u=datapath_u,
            datapath=datapath,
            max_neigh=50,
            add_graph_feat=True,
            feat_json=feat_path,
            add_atom_feat=True,
            use_cycle=False,
            batchsize=4,
            num_workers=num_workers,
        )
