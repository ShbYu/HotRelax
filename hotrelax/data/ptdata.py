from .utils import AtomsDataset, register_dataset
from typing import List, Optional, Dict, Union, Tuple
import torch


PtDict = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Union[float, int]],
]


@register_dataset("pt")
class PtData(AtomsDataset):
    def __init__(
        self,
        ptdata: Union[PtDict, str, None] = None,
        indices: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> None:
        if ptdata is None:
            self.collect_data = None
            self.collated_number = None
            self.meta_data = {"n_data": 0}
        elif isinstance(ptdata, str):
            self.collect_data, self.collated_number, self.meta_data = torch.load(ptdata, weights_only=False)
        else:
            self.collect_data, self.collated_number, self.meta_data = ptdata
        self.indices = indices
        self.data_list = [None] * self.meta_data["n_data"]

    def __len__(self):
        if self.indices is None:
            return self.meta_data["n_data"]
        else:
            return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.indices is not None:
            idx = int(self.indices[idx])
        if idx < 0:
            idx = len(self) + idx
        if self.data_list[idx] is None:
            data = {}
            for k in self.collect_data:
                k_idx = self.collated_number[k]
                data[k] = self.collect_data[k][k_idx[idx] : k_idx[idx + 1]]
            self.data_list[idx] = data
        return self.data_list[idx]

    def extend(
        self,
        ptdata: Union[PtDict, str],
    ):
        if isinstance(ptdata, str):
            collect_data, collated_number, meta_data = torch.load(ptdata, weights_only=False)
        else:
            collect_data, collated_number, meta_data = ptdata
        if self.collect_data is None:
            self.collect_data, self.collated_number, self.meta_data = (
                collect_data,
                collated_number,
                meta_data,
            )
            self.data_list = [None] * len(self)
        else:
            assert (
                meta_data["cutoff"] == self.meta_data["cutoff"]
            ), "Cutoff of two set must be the same!"
            self.meta_data["n_data"] += meta_data["n_data"]
            self.data_list.extend([None] * meta_data["n_data"])
            for key in self.collect_data:
                self.collect_data[key] = torch.cat(
                    [self.collect_data[key], collect_data[key]], dim=0
                )
                self.collated_number[key] = torch.cat(
                    [
                        self.collated_number[key],
                        collated_number[key][1:] + self.collated_number[key][-1],
                    ],
                    dim=0,
                )
