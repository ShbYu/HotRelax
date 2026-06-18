from ast import Dict
import os, logging
from typing import Union, List, Tuple, Dict
from . import *
import numpy as np
from ase.io import read
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from .utils import MaxEdgeSampler, MaxNodeSampler, atoms_collate_fn, AtomsDataset


log = logging.getLogger(__name__)

dataparser_mapping = {}


def register_dataparser(name: str):
    def decorator(cls):
        dataparser_mapping[name] = cls
        return cls

    return decorator


class DatasetParser:
    def __init__(self, p_dict: Dict):
        self.p_dict = p_dict
        self.data_dict = p_dict["Data"]

    def get_datapath(self) -> Union[str, List[str], None]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'get_datapath'!"
        )

    def get_dataset(self) -> AtomsDataset:
        return get_dataset(
            datapath=self.get_datapath(),
            cutoff=self.p_dict['cutoff'],
            properties=self.p_dict['Train']['targetProp'],
            spin=self.p_dict['Model']['Spin'],
            datatype=self.data_dict["type"],
        )

    def split_dataset(self, dataset: AtomsDataset) -> Tuple[AtomsDataset, AtomsDataset]:
        if ("trainSplit" in self.data_dict) and ("testSplit" in self.data_dict):
            return self.split_by_index(dataset)
        if ("trainNum" in self.data_dict) and (("testNum" in self.data_dict)):
            return self.split_by_number(dataset)
        if ("trainSet" in self.data_dict) and ("testSet" in self.data_dict):
            return self.split_by_set(dataset)
        raise ValueError("Unkonwn split method!")

    def split_by_index(
        self, dataset: AtomsDataset
    ) -> Tuple[AtomsDataset, AtomsDataset]:
        log.info(
            f"Load split from {self.data_dict['trainSplit']} and {self.data_dict['testSplit']}"
        )
        train_idx = np.loadtxt(self.data_dict["trainSplit"], dtype=int)
        test_idx = np.loadtxt(self.data_dict["testSplit"], dtype=int)
        return dataset.subset(train_idx), dataset.subset(test_idx)

    def split_by_number(
        self, dataset: AtomsDataset
    ) -> Tuple[AtomsDataset, AtomsDataset]:
        log.info(
            f"Random split, train num: {self.data_dict['trainNum']}, test num: {self.data_dict['testNum']}"
        )
        assert self.data_dict['trainNum'] + self.data_dict['testNum'] <= len(
            self.dataset
        )
        idx = np.random.choice(
            len(dataset),
            self.data_dict['trainNum'] + self.data_dict['testNum'],
            replace=False,
        )
        train_idx = idx[: self.data_dict['trainNum']]
        test_idx = idx[self.data_dict['trainNum'] :]
        return dataset.subset(train_idx), dataset.subset(test_idx)

    def split_by_set(self, dataset: AtomsDataset) -> Tuple[AtomsDataset, AtomsDataset]:
        raise ValueError(
            "Now trainSet and testSet only support datatype 'ase' and 'dpmd'!"
        )

    def get_dataloader(
        self, dataset: AtomsDataset, batch: int, shuffle: bool = True
    ) -> DataLoader:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        if self.data_dict["batchType"] == "structure":
            batch_sampler = BatchSampler(
                sampler,
                batch_size=self.data_dict["trainBatch"],
                drop_last=False,
            )
        elif self.data_dict["batchType"] == "edge":
            batch_sampler = MaxEdgeSampler(sampler, batch_size=batch)
        elif self.data_dict["batchType"] == "node":
            batch_sampler = MaxNodeSampler(sampler, batch_size=batch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=atoms_collate_fn,
            num_workers=self.data_dict["numWorkers"],
            pin_memory=self.data_dict["pinMemory"],
        )
        log.debug(f'numWorkers: {self.data_dict["numWorkers"]}')
        return dataloader


@register_dataparser("ase")
class ASEParser(DatasetParser):

    def get_datapath(self) -> Union[str, List[str], None]:
        if "name" in self.data_dict:
            return os.path.join(self.data_dict['path'], self.data_dict['name'])
        return None

    def get_dataset(self):
        return ASEData(
            datapath=self.get_datapath(),
            cutoff=self.p_dict['cutoff'],
            properties=self.p_dict['Train']['targetProp'],
            spin=self.p_dict['Model']['Spin'],
        )

    def split_by_set(self, dataset):
        already_in = len(dataset)
        dataset.extend(os.path.join(self.data_dict['path'], self.data_dict['trainSet']))
        train_idx = [i for i in range(already_in, len(dataset))]
        dataset.extend(os.path.join(self.data_dict['path'], self.data_dict['testSet']))
        test_idx = [i for i in range(train_idx[-1] + 1, len(dataset))]
        return dataset.subset(train_idx), dataset.subset(test_idx)


@register_dataparser("ase-db")
class ASEDBParser(DatasetParser):

    def get_datapath(self) -> Union[str, List[str], None]:

        if "name" in self.data_dict:
            return os.path.join(self.data_dict['path'], self.data_dict['name'])
        raise ValueError("For ase-db data, 'name' must be provided in data_dict.")

    def get_dataset(self):
        return ASEDBData(
            datapath=self.get_datapath(),
            cutoff=self.p_dict['cutoff'],
            properties=self.p_dict['Train']['targetProp'],
            spin=self.p_dict['Model']['Spin'],
        )


@register_dataparser("rmd17")
class RMD17Parser(DatasetParser):

    def get_datapath(self) -> Union[str, List[str], None]:

        if "name" in self.data_dict:
            return os.path.join(
                self.data_dict['path'], f"rmd17_{self.data_dict['name']}.npz"
            )
        raise ValueError("For ase-db data, 'name' must be provided in data_dict.")

    def get_dataset(self):
        return RevisedMD17(
            datapath=self.get_datapath(),
            cutoff=self.p_dict['cutoff'],
        )


@register_dataparser("dpmd")
class DeePMDParser(DatasetParser):

    def get_datapath(self) -> Union[str, List[str], None]:

        if "name" in self.data_dict:
            if isinstance(name, str):
                name = [name]
            return [
                os.path.join(self.data_dict['path'], n) for n in self.data_dict['name']
            ]

        raise ValueError("For ase-db data, 'name' must be provided in data_dict.")

    def get_dataset(self):
        return DeePMDData(
            datapath=self.get_datapath(),
            cutoff=self.p_dict['cutoff'],
            properties=self.p_dict['Train']['targetProp'],
            spin=self.p_dict['Model']['Spin'],
        )

    def split_by_set(self, dataset):
        already_in = len(dataset)
        train_list = [
            os.path.join(self.data_dict['path'], name)
            for name in self.data_dict['trainSet']
        ]
        dataset.extend(datapath=train_list)
        train_idx = [i for i in range(already_in, len(dataset))]
        test_list = [
            os.path.join(self.data_dict['path'], name)
            for name in self.data_dict['testSet']
        ]
        dataset.extend(datapath=test_list)
        test_idx = [i for i in range(train_idx[-1] + 1, len(dataset))]
        return dataset.subset(train_idx), dataset.subset(test_idx)


@register_dataparser("pt")
class PtDataParser(DatasetParser):

    def get_datapath(self) -> Union[str, List[str], None]:
        if "name" in self.data_dict:
            return os.path.join(self.data_dict['path'], self.data_dict['name'])
        return None

    def get_dataset(self):
        return PtData(
            ptdata=self.get_datapath(),
        )

    def split_by_set(self, dataset):
        already_in = len(dataset)
        dataset.extend(os.path.join(self.data_dict['path'], self.data_dict['trainSet']))
        train_idx = [i for i in range(already_in, len(dataset))]
        dataset.extend(os.path.join(self.data_dict['path'], self.data_dict['testSet']))
        test_idx = [i for i in range(train_idx[-1] + 1, len(dataset))]
        return dataset.subset(train_idx), dataset.subset(test_idx)
