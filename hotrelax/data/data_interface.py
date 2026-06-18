import logging
import numpy as np
from copy import copy
from typing import Optional, List, Dict, Tuple, Union
from .parser import dataparser_mapping
from ase.io import read
import pytorch_lightning as pl


log = logging.getLogger(__name__)


class LitAtomsDataset(pl.LightningDataModule):

    def __init__(self, p_dict):
        super().__init__()
        self.data_parser = dataparser_mapping[p_dict["Data"]["type"]](p_dict)
        self.train_batch = p_dict["Data"]["trainBatch"]
        self.test_batch = p_dict["Data"]["testBatch"]
        self._train_dataloader = None
        self._test_dataloader = None
        self.stats = {}

    def setup(self, stage: Optional[str] = None):
        dataset = self.data_parser.get_dataset()
        self._trainset, self._testset = self.data_parser.split_dataset(dataset)
        # self.calculate_stats()

    @property
    def trainset(self):
        if self._trainset is None:
            self.setup()
        return self._trainset

    @property
    def testset(self):
        if self._testset is None:
            self.setup()
        return self._testset

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = self.data_parser.get_dataloader(
                self.trainset, self.train_batch, shuffle=True
            )
        return self._train_dataloader

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        if self._test_dataloader is None:
            self._test_dataloader = self.data_parser.get_dataloader(
                self.testset, self.test_batch, shuffle=False
            )
        return self._test_dataloader

    def calculate_stats(self):
        element_count = {0: []}
        energy, n_neighbor, forces = np.empty(0), np.empty(0), np.empty((0, 3))
        for i_batch, batch_data in enumerate(self.train_dataloader()):
            if i_batch % 1000 == 0:
                log.debug(f"Now {i_batch}")
            # all elemetns
            atomic_numbers = np.split(
                batch_data['atomic_number'].detach().cpu().numpy(),
                np.cumsum(batch_data['n_atoms'].detach().cpu().numpy()),
            )
            # print("!!!!!!", len(atomic_numbers), batch_data["energy_t"].detach().cpu().numpy().shape)
            for atomic_number in atomic_numbers[:-1]:
                for i, n in enumerate(
                    np.bincount(atomic_number, minlength=max(element_count.keys()) + 1)
                ):
                    if i in element_count:
                        element_count[i].append(n)
                    else:
                        element_count[i] = [0] * (len(element_count[0]) - 1) + [n]
            if "energy_t" in batch_data:
                energy = np.concatenate(
                    (energy, batch_data["energy_t"].detach().cpu().numpy())
                )
            n_neighbor = np.concatenate(
                (n_neighbor, np.bincount(batch_data["idx_i"].detach().cpu().numpy()))
            )
            if "forces_t" in batch_data:
                forces = np.concatenate(
                    (forces, batch_data["forces_t"].detach().cpu().numpy())
                )

        self.stats["n_neighbor_mean"] = float(np.mean(n_neighbor))
        if len(forces) > 0:
            self.stats["forces_std"] = float(np.std(forces))
        else:
            self.stats["forces_std"] = 1.0
        self.stats["all_elements"] = [
            int(e) for e, n in element_count.items() if np.sum(n) > 0
        ]
        log.debug("Calculating ground energy...")
        if len(energy) > 0:
            A = np.array([element_count[k] for k in self.stats["all_elements"]]).T
            self.stats["ground_energy"] = np.linalg.lstsq(A, energy, rcond=None)[
                0
            ].tolist()
        else:
            self.stats["ground_energy"] = [0.0]

    @property
    def forces_std(self):
        if "forces_std" not in self.stats:
            self.calculate_stats()
        return self.stats["forces_std"]

    @property
    def n_neighbor_mean(self):
        if "n_neighbor_mean" not in self.stats:
            self.calculate_stats()
        return self.stats["n_neighbor_mean"]

    @property
    def all_elements(self):
        if "all_elements" not in self.stats:
            self.calculate_stats()
        return self.stats["all_elements"]

    @property
    def ground_energy(self):
        if "ground_energy" not in self.stats:
            self.calculate_stats()
        return self.stats["ground_energy"]
