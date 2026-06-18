from .utils import AtomsDataset, register_dataset
from typing import List, Optional, Union
from ase import Atoms
from ase.io import read


@register_dataset("ase")
class ASEData(AtomsDataset):

    def __init__(
        self,
        frames: Union[List[Atoms], str, None] = None,
        indices: Optional[List[int]] = None,
        properties: Optional[List[str]] = ["energy", "forces"],
        spin: bool = False,
        cutoff: float = 6.0,
        max_neigh: int = 20,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        if frames is None:
            self.frames = []
        elif isinstance(frames, str):
            self.frames = read(frames, index=":")
        else:
            self.frames = frames
        self.properties = properties
        self.spin = spin
        self.max_neigh = max_neigh

    def __len__(self):
        if self.indices is None:
            return len(self.frames)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        data = self.atoms_to_data(
            self.frames[idx],
            properties=self.properties,
            cutoff=self.cutoff,
            spin=self.spin,
            max_neigh=self.max_neigh,
        )
        return data

    def extend(self, frames: Union[List[Atoms], str]):
        if isinstance(frames, str):
            frames = read(frames, index=":")
            frames = read(frames, index=":")
        self.frames.extend(frames)