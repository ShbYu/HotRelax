from typing import Optional, List, Union
from .utils import dataset_mapping
from .asedata import ASEData
from .ptdata import PtData
from .data_interface import LitAtomsDataset

__all__ = ["ASEData", "PtData", "get_dataset"]


def get_dataset(
    cutoff: float,
    datatype: str,
    datapath: Union[None, str, List[str]],
    properties: Optional[List[str]] = None,
    spin: bool = False,
    indices: List[int] = None,
    add_feat: bool = False,
    feat_json: Optional[Union[str, List[str]]] = None,
    use_cycle: bool = False,
):

    kwargs = {
        "frames": datapath,  # ASEData
        "ptdata": datapath,  # PtData
        "datapath": datapath,
        "cutoff": cutoff,
        "properties": properties,
        "spin": spin,
        "indices": indices,
        "add_feat": add_feat,
        "feat_json": feat_json,
        "use_cycle": use_cycle,
    }

    return dataset_mapping[datatype](**kwargs)
