import torch
import torch.nn.functional as F
from typing import Dict, Callable
from .utils import expand_to


class Loss:

    atom_prop = ["forces", "spin_torques"]
    structure_prop = ["energy", "virial", "dipole", "polarizability"]
    list_prop = ["direct_pos", "direct_cell"]

    def __init__(self,
                 weight  : Dict[str, float]={"energy": 1.0, "forces": 1.0},
                 loss_fn : Callable=F.mse_loss,
                 ) -> None:
        self.weight = weight
        self.loss_fn = loss_fn

    def get_loss(self,
                 batch_data : Dict[str, torch.Tensor],
                 verbose    : bool=False,
                 log_mae    : bool=False,
                ):
        loss = {}
        mae = {}
        total_loss = 0.
        for prop in self.weight:
            if prop in self.atom_prop:
                loss[prop] = self.atom_prop_loss(batch_data, prop)
            elif prop in self.structure_prop:
                loss[prop] = self.structure_prop_loss(batch_data, prop)
            elif prop in self.list_prop:
                loss[prop], mae[prop] = self.list_prop_loss(batch_data, prop, log_mae)
            total_loss += loss[prop] * self.weight[prop]

        if verbose and log_mae:
            return total_loss, loss, mae
        elif verbose:
            return total_loss, loss
        return total_loss

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        return self.loss_fn(batch_data[f'{prop}_p'], batch_data[f'{prop}_t'])

    def structure_prop_loss(self,
                            batch_data : Dict[str, torch.Tensor],
                            prop       : str,
                            ) -> torch.Tensor:
        n_atoms = expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        return self.loss_fn(batch_data[f'{prop}_p'] / n_atoms,
                            batch_data[f'{prop}_t'] / n_atoms)
    
    def list_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       log_mae    : bool=False,
                       ) -> torch.Tensor:
        sum_loss = []
        for prop_single in batch_data[f'{prop}_p']:
            sum_loss.append(self.loss_fn(prop_single, batch_data[f'{prop}_t']))
        if log_mae:
            return sum(sum_loss), sum_loss[-1]
        return sum(sum_loss), None


class ForceScaledLoss(Loss):

    def __init__(self,
                 weight  : Dict[str, float]={"energy": 1.0, "forces": 1.0},
                 loss_fn : Callable=F.mse_loss,
                 scaled  : float=1.0,
                 ) -> None:
        super().__init__(weight, loss_fn)
        self.scaled = scaled

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        if 'prop' != 'forces':
            return super().atom_prop_loss(batch_data, prop)
        reweight = self.scaled / (torch.norm(batch_data['force_t'], dim=1) + self.scaled)
        return self.loss_fn(batch_data[f'forces_p'] * reweight, batch_data[f'forces_t'] * reweight)


class MissingValueLoss(Loss):

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        # idx = batch_data[f'{prop}_weight'][batch_data['batch']]
        # if not torch.any(idx):
        #     return torch.tensor(0.)
        # if torch.all(idx):
        #     return super().atom_prop_loss(batch_data, prop)
        # return self.loss_fn(batch_data[f'{prop}_p'][idx], batch_data[f'{prop}_t'][idx])
        return self.loss_fn(batch_data[f'{prop}_p'] * batch_data[f'{prop}_weight'],
                            batch_data[f'{prop}_t'] * batch_data[f'{prop}_weight'])

    def structure_prop_loss(self,
                            batch_data : Dict[str, torch.Tensor],
                            prop       : str,
                            ) -> torch.Tensor:
        # idx = batch_data[f'{prop}_weight']
        # if not torch.any(idx):
        #     return torch.tensor(0.)
        # if torch.all(idx):
        #     return super().structure_prop_loss(batch_data, prop)
        # n_atoms = expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        # return self.loss_fn(batch_data[f'{prop}_p'][idx] / n_atoms, 
        #                     batch_data[f'{prop}_t'][idx] / n_atoms)
        weight = batch_data[f'{prop}_weight'] / expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        return self.loss_fn(batch_data[f'{prop}_p'] * weight,
                            batch_data[f'{prop}_t'] * weight)


class MACEHuberLoss(Loss):
    """ 
    Only support energy, forces, and virials, use the same setting as universal model of MACE:
    Batatia, I. et al. A foundation model for atomistic materials chemistry. 
    Preprint at https://doi.org/10.48550/arXiv.2401.00096 (2024).
    """

    def __init__(self,
                 weight  : Dict[str, float]={"energy": 1.0, "forces": 10.0},
                 huber_delta  : float=0.01,
                 ) -> None:
        super().__init__(weight, loss_fn=F.huber_loss)
        self.huber_delta = huber_delta

    def get_loss(self,
                 batch_data : Dict[str, torch.Tensor],
                 verbose    : bool=False):
        loss = {}
        total_loss = 0.
        for prop in self.weight:
            if prop == "energy":
                loss["energy"] = self.loss_fn(
                    batch_data['energy_p'] / batch_data['n_atoms'], 
                    batch_data['energy_t'] / batch_data['n_atoms'],
                    reduction="mean", delta=self.huber_delta
                    )
            elif prop == "forces":
                loss["forces"] = self.conditional_huber_forces(
                    batch_data['forces_p'], batch_data['forces_t'],
                    )
            elif prop == "virial":
                loss["virial"] = self.loss_fn(
                    batch_data['virial_p'] / batch_data['n_atoms'], 
                    batch_data['virial_t'] / batch_data['n_atoms'],
                    reduction="mean", delta=self.huber_delta
                    )
            total_loss += loss[prop] * self.weight[prop]
        if verbose:
            return total_loss, loss
        return total_loss

    def conditional_huber_forces(self,
                                 pred_forces: torch.Tensor,
                                 ref_forces: torch.Tensor, 
                                 ) -> torch.Tensor:
        factors = self.huber_delta * torch.tensor([1.0, 0.7, 0.4, 0.1])
        c1 = torch.norm(ref_forces, dim=-1) < 100
        c2 = (torch.norm(ref_forces, dim=-1) >= 100) & (
            torch.norm(ref_forces, dim=-1) < 200
        )
        c3 = (torch.norm(ref_forces, dim=-1) >= 200) & (
            torch.norm(ref_forces, dim=-1) < 300
        )
        c4 = ~(c1 | c2 | c3)

        se = torch.zeros_like(pred_forces)

        se[c1] = self.loss_fn(ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0])
        se[c2] = self.loss_fn(ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1])
        se[c3] = self.loss_fn(ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2])
        se[c4] = self.loss_fn(ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3])

        return torch.mean(se)
