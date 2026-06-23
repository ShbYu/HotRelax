from ase.io import read
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from hotrelax.data import get_dataset
from hotrelax.data.utils import atoms_collate_fn
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.trajectory import Trajectory
import argparse
import time


def eval(model, data_loader, properties, device, output="results.txt"):
    with open(output, 'w') as fileobj:
        fileobj.write('cif_id\tmae_pos_dummy\tmae_pos_pred\tmae_cell_dummy\tmae_cell_pred\tmatch_rate\ttime\n')
    
    pred_traj = Trajectory(f"pred_test.traj", 'a')
    for batch_data in data_loader:
        batch_data = {key: value.to(device) for key, value in batch_data.items()}
        start_time = time.time()
        with torch.no_grad():
            model(batch_data, properties, create_graph=False)

        pos_unrelax = batch_data["pos_u"]
        pred_pos = batch_data["direct_pos_p"][-1] + pos_unrelax
        cell_unrelax = batch_data["cell_u"].squeeze()
        pred_cell = batch_data["direct_cell_p"][-1].squeeze() + cell_unrelax
        cell_uinv = torch.linalg.inv(cell_unrelax)
        cell_relax = batch_data["direct_cell_t"].squeeze() + cell_unrelax

        pos_relax = (batch_data["direct_pos_t"] + pos_unrelax) @ cell_uinv @ cell_relax
        pos_pred = pred_pos @ cell_uinv @ pred_cell

        end_time = time.time()
        compute_time = end_time - start_time

        name = batch_data["ats_index"].cpu().numpy()[0]
        volume_relax = torch.linalg.det(cell_relax)
        volume_unrelax = torch.linalg.det(cell_unrelax)
        pred_volume = torch.linalg.det(pred_cell)

        atoms_pred = Atoms(positions=pos_pred.cpu().numpy(), numbers=batch_data["atomic_number"].cpu().numpy(), cell=pred_cell.cpu().numpy(), pbc=np.array([1, 1, 1]))
        atoms_r = Atoms(positions=pos_relax.cpu().numpy(), numbers=batch_data["atomic_number"].cpu().numpy(), cell=cell_relax.cpu().numpy(), pbc=np.array([1, 1, 1]))
        matcher = StructureMatcher(stol=0.3, ltol=0.2)
        match_rate = int(matcher.fit(AseAtomsAdaptor.get_structure(atoms_pred), AseAtomsAdaptor.get_structure(atoms_r)))

        atoms_pred.info["index"] = int(name)
        pred_traj.write(atoms_pred, append=True)

        mae_pos_dummy = (pos_relax - pos_unrelax).abs().mean().item()
        mae_pos_pred = (pos_relax - pos_pred).abs().mean().item()
        mae_cell_dummy = (cell_relax - cell_unrelax).abs().mean().item()
        mae_cell_pred = (cell_relax - pred_cell).abs().mean().item()

        with open(output, 'a') as fileobj:
            content = str(name)+'\t'+str(mae_pos_dummy)+'\t'
            content += str(mae_pos_pred)+'\t'+str(mae_cell_dummy)+'\t'+str(mae_cell_pred)+'\t'
            content += str(match_rate)+'\t'+str(compute_time)+'\n'
            fileobj.write(content)
    pred_traj.close()

    return None


def main(*args, model_file='./outDir/best.pt', indices=None, input_file="input.yaml", 
         spin=False, pin_memory=True, **kwargs):
    if indices is not None:
        indices = np.loadtxt(indices, dtype=int)

    with open(input_file) as f:
        eval_dict = yaml.load(f, Loader=yaml.FullLoader)
    device = eval_dict["device"]
    cutoff = float(eval_dict["cutoff"])
    datatype = eval_dict["Data"]["type"]
    datapath = eval_dict["Data"]["evalSet"]
    properties = eval_dict["Train"]["targetProp"]
    batchsize = eval_dict["Data"]["evalBatch"]
    num_workers = eval_dict["Data"]["numWorkers"]

    model = torch.load(model_file, map_location=device, weights_only=False)
    model.eval()

    dataset = get_dataset(cutoff=cutoff,
                           datatype=datatype,
                           datapath=datapath,
                           properties=properties,
                           spin=spin,
                           indices=indices,
                           add_feat=eval_dict["Data"].get("addFeat", False),
                           feat_json=eval_dict["Data"].get("featJson"),
                           use_cycle=eval_dict["Data"].get("useCycle", False))

    data_loader = DataLoader(dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             collate_fn=atoms_collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    eval(model, data_loader, properties, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.yaml", help="input file path")
    parser.add_argument("--model_file", type=str, default="./outDir/best.pt", help="trained model path")
    args = parser.parse_args()
    main(model_file=args.model_file, input_file=args.input_file)
