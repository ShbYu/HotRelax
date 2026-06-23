import os
import sys
from ase.io.trajectory import Trajectory
from ase.io import read
import numpy as np
from crystgraph import HotRelax
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    data_root = sys.argv[1]
    data_path = sys.argv[2]

    train_id_prop_file = os.path.join(data_root, 'train.txt')
    val_id_prop_file = os.path.join(data_root, 'valid.txt')
    test_id_prop_file = os.path.join(data_root, 'test.txt')

    with open(train_id_prop_file) as f:
        reader = f.readlines()
        train_id_prop_data = [row.split(',')[0] for row in reader if row.split(',')[0].split('_')[-1] == "relaxed"]
    with open(val_id_prop_file) as f:
        reader = f.readlines()
        val_id_prop_data = [row.split(',')[0] for row in reader if row.split(',')[0].split('_')[-1] == "relaxed"]
    with open(test_id_prop_file) as f:
        reader = f.readlines()
        test_id_prop_data = [row.split(',')[0] for row in reader if row.split(',')[0].split('_')[-1] == "relaxed"]


    for dataset in ['train', 'val', 'test']:
        if dataset == 'train':
            id_prop_data = train_id_prop_data
            # traj_path_relax = os.path.join(data_path,'train', "traj_relax.traj")
            traj_path_unrelax = os.path.join(data_path,'train', "data.traj")
        elif dataset == 'val':
            id_prop_data = val_id_prop_data
            # traj_path_relax = os.path.join(data_path,'val', "traj_relax.traj")
            traj_path_unrelax = os.path.join(data_path,'val', "data.traj")
        elif dataset == 'test':
            id_prop_data = test_id_prop_data
            # traj_path_relax = os.path.join(data_path,'test', "traj_relax.traj")
            traj_path_unrelax = os.path.join(data_path,'test', "data.traj")

        # traj_relax = Trajectory(traj_path_relax, mode='a')
        traj_unrelax = Trajectory(traj_path_unrelax, mode='a')
        for index, cif_id in enumerate(id_prop_data):
            unrelaxed_path = os.path.join(data_root, 'CIF', cif_id.replace('relaxed', 'unrelaxed') + '.cif') 
            relaxed_path = os.path.join(data_root, 'CIF', cif_id + '.cif')

            atoms_u = read(unrelaxed_path)
            atoms_r = read(relaxed_path)
            atoms_u.info["cif_id"] = cif_id
            atoms_u.info["index"] = index

            hotR = HotRelax(atoms_u, atoms_r)
            hotR.get_relaxed_offset()
            atoms_u.info["direct_pos"] = hotR.pos_relax - hotR.pos_unrelax
            atoms_u.info["direct_cell"] = hotR.cell_relax - hotR.cell_unrelax

            traj_unrelax.write(atoms_u, append=True)
            # traj_relax.write(atoms_r, append=True)
        
        # traj_relax.close()
        traj_unrelax.close()
