import logging, time, yaml, os, shutil, argparse
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from hotrelax.utils import setup_seed, expand_para
from hotrelax.model import MiaoNet, LitAtomicModule
from hotrelax.layer.cutoff import *
from hotrelax.layer.embedding import AtomicEmbedding
from hotrelax.layer.radial import *
from hotrelax.data import LitAtomsDataset


# 别管Warning不Warning，只要能跑不就行
import warnings
warnings.filterwarnings(action='ignore', message='Checkpoint directory')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='The dirpath has changed from')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')

# torch.set_float32_matmul_precision("high")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", filename="log.txt", filemode="a",
)

DefaultPara = {
        "workDir": os.getcwd(),
        "seed": np.random.randint(0, 100000000),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "outputDir": os.path.join(os.getcwd(), "outDir"),
        "Data": {
            "path": os.getcwd(),
            "trainBatch": 32,
            "testBatch": 32,
            "std": "force",
            "mean": None,
            "nNeighbor": None,
            "elements": None,
            "numWorkers": 0,
            "pinMemory": False,
            "batchType": "structure",
            "meta": None,
        },
        "Model": {
            "net": "miao",
            "convMode": "node_j",
            "updateEdge": False,
            "mode": "normal",
            "bilinear": False,
            "activateFn": "silu",
            "nEmbedding": 64,
            "nLayer": 5,
            "maxRWay": 2,
            "maxMWay": 2,
            "maxOutWay": 2,
            "maxNBody": 3,
            "nHidden": 64,
            "targetWay": {0 : 'site_energy'},
            "CutoffLayer": {
                "type": "poly",
                "p": 5,
            },
            "RadialLayer": {
                "type": "besselMLP",
                "nBasis": 8,
                "nHidden": [64, 64, 64],
                "activateFn": "silu",
            },
            "Repulsion": 0,
            "Spin": False,
        },
        "Train": {
            "maxEpoch": 10000,
            "maxStep": 1000000,
            "allowMissing": False,
            "huberDelta": -1.0,
            "targetProp": ["drect_pos", "direct_cell"],
            "weight": [1.0, 1.0],
            "forceScale": 0.,
            "evalStepInterval": 50,
            "evalEpochInterval": 1,
            "logInterval": 50,
            "saveStart": 1000,
            "evalTest": True,
            "gradClip": None,
            "Optimizer": {
                "type": "Adam",
                "amsGrad": True,
                "weightDecay": 0.,
                "learningRate": 0.01,
                },
            "LrScheduler": {
                "type": "constant",
            },
            "emaDecay": 0.,
        },
    }

class SaveModelCheckpoint(ModelCheckpoint):
    """
    Saves model.pt for eval
    """
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        dirname = os.path.dirname(filepath)
        modelname = os.path.basename(filepath)[:-5]
        if trainer.is_global_zero:
            torch.save(trainer.lightning_module.model, os.path.join(dirname, f"{modelname}.pt"))
            shutil.copy(os.path.join(dirname, f"{modelname}.ckpt"), os.path.join(dirname, "best.ckpt"))
            shutil.copy(os.path.join(dirname, f"{modelname}.pt"), os.path.join(dirname, "best.pt"))

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        modelpath = filepath[:-4] + "pt"
        if trainer.is_global_zero:
            if os.path.exists(modelpath):
                os.remove(modelpath)


class LogAllLoss(pl.Callback):

    def __init__(self, properties) -> None:
        super().__init__()
        self.properties = properties
        self.train_loss = {p: [] for p in properties}
        self.train_loss['total'] = []
        self.title = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            loss_metrics = trainer.callback_metrics
            #self.train_loss['total'].append(np.sqrt(loss_metrics['train_loss'].detach().cpu().numpy()))
            self.train_loss['total'].append(loss_metrics['train_loss'].detach().cpu().numpy())
            for prop in self.properties:
                #self.train_loss[prop].append(np.sqrt(loss_metrics[f'train_{prop}'].detach().cpu().numpy()))
                self.train_loss[prop].append(loss_metrics[f'train_{prop}'].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if not self.title:
                content = f"{'epoch':^10}|{'step':^10}|{'lr':^10}|{'total':^21}"
                for prop in self.properties:
                    content += f"|{prop:^21}|{prop:^10}"
                logging.info(content)
                self.title = True
            epoch = trainer.current_epoch
            step = trainer.global_step
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            loss_metrics = trainer.callback_metrics
            train_loss = np.mean(self.train_loss['total'])
            #val_loss = np.sqrt(loss_metrics['val_loss'].detach().cpu().numpy())
            val_loss = loss_metrics['val_loss'].detach().cpu().numpy()
            content = f"{epoch:^10}|{step:^10}|{lr:^10.2e}|{train_loss:^10.4f}/{val_loss:^10.4f}"
            for prop in self.properties:
                train_prop_loss = np.mean(self.train_loss[prop])
                #val_prop_loss = np.sqrt(loss_metrics[f'val_{prop}'].detach().cpu().numpy())
                val_prop_loss = loss_metrics[f'val_{prop}'].detach().cpu().numpy()
                val_prop_mae = loss_metrics[f"mae_{prop}"].detach().cpu().numpy()
                content += f"|{train_prop_loss:^10.4f}/{val_prop_loss:^10.4f}|{val_prop_mae:^10.4f}"
            logging.info(content)
            for prop in self.train_loss:
                self.train_loss[prop] = []

def update_dict(d1, d2):
    for key in d2:
        if key in d1 and isinstance(d1[key], dict):
            update_dict(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def get_stats(data_dict):
    if type(data_dict["nNeighbor"]) is float:
        n_neighbor = data_dict["nNeighbor"]

    if isinstance(data_dict["elements"], list):
        elements = data_dict["elements"]

    mean_pos = data_dict["mean_pos"]
    mean_cell = data_dict["mean_cell"]
    std_pos = data_dict["std_pos"]
    std_cell = data_dict["std_cell"]

    logging.info(f"n_neighbor   : {n_neighbor}")
    logging.info(f"all_elements : {elements}")
    logging.info(f"mean_pos     : {mean_pos}")
    logging.info(f"mean_cell    : {mean_cell}")
    logging.info(f"std_pos      : {std_pos}")
    logging.info(f"std_cell     : {std_cell}")
    return mean_pos, mean_cell, std_pos, std_cell, n_neighbor, elements


def get_cutoff(p_dict):
    cutoff = p_dict['cutoff']
    cut_dict = p_dict['Model']['CutoffLayer']
    if cut_dict['type'] == "cos":
        return CosineCutoff(cutoff=cutoff)
    elif cut_dict['type'] == "cos2":
        return SmoothCosineCutoff(cutoff=cutoff, cutoff_smooth=cut_dict['smoothCutoff'])
    elif cut_dict['type'] == "poly":
        return PolynomialCutoff(cutoff=cutoff, p=cut_dict['p'])
    else:
        raise Exception("Unsupported cutoff type: {}, please choose from cos, cos2, and poly!".format(cut_dict['type']))
    

def get_radial(p_dict, cutoff_fn):
    cutoff = p_dict['cutoff']
    radial_dict = p_dict['Model']['RadialLayer']
    if "bessel" in radial_dict['type']:
        radial_fn = BesselPoly(r_max=cutoff, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    elif "chebyshev" in radial_dict['type']:
        if "minDist" in radial_dict:
            r_min = radial_dict['minDist']
        else:
            r_min = 0.5
            logging.warning("You are using chebyshev poly as basis function, but does not given 'minDist', "
                        "this may cause some problems!")
        radial_fn = ChebyshevPoly(r_max=cutoff, r_min=r_min, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    else:
        raise Exception("Unsupported radial type: {}!".format(radial_dict['type']))
    if "MLP" in radial_dict['type']:
        if radial_dict["activateFn"] == "silu":
            activate_fn = nn.SiLU()
        elif radial_dict["activateFn"] == "relu":
            activate_fn = nn.ReLU()
        else:
            raise Exception("Unsupported activate function in radial type: {}!".format(radial_dict["activateFn"]))
        return MLPPoly(n_hidden=radial_dict['nHidden'], radial_fn=radial_fn, activate_fn=activate_fn)
    else:
        return radial_fn


def get_model(p_dict, elements, mean_pos, mean_cell, std_pos, std_cell, n_neighbor):
    model_dict = p_dict['Model']
    target = p_dict['Train']['targetProp']
    target_way = {}
    if "direct_pos" in target:
        target_way["direct_pos"] = 1
    if "direct_cell" in target:
        target_way["direct_cell"] = 1
    cut_fn = get_cutoff(p_dict)
    emb = AtomicEmbedding(elements, model_dict['nEmbedding'])  # only support atomic embedding now
    radial_fn = get_radial(p_dict, cut_fn)
    max_r_way = expand_para(model_dict['maxRWay'], model_dict['nLayer'])
    max_out_way = expand_para(model_dict['maxOutWay'], model_dict['nLayer'])
    max_out_way[-1] = max(target_way.values())
    output_dim = expand_para(model_dict['nHidden'], model_dict['nLayer'])

    if model_dict['net'] == 'miao':
        model = MiaoNet(embedding_layer=emb,
                        radial_fn=radial_fn,
                        n_layers=model_dict['nLayer'],
                        max_r_way=max_r_way,
                        max_out_way=max_out_way,
                        output_dim=output_dim,
                        activate_fn=model_dict['activateFn'],
                        target_way=target_way,
                        mean_pos=mean_pos,
                        mean_cell=mean_cell,
                        std_pos=std_pos,
                        std_cell=std_cell,
                        norm_factor=n_neighbor,
                        bilinear=model_dict['bilinear'],
                        conv_mode=model_dict['convMode'],
                        update_edge=model_dict['updateEdge'],
                        ).to(p_dict['device'])

    assert isinstance(model_dict['Repulsion'], int), "Repulsion should be int!"

    return model


def main(*args, input_file='input.yaml', load_model=None, load_checkpoint=None, **kwargs):
    # Default values
    p_dict = DefaultPara
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))

    if os.path.exists(p_dict["outputDir"]):
        i = 1
        while os.path.exists(f"{p_dict['outputDir']}{i}"):
            i += 1
        shutil.move(p_dict["outputDir"], f"{p_dict['outputDir']}{i}")
        os.system(f"cp log.txt input.yaml allpara.yaml {p_dict['outputDir']}{i}")
    os.makedirs(p_dict["outputDir"])

    with open("allpara.yaml", "w") as f:
        yaml.dump(p_dict, f)

    setup_seed(p_dict["seed"])
    logging.info("Using seed {}".format(p_dict["seed"]))

    logging.info(f"Preparing data...")
    dataset = LitAtomsDataset(p_dict)
    dataset.setup()
    mean_pos, mean_cell, std_pos, std_cell, n_neighbor, elements = get_stats(p_dict["Data"])
    
    if load_model is not None and 'ckpt' not in load_model:
        logging.info(f"Load model from {load_model}")
        model = torch.load(load_model, weights_only=False)
    else:
        model = get_model(
            p_dict,
            elements,
            mean_pos,
            mean_cell,
            std_pos,
            std_cell,
            n_neighbor,
        )
        model.register_buffer('all_elements', torch.tensor(elements, dtype=torch.long))
        model.register_buffer('cutoff', torch.tensor(p_dict["cutoff"], dtype=torch.float64))

    if load_model is not None and 'ckpt' in load_model:
        lit_model = LitAtomicModule.load_from_checkpoint(load_model, model=model, p_dict=p_dict)
    else:
        lit_model = LitAtomicModule(model=model, p_dict=p_dict)

    if load_checkpoint is not None:
        ckpt = torch.load(load_checkpoint)
        p_dict["Train"]["maxEpoch"] += ckpt['epoch']
        p_dict["Train"]["maxStep"] += ckpt['global_step']

    logger = pl.loggers.TensorBoardLogger(save_dir=p_dict["outputDir"])
    callbacks = [
        SaveModelCheckpoint(
            dirpath=p_dict["outputDir"],
            filename='{epoch}-{step}-{val_loss:.4f}',
            save_top_k=5,
            monitor="val_loss"
        ),
        LearningRateMonitor(),
        LogAllLoss(p_dict["Train"]['targetProp']),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir='.',
        max_epochs=p_dict["Train"]["maxEpoch"],
        max_steps=p_dict["Train"]["maxStep"],
        enable_progress_bar=False,
        log_every_n_steps=p_dict["Train"]["logInterval"],
        val_check_interval=p_dict["Train"]["evalStepInterval"],
        check_val_every_n_epoch=p_dict["Train"]["evalEpochInterval"],
        gradient_clip_val=p_dict["Train"]["gradClip"],
        )
    
    if load_checkpoint is not None:
        logging.info(f"Load checkpoints from {load_checkpoint}")
        trainer.fit(lit_model, datamodule=dataset, ckpt_path=load_checkpoint)
    else:
        trainer.fit(lit_model, datamodule=dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.yaml", help="input file path")
    args = parser.parse_args()
    main(input_file=args.input_file)
