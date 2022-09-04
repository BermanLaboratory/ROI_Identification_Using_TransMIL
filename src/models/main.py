import argparse
from pathlib import Path
import numpy as np
import glob

from data.features_interface import FeaturesInterface
from models.architechture.TransMIL_interface import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers


from utils.utils import *
import os


from data.features import *
from torch.utils.data.sampler import SubsetRandomSampler

import yaml
from addict import Dict



#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',help = 'Path to the Configuration File' )
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    
    wandb_logger = pl_loggers.WandbLogger(name=cfg.General.run_name,project=cfg.General.project_name)
    

    #---->Data_Interface 

    dm = FeaturesInterface(cfg)
    
    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)

    callbacks = load_callbacks(cfg)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs= 10,
        gpus=cfg.General.gpus,  
        precision=cfg.General.precision,  
        check_val_every_n_epoch=2,
    )

    #---->train or test
    if cfg.General.mode == 'train':
        # print('Starting Training')
        trainer.fit(model = model, datamodule = dm)
    else:
    
        new_model = model.load_from_checkpoint(cfg.General.weights_file_path,cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    # cfg = read_yaml('/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/config/bermanlab.yaml')
    cfg = read_yaml(args['config'])

    #---->update
    cfg.config = args.config

    #---->main
    main(cfg)
 